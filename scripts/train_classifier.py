import argparse
import os
import shutil

import numpy as np
import torch
import torch.utils.tensorboard
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm
import wandb

import sys
# print(os.getcwd())
sys.path.append(os.getcwd())

import utils.misc as misc
import utils.train as utils_train
import utils.transforms as trans
from datasets.pl_data import FOLLOW_BATCH
from datasets.pl_pair_dataset import get_decomp_dataset
from models.decompdiff import DecompScorePosNet3D
from models.classifier import Classifier

torch.multiprocessing.set_sharing_strategy('file_system')


load_history = False


def get_auroc(y_true, y_pred, feat_mode):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    avg_auroc = 0.
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
        mapping = {
            'basic': trans.MAP_INDEX_TO_ATOM_TYPE_ONLY,
            'add_aromatic': trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC,
            'full': trans.MAP_INDEX_TO_ATOM_TYPE_FULL
        }
        print(f'atom: {mapping[feat_mode][c]} \t auc roc: {auroc:.4f}')
    return avg_auroc / len(y_true)


def get_bond_auroc(y_true, y_pred):
    avg_auroc = 0.
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
        bond_type = {
            0: 'none',
            1: 'single',
            2: 'double',
            3: 'triple',
            4: 'aromatic',
        }
        print(f'bond: {bond_type[c]} \t auc roc: {auroc:.4f}')
    return avg_auroc / len(y_true)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_classifier_full')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--train_report_iter', type=int, default=200)
    args = parser.parse_args()

    # Load configs
    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)

    # Logging
    log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    logger = misc.get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    shutil.copytree('./models', os.path.join(log_dir, 'models'))

    # init wandb
    wandb.init(
        # Set the project where this run will be logged
        project="guiding-decompdiff",
        name=f"train-classifier-decompdiff-with-test",
        # Track hyperparameters and run metadata
        config={
            "batch_size": config.train.batch_size,
        }
        )


    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode,
                                                  config.model.prior_types,)
    decomp_indicator = trans.AddDecompIndicator(
        max_num_arms=config.data.transform.max_num_arms,
        global_prior_index=ligand_featurizer.ligand_feature_dim,
        add_ord_feat=getattr(config.data.transform, 'add_ord_feat', True),
    )
    transform_list = [
        trans.ComputeLigandAtomNoiseDist(version=config.data.get('prior_mode', 'subpocket')),
        protein_featurizer,
        ligand_featurizer,
        decomp_indicator
    ]
    if config.data.transform.random_rot:
        transform_list.append(trans.RandomRotation())
    if getattr(config.model, 'bond_diffusion', False):
        transform_list.append(
            trans.FeaturizeLigandBond(mode=config.data.transform.ligand_bond_mode, set_bond_type=True)
        )
    transform = Compose(transform_list)

    # Datasets and loaders
    logger.info('Loading dataset...')
    dataset, subsets = get_decomp_dataset(
        config=config.data,
        transform=transform,
    )
    train_set, val_set = subsets['train'], subsets['test']
    logger.info(f'Training: {len(train_set)} Validation: {len(val_set)} Total: {len(dataset)}')

    collate_exclude_keys = ['ligand_nbh_list', 'pocket_atom_masks', 'pocket_prior_masks', 'scaffold_prior', 'arms_prior']
    # TODO: change the training loader back to a finite loader
    train_iterator = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys,
    )
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False,
                            follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys)

    # Model
    logger.info('Building model...')

    # keep the dim the same with decompdiff
    protein_feature_dim = sum([getattr(t, 'protein_feature_dim', 0) for t in transform_list])
    ligand_feature_dim = sum([getattr(t, 'ligand_feature_dim', 0) for t in transform_list])
    if getattr(config.model, 'add_valency_features', False):
        ligand_feature_dim += 3

    # define Graphformer classifer
    model = Classifier(
        config.model,
        protein_atom_feature_dim=protein_feature_dim,
        ligand_atom_feature_dim=ligand_feature_dim,
        device = args.device,
        prior_atom_types=ligand_featurizer.atom_types_prob, # TODO: does this gona be affect the training?
        prior_bond_types=ligand_featurizer.bond_types_prob, # TODO: does this gona be affect the training process?
        num_classes=ligand_featurizer.ligand_feature_dim
    ).to(args.device)

    # load history model
    if load_history == True:
        print("loading history model")
        state_dict_mod = torch.load(args.ckpt_dir+"/ckpt.pt",map_location = torch.device('cuda'))
        model.load_state_dict(state_dict_mod['model'])
        start_ep_load = state_dict_mod['iteration']
        del state_dict_mod
        print("finish loading model")


    print(f'protein feature dim: {protein_feature_dim} ligand feature dim: {ligand_feature_dim}')
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

    # Init Optimizer
    optimizer = utils_train.get_optimizer(config.train.optimizer, model)
    # load history optimizer
    if load_history == True:
        print("loading history optimizer")
        state_dict_opt = torch.load(args.ckpt_dir+"/ckpt.pt",map_location = torch.device('cuda'))
        optimizer.load_state_dict(state_dict_opt['optimizer'])
        del state_dict_opt

    # Init Scheduler
    scheduler = utils_train.get_scheduler(config.train.scheduler, optimizer)

    # load history scheduler
    if load_history == True:
        print("loading history scheduler")
        state_dict_opt = torch.load(args.ckpt_dir+"/ckpt.pt",map_location = torch.device('cuda'))
        scheduler.load_state_dict(state_dict_opt['scheduler'])
        del state_dict_opt

    # Init loss
    loss = utils_train.get_loss(config.train.loss)

    # define start epoch
    start_ep = 1
    if load_history == True:
        start_ep = start_ep_load + 1
    logger.info(f'start at epochs {start_ep}')
    logger.info(f'save ckpt every {config.train.ckpt_freq} epochs')

    try:
        best_val_loss = None
        train_loss_info = []
        val_loss_info = []
        for epoch in range(start_ep, start_ep + config.train.max_iters):

            # ##### Train ######
            print(f"training @ epoch {epoch}")
            model.train()

            batch_bar = tqdm(total=len(train_iterator), dynamic_ncols=True, leave=False, position=0, desc='Train')
            train_losses = 0
            for i, batch in tqdm(enumerate(train_iterator)):
                batch = batch.to(args.device)
                label = batch.y.reshape(-1,1) # (B,1)

                optimizer.zero_grad()

                results = model(
                    protein_pos=batch.protein_pos,
                    protein_v=batch.protein_atom_feature.float(),
                    batch_protein=batch.protein_element_batch,
                    
                    ligand_pos=batch.ligand_pos,
                    ligand_v=batch.ligand_atom_feature_full,
                    batch_ligand=batch.ligand_element_batch
                )

                # loss function
                weighting_mask = utils_train.create_mask(label,
                                                        Lcutoff=config.train.Lcutoff,
                                                        Rcutoff=config.train.Rcutoff,
                                                        weight=config.train.loss_weight)

                res_loss = loss(results*weighting_mask, label*weighting_mask)
                res_loss.backward()

                # TODO: need to see if needed clip grad norm
                optimizer.step()
                train_losses += res_loss
                batch_bar.set_postfix(
                    loss="{:.04f}".format(float(train_losses / (i + 1))),
                    lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))

                
                wandb.log({"train_loss": float(train_losses / (i + 1)),
                           "learning rate":float(optimizer.param_groups[0]['lr'])
                        })

                batch_bar.update()
                train_loss_info.append(((train_losses / (i + 1)).detach().cpu(), epoch))

            logger.info(
                    '[Train] epoch %d | Loss %.6f  | Lr: %.6f ' % (
                        epoch, float(train_losses / (i + 1)), optimizer.param_groups[0]['lr']
                    )
            )

            # del med variable to save mem
            del batch, label, res_loss
            torch.cuda.empty_cache()

            # if it % args.train_report_iter == 0:
            #     logger.info(
            #         '[Train] Iter %d | Loss %.6f (pos %.6f | v %.6f) | Lr: %.6f | Grad Norm: %.6f' % (
            #             it, loss, loss_pos, loss_v, optimizer.param_groups[0]['lr'], orig_grad_norm
            #         )
            #     )
            #     for k, v in results.items():
            #         if torch.is_tensor(v) and v.squeeze().ndim == 0:
            #             writer.add_scalar(f'train/{k}', v, it)
            #     writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
            #     writer.add_scalar('train/grad', orig_grad_norm, it)
            #     writer.flush()

            # ##### Val ######

            # if epoch % config.train.val_freq == 0 or epoch == config.train.max_iters:

            # print(f"Validation @ epoch {epoch}")
            # model.eval()
            # batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, leave=False, position=0, desc='Val')
            # val_losses = 0
            # with torch.no_grad():
            #     for i, batch in enumerate(val_loader):
            #         batch = batch.to(args.device)
            #         label = batch.y.reshape(-1, 1)  # (B,1)
            #         if debug:
            #             print("before entering val a cycle")
            #             print("protein_pos", batch.protein_pos.dtype)
            #             print("protein_v", batch.protein_atom_feature.float().dtype)
            #             print("batch_protein", batch.protein_element_batch.dtype)
            #             print("ligand_pos", batch.ligand_pos.dtype)
            #             print("ligand_v", batch.ligand_atom_feature_full.dtype)
            #             print("batch_ligand", batch.ligand_element_batch.dtype)

            #         results = model(
            #             protein_pos=batch.protein_pos,
            #             protein_v=batch.protein_atom_feature.float(),
            #             batch_protein=batch.protein_element_batch,

            #             ligand_pos=batch.ligand_pos,
            #             ligand_v=batch.ligand_atom_feature_full,
            #             batch_ligand=batch.ligand_element_batch
            #         )

            #         # TODO: place your loss function
            #         weighting_mask = utils_train.create_mask(label,
            #                                                  Lcutoff=config.train.Lcutoff,
            #                                                  Rcutoff=config.train.Rcutoff,
            #                                                  weight=config.train.loss_weight)
            #         val_loss = loss(results*weighting_mask, label*weighting_mask)
            #         val_loss_info.append((val_loss.detach().cpu(), epoch))
            #         val_losses += val_loss
            #         batch_bar.set_postfix(
            #             loss="{:.04f}".format(float(val_losses / (i + 1))),
            #             lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))

            #         batch_bar.update()

            #     batch_bar.close()

            #     # del med variable to save mem
            #     del batch, label, val_loss
            #     torch.cuda.empty_cache()

            #     if best_val_loss is None or val_losses <= best_val_loss:
            #         logger.info(f'[Validate] Best val loss achieved: {val_losses:.6f}')
            #         best_loss, best_iter = val_losses, epoch
            #         # update best val loss
            #         best_val_loss = val_losses

            #         ckpt_path = os.path.join(ckpt_dir, '%d.pt' % epoch)
            #         torch.save({
            #             'config': config,
            #             'model': model.state_dict(),
            #             'optimizer': optimizer.state_dict(),
            #             'scheduler': scheduler.state_dict(),
            #             'iteration': epoch,
            #         }, ckpt_path)
            #     else:
            #         logger.info(f'[Validate] Val loss is not improved. '
            #                     f'Best val loss: {best_loss:.6f} at iter {best_iter}')
            # step your scheduler
            scheduler.step(float(train_losses / (i + 1)))

            # FIXME: save ckpt every 5 epochs, for larger iter, disable this
            if epoch % config.train.ckpt_freq == 0:
                logger.info("saving ckpt")
                ckpt_path = os.path.join(ckpt_dir,'every_'+str(config.train.ckpt_freq)+'ep_not_the_best_%d.pt' % epoch)
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': epoch,
                }, ckpt_path)

        # after training, save loss list
        torch.save(train_loss_info, os.path.join(ckpt_dir, 'train_losses.pt'))
        # torch.save(val_loss_info, os.path.join(ckpt_dir, 'val_losses.pt'))

        # save the final epoch
        ckpt_path = os.path.join(ckpt_dir, 'every_ep_not_the_best_%d.pt' % epoch)
        torch.save({
            'config': config,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'iteration': epoch,
        }, ckpt_path)

    except KeyboardInterrupt:
        logger.info('Terminating...')