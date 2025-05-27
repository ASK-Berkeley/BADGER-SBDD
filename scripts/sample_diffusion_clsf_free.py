import argparse
import os
import shutil
import time
import sys
sys.path.append(os.getcwd())

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm
import wandb

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model_clsf_free_guide import ScorePosNet3D, log_sample_categorical
from utils.evaluation import atom_num
from models.classifier_t_conditioned import Classifier


debug = False

def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    all_step_v = [[] for _ in range(n_data)]
    for v in ligand_v_traj:  # step_i
        v_array = v.cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
    all_step_v = [np.stack(step_v) for step_v in all_step_v]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v


def sample_diffusion_ligand(model, data, num_samples, batch_size=16, device='cuda:0',
                            num_steps=None, pos_only=False, center_pos_mode='protein',
                            sample_num_atoms='prior',context=None, s=0, s_v=0, classifier=None, enable_wandb=False,clip=1e6,norm_type="l2",condition=None,w=0.):
    all_pred_pos, all_pred_v = [], []
    all_pred_pos_traj, all_pred_v_traj = [], []
    all_pred_v0_traj, all_pred_vt_traj = [], []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))
    current_i = 0
    with torch.no_grad():
        for i in tqdm(range(num_batch)):
            # try:
            n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)
            batch = Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)
            if debug:
                print("show batch", batch.protein_filename)
            t1 = time.time()
            # with torch.no_grad():
            batch_protein = batch.protein_element_batch
            if sample_num_atoms == 'prior':
                pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy())
                ligand_num_atoms = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_data)]
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'range':
                ligand_num_atoms = list(range(current_i + 1, current_i + n_data + 1))
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'ref':
                batch_ligand = batch.ligand_element_batch
                ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
            else:
                raise ValueError

            # init ligand pos
            center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
            batch_center_pos = center_pos[batch_ligand]
            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)

            # init ligand v
            if pos_only:
                init_ligand_v = batch.ligand_atom_feature_full
            else:
                uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
                init_ligand_v = log_sample_categorical(uniform_logits)
            # print("condition",condition)
            r = model.sample_diffusion(
                protein_pos=batch.protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch_protein,

                init_ligand_pos=init_ligand_pos,
                init_ligand_v=init_ligand_v,
                batch_ligand=batch_ligand,
                num_steps=num_steps,
                pos_only=pos_only,
                center_pos_mode=center_pos_mode,

                context=context, # Yue Jian: the scalar condition
                s=s, # guidance strength
                s_v=s_v,
                classifier=classifier,
                clip=float(clip),

                enable_wandb=enable_wandb,
                norm_type=norm_type,
                condition=condition,
                w=w
            )
            ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
            ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']
            # mu_traj, sigma_traj = r['mean_traj'], r['var_traj']
            # unbatch pos
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in
                            range(n_data)]  # num_samples * [num_atoms_i, 3]

            all_step_pos = [[] for _ in range(n_data)]
            for p in ligand_pos_traj:  # step_i
                p_array = p.cpu().numpy().astype(np.float64)
                for k in range(n_data):
                    all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_step_pos = [np.stack(step_pos) for step_pos in
                            all_step_pos]  # num_samples * [num_steps, num_atoms_i, 3]
            all_pred_pos_traj += [p for p in all_step_pos]

            # unbatch v
            ligand_v_array = ligand_v.cpu().numpy()
            all_pred_v += [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]

            all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
            all_pred_v_traj += [v for v in all_step_v]

            if not pos_only:
                all_step_v0 = unbatch_v_traj(ligand_v0_traj, n_data, ligand_cum_atoms)
                all_pred_v0_traj += [v for v in all_step_v0]
                all_step_vt = unbatch_v_traj(ligand_vt_traj, n_data, ligand_cum_atoms)
                all_pred_vt_traj += [v for v in all_step_vt]
            t2 = time.time()
            time_list.append(t2 - t1)
            current_i += n_data
            # except:
                # print("one down...")
    return all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj, all_pred_v0_traj, all_pred_vt_traj, time_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    # parser.add_argument('classifier_config', type=str)
    parser.add_argument('-si', '--start_data_id', type=int, default=75)
    parser.add_argument('-ei', '--end_data_id', type=int, default=76)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--result_path', type=str, default='N')
    parser.add_argument('--wandb', type=bool, default=False)
    parser.add_argument('--wandb_name', type=str, default="experiment")
    parser.add_argument('--other', type=str, default="none")
    # parser.add_argument('--ckpt_dir', type=str, default='./load_ckpt/transformer_60ep')
    args = parser.parse_args()

    # reduce cpu usage
    torch.set_num_threads(4)

    logger = misc.get_logger('sampling')

    # Load config
    config = misc.load_config(args.config)
    logger.info(config)
    misc.seed_all(config.sample.seed)

    # Load checkpoint
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    logger.info(f"Training Config: {ckpt['config']}")

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])

    # Load dataset
    dataset, subsets = get_dataset(
        config=ckpt['config'].data,
        transform=transform
    )
    train_set, test_set = subsets['train'], subsets['test']
    logger.info(f'Successfully load the dataset (size: {len(test_set)})!')


    # start a new wandb run to track this script
    if args.wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project=args.wandb_name,
            name=f"context={config.sample.context}_scale_on_pos={config.sample.s}_scale_on_v={config.sample.s_v}_#_of_samples{config.sample.num_samples}_clip{config.sample.clip}"+str(classifier_config.model.sample_time_method)+str(classifier_config.model.time_emb_mode)+"sin time embed dim"+str(classifier_config.model.time_emb_dim)+"norm_type_"+str(config.sample.norm_type)+"other"+str(args.other),
            # track hyperparameters and run metadata
            config={
                "batch_size": args.batch_size,
                "num of samples": config.sample.num_samples,
                "num of steps": config.sample.num_steps,
                "context": config.sample.context,
                "scale factor": config.sample.s,
                "clip": config.sample.clip
            }
        )


    # Load diffusion model and checkpoint
    model = ScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        # null_indicator=ckpt['config'].train.null_indicator
        null_indicator=-20.
    ).to(args.device)
    model.load_state_dict(ckpt['model'])
    logger.info(f'Successfully load the model! {config.model.checkpoint}')
    logger.info('Sampling following pockets!'+str(list(range(args.start_data_id,args.end_data_id))))
    logger.info('Results will be saving to' + str(args.result_path))

    # sample loops
    for id in list(range(args.start_data_id,args.end_data_id)):
        # get the pocket
        data = test_set[id]

        # sample
        pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list = sample_diffusion_ligand(
            model, data, config.sample.num_samples,
            batch_size=args.batch_size, device=args.device,
            num_steps=config.sample.num_steps,
            pos_only=config.sample.pos_only,
            center_pos_mode=config.sample.center_pos_mode,
            sample_num_atoms=config.sample.sample_num_atoms,
            context=config.sample.context,
            s=config.sample.s,
            s_v=config.sample.s_v,
            classifier=None,
            enable_wandb=args.wandb,
            clip = config.sample.clip,
            norm_type = config.sample.norm_type,
            condition = config.sample.condition,
            w = config.sample.w
        )
        result = {
            'data': data,
            'pred_ligand_pos': pred_pos,
            'pred_ligand_v': pred_v,
            'pred_ligand_pos_traj': pred_pos_traj,
            'pred_ligand_v_traj': pred_v_traj,
            'time': time_list
        }
        logger.info('Sample done!')

        result_path = args.result_path+"w="+str(int(config.sample.w))+"s="+str(int(config.sample.s))+"condition="+str(int(config.sample.condition))+"clip="+str(config.sample.clip)+"fixcontext"+"other"+str(args.other)
        logger.info("results are saved to this path" + result_path)
        os.makedirs(result_path, exist_ok=True)
        shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
        torch.save(result, os.path.join(result_path, f'result_{id}.pt'))
