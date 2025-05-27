# Copyright 2023 ByteDance and/or its affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0


import argparse
import os
import pickle
import shutil
import time

import numpy as np
import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_sum
from tqdm.auto import tqdm
import wandb

import sys
# print(os.getcwd())
sys.path.append(os.getcwd())

import utils.misc as misc
import utils.prior as utils_prior
import utils.reconstruct as recon
import utils.transforms as trans
from datasets.pl_data import FOLLOW_BATCH, torchify_dict
from datasets.pl_pair_dataset import get_decomp_dataset
from models.decompdiff import DecompScorePosNet3D, log_sample_categorical
from utils.data import ProteinLigandData, PDBProtein
from utils.evaluation import atom_num
from models.classifier import Classifier as clsfer

def pocket_pdb_to_pocket(pdb_path):
    protein = PDBProtein(pdb_path)
    protein_dict = torchify_dict(protein.to_dict_atom())
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=protein_dict,
        ligand_dict={
            'element': torch.empty([0, ], dtype=torch.long),
            'pos': torch.empty([0, 3], dtype=torch.float),
            'atom_feature': torch.empty([0, 8], dtype=torch.float),
            'bond_index': torch.empty([2, 0], dtype=torch.long),
            'bond_type': torch.empty([0, ], dtype=torch.long),
        }
    )
    return data


def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    all_step_v = [[] for _ in range(n_data)]
    for v in ligand_v_traj:  # step_i
        v_array = v.cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
    all_step_v = [np.stack(step_v) for step_v in all_step_v]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v


@torch.no_grad()
def sample_diffusion_ligand_decomp(
        model, data, init_transform, num_samples, batch_size=16, device='cuda:0', prior_mode='subpocket',
        num_steps=None, center_pos_mode='none', num_atoms_mode='prior',
        atom_prior_probs=None, bond_prior_probs=None,
        arms_natoms_config=None, scaffold_natoms_config=None, natoms_config=None,
        atom_enc_mode='add_aromatic', bond_fc_mode='fc', energy_drift_opt=None, context=0, scale_factor=0, classifier=None,clip=1,enable_wandb=False):
    all_pred_pos, all_pred_v, all_pred_bond, all_pred_bond_index = [], [], [], []
    all_pred_pos_traj, all_pred_v_traj = [], []
    all_pred_v0_traj, all_pred_vt_traj = [], []
    all_pred_bt_traj, all_pred_b_traj = [], []
    all_decomp_ind = []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))
    current_i = 0

    if num_atoms_mode == 'stat':
        with open(natoms_config, 'rb') as f:
            natoms_config = pickle.load(f)
        natoms_sampler = utils_prior.NumAtomsSampler(natoms_config)

    for i in tqdm(range(num_batch)):
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)

        if prior_mode == 'subpocket':
            # init ligand pos
            arm_pocket_sizes = [
                atom_num.get_space_size(data.protein_pos[data.pocket_atom_masks[arm_i]].detach().cpu().numpy())
                for arm_i in range(data.num_arms)]
            scaffold_pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy())
            arm_centers = [data.protein_pos[pocket_mask].mean(0) for pocket_mask in
                           data.pocket_atom_masks]  # [num_arms, 3]
            scaffold_center = data.protein_pos.mean(0)
            data.ligand_decomp_centers = torch.cat(arm_centers + [scaffold_center], dim=0)

            batch_data_list = []
            batch_init_pos = []
            ligand_num_atoms = []
            batch_decomp_ind = []
            for data_idx in range(n_data):
                n_atoms = 0
                init_ligand_pos, ligand_atom_mask = [], []
                for arm_i in range(data.num_arms):
                    if num_atoms_mode == 'prior':
                        arm_num_atoms = atom_num.sample_atom_num(arm_pocket_sizes[arm_i], arms_natoms_config).astype(
                            int)
                    elif num_atoms_mode == 'ref':
                        arm_num_atoms = int((data.ligand_atom_mask == arm_i).sum())
                    elif num_atoms_mode == 'ref_large':
                        inc = np.ceil(10 / (data.num_arms + 2))
                        arm_num_atoms = int((data.ligand_atom_mask == arm_i).sum().item() + inc)
                    else:
                        raise ValueError
                    init_arm_pos = arm_centers[arm_i] + torch.randn([arm_num_atoms, 3])
                    init_ligand_pos.append(init_arm_pos)
                    ligand_atom_mask += [arm_i] * arm_num_atoms
                    n_atoms += arm_num_atoms

                if num_atoms_mode == 'prior':
                    scaffold_num_atoms = atom_num.sample_atom_num(scaffold_pocket_size, scaffold_natoms_config).astype(
                        int)
                elif num_atoms_mode == 'ref':
                    scaffold_num_atoms = int((data.ligand_atom_mask == -1).sum())
                elif num_atoms_mode == 'ref_large':
                    inc = np.ceil(10 / (data.num_arms + 2)) * 2
                    scaffold_num_atoms = int((data.ligand_atom_mask == -1).sum().item() + inc)
                else:
                    raise ValueError
                init_scaffold_pos = scaffold_center + torch.randn([scaffold_num_atoms, 3])
                init_ligand_pos.append(init_scaffold_pos)
                ligand_atom_mask += [-1] * scaffold_num_atoms
                n_atoms += scaffold_num_atoms
                new_data = data.clone()
                new_data.ligand_atom_mask = torch.tensor(ligand_atom_mask, dtype=torch.long)
                new_data = init_transform(new_data)

                # init bond type
                if getattr(new_data, 'ligand_fc_bond_index') is not None:
                    if bond_prior_probs is not None:
                        new_data.ligand_fc_bond_type = torch.multinomial(torch.from_numpy(
                            bond_prior_probs.astype(np.float32)),
                            new_data.ligand_fc_bond_index.size(1), replacement=True)
                    else:
                        uniform_logits = torch.zeros(new_data.ligand_fc_bond_index.size(1), model.num_bond_classes)
                        new_data.ligand_fc_bond_type = log_sample_categorical(uniform_logits)
                batch_data_list.append(new_data)

                batch_init_pos.append(torch.cat(init_ligand_pos, dim=0))
                ligand_num_atoms.append(n_atoms)
                batch_decomp_ind.append(new_data.ligand_atom_mask.tolist())

            all_decomp_ind += batch_decomp_ind

        elif prior_mode == 'ref_prior':
            old_data = data.clone()
            old_data = init_transform(old_data)

            arm_centers = old_data.ligand_decomp_centers[:old_data.num_arms, :]
            arms_stds = old_data.ligand_decomp_stds[:old_data.num_arms, :]
            scaffold_center = old_data.ligand_decomp_centers[-1, :]
            scaffold_std = old_data.ligand_decomp_stds[-1, :]

            batch_data_list = []
            batch_init_pos = []
            ligand_num_atoms = []
            # batch_noise_stds = []
            batch_decomp_ind = []
            for data_idx in range(n_data):
                n_atoms = 0
                init_ligand_pos, ligand_atom_mask = [], []
                for arm_i in range(data.num_arms):
                    arm_num_atoms = int(old_data.arms_prior[arm_i][0])
                    init_arm_pos = arm_centers[arm_i] + torch.randn([arm_num_atoms, 3]) * arms_stds[arm_i, :].unsqueeze(
                        0)
                    init_ligand_pos.append(init_arm_pos)
                    ligand_atom_mask += [arm_i] * arm_num_atoms
                    n_atoms += arm_num_atoms

                scaffold_num_atoms = int(old_data.scaffold_prior[0][0]) if len(old_data.scaffold_prior) == 1 else 0
                init_scaffold_pos = scaffold_center + torch.randn([scaffold_num_atoms, 3]) * scaffold_std.unsqueeze(0)

                init_ligand_pos.append(init_scaffold_pos)
                ligand_atom_mask += [-1] * scaffold_num_atoms
                n_atoms += scaffold_num_atoms

                new_data = data.clone()
                new_data.ligand_atom_mask = torch.tensor(ligand_atom_mask, dtype=torch.long)
                new_data = init_transform(new_data)
                # init bond type
                if getattr(new_data, 'ligand_fc_bond_index') is not None:
                    if bond_prior_probs is not None:
                        new_data.ligand_fc_bond_type = torch.multinomial(torch.from_numpy(
                            bond_prior_probs.astype(np.float32)),
                            new_data.ligand_fc_bond_index.size(1), replacement=True)
                    else:
                        uniform_logits = torch.zeros(new_data.ligand_fc_bond_index.size(1), model.num_bond_classes)
                        new_data.ligand_fc_bond_type = log_sample_categorical(uniform_logits)

                batch_data_list.append(new_data)
                batch_init_pos.append(torch.cat(init_ligand_pos, dim=0))
                ligand_num_atoms.append(n_atoms)
                batch_decomp_ind.append(new_data.ligand_atom_mask.tolist())
                # assert n_atoms == _n_atoms
            all_decomp_ind += batch_decomp_ind

        elif prior_mode == 'beta_prior':
            old_data = data.clone()
            old_data = init_transform(old_data)
            arm_centers = old_data.ligand_decomp_centers[:old_data.num_arms, :]
            scaffold_center = old_data.ligand_decomp_centers[-1, :]
            if num_atoms_mode in ['old', 'v2']:
                arms_stds = old_data.ligand_decomp_stds[:old_data.num_arms, :]
                scaffold_std = old_data.ligand_decomp_stds[-1, :]

            batch_data_list = []
            batch_init_pos = []
            ligand_num_atoms = []
            batch_noise_stds = []
            batch_decomp_ind = []
            for data_idx in range(n_data):
                n_atoms = 0
                init_ligand_pos, ligand_atom_mask = [], []

                if num_atoms_mode == 'stat':
                    arm_natoms, arms_stds = natoms_sampler.sample_arm_natoms(arm_centers, data.protein_pos)
                    if len(data.scaffold_prior) > 0:
                        scaffold_center = [p[1] for p in data.scaffold_prior][0]
                        scaffold_natoms, scaffold_std = natoms_sampler.sample_sca_natoms(scaffold_center, arm_centers,
                                                                                         arms_stds,
                                                                                         data.protein_pos)
                    else:
                        scaffold_center = data.protein_pos.mean(0)
                        scaffold_natoms, scaffold_std = 0, torch.tensor([0.])

                for arm_i in range(data.num_arms):
                    if num_atoms_mode == 'old':
                        arm_std = arms_stds[arm_i]
                        _m = 12.41
                        _b = -4.98
                        arm_num_atoms_lower = torch.clamp(np.floor((_m - 2.0) * arm_std[0] + _b), min=2).long()
                        arm_num_atoms_upper = torch.clamp(np.ceil((_m + 3.0) * arm_std[0] + _b), min=2).long()
                        arm_num_atoms = int(
                            torch.randint(low=arm_num_atoms_lower, high=arm_num_atoms_upper + 1, size=(1,)))
                    elif num_atoms_mode == 'v2':
                        arm_num_atoms = int(data.arms_prior[arm_i][0])
                    elif num_atoms_mode == 'stat':
                        arm_num_atoms = int(arm_natoms[arm_i])

                    init_arm_pos = arm_centers[arm_i] + torch.randn([arm_num_atoms, 3]) * arms_stds[arm_i, :].unsqueeze(
                        0)
                    init_ligand_pos.append(init_arm_pos)
                    ligand_atom_mask += [arm_i] * arm_num_atoms
                    n_atoms += arm_num_atoms
                    init_arm_std = arms_stds[arm_i, :].unsqueeze(0).expand(1, 3)
                    batch_noise_stds.append(init_arm_std)

                if num_atoms_mode == 'old':
                    _m = 12.41
                    _b = -4.98
                    scaffold_num_atoms_lower = torch.clamp(np.ceil((_m - 2.0) * scaffold_std[0] + _b), min=2).long()
                    scaffold_num_atoms_upper = torch.clamp(np.ceil((_m + 3.0) * scaffold_std[0] + _b), min=2).long()
                    scaffold_num_atoms = int(
                        torch.randint(low=scaffold_num_atoms_lower, high=scaffold_num_atoms_upper + 1, size=(1,)))
                elif num_atoms_mode == 'v2':
                    if len(data.scaffold_prior) > 0:
                        scaffold_num_atoms = int(data.scaffold_prior[0][0])
                    else:
                        scaffold_num_atoms = 0
                elif num_atoms_mode == 'stat':
                    scaffold_num_atoms = int(scaffold_natoms)

                init_scaffold_pos = scaffold_center + torch.randn([scaffold_num_atoms, 3]) * scaffold_std.unsqueeze(0)
                init_scaffold_std = scaffold_std.unsqueeze(0).expand(1, 3)
                batch_noise_stds.append(init_scaffold_std)

                init_ligand_pos.append(init_scaffold_pos)
                ligand_atom_mask += [-1] * scaffold_num_atoms
                n_atoms += scaffold_num_atoms

                new_data = data.clone()
                new_data.ligand_atom_mask = torch.tensor(ligand_atom_mask, dtype=torch.long)
                new_data = init_transform(new_data)
                # init bond type
                if getattr(new_data, 'ligand_fc_bond_index') is not None:
                    if bond_prior_probs is not None:
                        new_data.ligand_fc_bond_type = torch.multinomial(torch.from_numpy(
                            bond_prior_probs.astype(np.float32)),
                            new_data.ligand_fc_bond_index.size(1), replacement=True)
                    else:
                        uniform_logits = torch.zeros(new_data.ligand_fc_bond_index.size(1), model.num_bond_classes)
                        new_data.ligand_fc_bond_type = log_sample_categorical(uniform_logits)

                batch_data_list.append(new_data)
                batch_init_pos.append(torch.cat(init_ligand_pos, dim=0))
                ligand_num_atoms.append(n_atoms)
                batch_decomp_ind.append(new_data.ligand_atom_mask.tolist())
                # assert n_atoms == _n_atoms
            all_decomp_ind += batch_decomp_ind

        else:
            raise ValueError(prior_mode)

        logger.info(f"ligand_num_atoms={ligand_num_atoms}")
        init_ligand_pos = torch.cat(batch_init_pos, dim=0).to(device)
        batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
        assert len(init_ligand_pos) == len(batch_ligand)

        # init ligand v
        if atom_prior_probs is not None:
            init_ligand_v = torch.multinomial(torch.from_numpy(
                atom_prior_probs.astype(np.float32)),
                len(batch_ligand), replacement=True).to(device)
        else:
            uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
            init_ligand_v = log_sample_categorical(uniform_logits)

        collate_exclude_keys = ['scaffold_prior', 'arms_prior']
        batch = Batch.from_data_list([d for d in batch_data_list], exclude_keys=collate_exclude_keys,
                                     follow_batch=FOLLOW_BATCH).to(device)
        batch_protein = batch.protein_element_batch
        batch_full_protein_pos = full_protein_pos.repeat(n_data, 1).to(device)
        full_batch_protein = torch.arange(n_data).repeat_interleave(len(full_protein_pos)).to(device)

        # batch.ligand_decomp_stds = torch.ones_like(batch.ligand_decomp_centers)
        if num_atoms_mode == 'stat':
            batch.ligand_decomp_stds = torch.cat(batch_noise_stds, dim=0).to(device)

        t1 = time.time()
        r = model.sample_diffusion(
            protein_pos=batch.protein_pos,
            protein_v=batch.protein_atom_feature.float(),
            batch_protein=batch_protein,
            protein_group_idx=batch.protein_decomp_group_idx,

            init_ligand_pos=init_ligand_pos,
            init_ligand_v=init_ligand_v,
            ligand_v_aux=batch.ligand_atom_aux_feature.float(),
            batch_ligand=batch_ligand,
            ligand_group_idx=batch.ligand_decomp_group_idx,
            ligand_atom_mask=None,

            prior_centers=batch.ligand_decomp_centers,
            prior_stds=batch.ligand_decomp_stds,
            prior_num_atoms=batch.ligand_decomp_num_atoms,
            batch_prior=batch.ligand_decomp_centers_batch,
            prior_group_idx=batch.prior_group_idx,

            ligand_fc_bond_index=getattr(batch, 'ligand_fc_bond_index', None),
            init_ligand_fc_bond_type=getattr(batch, 'ligand_fc_bond_type', None),
            batch_ligand_bond=getattr(batch, 'ligand_fc_bond_type_batch', None),
            ligand_decomp_batch=batch.ligand_decomp_mask,
            ligand_decomp_index=batch.ligand_atom_mask,

            num_steps=num_steps,
            center_pos_mode=center_pos_mode,
            energy_drift_opt=energy_drift_opt,

            full_protein_pos=batch_full_protein_pos,
            full_batch_protein=full_batch_protein,

            # Added by Yue: guidance argument
            context=context,
            scale_factor=scale_factor,
            classifier=classifier,
            clip=clip,
            enable_wandb=enable_wandb
        )
        ligand_pos, ligand_v, ligand_bond = r['pos'], r['v'], r['bond']
        ligand_pos_traj, ligand_v_traj = r['pos_traj'], r['v_traj']
        ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']
        ligand_bt_traj, ligand_b_traj = r['bt_traj'], r['bond_traj']

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
        all_step_pos = [np.stack(step_pos) for step_pos in all_step_pos]  # num_samples * [num_steps, num_atoms_i, 3]
        all_pred_pos_traj += [p for p in all_step_pos]

        # unbatch v
        ligand_v_array = ligand_v.cpu().numpy()
        all_pred_v += [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]

        # unbatch bond
        if getattr(model, 'bond_diffusion', False):
            ligand_bond_array = ligand_bond.cpu().numpy()
            ligand_bond_index_array = batch.ligand_fc_bond_index.cpu().numpy()
            # cum_bonds = batch.ligand_fc_bond_type_ptr
            ligand_num_bonds = scatter_sum(torch.ones_like(batch.ligand_fc_bond_type_batch),
                                           batch.ligand_fc_bond_type_batch).tolist()
            cum_bonds = np.cumsum([0] + ligand_num_bonds)
            all_pred_bond_index += [ligand_bond_index_array[:, cum_bonds[k]:cum_bonds[k + 1]] - ligand_cum_atoms[k] for
                                    k in range(n_data)]
            all_pred_bond += [ligand_bond_array[cum_bonds[k]:cum_bonds[k + 1]] for k in range(n_data)]
            # assert all_pred_bond_index[-1].size(1) == all_pred_bond[-1].size(0)
            all_step_b = unbatch_v_traj(ligand_b_traj, n_data, cum_bonds)
            all_pred_b_traj += [b for b in all_step_b]

            all_step_bt = unbatch_v_traj(ligand_bt_traj, n_data, cum_bonds)
            all_pred_bt_traj += [b for b in all_step_bt]
        else:
            all_pred_bond_index += []
            all_pred_bond += []

        all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
        all_pred_v_traj += [v for v in all_step_v]
        all_step_v0 = unbatch_v_traj(ligand_v0_traj, n_data, ligand_cum_atoms)
        all_pred_v0_traj += [v for v in all_step_v0]
        all_step_vt = unbatch_v_traj(ligand_vt_traj, n_data, ligand_cum_atoms)
        all_pred_vt_traj += [v for v in all_step_vt]

        t2 = time.time()
        time_list.append(t2 - t1)
        current_i += n_data

    n_recon_success, n_complete = 0, 0
    results = []
    for i, (pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_bond_index, pred_bond_type, pred_b_traj,
            pred_bt_traj) in enumerate(
        zip(all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj, all_pred_bond_index, all_pred_bond,
            all_pred_b_traj, all_pred_bt_traj)):
        pred_atom_type = trans.get_atomic_number_from_index(pred_v, mode=atom_enc_mode)
        pred_bond_index = pred_bond_index.tolist()
        # reconstruction
        try:
            pred_aromatic = trans.is_aromatic_from_index(pred_v, mode=atom_enc_mode)
            if args.recon_with_bond:
                mol = recon.reconstruct_from_generated_with_bond(pred_pos, pred_atom_type, pred_bond_index,
                                                                 pred_bond_type)
            else:
                mol = recon.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
            smiles = Chem.MolToSmiles(mol)
            n_recon_success += 1

        except recon.MolReconsError:
            logger.warning('Reconstruct failed %s' % f'{i}')
            mol = None
            smiles = ''

        if mol is not None and '.' not in smiles:
            n_complete += 1
        results.append(
            {
                'mol': mol,
                'smiles': smiles,
                'pred_pos': pred_pos,
                'pred_v': pred_v,
                'pred_pos_traj': pred_pos_traj,
                'pred_v_traj': pred_v_traj,
                'decomp_mask': all_decomp_ind[i],

                'pred_bond_index': pred_bond_index,
                'pred_bond_type': pred_bond_type
            }
        )
    logger.info(f'n_reconstruct: {n_recon_success} n_complete: {n_complete}')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('-si', '--start_data_id', type=int, default=0)
    parser.add_argument('-ei', '--end_data_id', type=int, default=1)
    parser.add_argument('--ori_data_path', type=str, default='./data/test_set')
    parser.add_argument('--index_path', type=str, default='./data/test_index.pkl')
    parser.add_argument('--beta_prior_path', type=str, default='./pregen_info/beta_priors')
    parser.add_argument('--natom_models_path', type=str, default='./pregen_info/natom_models.pkl')
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--outdir', type=str, default='./outputs_test')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--prior_mode', type=str, choices=['subpocket', 'ref_prior', 'beta_prior'])
    parser.add_argument('--num_atoms_mode', type=str, choices=['prior', 'ref', 'ref_large'], default='ref')
    parser.add_argument('--bp_num_atoms_mode', type=str, choices=['old', 'stat', 'v2'], default='v2')
    parser.add_argument('--suffix', default=None)
    parser.add_argument('--recon_with_bond', type=eval, default=True)
    parser.add_argument('--wandb', type=bool, default=False)
    parser.add_argument('--wandb_name', type=str, default="visualizing sampling process for guided decompdiff on pocket 50")
    args = parser.parse_args()
    RDLogger.DisableLog('rdApp.*')

    torch.set_num_threads(8)

    # Load config
    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.sample.seed)

    # print(f"here{config_name}")
    log_dir = args.outdir
    os.makedirs(log_dir, exist_ok=True)
    logger = misc.get_logger('evaluate', log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))

    # Load checkpoint
    assert config.model.checkpoint or args.ckpt_path
    ckpt_path = args.ckpt_path if args.ckpt_path is not None else config.model.checkpoint
    ckpt = torch.load(ckpt_path, map_location=args.device)
    if 'train_config' in config.model:
        logger.info(f"Load training config from: {config.model['train_config']}")
        ckpt['config'] = misc.load_config(config.model['train_config'])
    logger.info(f"Training Config: {ckpt['config']}")

    # start a new wandb run to track this script
    if args.wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project=args.wandb_name,
            name=f"context={config.sample.delta_context}_clip{config.sample.clip}_scale_on_pos={config.sample.scale_factor}#_of_samples{config.sample.num_samples}",
            # track hyperparameters and run metadata
            config={
                "batch_size": args.batch_size,
                "num of samples": config.sample.num_samples,
                "num of steps": config.sample.num_steps,
                "context": config.sample.delta_context,
                "scale factor": config.sample.scale_factor,
                "clip":config.sample.clip
            }
        )

    # Transforms
    cfg_transform = ckpt['config'].data.transform
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = cfg_transform.ligand_atom_mode
    ligand_bond_mode = cfg_transform.ligand_bond_mode
    max_num_arms = cfg_transform.max_num_arms
    ligand_featurizer = trans.FeaturizeLigandAtom(
        ligand_atom_mode, prior_types=ckpt['config'].model.get('prior_types', False))
    decomp_indicator = trans.AddDecompIndicator(
        max_num_arms=max_num_arms,
        global_prior_index=ligand_featurizer.ligand_feature_dim,
        add_ord_feat=getattr(ckpt['config'].data.transform, 'add_ord_feat', True),
    )
    transform = Compose([
        protein_featurizer
    ])

    prior_mode = config.sample.prior_mode if args.prior_mode is None else args.prior_mode
    init_transform_list = [
        trans.ComputeLigandAtomNoiseDist(version=prior_mode),
        decomp_indicator
    ]
    if getattr(ckpt['config'].model, 'bond_diffusion', False):
        init_transform_list.append(trans.FeaturizeLigandBond(mode=ligand_bond_mode, set_bond_type=False))
    init_transform = Compose(init_transform_list)

    # load classifier
    # classifier first
    print(f"protein dim {protein_featurizer.protein_feature_dim + decomp_indicator.protein_feature_dim}, ligand dim {ligand_featurizer.ligand_feature_dim + decomp_indicator.ligand_feature_dim}")
    print(f"prior atom {ligand_featurizer.atom_types_prob}, prior bond {ligand_featurizer.bond_types_prob}")
    classifier = clsfer(
        config.model,
        protein_atom_feature_dim=protein_featurizer.protein_feature_dim + decomp_indicator.protein_feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.ligand_feature_dim + decomp_indicator.ligand_feature_dim,
        device = args.device,
        prior_atom_types=ligand_featurizer.atom_types_prob, # TODO: does this gona be affect the training?
        prior_bond_types=ligand_featurizer.bond_types_prob, # TODO: does this gona be affect the training process?
        num_classes=ligand_featurizer.ligand_feature_dim
    ).to(args.device)

    # load history model
    print("loading history model")
    print("loading classifier from",config.sample.ckpt + "/ckpt.pt")
    state_dict_mod = torch.load(config.sample.ckpt + "/ckpt.pt", map_location=torch.device('cuda'))
    classifier.load_state_dict(state_dict_mod['model'])
    start_ep_load = state_dict_mod['iteration']
    del state_dict_mod
    print("finish loading model")


    # Load model
    model = DecompScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.protein_feature_dim + decomp_indicator.protein_feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.ligand_feature_dim + decomp_indicator.ligand_feature_dim,
        num_classes=ligand_featurizer.ligand_feature_dim,
        prior_atom_types=ligand_featurizer.atom_types_prob, prior_bond_types=ligand_featurizer.bond_types_prob
    ).to(args.device)
    model.load_state_dict(ckpt['model'], strict=True)
    logger.info(f'Successfully load the model! {config.model.checkpoint}')

    # load num of atoms config
    with open(config.sample.arms_num_atoms_config, 'rb') as f:
        arms_num_atoms_config = pickle.load(f)
    logger.info(f'Successfully load arms num atoms config from {config.sample.arms_num_atoms_config}')
    with open(config.sample.scaffold_num_atoms_config, 'rb') as f:
        scaffold_num_atoms_config = pickle.load(f)
    logger.info(f'Successfully load scaffold num atoms config from {config.sample.scaffold_num_atoms_config}')

    # data = pocket_pdb_to_pocket(index['data']['protein_file'])
    dataset, subsets = get_decomp_dataset(
        config=ckpt['config'].data,
        transform=None,
    )
    train_set, val_set = subsets['train'], subsets['test']
    logger.info(f'prior mode{prior_mode}')
    logger.info(f'scale factor{config.sample.scale_factor}')
    # every thing related to data id start here
    for data_id in list(range(args.start_data_id,args.end_data_id)):
        with open(args.index_path, 'rb') as f:
            test_index = pickle.load(f)
        index = test_index[data_id]
        pdb_path = index['data']['protein_file']

        data = val_set[data_id]


        full_protein = PDBProtein(os.path.join(args.ori_data_path, data.src_protein_filename))
        full_protein_pos = torch.from_numpy(full_protein.to_dict_atom()['pos'].astype(np.float32))

        # Setup prior
        if prior_mode == 'subpocket':
            logger.info(f'Apply subpocket prior for data id {data_id}')
        elif prior_mode == 'beta_prior':
            beta_prior_path = os.path.join(args.beta_prior_path, f"{data_id:08d}.pkl")
            utils_prior.substitute_golden_prior_with_beta_prior(data, beta_prior_path=beta_prior_path)
            logger.info(f'Apply beta prior for data id {data_id}')
        elif prior_mode == 'ref_prior':
            utils_prior.compute_golden_prior_from_data(data)
            logger.info(f'Apply golden prior for data id {data_id}')
        else:
            raise ValueError(prior_mode)

        mean_context = 0
        delta_context = config.sample.delta_context
        context = mean_context + delta_context
        scale_factor = config.sample.scale_factor

        data = transform(data)
        raw_results = sample_diffusion_ligand_decomp(
            model, data,
            init_transform=init_transform,
            num_samples=config.sample.num_samples,
            batch_size=args.batch_size, device=args.device,
            num_steps=config.sample.num_steps,
            center_pos_mode=config.sample.center_pos_mode,
            num_atoms_mode=args.bp_num_atoms_mode if prior_mode == 'beta_prior' else args.num_atoms_mode,
            arms_natoms_config=arms_num_atoms_config,
            scaffold_natoms_config=scaffold_num_atoms_config,
            atom_enc_mode=ligand_atom_mode,
            bond_fc_mode=ligand_bond_mode,
            energy_drift_opt=config.sample.energy_drift,
            prior_mode=prior_mode,
            atom_prior_probs=ligand_featurizer.atom_types_prob, bond_prior_probs=ligand_featurizer.bond_types_prob,
            natoms_config=args.natom_models_path,
            context=context,
            scale_factor=scale_factor,
            classifier=classifier,
            clip = float(config.sample.clip),
            enable_wandb=args.wandb
        )
        results = []
        for r in raw_results:
            results.append({
                **r,
                'ligand_filename': index['src_ligand_filename']
            })
        logger.info('Sample done!')
        # add data id here!
        if args.suffix:
            torch.save(results, os.path.join(log_dir, f'result_{args.suffix}.pt'))
        else:
            torch.save(results, os.path.join(log_dir, f'result_{data_id}.pt'))
