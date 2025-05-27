import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
from tqdm.auto import tqdm

from models.common_old import compose_context, ShiftedSoftplus


from models.encoders import get_refine_net
from models.common import compose_context, compose_context_with_prior, ShiftedSoftplus, GaussianSmearing, \
    to_torch_const, extract
from models.transitions import cosine_beta_schedule, get_beta_schedule, DiscreteTransition, index_to_log_onehot, \
    log_sample_categorical



def center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, mode='protein'):
    if mode == 'none':
        offset = 0.
        pass
    elif mode == 'protein':
        offset = scatter_mean(protein_pos, batch_protein, dim=0)
        protein_pos = protein_pos - offset[batch_protein]
        if ligand_pos is not None:
            ligand_pos = ligand_pos - offset[batch_ligand]
    else:
        raise NotImplementedError
    return protein_pos, ligand_pos, offset


def categorical_kl(log_prob1, log_prob2):
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
    return kl


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + (mean1 - mean2) ** 2 * torch.exp(-logvar2))
    return kl.sum(-1)


def log_normal(values, means, log_scales):
    var = torch.exp(log_scales * 2)
    log_prob = -((values - means) ** 2) / (2 * var) - log_scales - np.log(np.sqrt(2 * np.pi))
    return log_prob.sum(-1)


# Time embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# Model
class Classifier(nn.Module):
    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim, num_classes,
                 prior_atom_types=None, prior_bond_types=None, device='cpu'):
        super().__init__()
        self.config = config

        # variance schedule
        self.model_mean_type = config.model_mean_type  # ['noise', 'C0']

        self.add_prior_node = getattr(config, 'add_prior_node', False)
        self.bond_diffusion = getattr(config, 'bond_diffusion', False)
        self.bond_net_type = getattr(config, 'bond_net_type', 'mlp')
        self.center_prox_loss = getattr(config, 'center_prox_loss', False)
        self.armsca_prox_loss = getattr(config, 'armsca_prox_loss', False)
        self.clash_loss = getattr(config, 'clash_loss', False)

        self.sample_time_method = config.sample_time_method  # ['importance', 'symmetric']
        self.loss_pos_type = config.loss_pos_type  # ['mse', 'kl']
        print(f'Loss pos mode {self.loss_pos_type} applied!')

        if config.beta_schedule == 'cosine':
            alphas = cosine_beta_schedule(config.num_diffusion_timesteps, config.pos_beta_s) ** 2
            print('cosine pos alpha schedule applied!')
            betas = 1. - alphas
        else:
            betas = get_beta_schedule(
                beta_schedule=config.beta_schedule,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                num_diffusion_timesteps=config.num_diffusion_timesteps,
            )
            alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.betas = to_torch_const(betas)
        self.num_timesteps = self.betas.size(0)
        self.alphas_cumprod = to_torch_const(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch_const(alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch_const(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch_const(np.sqrt(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_c0_coef = to_torch_const(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_ct_coef = to_torch_const(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_var = to_torch_const(posterior_variance)
        # self.posterior_logvar = to_torch_const(np.log(np.maximum(posterior_variance, 1e-10)))
        self.posterior_logvar = to_torch_const(np.log(np.append(self.posterior_var[1], self.posterior_var[1:])))
        self.pos_score_coef = to_torch_const(betas / np.sqrt(alphas))

        # atom / bond type transition
        self.num_classes = num_classes
        self.num_bond_classes = getattr(config, 'num_bond_classes', 1)

        self.atom_type_trans = DiscreteTransition(
            config.v_beta_schedule, self.num_timesteps,
            s=config.v_beta_s, num_classes=self.num_classes, prior_probs=prior_atom_types
        )
        self.bond_type_trans = DiscreteTransition(
            config.v_beta_schedule, self.num_timesteps,
            s=config.v_beta_s, num_classes=self.num_bond_classes, prior_probs=prior_bond_types
        )

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))

        # model definition
        self.hidden_dim = config.hidden_dim
 
        if self.config.node_indicator:
            if self.add_prior_node:
                emb_dim = self.hidden_dim - 3
            else:
                emb_dim = self.hidden_dim - 1
        else:
            emb_dim = self.hidden_dim

        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)
        if self.add_prior_node:
            self.prior_std_expansion = GaussianSmearing(0., 5., num_gaussians=20, fix_offset=False)
            self.prior_atom_emb = nn.Linear(20, emb_dim)

        # center pos
        self.center_pos_mode = config.center_pos_mode  # ['none', 'protein']
        self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim-2, emb_dim)

        self.refine_net_type = config.model_type
        self.refine_net = get_refine_net(self.refine_net_type, config)

        if self.refine_net_type == 'uni_o2_bond':
            self.ligand_bond_emb = nn.Linear(self.num_bond_classes, self.hidden_dim)

        # atom type prediction
        self.v_inference = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

        # TODO: define your pooling layer here
        pool = config.pool
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'add':
            self.pool = global_add_pool
        elif pool == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError('Not defined pooling!')

        self.pre_head_dim = self.hidden_dim + 3
        self.head = nn.Sequential(
            nn.Linear(self.pre_head_dim, 2*self.hidden_dim),
            ShiftedSoftplus(),# TODO: may need to change this activation
            nn.Linear(2*self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),  # TODO: may need to change this activation
            nn.Linear(self.hidden_dim,1)
        )


    def forward(self, protein_pos, protein_v, batch_protein,
            ligand_pos, ligand_v, batch_ligand,ligand_atom_mask=None, return_all=False):

        # move conformation to the center
        protein_pos, ligand_pos, offset = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand, mode=self.center_pos_mode)
        
        ###
        init_ligand_v = F.one_hot(ligand_v, self.num_classes).float()

        input_ligand_feat = init_ligand_v

        h_protein = self.protein_atom_emb(protein_v)
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)

        if self.config.node_indicator:
            pro_ind = torch.tensor([0]).unsqueeze(0).repeat(len(h_protein), 1).to(h_protein)
            lig_ind = torch.tensor([1]).unsqueeze(0).repeat(len(init_ligand_h), 1).to(h_protein)

            h_protein = torch.cat([h_protein, pro_ind], -1)
            init_ligand_h = torch.cat([init_ligand_h, lig_ind], -1)

        # TODO: should we change back to the original version of compose_context
        h_all, pos_all, batch_all, mask_ligand, mask_ligand_atom, p_index_in_ctx, l_index_in_ctx = compose_context(
            h_protein=h_protein,
            h_ligand=init_ligand_h,
            pos_protein=protein_pos,
            pos_ligand=ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
            ligand_atom_mask=ligand_atom_mask
        )


        outputs = self.refine_net(h_all, pos_all, mask_ligand, batch_all, return_all=return_all)
        final_pos, final_h = outputs['x'], outputs['h']
        combine_final = torch.cat((final_pos,final_h),dim=1)
        after_pool = self.pool(x = combine_final, batch = batch_all)
        pred = self.head(after_pool)

        return pred