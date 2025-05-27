import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
from tqdm.auto import tqdm

from models.common import compose_context, ShiftedSoftplus
from models.egnn import EGNN
from models.uni_transformer import UniTransformerO2TwoUpdateGeneral

debug = False


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)

    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.
    alphas = np.sqrt(alphas)
    return alphas

def get_distance(pos, edge_index):
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)

def to_torch_const(x):
    x = torch.from_numpy(x).float()
    x = nn.Parameter(x, requires_grad=False)
    return x

def log_1_min_a(a):
    return np.log(1 - np.exp(a) + 1e-40)

# Time embedding: sinusoidal
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

def get_refine_net(refine_net_type, config, device):
    if refine_net_type == 'uni_o2':
        refine_net = UniTransformerO2TwoUpdateGeneral(
            num_blocks=config.num_blocks,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            k=config.knn,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=config.num_r_gaussian,
            num_node_types=config.num_node_types,
            act_fn=config.act_fn,
            norm=config.norm,
            cutoff_mode=config.cutoff_mode,
            ew_net_type=config.ew_net_type,
            num_x2h=config.num_x2h,
            num_h2x=config.num_h2x,
            r_max=config.r_max,
            x2h_out_fc=config.x2h_out_fc,
            sync_twoup=config.sync_twoup,
            device = device
        ).to(device)
    elif refine_net_type == 'egnn':
        refine_net = EGNN(
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=1,
            k=config.knn,
            cutoff_mode=config.cutoff_mode,
            device = device
        ).to(device)
    else:
        raise ValueError(refine_net_type)
    return refine_net

def center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, mode='protein'):
    if mode == 'none':
        offset = 0.
        pass
    elif mode == 'protein':
        offset = scatter_mean(protein_pos, batch_protein, dim=0)
        protein_pos = protein_pos - offset[batch_protein]
        ligand_pos = ligand_pos - offset[batch_ligand]
    else:
        raise NotImplementedError
    return protein_pos, ligand_pos, offset

def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)

    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x

def log_sample_categorical(logits):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample_index = (gumbel_noise + logits).argmax(dim=-1)
    # sample_onehot = F.one_hot(sample, self.num_classes)
    # log_sample = index_to_log_onehot(sample, self.num_classes)
    return sample_index

def extract(coef, t, batch):
    out = coef[t][batch]
    return out.unsqueeze(-1)

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

# Model
class Classifier(nn.Module):
    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim, multi_properties = True, device='cpu'):
        super().__init__()
        print("initializing time conditioned classifier")
        self.config = config
        self.device = device

        # model definition
        self.hidden_dim = config.hidden_dim
        self.num_classes = ligand_atom_feature_dim
        self.multi_properties = multi_properties
        if self.config.node_indicator:
            emb_dim = self.hidden_dim - 1
        else:
            emb_dim = self.hidden_dim

        # atom embedding
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)
        # self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, emb_dim)

        # center pos
        # TODO: think about if we need to subtract center position for classifier
        self.center_pos_mode = config.center_pos_mode  # ['none', 'protein']

        self.refine_net_type = config.model_type
        self.refine_net = get_refine_net(self.refine_net_type, config, device)

        # introducing time embedding
        self.time_emb_dim = config.time_emb_dim
        self.time_emb_mode = config.time_emb_mode  # ['simple', 'sin']
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                print("Using simple time embedding mode")
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + 1, emb_dim)
            elif self.time_emb_mode == 'sin':
                self.time_emb = nn.Sequential(
                    SinusoidalPosEmb(self.time_emb_dim),
                    nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                    nn.GELU(),
                    nn.Linear(self.time_emb_dim * 4, self.time_emb_dim)
                )
                print("show time emb layer",self.time_emb)
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + self.time_emb_dim, emb_dim)
            else:
                raise NotImplementedError
        else:
            print("no time embedding")
            self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, emb_dim)
        
        # introducing timestep sampler for adding noise to the image
        self.sample_time_method = config.sample_time_method  # ['importance', 'symmetric','uniform']


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
        if not multi_properties:
            self.head = nn.Sequential(
                nn.Linear(self.pre_head_dim, 2*self.hidden_dim),
                ShiftedSoftplus(),# TODO: may need to change this activation
                nn.Linear(2*self.hidden_dim, self.hidden_dim),
                ShiftedSoftplus(),  # TODO: may need to change this activation
                nn.Linear(self.hidden_dim,1)
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(self.pre_head_dim, 2*self.hidden_dim),
                ShiftedSoftplus(),# TODO: may need to change this activation
                nn.Linear(2*self.hidden_dim, self.hidden_dim),
                ShiftedSoftplus(),  # TODO: may need to change this activation
                nn.Linear(self.hidden_dim,3)
            )
        

        # basic diffusion time schedule calculation, alphas and betas
        if config.beta_schedule == 'cosine':
            alphas = cosine_beta_schedule(config.num_diffusion_timesteps, config.pos_beta_s) ** 2
            # print('cosine pos alpha schedule applied!')
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
        if debug:
            print("show alpha and beta:")
            print("betas",betas.shape,betas)
            print("alphas",alphas.shape,alphas)
            print("alphas_cumprod",alphas_cumprod.shape,alphas_cumprod)
            print("alphas_cumprod_prev",alphas_cumprod_prev.shape,alphas_cumprod_prev)

        self.betas = to_torch_const(betas)
        # self.alphas = to_torch_const(alphas)
        self.num_timesteps = self.betas.size(0)
        self.alphas_cumprod = to_torch_const(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch_const(alphas_cumprod_prev)
        if debug:
            print("show self alpha and beta:")
            print("self.betas",self.betas.shape,self.betas)
            # print("self.alphas",self.alphas.shape,self.alphas)
            print("self.alphas_cumprod",self.alphas_cumprod.shape,self.alphas_cumprod)
            print("self.alphas_cumprod_prev",self.alphas_cumprod_prev.shape,self.alphas_cumprod_prev)

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
        self.posterior_logvar = to_torch_const(np.log(np.append(self.posterior_var[1], self.posterior_var[1:])))

        # atom type diffusion schedule in log space
        if config.v_beta_schedule == 'cosine':
            alphas_v = cosine_beta_schedule(self.num_timesteps, config.v_beta_s)
            # print('cosine v alpha schedule applied!')
        else:
            raise NotImplementedError
        log_alphas_v = np.log(alphas_v)
        log_alphas_cumprod_v = np.cumsum(log_alphas_v)
        self.log_alphas_v = to_torch_const(log_alphas_v)
        self.log_one_minus_alphas_v = to_torch_const(log_1_min_a(log_alphas_v))
        self.log_alphas_cumprod_v = to_torch_const(log_alphas_cumprod_v)
        self.log_one_minus_alphas_cumprod_v = to_torch_const(log_1_min_a(log_alphas_cumprod_v))

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))
    
    def q_v_sample(self, log_v0, t, batch):
        log_qvt_v0 = self.q_v_pred(log_v0, t, batch)
        sample_index = log_sample_categorical(log_qvt_v0)
        log_sample = index_to_log_onehot(sample_index, self.num_classes)
        return sample_index, log_sample

    def q_v_pred(self, log_v0, t, batch):
        # compute q(vt | v0)
        log_cumprod_alpha_t = extract(self.log_alphas_cumprod_v, t, batch)
        log_1_min_cumprod_alpha = extract(self.log_one_minus_alphas_cumprod_v, t, batch)

        log_probs = log_add_exp(
            log_v0 + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes)
        )
        return log_probs
    
    def sample_time(self, num_graphs, device, method):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(num_graphs, device, method='symmetric')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            time_step = torch.multinomial(pt_all, num_samples=num_graphs, replacement=True)
            pt = pt_all.gather(dim=0, index=time_step)
            return time_step, pt

        elif method == 'symmetric':
            time_step = torch.randint(
                0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device)
            # print("1",time_step.shape,time_step)
            time_step = torch.cat(
                [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
            # print("2",time_step.shape,time_step)
            pt = torch.ones_like(time_step).float() / self.num_timesteps
            return time_step, pt
        
        elif method == 'uniform':
            time_step = torch.randint(
                0, self.num_timesteps, size=(num_graphs,), device=device)
            pt = None
            return time_step, pt
        else:
            raise ValueError


    def forward(self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand,
                time_step=None, return_all=False, fix_x=False,train_on_xt=True):

        num_graphs = batch_protein.max().item() + 1
        # FIXME: we remove the center of mass here this time when training a classifier, 
        # we didn't apply this last time when training a classifier
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand, mode=self.center_pos_mode)

        if train_on_xt:
            # 1. sample noise levels
            if time_step is None:
                time_step, pt = self.sample_time(num_graphs, protein_pos.device, self.sample_time_method)
            else:
                pt = torch.ones_like(time_step).float() / self.num_timesteps

            a = self.alphas_cumprod.index_select(0, time_step)  # (num_graphs, )

            # 2. perturb pos and v
            a_pos = a[batch_ligand].unsqueeze(-1)  # (num_ligand_atoms, 1)
            pos_noise = torch.zeros_like(ligand_pos)
            pos_noise.normal_()
            # Xt = a.sqrt() * X0 + (1-a).sqrt() * eps
            ligand_pos_perturbed = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise  # pos_noise * std
            # Vt = a * V0 + (1-a) / K
            log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)
            ligand_v_perturbed, _ = self.q_v_sample(log_ligand_v0, time_step, batch_ligand)
            # print(ligand_v_perturbed)
            ligand_v_perturbed = F.one_hot(ligand_v_perturbed, self.num_classes).float()
            init_ligand_v = ligand_v_perturbed

            # time embedding in forward path
            if self.time_emb_dim > 0:
                if self.time_emb_mode == 'simple':
                    if debug:
                        print("preconcate",init_ligand_v.shape,((time_step / self.num_timesteps)[batch_ligand].unsqueeze(-1)).shape)
                    input_ligand_feat = torch.cat([
                        init_ligand_v,
                        (time_step / self.num_timesteps)[batch_ligand].unsqueeze(-1)
                    ], -1)
                elif self.time_emb_mode == 'sin':
                    # print(time_step.shape)
                    # time_feat = self.time_emb(time_step)
                    time_feat = self.time_emb(time_step[batch_ligand])
                    # print("time",time_feat.shape)
                    input_ligand_feat = torch.cat([init_ligand_v, time_feat], -1)
                else:
                    raise NotImplementedError
            else:
                # print("no forward time embeddding")
                input_ligand_feat = init_ligand_v
        else:
            ligand_pos_perturbed = ligand_pos
            init_ligand_v = ligand_v

            input_ligand_feat = init_ligand_v



        h_protein = self.protein_atom_emb(protein_v)
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)


        if self.config.node_indicator:
            """
            expand 1 dim at end:
            "1" for ligand graph node
            "0" for protein graph node
            """
            h_protein = torch.cat([h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1)
            init_ligand_h = torch.cat([init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein)], -1)

        h_all, pos_all, batch_all, mask_ligand = compose_context(
            h_protein=h_protein,
            h_ligand=init_ligand_h,
            pos_protein=protein_pos,
            pos_ligand=ligand_pos_perturbed,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )


        if self.refine_net_type == "uni_o2":
            outputs = self.refine_net(h_all, pos_all, mask_ligand, batch_all, return_all=return_all, fix_x=fix_x)
        elif self.refine_net_type == "egnn":
            outputs = self.refine_net(h_all, pos_all, mask_ligand, batch_all, return_all=return_all)

        final_pos, final_h = outputs['x'], outputs['h'] # combined graph between ligand and protein pocket, use this as output head feeding
        final_ligand_pos, final_ligand_h = final_pos[mask_ligand], final_h[mask_ligand] # extract the ligand graph

        # instruction: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.global_max_pool.html
        combine_final = torch.cat((final_pos,final_h),dim=1)
        after_pool = self.pool(x = combine_final, batch = batch_all)

        pred = self.head(after_pool)

        return pred