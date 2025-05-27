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
from models.egtf import EGTF



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
    elif refine_net_type == 'egtf':
        refine_net = EGTF(
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=1,
            k=config.knn,
            cutoff_mode=config.cutoff_mode,
             # EGTF parameters
            num_encoder = config.num_encoder,
            num_heads = config.num_heads,
            num_ffn = config.num_ffn,
            act_fn_ecd = config.act_fn_ecd,
            dropout_r = config.dropout_r,
            device = device
        ).to(device)
    else:
        raise ValueError(refine_net_type)
    return refine_net


# Model
class Classifier(nn.Module):
    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim, device='cpu'):
        super().__init__()
        self.config = config
        self.device = device

        # model definition
        self.hidden_dim = config.hidden_dim
        self.num_classes = ligand_atom_feature_dim
        self.num_ffn_head = config.num_ffn_head

        if self.config.node_indicator:
            emb_dim = self.hidden_dim - 1
        else:
            emb_dim = self.hidden_dim

        # atom embedding
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)
        self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, emb_dim)

        # center pos
        self.center_pos_mode = config.center_pos_mode  # ['none', 'protein']

        self.refine_net_type = config.model_type
        self.refine_net = get_refine_net(self.refine_net_type, config, device)

        # Ignore position when passing into the final energy head
        self.ignore_pos = config.ignore_pos

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

        # Ignore position dimension if ignoring positions for the head
        self.pre_head_dim = self.hidden_dim if self.ignore_pos else self.hidden_dim + 3
        self.head = nn.Sequential(
            nn.Linear(self.pre_head_dim, self.num_ffn_head),
            ShiftedSoftplus(),# TODO: may need to change this activation
            nn.Linear(self.num_ffn_head, self.hidden_dim),
            ShiftedSoftplus(),  # TODO: may need to change this activation
            nn.Linear(self.hidden_dim,1)
        )


    def forward(self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand,
                time_step=None, return_all=False, fix_x=False):

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
            pos_ligand=ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )

        if self.refine_net_type == "uni_o2":
            outputs = self.refine_net(h_all, pos_all, mask_ligand, batch_all, return_all=return_all, fix_x=fix_x)
        elif self.refine_net_type == "egnn" or self.refine_net_type == "egtf":
            outputs = self.refine_net(h_all, pos_all, mask_ligand, batch_all, return_all=return_all)

        final_pos, final_h = outputs['x'], outputs['h'] # combined graph between ligand and protein pocket, use this as output head feeding
        final_ligand_pos, final_ligand_h = final_pos[mask_ligand], final_h[mask_ligand] # extract the ligand graph

        # instruction: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.global_max_pool.html
        if not self.ignore_pos:
            combine_final = torch.cat((final_pos,final_h),dim=1)
        else:
            combine_final = final_h

        after_pool = self.pool(x = combine_final, batch = batch_all)

        pred = self.head(after_pool)

        return pred
    
    def train_forward(self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand,
                time_step=None, return_all=False, fix_x=False):
        """
        Forward pass for training.
        """
        ligand_v = F.one_hot(ligand_v, self.num_classes).float()
        pred = self.forward(protein_pos, protein_v, batch_protein, ligand_pos, ligand_v=ligand_v, batch_ligand=batch_ligand,
                            time_step=time_step, return_all=return_all, fix_x=fix_x)
        return pred


class Classifier_Multi(nn.Module):
    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim, device='cpu'):
        super().__init__()
        self.config = config
        self.device = device

        # model definition
        self.hidden_dim = config.hidden_dim
        self.num_classes = ligand_atom_feature_dim
        self.num_ffn_head = config.num_ffn_head

        if self.config.node_indicator:
            emb_dim = self.hidden_dim - 1
        else:
            emb_dim = self.hidden_dim

        # atom embedding
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)
        self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, emb_dim)

        # center pos
        # TODO: think about if we need to subtract center position for classifier
        self.center_pos_mode = config.center_pos_mode  # ['none', 'protein']

        self.refine_net_type = config.model_type
        self.refine_net = get_refine_net(self.refine_net_type, config, device)

        # Ignore position when passing into the final energy head
        self.ignore_pos = config.ignore_pos

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

        # Ignore position dimension if ignoring positions for the head
        self.pre_head_dim = self.hidden_dim if self.ignore_pos else self.hidden_dim + 3
        self.head = nn.Sequential(
            nn.Linear(self.pre_head_dim, self.num_ffn_head),
            ShiftedSoftplus(),# TODO: may need to change this activation
            nn.Linear(self.num_ffn_head, self.hidden_dim),
            ShiftedSoftplus(),  # TODO: may need to change this activation
            nn.Linear(self.hidden_dim,3)
        )


    def forward(self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand,
                time_step=None, return_all=False, fix_x=False, train_on_xt=True):

            
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
            pos_ligand=ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )


        if self.refine_net_type == "uni_o2":
            outputs = self.refine_net(h_all, pos_all, mask_ligand, batch_all, return_all=return_all, fix_x=fix_x)
        elif self.refine_net_type == "egnn" or self.refine_net_type == "egtf":
            outputs = self.refine_net(h_all, pos_all, mask_ligand, batch_all, return_all=return_all)

        final_pos, final_h = outputs['x'], outputs['h'] # combined graph between ligand and protein pocket, use this as output head feeding
        final_ligand_pos, final_ligand_h = final_pos[mask_ligand], final_h[mask_ligand] # extract the ligand graph

        # instruction: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.global_max_pool.html
        if not self.ignore_pos:
            combine_final = torch.cat((final_pos,final_h),dim=1)
        else:
            combine_final = final_h

        after_pool = self.pool(x = combine_final, batch = batch_all)

        pred = self.head(after_pool)

        return pred
    

