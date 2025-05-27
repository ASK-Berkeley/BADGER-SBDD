import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
from torch_geometric.nn import radius_graph, knn_graph
from models.common import GaussianSmearing, MLP, batch_hybrid_edge_connection, NONLINEARITIES

debug = False

class EncoderLayer(nn.Module):
    """
    Attention is All you Need
    Reference: A. Vaswani et al., https://arxiv.org/abs/1706.03762
    """
    def __init__(self, d_model, num_heads, num_ffn, dropout_r, act_fn_ecd):
        super(EncoderLayer, self).__init__()
        self.act_fn_ecd = nn.ReLU if act_fn_ecd == "ReLU" else nn.SiLU()

        self.attention = MultiHeadAttention(d_model = d_model, num_heads = num_heads)
        self.norm1 = LayerNormalization(parameters_shape = [d_model])
        self.dropout = nn.Dropout(p = dropout_r)
        self.ffn = MLP(d_model = d_model, num_ffn = num_ffn, act_fn = act_fn_ecd, dropout_r = dropout_r)
        self.norm2 = LayerNormalization(parameters_shape = [d_model])
        self.act_fn = act_fn_ecd

    def forward(self, x):
        # Input value that's untouched and will be added to normalization layer
        residual_x = x
        # Calcualte self-attention matrix
        x = self.attention(x)
        # Dropout
        x = self.dropout(x)
        # Add residual_x to x and normalize
        x = self.norm1(x + residual_x)
        # Create another untouched residual input x
        residual_x = x
        # Pass to the feedforward neural network
        x = self.ffn(x)
        # Dropout
        x = self.dropout(x)
        # Add residual_x to the x and normalize again
        x = self.norm2(x + residual_x)
        # Output of encoder layer
        return x

# Multi-head attention class
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model                              # for example 512
        self.num_heads = num_heads                          # for example for 8 heads
        self.head_dim = d_model // num_heads                # head_dim will be 64
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)    # 512 x 1536
        self.linear_layer = nn.Linear(d_model, d_model)     # 512 x 512

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add a batch dimension at the front
        batch_size, sequence_length, d_model = x.size()     # for example 30 x 200 x 512
        #sequence_length, d_model = x.size()     # for example 30 x 200 x 512
        qkv = self.qkv_layer(x) # 30 x 200 x 1536
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim) # 30 x 200 x 8 x 192
        #qkv = qkv.reshape(sequence_length, self.num_heads, 3 * self.head_dim) # 30 x 200 x 8 x 192
        qkv = qkv.permute(0, 2, 1, 3) # 30 x 8 x 200 x 192
        q, k, v = qkv.chunk(3, dim = -1) # breakup using the last dimension, each are 30 x 8 x 200 x 64

        values, attention = single_head_attention(q, k, v)
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        #values = values.reshape(sequence_length, self.num_heads * self.head_dim)
        out = self.linear_layer(values)

        return out
        
# Layer normalization
class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps = 1e-8):
        super().__init__()
        self.eps = eps # to take care of zero division
        self.parameters_shape = parameters_shape
        self.gamma = nn.Parameter(torch.ones(parameters_shape)) # learnable parameter "std" (512,)
        self.beta = nn.Parameter(torch.zeros(parameters_shape)) # learnable parameter "mean" (512,)

    def forward(self, inputs):
        dims = [-(i+1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim = dims, keepdim = True) # eg. for (30, 200, 512) inputs, mean -> (30, 200, 1)
        var = ((inputs - mean)**2).mean(dim = dims, keepdim = True) # (30, 200, 1) 
        std = (var + self.eps).sqrt() # (30, 200, 1)
        y = (inputs - mean) / std # Normalized output (30, 200, 512)
        out = self.gamma*y + self.beta # Apply learnable parameters

        return out

# Feedforward MLP
class MLP(nn.Module):
    def __init__(self, d_model, num_ffn, act_fn, dropout_r = 0.1):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(d_model, num_ffn)
        self.linear2 = nn.Linear(num_ffn, d_model)
        self.act_fn = act_fn
        self.dropout = nn.Dropout(p = dropout_r)

    def forward(self, x):
        x = self.linear1(self.act_fn(x))
        x = self.dropout(x)
        x = self.linear2(self.act_fn(x))

        return x
    

def single_head_attention(q, k, v):
    # attention(q, k, v) = softmax(qK.T/sqrt(dk)V)
    d_k = q.size()[-1] # 64
    # Only transpose the last 2 dimensions, because the first dimension is the batch size
    # scale the value with square root of d_k which is a constant value
    val_before_softmax = torch.matmul(q, k.transpose(-1,-2))/math.sqrt(d_k)
    attention = F.softmax(val_before_softmax, dim = -1) # 200 x 200
    # Multiply attention matrix with value matrix
    values = torch.matmul(attention, v) # 200 x 64
    return values, attention

class EnBaseLayer(nn.Module):
    def __init__(self, hidden_dim, edge_feat_dim, num_r_gaussian, update_x=True, act_fn='silu', norm=False):
        super().__init__()
        self.r_min = 0.
        self.r_max = 10.
        self.hidden_dim = hidden_dim
        self.num_r_gaussian = num_r_gaussian
        self.edge_feat_dim = edge_feat_dim
        self.update_x = update_x
        self.act_fn = act_fn
        self.norm = norm
        if num_r_gaussian > 1:
            self.distance_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian)
        self.edge_mlp = MLP(2 * hidden_dim + edge_feat_dim + num_r_gaussian, hidden_dim, hidden_dim,
                            num_layer=2, norm=norm, act_fn=act_fn, act_last=True)
        self.edge_inf = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        if self.update_x:
            # self.x_mlp = MLP(hidden_dim, 1, hidden_dim, num_layer=2, norm=norm, act_fn=act_fn)
            x_mlp = [nn.Linear(hidden_dim, hidden_dim), NONLINEARITIES[act_fn]]
            layer = nn.Linear(hidden_dim, 1, bias=False)
            torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
            x_mlp.append(layer)
            x_mlp.append(nn.Tanh())
            self.x_mlp = nn.Sequential(*x_mlp)

        self.node_mlp = MLP(2 * hidden_dim, hidden_dim, hidden_dim, num_layer=2, norm=norm, act_fn=act_fn)

    def forward(self, h, x, edge_index, mask_ligand, edge_attr=None):
        src, dst = edge_index
        hi, hj = h[dst], h[src]
        # \phi_e in Eq(3)
        rel_x = x[dst] - x[src]
        d_sq = torch.sum(rel_x ** 2, -1, keepdim=True)
        if self.num_r_gaussian > 1:
            d_feat = self.distance_expansion(torch.sqrt(d_sq + 1e-8))
        else:
            d_feat = d_sq
        if edge_attr is not None:
            edge_feat = torch.cat([d_feat, edge_attr], -1)
        else:
            edge_feat = d_sq

        mij = self.edge_mlp(torch.cat([hi, hj, edge_feat], -1))
        eij = self.edge_inf(mij)
        mi = scatter_sum(mij * eij, dst, dim=0, dim_size=h.shape[0])

        # h update in Eq(6)
        h = h + self.node_mlp(torch.cat([mi, h], -1))
        if self.update_x:
            # x update in Eq(4)
            xi, xj = x[dst], x[src]
            # (xi - xj) / (\|xi - xj\| + C) to make it more stable
            delta_x = scatter_sum((xi - xj) / (torch.sqrt(d_sq + 1e-8) + 1) * self.x_mlp(mij), dst, dim=0)
            x = x + delta_x * mask_ligand[:, None]  # only ligand positions will be updated

        return h, x


class EGTF(nn.Module):
    def __init__(self, num_layers, hidden_dim, edge_feat_dim, num_r_gaussian, k=32, cutoff=10.0, cutoff_mode='knn',
                 update_x=True, act_fn='silu', norm=False, 
                 # Transformer Encoder parameters
                 num_encoder = 1, num_heads = 8, num_ffn = 128, act_fn_ecd = 'ReLU', dropout_r = 0.1,
                 device = 'cpu'):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.edge_feat_dim = edge_feat_dim
        self.num_r_gaussian = num_r_gaussian
        self.update_x = update_x
        self.act_fn = act_fn
        self.norm = norm
        self.k = k
        self.cutoff = cutoff
        self.cutoff_mode = cutoff_mode
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=num_r_gaussian)
        self.net = self._build_network()
        self.device = device

        self.encoder_layers = \
            nn.ModuleList([EncoderLayer(hidden_dim, num_heads,
                                        num_ffn, dropout_r, act_fn_ecd) 
                                        for _ in range(num_encoder)])

    def _build_network(self):
        # Equivariant layers
        layers = []
        for l_idx in range(self.num_layers):
            layer = EnBaseLayer(self.hidden_dim, self.edge_feat_dim, self.num_r_gaussian,
                                update_x=self.update_x, act_fn=self.act_fn, norm=self.norm)
            layers.append(layer)
        return nn.ModuleList(layers)

    # todo: refactor
    def _connect_edge(self, x, mask_ligand, batch):
        # if self.cutoff_mode == 'radius':
        #     edge_index = radius_graph(x, r=self.r, batch=batch, flow='source_to_target')
        if self.cutoff_mode == 'knn':
            if debug:
                edge_index = knn_graph(x.to('cuda:0'), k=self.k, batch=batch.to("cuda:0"), flow='source_to_target')
                edge_index = edge_index.to(self.device)
                x = x.to(self.device)
            else:
                edge_index = knn_graph(x, k=self.k, batch=batch, flow='source_to_target')


        elif self.cutoff_mode == 'hybrid':
            edge_index = batch_hybrid_edge_connection(
                x, k=self.k, mask_ligand=mask_ligand, batch=batch, add_p_index=True)
        else:
            raise ValueError(f'Not supported cutoff mode: {self.cutoff_mode}')
        return edge_index

    # todo: refactor
    @staticmethod
    def _build_edge_type(edge_index, mask_ligand):
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1
        n_dst = mask_ligand[dst] == 1
        edge_type[n_src & n_dst] = 0
        edge_type[n_src & ~n_dst] = 1
        edge_type[~n_src & n_dst] = 2
        edge_type[~n_src & ~n_dst] = 3
        edge_type = F.one_hot(edge_type, num_classes=4)
        return edge_type

    def forward(self, h, x, mask_ligand, batch, return_all=False):

        all_x = [x]
        all_h = [h]

        for l_idx, layer in enumerate(self.net):
            edge_index = self._connect_edge(x, mask_ligand, batch)
            edge_type = self._build_edge_type(edge_index, mask_ligand)
            h, x = layer(h, x, edge_index, mask_ligand, edge_attr=edge_type)
            all_x.append(x)
            all_h.append(h)

        for layer in self.encoder_layers:
            h = layer(h)

        outputs = {'x': x, 'h': h}

        if return_all:
            outputs.update({'all_x': all_x, 'all_h': all_h})

        return outputs
