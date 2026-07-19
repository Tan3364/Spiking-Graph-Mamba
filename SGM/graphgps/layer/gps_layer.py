import os
import json
import numpy as np
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from performer_pytorch import SelfAttention
from torch_geometric.data import Batch
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch
from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvESLapPE
from graphgps.layer.bigbird_layer import SingleBigBirdLayer
from mamba_ssm import Mamba

from graphgps.layer.neuron import SDNNeuron, BPTTNueron, SLTTNueron
from graphgps.layer.surrogate import quant4_surrogate
from graphgps.layer.surrogate import ternary_piecewise_quadratic_surrogate
from graphgps.layer.bayesian_linear import bayesian_linear
from torch_geometric.utils import degree, sort_edge_index
from typing import List

import numpy as np
import torch
from torch import Tensor
from spikingjelly.clock_driven import encoding


def record_fire_rate_json(fire_rate, phase, model, json_path):
    """
    将当前 fire_rate 和阶段信息保存到 JSON 文件中。
    如果文件不存在则创建；存在则追加。
    """
    # 如果文件存在，加载旧数据；否则新建空字典
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    # 生成唯一 key
    step_key = f"batch_{len(data)}"

    # 添加一条新记录
    data[step_key] = {
        "phase": phase,
        "model": model,
        "fire_rate": round(float(fire_rate), 6)
    }

    # 写回 JSON 文件（缩进更美观）
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)


def permute_nodes_within_identity(identities):
    unique_identities, inverse_indices = torch.unique(identities, return_inverse=True)
    node_indices = torch.arange(len(identities), device=identities.device)

    masks = identities.unsqueeze(0) == unique_identities.unsqueeze(1)

    # Generate random indices within each identity group using torch.randint
    permuted_indices = torch.cat([
        node_indices[mask][torch.randperm(mask.sum(), device=identities.device)] for mask in masks
    ])
    return permuted_indices

def sort_rand_gpu(pop_size, num_samples, neighbours):
    # Randomly generate indices and select num_samples in neighbours
    idx_select = torch.argsort(torch.rand(pop_size, device=neighbours.device))[:num_samples]
    neighbours = neighbours[idx_select]
    return neighbours

def augment_seq(edge_index, batch, num_k = -1):
    unique_batches = torch.unique(batch)
    # Initialize list to store permuted indices
    permuted_indices = []
    mask = []

    for batch_index in unique_batches:
        # Extract indices for the current batch
        indices_in_batch = (batch == batch_index).nonzero().squeeze()
        for k in indices_in_batch:
            neighbours = edge_index[1][edge_index[0]==k]
            if num_k > 0 and len(neighbours) > num_k:
                neighbours = sort_rand_gpu(len(neighbours), num_k, neighbours)
            permuted_indices.append(neighbours)
            mask.append(torch.zeros(neighbours.shape, dtype=torch.bool, device=batch.device))
            permuted_indices.append(torch.tensor([k], device=batch.device))
            mask.append(torch.tensor([1], dtype=torch.bool, device=batch.device))
    permuted_indices = torch.cat(permuted_indices)
    mask = torch.cat(mask)
    return permuted_indices.to(device=batch.device), mask.to(device=batch.device)

def lexsort(
    keys: List[Tensor],
    dim: int = -1,
    descending: bool = False,
) -> Tensor:
    r"""Performs an indirect stable sort using a sequence of keys.

    Given multiple sorting keys, returns an array of integer indices that
    describe their sort order.
    The last key in the sequence is used for the primary sort order, the
    second-to-last key for the secondary sort order, and so on.

    Args:
        keys ([torch.Tensor]): The :math:`k` different columns to be sorted.
            The last key is the primary sort key.
        dim (int, optional): The dimension to sort along. (default: :obj:`-1`)
        descending (bool, optional): Controls the sorting order (ascending or
            descending). (default: :obj:`False`)
    """
    assert len(keys) >= 1

    out = keys[0].argsort(dim=dim, descending=descending, stable=True)
    for k in keys[1:]:
        index = k.gather(dim, out)
        index = index.argsort(dim=dim, descending=descending, stable=True)
        out = out.gather(dim, index)
    return out


def permute_within_batch(batch):
    # Enumerate over unique batch indices
    unique_batches = torch.unique(batch)

    # Initialize list to store permuted indices
    permuted_indices = []

    for batch_index in unique_batches:
        # Extract indices for the current batch
        indices_in_batch = (batch == batch_index).nonzero().squeeze()

        # Permute indices within the current batch
        permuted_indices_in_batch = indices_in_batch[torch.randperm(len(indices_in_batch))]

        # Append permuted indices to the list
        permuted_indices.append(permuted_indices_in_batch)

    # Concatenate permuted indices into a single tensor
    permuted_indices = torch.cat(permuted_indices)

    return permuted_indices

class GPSLayer(nn.Module):
    """Local MPNN + full graph attention x-former layer + Spiking Module.
    """

    def __init__(self, dim_h,
                 local_gnn_type, global_model_type, num_heads,
                 pna_degrees=None, equivstable_pe=False, dropout=0.0,
                 attn_dropout=0.0, layer_norm=True, batch_norm=False,
                 bigbird_cfg=None, neuron_type="sdn",
                learnable_vth=True, shared_vth=False):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.NUM_BUCKETS = 3
        self.learnable_vth = learnable_vth

        #存档single_layer_fire_rate
        self.fire_rate_list = {
            "pos_gatedgcn_fire_rate": [],
            "neg_gatedgcn_fire_rate": [],
            "pos_train_mamba_fire_rate": [],
            "neg_train_mamba_fire_rate": [],
            "h_abs_1_count": [],
            "h_abs_2_count": [],
        }

        #self.tau = tau
        #self.v_reset = v_reset
        #self.v_threshold = v_threshold
        #device = torch.device("cuda:0")
        #self.device = device

        # Local message-passing model.
        if local_gnn_type == 'None':
            self.local_model = None
        elif local_gnn_type == 'GENConv':
            self.local_model = pygnn.GENConv(dim_h, dim_h)
        elif local_gnn_type == 'GINE':
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                   nn.ReLU(),
                                   Linear_pyg(dim_h, dim_h))
            if self.equivstable_pe:  # Use specialised GINE layer for EquivStableLapPE.
                self.local_model = GINEConvESLapPE(gin_nn)
            else:
                self.local_model = pygnn.GINEConv(gin_nn)
        elif local_gnn_type == 'GAT':
            self.local_model = pygnn.GATConv(in_channels=dim_h,
                                             out_channels=dim_h // num_heads,
                                             heads=num_heads,
                                             edge_dim=dim_h)
        elif local_gnn_type == 'PNA':
            # Defaults from the paper.
            # aggregators = ['mean', 'min', 'max', 'std']
            # scalers = ['identity', 'amplification', 'attenuation']
            aggregators = ['mean', 'max', 'sum']
            scalers = ['identity']
            deg = torch.from_numpy(np.array(pna_degrees))
            self.local_model = pygnn.PNAConv(dim_h, dim_h,
                                             aggregators=aggregators,
                                             scalers=scalers,
                                             deg=deg,
                                             edge_dim=16, # dim_h,
                                             towers=1,
                                             pre_layers=1,
                                             post_layers=1,
                                             divide_input=False)
        elif local_gnn_type == 'CustomGatedGCN':
            self.local_model = GatedGCNLayer(dim_h, dim_h,
                                             dropout=dropout,
                                             residual=True,
                                             equivstable_pe=equivstable_pe)
        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        self.local_gnn_type = local_gnn_type

        # Global attention transformer-style model.
        if global_model_type == 'None':
            self.self_attn = None
        elif global_model_type == 'Transformer':
            self.self_attn = torch.nn.MultiheadAttention(
                dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
            # self.global_model = torch.nn.TransformerEncoderLayer(
            #     d_model=dim_h, nhead=num_heads,
            #     dim_feedforward=2048, dropout=0.1, activation=F.relu,
            #     layer_norm_eps=1e-5, batch_first=True)
        elif global_model_type == 'Performer':
            self.self_attn = SelfAttention(
                dim=dim_h, heads=num_heads,
                dropout=self.attn_dropout, causal=False)
        elif global_model_type == "BigBird":
            bigbird_cfg.dim_hidden = dim_h
            bigbird_cfg.n_heads = num_heads
            bigbird_cfg.dropout = dropout
            self.self_attn = SingleBigBirdLayer(bigbird_cfg)
            # layer_type: CustomGatedGCN+Mamba_Hybrid_Degree_Noise
        elif 'Mamba' in global_model_type:
            if global_model_type.split('_')[-1] == '2':
                self.self_attn = Mamba(d_model=dim_h, # Model dimension d_model
                        d_state=8,  # SSM state expansion factor
                        d_conv=4,    # Local convolution width
                        expand=2,    # Block expansion factor
                    )
            elif global_model_type.split('_')[-1] == '4':
                self.self_attn = Mamba(d_model=dim_h, # Model dimension d_model
                        d_state=4,  # SSM state expansion factor
                        d_conv=4,    # Local convolution width
                        expand=4,    # Block expansion factor
                    )
            elif global_model_type.split('_')[-1] == 'Multi':
                self.self_attn = []
                for i in range(4):
                    self.self_attn.append(Mamba(d_model=dim_h, # Model dimension d_model
                    d_state=16,  # SSM state expansion factor
                    d_conv=4,    # Local convolution width
                    expand=1,    # Block expansion factor
                    ))
            elif global_model_type.split('_')[-1] == 'SmallConv':
                self.self_attn = Mamba(d_model=dim_h, # Model dimension d_model
                        d_state=16,  # SSM state expansion factor
                        d_conv=2,    # Local convolution width
                        expand=1,    # Block expansion factor
                    )
            elif global_model_type.split('_')[-1] == 'SmallState':
                self.self_attn = Mamba(d_model=dim_h, # Model dimension d_model
                        d_state=8,  # SSM state expansion factor
                        d_conv=4,    # Local convolution width
                        expand=1,    # Block expansion factor
                    )
            else:
                self.self_attn = Mamba(d_model=dim_h, # Model dimension d_model
                        d_state=16,  # SSM state expansion factor
                        d_conv=4,    # Local convolution width
                        expand=1,    # Block expansion factor
                    )
        else:
            raise ValueError(f"Unsupported global x-former model: "
                             f"{global_model_type}")
        self.global_model_type = global_model_type

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            # self.norm1_local = pygnn.norm.LayerNorm(dim_h)
            self.norm1_attn = pygnn.norm.LayerNorm(dim_h)
            self.norm1_local = pygnn.norm.GraphNorm(dim_h)
            #self.norm1_attn = pygnn.norm.GraphNorm(dim_h)
            #self.norm1_attn = nn.RMSNorm(normalized_shape=dim_h)
            # self.norm1_local = pygnn.norm.InstanceNorm(dim_h)
            # self.norm1_attn = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # norm2为LayerNorm for Mamba
        #self.norm2_attn = pygnn.norm.LayerNorm(dim_h)
        # norm3为BtachNorm for Mamba
        #self.norm3_attn = nn.BatchNorm1d(dim_h)
        # norm4为LayerNorm for Graph
        #self.norm4_local = pygnn.norm.GraphNorm(dim_h)
        # norm5为BtachNorm for Graph
        #self.norm5_local = nn.BatchNorm1d(dim_h)

        # neuron module
        if neuron_type == "sdn":
            #self.neuron = SDNNeuron(piecewise_quadratic_surrogate(),threshold)
            self.neuron_1 = SDNNeuron(ternary_piecewise_quadratic_surrogate())
            self.neuron_2 = SDNNeuron(ternary_piecewise_quadratic_surrogate())
            #print('sdn')
            #减小阈值
        elif neuron_type == "bptt":
            self.neuron_1 = BPTTNueron(ternary_piecewise_quadratic_surrogate())
            self.neuron_2 = BPTTNueron(ternary_piecewise_quadratic_surrogate())
            #print('bptt')
        elif neuron_type == "sltt":
            self.neuron_1 = SLTTNueron(ternary_piecewise_quadratic_surrogate())
            self.neuron_2 = SLTTNueron(ternary_piecewise_quadratic_surrogate())
            #print('sltt')
        else:
            raise ValueError(f"Please choose one neuron.")

        # learnable_vth
        if learnable_vth:
            if shared_vth:
                self.ln_vth = nn.Parameter(torch.zeros(1))
            else:
                self.ln_vth = nn.Parameter(torch.zeros(dim_h, 1))

        # Feed Forward block. (MLP layer)
        self.activation = F.relu
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        if self.layer_norm:
            # self.norm2 = pygnn.norm.LayerNorm(dim_h)
            self.norm2 = pygnn.norm.GraphNorm(dim_h)
            # self.norm2 = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

        #self.to(self.device)

        #neuron dropout
        #dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        #dropout_fn = DropoutNd
        #self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, batch):
        #print("batch:",batch)
        h = batch.x
        #print("batch.x",h)
        encoder = encoding.PoissonEncoder()
        #h = encoder(h)
        h_in1 = batch.x  # for first residual connection
        #print("h_inital:",h)
        h_out_list = []

        # Local MPNN with edge attributes.
        if self.local_model is not None:
            self.local_model: pygnn.conv.MessagePassing  # Typing hint.
            if self.local_gnn_type == 'CustomGatedGCN':
                es_data = None
                if self.equivstable_pe:
                    es_data = batch.pe_EquivStableLapPE
                local_out = self.local_model(
                    Batch(
                        batch=batch,
                        x=h,
                        edge_index=batch.edge_index,
                        edge_attr=batch.edge_attr,
                        pe_EquivStableLapPE=es_data
                    )
                )
                #print("local_out:",local_out, sigma_fire_rate)
                # GatedGCN does residual connection and dropout internally.
                #self.fire_rate_list["pos_sigma_fire_rate"].append(pos_sigma_fire_rate.detach().cpu().item())
                #self.fire_rate_list["neg_sigma_fire_rate"].append(neg_sigma_fire_rate.detach().cpu().item())
                h_local = local_out.x
                batch.edge_attr = local_out.edge_attr
                #print("1", h_local)

                # 泊松编码器
                ##encoder = encoding.PoissonEncoder()

                ##h_local = self.norm4_local(h_local)
                # B_Plan(L维度展开)
                # 转置
                ##h_local = h_local.transpose(-2, -1)  # [dim,l]
                ##if self.learnable_vth:
                    ##h_local = h_local / torch.exp(self.ln_vth)

                #normalization
                #print(h_local)
                ##h_local, pos_gatedgcn_fire_rate, neg_gatedgcn_fire_rate = self.neuron_1(h_local, 0.625 , name="graph")  # [l, dim]
                #h_local, pos_gatedgcn_fire_rate, neg_gatedgcn_fire_rate = self.neuron_1(h_local, name="graph")  # [l, dim]
                #print("h_local:",h_local,h_local.shape)
                ##self.fire_rate_list["pos_gatedgcn_fire_rate"].append(pos_gatedgcn_fire_rate.detach().cpu().item())
                ##self.fire_rate_list["neg_gatedgcn_fire_rate"].append(neg_gatedgcn_fire_rate.detach().cpu().item())

                #new add    # 去掉残差
                ##h_local = self.dropout_local(h_local)
                #h_local = h_in1 + h_local

            else:
                if self.equivstable_pe:
                    h_local = self.local_model(h, batch.edge_index, batch.edge_attr,
                                               batch.pe_EquivStableLapPE)
                else:
                    h_local = self.local_model(h, batch.edge_index, batch.edge_attr)
                # [l,dim]
                h_local = self.dropout_local(h_local)
                h_local = h_in1 + h_local  # Residual connection.

            #print("h_local",h_local)
            # new add
            #h_local = self.norm4_local(h_local)

            # 1. 标准化到合适范围
            #h_mean = h_local.mean(dim=-1, keepdim=True)
            #h_std = h_local.std(dim=-1, keepdim=True) + 1e-8
            #h_local = (h_local - h_mean) / h_std
            #h_local = torch.tanh(h_local)
            if self.layer_norm:
                h_local = self.norm1_local(h_local, batch.batch)
            if self.batch_norm:
                h_local = self.norm1_local(h_local)

            h_local = h_local.transpose(-2, -1)  # [dim,l]

            if self.learnable_vth:
                h_local = h_local / torch.exp(self.ln_vth)

            #normalization
            #print(h_local)
            h_local, pos_gatedgcn_fire_rate, neg_gatedgcn_fire_rate = self.neuron_1(h_local, 1.0, name="graph")  # [l, dim]
            #h_local, pos_gatedgcn_fire_rate, neg_gatedgcn_fire_rate = self.neuron_1(h_local, name="graph")  # [l, dim]
            #print("h_local:",h_local,h_local.shape)
            self.fire_rate_list["pos_gatedgcn_fire_rate"].append(pos_gatedgcn_fire_rate)
            self.fire_rate_list["neg_gatedgcn_fire_rate"].append(neg_gatedgcn_fire_rate)
            #print("h_final",h_local)
            h_out_list.append(h_local)

        # Multi-head attention.
        if self.self_attn is not None:
            if self.global_model_type in ['Transformer', 'Performer', 'BigBird', 'Mamba']:
                h_dense, mask = to_dense_batch(h, batch.batch)
            if self.global_model_type == 'Transformer':
                h_attn = self._sa_block(h_dense, None, ~mask)[mask]
            elif self.global_model_type == 'Performer':
                h_attn = self.self_attn(h_dense, mask=mask)[mask]
            elif self.global_model_type == 'BigBird':
                h_attn = self.self_attn(h_dense, attention_mask=mask)

            elif self.global_model_type == 'Mamba':
                h_attn = self.self_attn(h_dense)[mask]

            elif self.global_model_type == 'Mamba_Permute':
                h_ind_perm = permute_within_batch(batch.batch)
                h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]

            elif self.global_model_type == 'Mamba_Degree':
                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                # indcies that sort by batch and then deg, by ascending order
                h_ind_perm = lexsort([deg, batch.batch])
                h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]

            elif self.global_model_type == 'Mamba_Hybrid':
                if batch.split == 'train':
                    h_ind_perm = permute_within_batch(batch.batch)
                    h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    for i in range(5):
                        h_ind_perm = permute_within_batch(batch.batch)
                        h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif 'Mamba_Hybrid_Degree' == self.global_model_type:
                if batch.split == 'train':
                    h_ind_perm = permute_within_batch(batch.batch)
                    #h_ind_perm = permute_nodes_within_identity(batch.batch)
                    deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                    h_ind_perm_1 = lexsort([deg[h_ind_perm], batch.batch[h_ind_perm]])
                    h_ind_perm = h_ind_perm[h_ind_perm_1]
                    h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    if self.global_model_type.split('_')[-1] == 'Multi':
                        h_attn_list = []
                        for mod in self.self_attn:
                            mod = mod.to(h_dense.device)
                            h_attn = mod(h_dense)[mask][h_ind_perm_reverse]
                            h_attn_list.append(h_attn)
                        h_attn = sum(h_attn_list) / len(h_attn_list)
                    else:
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    for i in range(5):
                        #h_ind_perm = permute_nodes_within_identity(batch.batch)
                        h_ind_perm = permute_within_batch(batch.batch)
                        deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                        h_ind_perm_1 = lexsort([deg[h_ind_perm], batch.batch[h_ind_perm]])
                        h_ind_perm = h_ind_perm[h_ind_perm_1]
                        h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        if self.global_model_type.split('_')[-1] == 'Multi':
                            h_attn_list = []
                            for mod in self.self_attn:
                                mod = mod.to(h_dense.device)
                                h_attn = mod(h_dense)[mask][h_ind_perm_reverse]
                                h_attn_list.append(h_attn)
                            h_attn = sum(h_attn_list) / len(h_attn_list)
                        else:
                            h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        #h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif 'Mamba_Hybrid_Degree_Noise' == self.global_model_type:
                if batch.split == 'train':
                    deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                    #deg_noise = torch.std(deg)*torch.randn(deg.shape).to(deg.device)
                    # Potentially use torch.rand_like?
                    #deg_noise = torch.std(deg)*torch.randn(deg.shape).to(deg.device)
                    #deg_noise = torch.randn(deg.shape).to(deg.device)
                    deg_noise = torch.rand_like(deg).to(deg.device)
                    h_ind_perm = lexsort([deg+deg_noise, batch.batch])
                    h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse] #[l,dim]


                    #h_attn = self.norm2_attn(h_attn)
                    #print("layer norm",h_attn)

                    ##h_attn = self.norm2_attn(h_attn)


                    # 转置
                    ##h_attn = h_attn.transpose(-2, -1)  # [dim,l]
                    #print("mamba",h_attn)
                    # save
#                    save_dir = "./h_attn_outputs"
#                    os.makedirs(save_dir, exist_ok=True)
#                    filepath = os.path.join(save_dir, "h_attn_ave.pt")
#
#                    if os.path.exists(filepath):
#                        saved_data = torch.load(filepath)
#                    else:
#                        saved_data = []
#
#                    h_attn_ave = h_attn.mean(dim=0)
#                    if h_attn_ave.numel() > 1024:
#                        h_attn_ave = h_attn_ave[:1024]
#
#                    # 添加到缓存列表
#                    saved_data.append(h_attn_ave)
#                    # ⭐ 当累计达到 5000 条数据，保存并清空缓存 ⭐
#                    if len(saved_data) >= 5000:
#                        print(f"Saving 5000 entries to {filepath} ...")
#                        torch.save(saved_data, filepath)
#                        saved_data = []  # 清空缓存（防止继续无限增长）


                    #print("2",h_attn.shape)
                    #B_Plan for Spike Module(L维度展开)
                    # neuron part
                    ##if self.learnable_vth:
                        ##h_attn = h_attn / torch.exp(self.ln_vth)
                    #h_attn = self.dropout(self.neuron(h_attn))

                    ##h_attn, pos_train_mamba_fire_rate, neg_train_mamba_fire_rate = self.neuron_2(h_attn, 0.3, name="mamba")  # [l, dim]

                    ##self.fire_rate_list["pos_train_mamba_fire_rate"].append(pos_train_mamba_fire_rate.detach().cpu().item())
                    ##self.fire_rate_list["neg_train_mamba_fire_rate"].append(neg_train_mamba_fire_rate.detach().cpu().item())
                    #print(f"Train Mamba Firing rate: {fire_rate:.4f}")
                    #record_fire_rate_json(fire_rate, phase="train", model="mamba", json_path="mamba_train_firing_rate_log.json")
                    #print("3",h_attn.shape)
                    #h_attn = self.output_linear(h_attn)

                else:
                    mamba_arr = []
                    deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                    for i in range(5):
                        #deg_noise = torch.std(deg)*torch.randn(deg.shape).to(deg.device)
                        #deg_noise = torch.randn(deg.shape).to(deg.device)
                        deg_noise = torch.rand_like(deg).to(deg.device)
                        h_ind_perm = lexsort([deg+deg_noise, batch.batch])
                        h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse] #[batch,l,dim]

                        #h_attn = self.norm2_attn(h_attn)
                        #print("layer_norm_eval",h_attn)
                        ##h_attn = self.norm2_attn(h_attn)
                        # 转置
                        ##h_attn = h_attn.transpose(-2, -1)  # [dim,l]

                        #print("mamba",h_attn)
                        #print("4",h_attn.shape)
                        # neuron part
                        ##if self.learnable_vth:
                            ##h_attn = h_attn / torch.exp(self.ln_vth)
                        #h_attn = self.dropout(self.neuron(h_attn))
                        ##h_attn, pos_eval_mamba_fire_rate, neg_eval_mamba_fire_rate = self.neuron_2(h_attn, 0.3, name="mamba_eval") # [l, dim]
                        #print(name)
                        ##self.fire_rate_list["pos_eval_mamba_fire_rate_single"].append(pos_eval_mamba_fire_rate.detach().cpu().item())
                        ##self.fire_rate_list["neg_eval_mamba_fire_rate_single"].append(neg_eval_mamba_fire_rate.detach().cpu().item())
                        #print(f"Else Mamba Firing rate: {fire_rate:.4f}")
                        #record_fire_rate_json(fire_rate, phase="eval", model="mamba", json_path="mamba_eval_firing_rate_log.json")
                        #print("5",h_attn.shape)
                        #h_attn = self.output_linear(h_attn)
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5
                    ##pos_eval_mamba_fire_rate_ave = sum(self.fire_rate_list["pos_eval_mamba_fire_rate_single"]) / len(self.fire_rate_list["pos_eval_mamba_fire_rate_single"])
                    ##neg_eval_mamba_fire_rate_ave = sum(self.fire_rate_list["neg_eval_mamba_fire_rate_single"]) / len(self.fire_rate_list["neg_eval_mamba_fire_rate_single"])
                    ##self.fire_rate_list["pos_eval_mamba_fire_rate_ave"].append(pos_eval_mamba_fire_rate_ave)
                    ##self.fire_rate_list["neg_eval_mamba_fire_rate_ave"].append(neg_eval_mamba_fire_rate_ave)

            elif 'Mamba_Hybrid_Degree_Noise_Bucket' == self.global_model_type:
                if batch.split == 'train':
                    deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                    #deg_noise = torch.std(deg)*torch.randn(deg.shape).to(deg.device)
                    deg_noise = torch.rand_like(deg).to(deg.device)
                    #deg_noise = torch.randn(deg.shape).to(deg.device)
                    deg = deg + deg_noise
                    indices_arr, emb_arr = [],[]
                    bucket_assign = torch.randint_like(deg, 0, self.NUM_BUCKETS).to(deg.device)
                    for i in range(self.NUM_BUCKETS):
                        ind_i = (bucket_assign==i).nonzero().squeeze()
                        h_ind_perm_sort = lexsort([deg[ind_i], batch.batch[ind_i]])
                        h_ind_perm_i = ind_i[h_ind_perm_sort]
                        h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                        h_dense = self.self_attn(h_dense)[mask]
                        indices_arr.append(h_ind_perm_i)
                        emb_arr.append(h_dense)
                    h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                    h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                    ##h_attn = self.norm2_attn(h_attn)
                    ##h_attn = h_attn.transpose(-2, -1)  # [dim,l]
                    ##if self.learnable_vth:
                        ##h_attn = h_attn / torch.exp(self.ln_vth)
                    ##h_attn, pos_train_mamba_fire_rate, neg_train_mamba_fire_rate = self.neuron_2(h_attn, 0.3, name="mamba")  # [l, dim]
                    ##self.fire_rate_list["pos_train_mamba_fire_rate"].append(pos_train_mamba_fire_rate.detach().cpu().item())
                    ##self.fire_rate_list["neg_train_mamba_fire_rate"].append(neg_train_mamba_fire_rate.detach().cpu().item())
                    #print(f"Train Mamba Firing rate: {fire_rate:.4f}")
                    #record_fire_rate_json(fire_rate, phase="train", model="mamba", json_path="mamba_train_firing_rate_log.json")

                else:
                    mamba_arr = []
                    #batch.x = node features
                    deg_ = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                    for i in range(5):
                        #deg_noise = torch.std(deg)*torch.randn(deg.shape).to(deg.device)
                        deg_noise = torch.rand_like(deg_).to(deg_.device)
                        #deg_noise = torch.randn(deg.shape).to(deg.device)
                        deg = deg_ + deg_noise
                        indices_arr, emb_arr = [],[]
                        bucket_assign = torch.randint_like(deg, 0, self.NUM_BUCKETS).to(deg.device)
                        for i in range(self.NUM_BUCKETS):
                            ind_i = (bucket_assign==i).nonzero().squeeze()
                            h_ind_perm_sort = lexsort([deg[ind_i], batch.batch[ind_i]])
                            h_ind_perm_i = ind_i[h_ind_perm_sort]
                            h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                            h_dense = self.self_attn(h_dense)[mask]
                            indices_arr.append(h_ind_perm_i)
                            emb_arr.append(h_dense)
                        h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                        h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                        ##h_attn = self.norm2_attn(h_attn)
                        ##h_attn = h_attn.transpose(-2, -1)  # [dim,l]
                        #print("4",h_attn.shape)
                        # neuron part
                        ##if self.learnable_vth:
                            ##h_attn = h_attn / torch.exp(self.ln_vth)
                        #h_attn = self.dropout(self.neuron(h_attn))
                        ##h_attn, pos_eval_mamba_fire_rate, neg_eval_mamba_fire_rate = self.neuron_2(h_attn, 0.3, name="mamba_eval") # [l, dim]

                        ##self.fire_rate_list["pos_eval_mamba_fire_rate_single"].append(pos_eval_mamba_fire_rate.detach().cpu().item())
                        ##self.fire_rate_list["neg_eval_mamba_fire_rate_single"].append(neg_eval_mamba_fire_rate.detach().cpu().item())
                        #print(f"Else Mamba Firing rate: {fire_rate:.4f}")
                        #record_fire_rate_json(fire_rate, phase="eval", model="mamba", json_path="mamba_eval_firing_rate_log.json")
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5
                    ##pos_eval_mamba_fire_rate_ave = sum(self.fire_rate_list["pos_eval_mamba_fire_rate_single"]) / len(self.fire_rate_list["pos_eval_mamba_fire_rate_single"])
                    ##neg_eval_mamba_fire_rate_ave = sum(self.fire_rate_list["neg_eval_mamba_fire_rate_single"]) / len(self.fire_rate_list["neg_eval_mamba_fire_rate_single"])
                    ##self.fire_rate_list["pos_eval_mamba_fire_rate_ave"].append(pos_eval_mamba_fire_rate_ave)
                    ##self.fire_rate_list["neg_eval_mamba_fire_rate_ave"].append(neg_eval_mamba_fire_rate_ave)

            elif 'Mamba_Hybrid_Noise' == self.global_model_type:
                if batch.split == 'train':
                    deg_noise = torch.rand_like(batch.batch.to(torch.float)).to(batch.batch.device)
                    indices_arr, emb_arr = [],[]
                    bucket_assign = torch.randint_like(deg_noise, 0, self.NUM_BUCKETS).to(deg_noise.device)
                    for i in range(self.NUM_BUCKETS):
                        ind_i = (bucket_assign==i).nonzero().squeeze()
                        h_ind_perm_sort = lexsort([deg_noise[ind_i], batch.batch[ind_i]])
                        h_ind_perm_i = ind_i[h_ind_perm_sort]
                        h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                        h_dense = self.self_attn(h_dense)[mask]
                        indices_arr.append(h_ind_perm_i)
                        emb_arr.append(h_dense)
                    h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                    h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    deg = batch.batch.to(torch.float)
                    for i in range(5):
                        deg_noise = torch.rand_like(batch.batch.to(torch.float)).to(batch.batch.device)
                        indices_arr, emb_arr = [],[]
                        bucket_assign = torch.randint_like(deg_noise, 0, self.NUM_BUCKETS).to(deg_noise.device)
                        for i in range(self.NUM_BUCKETS):
                            ind_i = (bucket_assign==i).nonzero().squeeze()
                            h_ind_perm_sort = lexsort([deg_noise[ind_i], batch.batch[ind_i]])
                            h_ind_perm_i = ind_i[h_ind_perm_sort]
                            h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                            h_dense = self.self_attn(h_dense)[mask]
                            indices_arr.append(h_ind_perm_i)
                            emb_arr.append(h_dense)
                        h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                        h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif 'Mamba_Hybrid_Noise_Bucket' == self.global_model_type:
                if batch.split == 'train':
                    deg_noise = torch.rand_like(batch.batch.to(torch.float)).to(batch.batch.device)
                    h_ind_perm = lexsort([deg_noise, batch.batch])
                    h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    deg = batch.batch.to(torch.float)
                    for i in range(5):
                        deg_noise = torch.rand_like(batch.batch.to(torch.float)).to(batch.batch.device)
                        h_ind_perm = lexsort([deg_noise, batch.batch])
                        h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif self.global_model_type == 'Mamba_Eigen':
                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                centrality = batch.EigCentrality
                if batch.split == 'train':
                    # Shuffle within 1 STD
                    centrality_noise = torch.std(centrality)*torch.rand(centrality.shape).to(centrality.device)
                    # Order by batch, degree, and centrality
                    h_ind_perm = lexsort([centrality+centrality_noise, batch.batch])
                    h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    for i in range(5):
                        centrality_noise = torch.std(centrality)*torch.rand(centrality.shape).to(centrality.device)
                        h_ind_perm = lexsort([centrality+centrality_noise, batch.batch])
                        h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif 'Mamba_Eigen_Bucket' == self.global_model_type:
                centrality = batch.EigCentrality
                if batch.split == 'train':
                    centrality_noise = torch.std(centrality)*torch.rand(centrality.shape).to(centrality.device)
                    indices_arr, emb_arr = [],[]
                    bucket_assign = torch.randint_like(centrality, 0, self.NUM_BUCKETS).to(centrality.device)
                    for i in range(self.NUM_BUCKETS):
                        ind_i = (bucket_assign==i).nonzero().squeeze()
                        h_ind_perm_sort = lexsort([(centrality+centrality_noise)[ind_i], batch.batch[ind_i]])
                        h_ind_perm_i = ind_i[h_ind_perm_sort]
                        h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                        h_dense = self.self_attn(h_dense)[mask]
                        indices_arr.append(h_ind_perm_i)
                        emb_arr.append(h_dense)
                    h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                    h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    for i in range(5):
                        centrality_noise = torch.std(centrality)*torch.rand(centrality.shape).to(centrality.device)
                        indices_arr, emb_arr = [],[]
                        bucket_assign = torch.randint_like(centrality, 0, self.NUM_BUCKETS).to(centrality.device)
                        for i in range(self.NUM_BUCKETS):
                            ind_i = (bucket_assign==i).nonzero().squeeze()
                            h_ind_perm_sort = lexsort([(centrality+centrality_noise)[ind_i], batch.batch[ind_i]])
                            h_ind_perm_i = ind_i[h_ind_perm_sort]
                            h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                            h_dense = self.self_attn(h_dense)[mask]
                            indices_arr.append(h_ind_perm_i)
                            emb_arr.append(h_dense)
                        h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                        h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif self.global_model_type == 'Mamba_RWSE':
                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                RWSE_sum = torch.sum(batch.pestat_RWSE, dim=1)
                if batch.split == 'train':
                    # Shuffle within 1 STD
                    RWSE_noise = torch.std(RWSE_sum)*torch.randn(RWSE_sum.shape).to(RWSE_sum.device)
                    # Sort in descending order
                    # Nodes with more local connections -> larger sum in RWSE
                    # Nodes with more global connections -> smaller sum in RWSE
                    h_ind_perm = lexsort([-RWSE_sum+RWSE_noise, batch.batch])
                    # h_ind_perm = lexsort([-RWSE_sum+RWSE_noise, deg, batch.batch])
                    h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    # Sort in descending order
                    # Nodes with more local connections -> larger sum in RWSE
                    # Nodes with more global connections -> smaller sum in RWSE
                    # h_ind_perm = lexsort([-RWSE_sum, deg, batch.batch])
                    mamba_arr = []
                    for i in range(5):
                        RWSE_noise = torch.std(RWSE_sum)*torch.randn(RWSE_sum.shape).to(RWSE_sum.device)
                        h_ind_perm = lexsort([-RWSE_sum+RWSE_noise, batch.batch])
                        h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif self.global_model_type == 'Mamba_Cluster':
                h_ind_perm = permute_within_batch(batch.batch)
                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                if batch.split == 'train':
                    unique_cluster_n = len(torch.unique(batch.LouvainCluster))
                    permuted_louvain = torch.zeros(batch.LouvainCluster.shape).long().to(batch.LouvainCluster.device)
                    random_permute = torch.randperm(unique_cluster_n+1).long().to(batch.LouvainCluster.device)
                    for i in range(unique_cluster_n):
                        indices = torch.nonzero(batch.LouvainCluster == i).squeeze()
                        permuted_louvain[indices] = random_permute[i]
                    #h_ind_perm_1 = lexsort([deg[h_ind_perm], permuted_louvain[h_ind_perm], batch.batch[h_ind_perm]])
                    #h_ind_perm_1 = lexsort([permuted_louvain[h_ind_perm], deg[h_ind_perm], batch.batch[h_ind_perm]])
                    h_ind_perm_1 = lexsort([permuted_louvain[h_ind_perm], batch.batch[h_ind_perm]])
                    h_ind_perm = h_ind_perm[h_ind_perm_1]
                    h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    #h_ind_perm = lexsort([batch.LouvainCluster, deg, batch.batch])
                    #h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                    #h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    #h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                    mamba_arr = []
                    for i in range(5):
                        unique_cluster_n = len(torch.unique(batch.LouvainCluster))
                        permuted_louvain = torch.zeros(batch.LouvainCluster.shape).long().to(batch.LouvainCluster.device)
                        random_permute = torch.randperm(unique_cluster_n+1).long().to(batch.LouvainCluster.device)
                        for i in range(len(torch.unique(batch.LouvainCluster))):
                            indices = torch.nonzero(batch.LouvainCluster == i).squeeze()
                            permuted_louvain[indices] = random_permute[i]
                        # potentially permute it 5 times and average
                        # on the cluster level
                        #h_ind_perm_1 = lexsort([deg[h_ind_perm], permuted_louvain[h_ind_perm], batch.batch[h_ind_perm]])
                        #h_ind_perm_1 = lexsort([permuted_louvain[h_ind_perm], deg[h_ind_perm], batch.batch[h_ind_perm]])
                        h_ind_perm_1 = lexsort([permuted_louvain[h_ind_perm], batch.batch[h_ind_perm]])
                        h_ind_perm = h_ind_perm[h_ind_perm_1]
                        h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif self.global_model_type == 'Mamba_Hybrid_Degree_Bucket':
                if batch.split == 'train':
                    h_ind_perm = permute_within_batch(batch.batch)
                    deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                    indices_arr, emb_arr = [],[]
                    for i in range(self.NUM_BUCKETS):
                        ind_i = h_ind_perm[h_ind_perm%self.NUM_BUCKETS==i]
                        h_ind_perm_sort = lexsort([deg[ind_i], batch.batch[ind_i]])
                        h_ind_perm_i = ind_i[h_ind_perm_sort]
                        h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                        h_dense = self.self_attn(h_dense)[mask]
                        indices_arr.append(h_ind_perm_i)
                        emb_arr.append(h_dense)
                    h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                    h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    for i in range(5):
                        h_ind_perm = permute_within_batch(batch.batch)
                        deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                        indices_arr, emb_arr = [],[]
                        for i in range(self.NUM_BUCKETS):
                            ind_i = h_ind_perm[h_ind_perm%self.NUM_BUCKETS==i]
                            h_ind_perm_sort = lexsort([deg[ind_i], batch.batch[ind_i]])
                            h_ind_perm_i = ind_i[h_ind_perm_sort]
                            h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                            h_dense = self.self_attn(h_dense)[mask]
                            indices_arr.append(h_ind_perm_i)
                            emb_arr.append(h_dense)
                        h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                        h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif self.global_model_type == 'Mamba_Cluster_Bucket':
                h_ind_perm = permute_within_batch(batch.batch)
                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                if batch.split == 'train':
                    indices_arr, emb_arr = [],[]
                    unique_cluster_n = len(torch.unique(batch.LouvainCluster))
                    permuted_louvain = torch.zeros(batch.LouvainCluster.shape).long().to(batch.LouvainCluster.device)
                    random_permute = torch.randperm(unique_cluster_n+1).long().to(batch.LouvainCluster.device)
                    for i in range(len(torch.unique(batch.LouvainCluster))):
                        indices = torch.nonzero(batch.LouvainCluster == i).squeeze()
                        permuted_louvain[indices] = random_permute[i]
                    for i in range(self.NUM_BUCKETS):
                        ind_i = h_ind_perm[h_ind_perm%self.NUM_BUCKETS==i]
                        h_ind_perm_sort = lexsort([permuted_louvain[ind_i], deg[ind_i], batch.batch[ind_i]])
                        h_ind_perm_i = ind_i[h_ind_perm_sort]
                        h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                        h_dense = self.self_attn(h_dense)[mask]
                        indices_arr.append(h_ind_perm_i)
                        emb_arr.append(h_dense)
                    h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                    h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    for i in range(5):
                        indices_arr, emb_arr = [],[]
                        unique_cluster_n = len(torch.unique(batch.LouvainCluster))
                        permuted_louvain = torch.zeros(batch.LouvainCluster.shape).long().to(batch.LouvainCluster.device)
                        random_permute = torch.randperm(unique_cluster_n+1).long().to(batch.LouvainCluster.device)
                        for i in range(len(torch.unique(batch.LouvainCluster))):
                            indices = torch.nonzero(batch.LouvainCluster == i).squeeze()
                            permuted_louvain[indices] = random_permute[i]
                        for i in range(self.NUM_BUCKETS):
                            ind_i = h_ind_perm[h_ind_perm%self.NUM_BUCKETS==i]
                            h_ind_perm_sort = lexsort([permuted_louvain[ind_i], deg[ind_i], batch.batch[ind_i]])
                            h_ind_perm_i = ind_i[h_ind_perm_sort]
                            h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                            h_dense = self.self_attn(h_dense)[mask]
                            indices_arr.append(h_ind_perm_i)
                            emb_arr.append(h_dense)
                        h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                        h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif self.global_model_type == 'Mamba_Augment':
                aug_idx, aug_mask = augment_seq(batch.edge_index, batch.batch, 3)
                h_dense, mask = to_dense_batch(h[aug_idx], batch.batch[aug_idx])
                aug_idx_reverse = torch.nonzero(aug_mask).squeeze()
                h_attn = self.self_attn(h_dense)[mask][aug_idx_reverse]
            else:
                raise RuntimeError(f"Unexpected {self.global_model_type}")

            h_attn = self.dropout_attn(h_attn) #[l,dim]
            #print("6",h_attn.shape)

            #12_30_change##
            h_attn = h_in1 + h_attn  # Residual connection.
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn, batch.batch)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn)
            #print("7",h_attn.shape) #[l,dim]

            #print("final1:",h_attn) #[l,dim]

            #h_attn = self.norm2_attn(h_attn)

            # 1. 标准化到合适范围
            #h_mean = h_attn.mean(dim=-1, keepdim=True)
            #h_std = h_attn.std(dim=-1, keepdim=True) + 1e-8
            #h_attn = (h_attn - h_mean) / h_std

            #h_attn = torch.tanh(h_attn)

            h_attn = h_attn.transpose(-2, -1)  # [dim,l]
            if self.learnable_vth:
                h_attn = h_attn / torch.exp(self.ln_vth)
            #print("6",h_attn)
            h_attn, pos_train_mamba_fire_rate, neg_train_mamba_fire_rate = self.neuron_2(h_attn, 1.0, name="mamba")  # [l, dim]
            self.fire_rate_list["pos_train_mamba_fire_rate"].append(pos_train_mamba_fire_rate)
            self.fire_rate_list["neg_train_mamba_fire_rate"].append(neg_train_mamba_fire_rate)

            h_out_list.append(h_attn) #[l,dim]
            #print("h_final:",h_out_list[0].shape,h_out_list[1].shape)

        # Combine local and global outputs.
        # h = torch.cat(h_out_list, dim=-1)

        #特征维度求和
        h = sum(h_out_list)
        total_elements = h.numel()
        # 统计绝对值为 1 和 2 的数量
        count_abs_1 = torch.sum(torch.abs(torch.abs(h) - 1.0) < 1e-5)
        count_abs_2 = torch.sum(torch.abs(torch.abs(h) - 2.0) < 1e-5)
            
        # 计算比例，并保持为 Tensor 格式
        # 使用 .float() 确保是浮点 Tensor，而不是 Python float
        rate_abs_1 = count_abs_1.float() / total_elements
        rate_abs_2 = count_abs_2.float() / total_elements
            
        # 记录到列表（此时存入的是 Tensor）
        self.fire_rate_list["h_abs_1_count"].append(rate_abs_1)
        self.fire_rate_list["h_abs_2_count"].append(rate_abs_2)
        #print(h.min(), h.max())


        #加入SDN_Plan 1


        #将特征求和替换为哈达玛积(结构×时序)[l,dim]
        #h = h_local * h_attn
        #print(h.shape) #[l,dim]


        # Feed Forward block. MLP Module
        h = h_in1 + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)

        #h = self.norm2_attn(h)

        #加入SDN_Plan 2
#        h = h.transpose(-2, -1)
#        h, pos_fire_rate, neg_fire_rate = self.neuron_2(h, 0.5, name="mamba")
#        self.fire_rate_list["pos_fire_rate"].append(pos_fire_rate.detach().cpu().item())
#        self.fire_rate_list["neg_fire_rate"].append(neg_fire_rate.detach().cpu().item())

        batch.x = h
        #print("final:",batch.x.shape,batch)
        #print(self.sigma_fire_rate)
        #print(self.gatedgcn_fire_rate)
        #print(self.train_mamba_fire_rate)
        #print(self.eval_mamba_fire_rate_single)
        #print(self.eval_mamba_fire_rate_ave)
        #print("all",batch.x)
        #print("final_h:", batch.x)
        return batch

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block.
        """
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return x

    def _ff_block(self, x):
        """Feed Forward block.MLP Module
        """
        x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = f'summary: dim_h={self.dim_h}, ' \
            f'local_gnn_type={self.local_gnn_type}, ' \
            f'global_model_type={self.global_model_type}, ' \
            f'heads={self.num_heads}'
        return s
