import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_scatter import scatter
from graphgps.layer.neuron import SDNNeuron, BPTTNueron, SLTTNueron
#from graphgps.layer.surrogate import piecewise_quadratic_surrogate
from graphgps.layer.surrogate import ternary_piecewise_quadratic_surrogate
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_layer
from spikingjelly.clock_driven import encoding

#def record_fire_rate_json(fire_rate, phase, model, json_path="sigmoid_firing_rate_log.json"):
#    """
#    将当前 fire_rate 和阶段信息保存到 JSON 文件中。
#    如果文件不存在则创建；存在则追加。
#    """
#    # 如果文件存在，加载旧数据；否则新建空字典
#    if os.path.exists(json_path):
#        with open(json_path, "r") as f:
#            try:
#                data = json.load(f)
#            except json.JSONDecodeError:
#                data = {}
#    else:
#        data = {}
#
#    # 生成唯一 key
#    step_key = f"batch_{len(data)}"
#
#    # 添加一条新记录
#    data[step_key] = {
#        "phase": phase,
#        "model": model,
#        "fire_rate": round(float(fire_rate), 6)
#    }
#
#    # 写回 JSON 文件（缩进更美观）
#    with open(json_path, "w") as f:
#        json.dump(data, f, indent=4)

class GatedGCNLayer(pyg_nn.conv.MessagePassing):
    """
        GatedGCN layer
        Residual Gated Graph ConvNets
        https://arxiv.org/pdf/1711.07553.pdf
    """
    def __init__(self, in_dim, out_dim, dropout, residual,
                 equivstable_pe=False, **kwargs):
        super().__init__(**kwargs)
        self.A = pyg_nn.Linear(in_dim, out_dim, bias=True) #用于中心节点更新
        self.B = pyg_nn.Linear(in_dim, out_dim, bias=True) #用于邻居聚合
        self.C = pyg_nn.Linear(in_dim, out_dim, bias=True) #用于边特征变换
        self.D = pyg_nn.Linear(in_dim, out_dim, bias=True) #用于目标节点（门控计算）
        self.E = pyg_nn.Linear(in_dim, out_dim, bias=True) #用于源节点（门控计算）


        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        self.EquivStablePE = equivstable_pe
        if self.EquivStablePE:
            self.mlp_r_ij = nn.Sequential(
                nn.Linear(1, out_dim), nn.ReLU(),
                nn.Linear(out_dim, 1),
                nn.Sigmoid())

        self.bn_node_x = nn.BatchNorm1d(out_dim)
        self.bn_edge_e = nn.BatchNorm1d(out_dim)
        self.dropout = dropout
        self.residual = residual
        #self.neuron = SDNNeuron(piecewise_quadratic_surrogate(),0.85)
        #self.ternary_neuron = SDNNeuron(ternary_piecewise_quadratic_surrogate())
        self.ln_vth = nn.Parameter(torch.zeros(in_dim, 1))
        self.encoder= encoding.PoissonEncoder()
        self.e = None

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        num_edges = edge_index.size(1)  # edge_index的形状是 [2, num_edges]  

        # 安全地获取边特征
        if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
             e = batch.edge_attr
        else:
        # 如果没有边特征，创建默认的边特征
             e = torch.ones(num_edges, x.size(1)).to(x.device)  # 保持与节点特征相同维度
             

        ##Spike_e = self.encoder(e)

        """
        x               : [n_nodes, in_dim]
        e               : [n_edges, in_dim]
        edge_index      : [2, n_edges]
        """
        if self.residual:
        # False
            x_in = x
        # e_in = e
            e_in = e
            #e_in = e

        Ax = self.A(x) #用于中心节点更新
        ##Spike_Ax= self.encoder(Ax)
        Bx = self.B(x) #用于邻居聚合
        ##Spike_Bx = self.encoder(Bx)
        Ce = self.C(e) #用于边特征变换
        ##Spike_Ce = self.encoder(Ce)
        Dx = self.D(x) #用于目标节点（门控计算）
        ##Spike_Dx = self.encoder(Dx)
        Ex = self.E(x) #用于源节点（门控计算）
        ##Spike_Ex = self.encoder(Ex)
        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        pe_LapPE = batch.pe_EquivStableLapPE if self.EquivStablePE else None

        # 消息传递（依次调用 message → aggregate聚合 → update）
        x, e = self.propagate(edge_index, Bx=Bx, Dx=Dx, Ex=Ex, Ce=Ce, e=e, Ax=Ax, PE=pe_LapPE)
        ##x, e = self.propagate(edge_index, Bx=Spike_Bx, Dx=Spike_Dx, Ex=Spike_Ex, Ce=Spike_Ce, e=Spike_e, Ax=Spike_Ax, PE=pe_LapPE)

        # 读取在 message() 中保存的 fire rate（如果存在）
        #pos_sigma_fire_rate = getattr(self, 'pos_sigma_fire_rate', None)
        #neg_sigma_fire_rate = getattr(self, 'neg_sigma_fire_rate', None)
        # 清理，避免下个 batch 污染
        #if hasattr(self, 'sigma_fire_rate'):
            #del self.sigma_fire_rate

        x = self.bn_node_x(x)
        e = self.bn_edge_e(e)

        x = F.relu(x)
        # 替换 ReLU 为 LIF 神经元, B_Plan(L维度展开)
        # 转置
        ##x = x.transpose(-2, -1)  # [dim,l]
        # neuron part
        ##x = x / torch.exp(self.ln_vth)
#        x = self.neuron(x).transpose(-2, -1)  # [l, dim]
        ##x, fire_rate = self.neuron(x)
        #print(f"Inside GatedGCN x Firing rate: {fire_rate:.4f}")

        e = F.relu(e)
        # 替换 ReLU 为 LIF 神经元, B_Plan(L维度展开)
        # 转置
        ##e = e.transpose(-2, -1)  # [dim,l]
        # neuron part
        ##e = e / torch.exp(self.ln_vth)
#        e = self.neuron(e).transpose(-2, -1)  # [l, dim]
        ##e, fire_rate = self.neuron(e)
        #print(f"Inside GatedGCN e Firing rate: {fire_rate:.4f}")

        # Dropout（可选，脉冲数据稀疏，可能不需要）
        x = F.dropout(x, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        if self.residual:
#           False_x = x_in + x
            e = e_in + e
            x = x_in + x
            #x = self.encoder(x_in) + x


        batch.x = x
        batch.edge_attr = e

        
        
        ##return batch, pos_sigma_fire_rate, neg_sigma_fire_rate
        return batch

    def message(self, Dx_i, Ex_j, PE_i, PE_j, Ce):
        #动态控制邻居j对中心节点的信息贡献
        """
        {}x_i           : [n_edges, out_dim]
        {}x_j           : [n_edges, out_dim]
        {}e             : [n_edges, out_dim]
        """
        # 计算门控权重 sigma_ij
        e_ij = Dx_i + Ex_j + Ce # 目标节点 + 源节点 + 边特征
        #print("Original_e_ij:",e_ij)
        ##e_ij = self.encoder(e_ij)
        #print("e_ij:",e_ij)

        sigma_ij = torch.sigmoid(e_ij) # 门控（0~1）

        # 修改其sigmoid为SDN node
        # 替换 ReLU 为 LIF 神经元, B_Plan(L维度展开)
        # 转置 + neuron part
        ##sigma_ij, pos_sigma_fire_rate, neg_sigma_fire_rate = self.ternary_neuron(e_ij.transpose(-2, -1), 0.875, name="sigmoid")  # [l, dim]

#        try:
#            # 保持 tensor（不转 .item()，以便在 GPU 上短时间存储），并 detach 防止梯度流
#            self.pos_sigma_fire_rate = pos_sigma_fire_rate.detach()
#            self.neg_sigma_fire_rate = neg_sigma_fire_rate.detach()
#        except Exception:
#            # 容错：如果 sigma_fire_rate 已经是标量或 python 值
#            self.pos_sigma_fire_rate = torch.tensor(pos_sigma_fire_rate).detach()
#            self.neg_sigma_fire_rate = torch.tensor(neg_sigma_fire_rate).detach()
            
        #print(f"Inside GatedGCN sigma_ij Firing rate: {fire_rate:.4f}")
        #record_fire_rate_json(fire_rate, phase="train", model="sigma_ij")
        #print("sigma_ij:",sigma_ij)
#        sigma_ij = self.neuron(e_ij)  # 用 LIF 替代 sigmoid

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        if self.EquivStablePE:
            r_ij = ((PE_i - PE_j) ** 2).sum(dim=-1, keepdim=True)
            r_ij = self.mlp_r_ij(r_ij)  # the MLP is 1 dim --> hidden_dim --> 1 dim
            sigma_ij = sigma_ij * r_ij

        self.e = e_ij  # 保存边信息（用于 update）
        
        return sigma_ij

    def aggregate(self, sigma_ij, index, Bx_j, Bx):
        """
        sigma_ij        : [n_edges, out_dim]  ; is the output from message() function
        index           : [n_edges]
        {}x_j           : [n_edges, out_dim]
        """
        #用门控权重 sigma_ij 对邻居特征 Bx_j 加权
        dim_size = Bx.shape[0]  # or None ??   <--- Double check this


        # 加权聚合：sigma_ij * Bx_j
        sum_sigma_x = sigma_ij * Bx_j
        numerator_eta_xj = scatter(sum_sigma_x, index, 0, None, dim_size,
                                   reduce='sum')

        # 归一化分母：sum(sigma_ij)
        sum_sigma = sigma_ij
        denominator_eta_xj = scatter(sum_sigma, index, 0, None, dim_size,
                                     reduce='sum')

        # 加权平均
        out = numerator_eta_xj / (denominator_eta_xj + 1e-6)
        return out

    def update(self, aggr_out, Ax):
        """
        aggr_out        : [n_nodes, out_dim] ; is the output from aggregate() function after the aggregation
        {}x             : [n_nodes, out_dim]
        """
        # 中心节点更新：Ax + 聚合结果
        x = Ax + aggr_out
        e_out = self.e
        del self.e
        return x, e_out


@register_layer('gatedgcnconv')
class GatedGCNGraphGymLayer(nn.Module):
    """GatedGCN layer.
    Residual Gated Graph ConvNets
    https://arxiv.org/pdf/1711.07553.pdf
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = GatedGCNLayer(in_dim=layer_config.dim_in,
                                   out_dim=layer_config.dim_out,
                                   dropout=0.,  # Dropout is handled by GraphGym's `GeneralLayer` wrapper
                                   residual=False,  # Residual connections are handled by GraphGym's `GNNStackStage` wrapper
                                   **kwargs)

    def forward(self, batch):
        return self.model(batch)
