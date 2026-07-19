import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
from torch_geometric.graphgym.register import register_network
from graphgps.encoder.ER_edge_encoder import EREdgeEncoder

from graphgps.layer.gps_layer import GPSLayer


class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
            # Update dim_in to reflect the new dimension for the node features
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            # Hard-set edge dim for PNA.
            cfg.gnn.dim_edge = 16 if 'PNA' in cfg.gt.layer_type else cfg.gnn.dim_inner
            if cfg.dataset.edge_encoder_name == 'ER':
                self.edge_encoder = EREdgeEncoder(cfg.gnn.dim_edge)
            elif cfg.dataset.edge_encoder_name.endswith('+ER'):
                EdgeEncoder = register.edge_encoder_dict[
                    cfg.dataset.edge_encoder_name[:-3]]
                self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge - cfg.posenc_ERE.dim_pe)
                self.edge_encoder_er = EREdgeEncoder(cfg.posenc_ERE.dim_pe, use_edge_attr=True)
            else:
                EdgeEncoder = register.edge_encoder_dict[
                    cfg.dataset.edge_encoder_name]
                self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)

            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_edge, -1, -1, has_act=False,
                                    has_bias=False, cfg=cfg))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


@register_network('GPSModel')
class GPSModel(torch.nn.Module):
    """Multi-scale graph x-former.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        try:
            local_gnn_type, global_model_type = cfg.gt.layer_type.split('+')
        except:
            raise ValueError(f"Unexpected layer type: {cfg.gt.layer_type}")
        layers = []
        for _ in range(cfg.gt.layers):
            layers.append(GPSLayer(
                dim_h=cfg.gt.dim_hidden,
                local_gnn_type=local_gnn_type,
                global_model_type=global_model_type,
                num_heads=cfg.gt.n_heads,
                pna_degrees=cfg.gt.pna_degrees,
                equivstable_pe=cfg.posenc_EquivStableLapPE.enable,
                dropout=cfg.gt.dropout,
                attn_dropout=cfg.gt.attn_dropout,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
                bigbird_cfg=cfg.gt.bigbird,
                neuron_type=cfg.gt.neuron_type,
                learnable_vth=cfg.gt.learnable_vth,
                shared_vth=cfg.gt.shared_vth,
            ))
        self.layers = torch.nn.Sequential(*layers)
        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

#    def get_layer_fire_rate_avg(self):
#        """
#        计算模型中每层的不同 fire rate 的平均值。
#        返回：
#            avg_dict: dict, key 为 fire_rate 类型，value 为 list，
#                      每个元素是对应层的平均 fire rate。
#        """
#        # 定义需要统计的 fire rate 类型
#        fire_rate_keys = [
#            "pos_gatedgcn_fire_rate",
#            "neg_gatedgcn_fire_rate",
#            "pos_train_mamba_fire_rate",
#            "neg_train_mamba_fire_rate",
#        ]
#
#        # 初始化结果字典：每种 fire rate 都对应一个 list
#        avg_dict = {key: [] for key in fire_rate_keys}
#
#        # 遍历模型的每一层（假设 self.layers 是 nn.ModuleList）
#        for layer in self.layers:
#            # 如果该层有 fire_rate_list 属性
#            if hasattr(layer, "fire_rate_list"):
#                for key in fire_rate_keys:
#                    values = layer.fire_rate_list.get(key, [])                  
#                    if isinstance(values, (list, tuple)) and len(values) > 0:
#                        avg_value = sum(values) / len(values)
#                    else:
#                        avg_value = 0.0
#                    avg_dict[key].append(avg_value)
#            else:
#                # 如果该层没有 fire_rate_list 属性，则填 0
#                for key in fire_rate_keys:
#                    avg_dict[key].append(0.0)
#
#        #print(avg_dict)  # 调试用，可删
#        return avg_dict

    def get_layer_fire_rate_avg(self):
        fire_rate_keys = [
            "pos_gatedgcn_fire_rate",
            "neg_gatedgcn_fire_rate",
            "pos_train_mamba_fire_rate",
            "neg_train_mamba_fire_rate",
            "h_abs_1_count",
            "h_abs_2_count",
        ]

        avg_dict = {key: [] for key in fire_rate_keys}

        # 获取模型当前的设备，确保生成的 0 也是 Tensor 且在正确的设备上
        device = next(self.parameters()).device

        for layer in self.layers:
            if hasattr(layer, "fire_rate_list"):
                for key in fire_rate_keys:
                    values = layer.fire_rate_list.get(key, [])
                    
                    # --- 核心修改：使用 torch.stack 保持计算图 ---
                    if isinstance(values, (list, tuple)) and len(values) > 0:
                        # stack 会把 list[Tensor] 变成一个新维度的 Tensor，mean 保持梯度
                        avg_value = torch.stack(values).mean()
                    else:
                        # 必须返回带梯度的 Tensor 0，或者至少是 Tensor 0
                        avg_value = torch.tensor(0.0, device=device, requires_grad=True)
                    
                    avg_dict[key].append(avg_value)
            else:
                for key in fire_rate_keys:
                    avg_dict[key].append(torch.tensor(0.0, device=device, requires_grad=True))

        return avg_dict

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
