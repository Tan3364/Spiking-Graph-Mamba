from torch import nn
import torch
import os
from torch_geometric.graphgym.register import register_act
import time
import uuid
import matplotlib.pyplot as plt

@register_act('sdn')
#class SDNNeuron(nn.Module):
#    ckt_path = "mem_pred_d8k8_c3(relu(c2(c1(x))+c1(x)))_125.pkl"
#
#    def __init__(self, surrogate_function, threshold):
#        super().__init__()
#        self.surrogate_function = surrogate_function
#        self.threshold = threshold
#        ckt_path = os.path.join(os.path.dirname(__file__), self.ckt_path)
#        self.model = torch.jit.load(ckt_path).eval()
#
#    def forward(self, x):
#        #print("x",x.shape)
#        mem = self.pred(x)
#        s = self.surrogate_function(mem+x - self.threshold)
#        #print("s",s.shape)
#        #average population spike
#        fire_rate = s.mean(dim=0).mean()  # batch平均fire rate
#        #print("s_mean",s_mean.shape)
#        #fire_rate = (s_mean != 0).float().mean().item()
#        s = s.transpose(-2, -1)
#        return s, fire_rate
#
#    @torch.no_grad()
#    def pred(self, x):
#        shape = x.shape
#        L = x.size(-1)
#        return self.model(x.detach().view(-1, 1, L)).view(shape)

class SDNNeuron(nn.Module):
    ckt_path = "exp_ternary_0_1.0.pkl"

    def __init__(self, surrogate_function):
        super().__init__()
        self.surrogate_function = surrogate_function
        #self.threshold = threshold      
        #self.neg_th = neg_th      # 负阈值
        ckt_path = os.path.join(os.path.dirname(__file__), self.ckt_path)
        self.model = torch.jit.load(ckt_path).train()
        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x, threshold=None, name=None):
    #def forward(self, x, name):
        """
        x: 输入电流或膜电位更新量
        输出:
          s: 三值spike {-1,0,1}
          fire_rate: 平均放电率
        """
        #if name == "graph":
            #decay = 1
            #mem = self.pred(x)
            #mem_all = x + decay * mem
            #print("mem_all", mem_all)
        #else:
            #decay = 1
       
        # 预测膜电位
        #mem = self.pred(x)
        #mem_all = x +  mem
#        V_np = mem_all.detach().cpu().numpy().flatten()
#        plt.figure(figsize=(6, 5))
#        plt.hist(V_np, bins=100, edgecolor='none', color='#1f77b4', alpha=0.7, density=True)
#        t = threshold
#        plt.axvline(x=t, color='red', linestyle='--', linewidth=1.5, label=f'Threshold $\pm${t}')
#        plt.axvline(x=-t, color='red', linestyle='--', linewidth=1.5)
#
#        d = 0.125  # 这里的 d 应该对应你训练时的 margin
#        plt.axvspan(t - d, t + d, color='gold', alpha=0.2, label='Margin Zone')
#        plt.axvspan(-t - d, -t + d, color='gold', alpha=0.2)
#
#        plt.xlabel('Membrane Potential', fontsize=12)
#        plt.ylabel('Density', fontsize=12)
#        #plt.title('Potential Distribution Analysis', fontsize=13)
#        plt.grid(axis='y', linestyle=':', alpha=0.6)
#        plt.legend(loc='upper right', fontsize=10)
#
#        # 限制 X 轴范围，聚焦在阈值附近
#        plt.xlim([-t - 0.5, t + 0.5])
#
#        plt.tight_layout()
#
#        # 🔁 保存图片
#        timestamp = time.strftime("%Y%m%d_%H%M%S")
#        filename = f"Potential_{name}_Hist_{timestamp}_{uuid.uuid4().hex[:6]}.png"
#        plt.savefig(os.path.join("./", filename), dpi=300) # 300DPI 保证清晰
#        print(f"[*] Histogram saved: {filename}")
#        plt.show()
#        plt.close()
        # 三值激活逻辑（正/负阈值）
        #spike_pos = self.surrogate_function(mem + x - self.pos_th)      # 近似正脉冲
        #spike_neg = self.surrogate_function(-mem - x - self.neg_th)     # 近似负脉冲
        #s = spike_pos - spike_neg                                       # 组合成{-1,0,1}输出
        #x_mean = mem_all.mean(dim=0, keepdim=True)  # 每列均值
        #x_std  = mem_all.std(dim=0, keepdim=True)   # 每列标准差
        #aver_mem = (mem_all - x_mean) / (x_std + 1e-5)

        
        
        #("s:",(mem+x).shape)

        # ⚙️ 可视化膜电位 + 阈值 + delta
        # ------------------------------
                
#        v_all = (mem_all).mean(dim=0).detach().cpu().numpy()
#        plt.figure(figsize=(7,4))
#        plt.plot(v_all, label="Avg Neuron Membrane Potential")  # 膜电位曲线
#        #V_np = v_all.detach().cpu().numpy()
#        V_np = v_all
###
#        plt.figure(figsize=(6,4))
#        plt.hist(V_np, bins=50, edgecolor='black')
#        plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2)
#        plt.axvline(x=-threshold, color='red', linestyle='--', linewidth=2)
#        plt.xlabel('Membrane Potential')
#        plt.ylabel('Frequency')
#        plt.title('Histogram of Membrane Potentials Over Timesteps')

        # 固定四条阈值线
#        t = threshold
#        d = 0.125
#        plt.axhline(t - d, color='red', linestyle='--', label="Threshold Lines")
#        plt.axhline(t + d, color='red', linestyle='--')
#        plt.axhline(-t - d, color='red', linestyle='--')
#        plt.axhline(-t + d, color='red', linestyle='--')
#
#        plt.xlabel("Time")
#        plt.ylabel("Membrane Potential")
#        plt.title("Membrane Potential with Thresholds")
#        plt.legend()
#        plt.tight_layout()
#
#        # ------------------------------
#        # 🔁 生成唯一文件名并保存
#        # ------------------------------
#        timestamp = time.strftime("%Y%m%d_%H%M%S")
#        filename = f"{name}_membrane_viz_{timestamp}_{uuid.uuid4().hex[:6]}.png"
#        save_path = "./"
#        filepath = os.path.join(save_path, filename)
#        plt.savefig(filepath)
#        plt.close()

        # ------------------------------
        # 🔁 Reset：用 |s| 进行膜电位重置
        # ------------------------------
        # 注意这里不直接修改 mem，而是模拟重置逻辑
        #mem_reset = mem - (self.pos_th * torch.abs(s))

        # ------------------------------
        # 🔥 Fire rate统计
        # ------------------------------
        ##fire_rate = torch.abs(s).mean(dim=0).mean()  # 平均放电率 (batch维平均)
        #pos_fire_rate = torch.abs(spike_pos).mean(dim=0).mean()
        #neg_fire_rate = torch.abs(spike_neg).mean(dim=0).mean()
        #s = self.surrogate_function(mem_all, threshold)
        mem_all = x
        s = self.surrogate_function(mem_all, threshold)
        #pos_fire_rate = (s ==1).float().mean(dim=0).mean()
        pos_fire_rate = torch.relu(s).mean(dim=0).mean()
        #neg_fire_rate = (s == -1).float().mean(dim=0).mean()
        neg_fire_rate = torch.relu(-s).mean(dim=0).mean()
        # ------------------------------
        # ⚙️ 维度调整
        # ------------------------------
        s = s.transpose(-2, -1)
        #print("grad",s.requires_grad)
        return s, pos_fire_rate, neg_fire_rate

    #@torch.no_grad()
    def pred(self, x):
        shape = x.shape
        L = x.size(-1)
        
        # 使用JIT模型预测膜电位
        #return self.model(x.detach().view(-1, 1, L)).view(shape)
        return self.model(x.view(-1, 1, L)).view(shape)

#class SDNNeuron(nn.Module):
#    ckt_path = "mem_pred_d8k8_c3(relu(c2(c1(x))+c1(x)))_125.pkl"
#
#    def __init__(self, surrogate_function, threshold):
#        super().__init__()
#        self.surrogate_function = surrogate_function
#        self.threshold = threshold
#        #self.delta = delta
#        #self.save_path = save_path
#        ckt_path = os.path.join(os.path.dirname(__file__), self.ckt_path)
#        self.model = torch.jit.load(ckt_path).eval()
#
#    def forward(self, x, threshold):
#        """
#        x: 输入电流或膜电位更新量
#        输出:
#          s: 三值spike {-1,0,1}
#          pos_fire_rate: 平均正发放率
#          neg_fire_rate: 平均负发放率
#        """
#        # ------------------------------
#        # 预测膜电位
#        # ------------------------------
#        mem = self.model(x)  # 假设 model 预测膜电位
#
#        # ------------------------------
#        # 三值激活逻辑
#        # ------------------------------
#        s = self.surrogate_function(mem + x, self.threshold)
#
#        # ------------------------------
#        # 🔥 Fire rate统计
#        # ------------------------------
#        pos_fire_rate = (s == 1).float().mean(dim=0).mean()
#        neg_fire_rate = (s == -1).float().mean(dim=0).mean()
#
#        # ------------------------------
        # ⚙️ 可视化膜电位 + 阈值 + delta
        # ------------------------------
#        plt.figure(figsize=(7,4))
#        plt.plot(v_all, label="Avg Neuron Membrane Potential")  # 膜电位曲线
#
#        # 固定四条阈值线
#        t = self.threshold
#        d = 0.15
#        plt.axhline(t - d, color='red', linestyle='--', label="Threshold Lines")
#        plt.axhline(t + d, color='red', linestyle='--')
#        plt.axhline(-t - d, color='red', linestyle='--')
#        plt.axhline(-t + d, color='red', linestyle='--')
#
#        plt.xlabel("Time / Index")
#        plt.ylabel("Membrane Potential")
#        plt.title("Membrane Potential with Thresholds")
#        plt.legend()
#        plt.tight_layout()
#
#        # ------------------------------
#        # 🔁 生成唯一文件名并保存
#        # ------------------------------
#        timestamp = time.strftime("%Y%m%d_%H%M%S")
#        filename = f"membrane_viz_{timestamp}_{uuid.uuid4().hex[:6]}.png"
#        save_path = "./"
#        filepath = os.path.join(save_path, filename)
#        plt.savefig(filepath)
#        plt.close()
##
#        
#        # ------------------------------
#        # ⚙️ 维度调整
#        # ------------------------------
#        s = s.transpose(-2, -1)
#
#        return s, pos_fire_rate, neg_fire_rate
#
#@register_act('sltt')
#class SLTTNueron(nn.Module):
#    def __init__(self, surrogate_function, tau=0.125, vth=1.0, v_r=0):
#        super().__init__()
#        self.surrogate_function = surrogate_function
#        self.tau = tau
#        self.vth = vth
#        self.v_r = v_r
#
#    def forward(self, x, threshold = 1.0, name=None):
#        u = torch.zeros_like(x[..., 0])
#        #u_out = []
#        out = []
#
#        for i in range(x.size(-1)):
#            u = u.detach() * self.tau + x[..., i]
#            #s = self.surrogate_function(u - self.vth)
#            s = self.surrogate_function(u, self.vth)
#            out.append(s)
#            u = (1 - s.detach()) * u + s.detach() * self.v_r
#            #u_out.append(u)
#
#        #print("u_out_0", u_out[0].shape)
#        #print("23", torch.stack(out, -1).shape)
#
#        #s_mean = torch.stack(out, -1)[:,0] 
#        s_mean = torch.stack(out, -1).float().mean(dim=0).mean()
#        #print(s_mean.shape)
#
#
#        return torch.stack(out, -1), s_mean

@register_act('sltt')
class SLTTNueron(nn.Module):
    def __init__(self, surrogate_function, tau=0.125, vth=1.0, v_r=0):
        super().__init__()
        self.surrogate_function = surrogate_function
        self.tau = tau
        self.vth = vth
        self.v_r = v_r

    def forward(self, x, threshold=None, name=None):
        u = torch.zeros_like(x[..., 0])
        out = []
        pos_rate = []
        neg_rate = []
        for i in range(x.size(-1)):
            u = u.detach() * self.tau + x[..., i]
            s = self.surrogate_function(u, self.vth)
            pos_fire_rate = torch.relu(s).mean()
            pos_rate.append(pos_fire_rate)
            neg_fire_rate = torch.relu(-s).mean()
            neg_rate.append(neg_fire_rate)
            out.append(s)
            u = (1 - s.detach()) * u + s.detach() * self.v_r

        pos_rate_tensor = torch.stack(pos_rate)
        neg_rate_tensor = torch.stack(neg_rate)
        return torch.stack(out, -1).transpose(-2,-1), pos_rate_tensor, neg_rate_tensor
        
#class SLTTNueron(nn.Module):
#    def __init__(self, surrogate_function, tau=0.125, vth=None, v_r=0):
#        super().__init__()
#        self.surrogate_function = surrogate_function
#        self.tau = tau
#        #self.vth = vth
#        self.v_r = v_r
#
#    def forward(self, x, threshold=None, name=None):
#        # membrane potential
#        u = torch.zeros_like(x[..., 0])
#
#        out = []
#        pos_cnt = 0.0
#        neg_cnt = 0.0
#        T = x.size(-1)
#
#        for i in range(T):
#            u = u.detach() * self.tau + x[..., i]
#
#            # ===== 三值脉冲 =====
#            s_pos = self.surrogate_function(u, threshold)      # +1 branch
#            s_neg = self.surrogate_function(-u, threshold)     # -1 branch
#            s = s_pos - s_neg                                 # {-1, 0, +1}
#
#            out.append(s)
#
#            # firing statistics
#            pos_cnt += (s == 1).float().mean()
#            neg_cnt += (s == -1).float().mean()
#
#            # reset (对正负脉冲都 reset)
#            fired = (s != 0).float().detach()
#            u = (1 - fired) * u + fired * self.v_r
#
#        spikes = torch.stack(out, -1).transpose(-2,-1)
#
#        pos_fire_rate = pos_cnt / T
#        neg_fire_rate = neg_cnt / T
#
#        return spikes, pos_fire_rate, neg_fire_rate

@register_act('bptt')
class BPTTNueron(nn.Module):
    def __init__(self, surrogate_function, tau=0.125, vth=1.0, v_r=0):
        super().__init__()
        self.surrogate_function = surrogate_function
        self.tau = tau
        self.vth = vth
        self.v_r = v_r

    def forward(self, x, threshold=None, name=None):
        u = torch.zeros_like(x[..., 0])
        out = []
        pos_rate = []
        neg_rate = []
        for i in range(x.size(-1)):
            u = u * self.tau + x[..., i]
            s = self.surrogate_function(u, self.vth)
            pos_fire_rate = torch.relu(s).mean()
            pos_rate.append(pos_fire_rate)
            neg_fire_rate = torch.relu(-s).mean()
            neg_rate.append(neg_fire_rate)
            out.append(s)
            u = (1 - s.detach()) * u + s.detach() * self.v_r

        pos_rate_tensor = torch.stack(pos_rate)
        neg_rate_tensor = torch.stack(neg_rate)    
        return torch.stack(out, -1).transpose(-2,-1), pos_rate_tensor, neg_rate_tensor
        
#class BPTTNueron(nn.Module):
#    def __init__(self, surrogate_function, tau=0.125, vth=1.0, v_r=0):
#        super().__init__()
#        self.surrogate_function = surrogate_function
#        self.tau = tau
#        self.vth = vth
#        self.v_r = v_r
#
#    def forward(self, x):
#        u = torch.zeros_like(x[..., 0])
#        #u_out = []
#        out = []
#        for i in range(x.size(-1)):
#            u = u * self.tau + x[..., i]
#            s = self.surrogate_function(u - self.vth)
#            out.append(s)
#            u = (1 - s.detach()) * u + s.detach() * self.v_r
#            #u_out.append(u)
#
#        return torch.stack(out, -1)



