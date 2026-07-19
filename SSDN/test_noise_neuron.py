import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from torch import Tensor

def get_conv1d_k1_bn(
    in_channels: int,
    out_channels: int,
):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, 1),
        nn.BatchNorm1d(out_channels),
    )
def get_conv1d_kernel_bn(
    in_channels: int, out_channels: int, kernel_size: int
):
    return nn.Sequential(
        nn.Conv1d(
            in_channels, out_channels, kernel_size, padding='same', groups=in_channels
        ),
        nn.BatchNorm1d(out_channels),
    )
class SDN(nn.Module):
    def __init__(self, d_model=8, kernel_size=8, n_layers=1):
        super(SDN, self).__init__()
        self.encoder = nn.Conv1d(1, d_model, kernel_size=1, bias=False)
        self.spatial = get_conv1d_kernel_bn(d_model, d_model, kernel_size=kernel_size)
        self.feature = get_conv1d_k1_bn(d_model, d_model)

        self.decoder = nn.Conv1d(d_model, 1, 1)
    def forward(self, x):
        """
        Input x is shape (B, 1, L)
        """
        x = self.encoder(x)
        x = F.relu(self.spatial(x))
        x = F.relu(self.feature(x) + x)
        return self.decoder(x).squeeze()

def heaviside(x: Tensor):
    """heaviside function

    Args:
        x (Tensor): u - vth

    Returns:
        Tensor: spike
    """
    return (x >= 0).int()

@torch.no_grad()
def hardreset(x: Tensor, tau: float = 0.2, v_th: float = 1.0):
    """perform lif evolution with hardreset mechanism

    Args:
        x (Tensor): input currents with (T, N)
        tau (float, optional): attenuation coefficient. Defaults to 0.2.
        v_th (float, optional): threshold when to spike. Defaults to 1.0.

    Returns:
        (Tensor, Tensor): spikes, attenuated membrane potential
    """
    y = []
    pre_mem = []
    post_mem = []
    u = torch.zeros_like(x[:, 0])
    for i in x.unbind(1):
        u = tau * u
        pre_mem.append(u)
        u = u + i
        post_mem.append(u)
        s = heaviside(u - v_th)
        y.append(s)
        u = u * (1 - s)
    y = torch.stack(y).int()
    pre_mem = torch.stack(pre_mem)
    post_mem = torch.stack(post_mem)
    return y.transpose(0,1), pre_mem.transpose(0,1), post_mem.transpose(0,1)

def random_signal(N, T, m=0, std=1.0):
    return np.random.randn(N, T) * std + m
class InputSignal(Dataset):
    def __init__(self, tau, length):
        self.signal = random_signal(5000, length)
        self.signal = torch.as_tensor(self.signal, dtype = torch.float32)
        self.s, self.pre_mem, self.post_mem = hardreset(self.signal, tau=tau)

    def __len__(self):
        return 5000

    def __getitem__(self, i):
        return self.signal[i], self.pre_mem[i], self.post_mem[i], self.s[i]

def sdn_compute_spike(output, signal):
    return (output.detach() + signal.squeeze() >= 1).float()
def spike_sn_compute_spike(output, signal):
    return (output.detach() >= 1).float()

from sklearn.metrics import precision_score, recall_score, f1_score

def compute_metrics(pred_s, s):
    pred_s = pred_s.detach().cpu().flatten().numpy()
    s = s.detach().cpu().flatten().numpy()

    precision = precision_score(s, pred_s, zero_division=0)
    recall = recall_score(s, pred_s, zero_division=0)
    f1 = f1_score(s, pred_s, zero_division=0)
    return precision, recall, f1


def eval(model, dataset, compute_spike):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for index, (signal, pre_mem, post_mem, s) in enumerate(dataset):
        signal = signal.unsqueeze(1).to(device)
        s = s.to(device)
        output = model(signal)
        pred_s = compute_spike(output, signal)
    acc = (pred_s == s).float().mean()
    precision, recall, f1 = compute_metrics(pred_s, s)
    print('Accuracy {} Precision {} Recall {} F1 {}'.format(acc, precision, recall, f1))

sdn_net = SDN()
sdn_net.load_state_dict(torch.load('exp_ternary_0.pt'))
spike_sn_net = SDN()
spike_sn_net.load_state_dict(torch.load('exp_ternary_1.0.pt'))
data_loader = torch.utils.data.DataLoader(InputSignal(tau = 0.2, length = 100), batch_size=32, shuffle=False)

eval(sdn_net, data_loader, sdn_compute_spike)

eval(spike_sn_net, data_loader, spike_sn_compute_spike)

def eval_input_gaussian(model, dataset, compute_spike, std=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for index, (signal, pre_mem, post_mem, s) in enumerate(dataset):
        signal = signal.unsqueeze(1).to(device)
        signal += torch.randn_like(signal) * (std ** 2)
        s = s.to(device)
        output = model(signal)
        pred_s = compute_spike(output, signal)
    acc = (pred_s == s).float().mean()
    precision, recall, f1 = compute_metrics(pred_s, s)
    # print('Accuracy {} Precision {} Recall {} F1 {}'.format(acc, precision, recall, f1))
    return acc.cpu(), precision, recall, f1

def eval_output_gaussian(model, dataset, compute_spike, std=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for index, (signal, pre_mem, post_mem, s) in enumerate(dataset):
        signal = signal.unsqueeze(1).to(device)
        s = s.to(device)
        output = model(signal)
        output += torch.randn_like(output) * (std ** 2)
        pred_s = compute_spike(output, signal)
    acc = (pred_s == s).float().mean()
    precision, recall, f1 = compute_metrics(pred_s, s)
    # print('Accuracy {} Precision {} Recall {} F1 {}'.format(acc, precision, recall, f1))
    return acc.cpu(), precision, recall, f1

import matplotlib.pyplot as plt
noise_std = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
def plot_metrics(eval):
    sdn_acc, sdn_precision, sdn_recall, sdn_f1 = [], [], [], []
    spike_sn_acc, spike_sn_precision, spike_sn_recall, spike_sn_f1 = [], [], [], []
    for std in noise_std:
        acc, precision, recall, f1 = eval(sdn_net, data_loader, sdn_compute_spike, std)
        sdn_acc.append(acc)
        sdn_precision.append(precision)
        sdn_recall.append(recall)
        sdn_f1.append(f1)
        acc, precision, recall, f1 = eval(spike_sn_net, data_loader, spike_sn_compute_spike, std)
        spike_sn_acc.append(acc)
        spike_sn_precision.append(precision)
        spike_sn_recall.append(recall)
        spike_sn_f1.append(f1)

    metrics = {
        "Accuracy": (sdn_acc, spike_sn_acc),
        "Precision": (sdn_precision, spike_sn_precision),
        "Recall": (sdn_recall, spike_sn_recall),
        "F1 Score": (sdn_f1, spike_sn_f1),
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2x2 grid
    axes = axes.flatten()  # flatten so we can index easily

    for i, (metric_name, (sdn_vals, spike_vals)) in enumerate(metrics.items()):
        ax = axes[i]
        ax.plot(noise_std, sdn_vals, marker='o', linestyle='-', linewidth=2, markersize=6, label="sdn")
        ax.plot(noise_std, spike_vals, marker='s', linestyle='-', linewidth=2, markersize=6, label="spike_sn")
        ax.set_xlabel("Gaussian Noise std", fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()
    plt.savefig('./noise_metrics.jpg', dpi=300)

plot_metrics(eval_input_gaussian)

plot_metrics(eval_output_gaussian)
