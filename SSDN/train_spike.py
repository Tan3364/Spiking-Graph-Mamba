import argparse
import os
import time
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from loguru import logger
import torch.nn.functional as F

from dataset import MemDataset
from model import SDN
from utils import Metrics, MetricsCheckpoint, parameters_count

parser = argparse.ArgumentParser(description="SDN Training")
parser.add_argument("--lr", default=0.1, type=float, help="Learning rate")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay")
parser.add_argument("--epochs", default=100, type=int, help="Training epochs")
parser.add_argument(
    "--num_workers", default=4, type=int, help="Number of workers to use for dataloader"
)
parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
# Model
parser.add_argument("--n_layers", default=1, type=int, help="Number of layers")
parser.add_argument("--d_model", default=8, type=int, help="Model dimension")
parser.add_argument("--k", "--kernel_size", default=8, type=int, help="Kernel size")

# Dataset
parser.add_argument(
    "--training", type=str, required=True, help="Path to training dataset"
)
parser.add_argument("--test", type=str, required=True, help="Path to test dataset")

# General
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--margin", default=None, type=float, help="Margin for margin_loss")

items_group = parser.add_mutually_exclusive_group()
items_group.add_argument("--resume", "-r", default=None, type=str)
items_group.add_argument("--save", "-s", default="exp", type=str)

args = parser.parse_args()


# =======================
# Seed (CRITICAL)
# =======================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(args.seed)


# =======================
# Device & Logger
# =======================
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Model will be trained on device: {device}")

if args.resume is not None:
    args.save = args.resume

os.makedirs(args.save, exist_ok=True)
logger.add(os.path.join(args.save, "exp.log"))


# =======================
# Data
# =======================
logger.info("==> Preparing data.")


def split_train_val(train, val_split):
    train_len = int(len(train) * (1.0 - val_split))
    train, val = torch.utils.data.random_split(
        train,
        (train_len, len(train) - train_len),
        generator=torch.Generator().manual_seed(args.seed),
    )
    return train, val


transform = transforms.Lambda(lambda x: x.view(1, -1))

trainset = MemDataset(args.training, transform=transform)
trainset, valset = split_train_val(trainset, val_split=0.1)

testset = MemDataset(args.test, transform=transform)

# Dataloader seed
g = torch.Generator()
g.manual_seed(args.seed)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    generator=g,
)

valloader = torch.utils.data.DataLoader(
    valset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
)


# =======================
# Model
# =======================
logger.info("==> Building model.")
model = SDN(
    d_model=args.d_model,
    kernel_size=args.k,
    n_layers=args.n_layers,
)

logger.info(model)
logger.info(f"Params: {parameters_count(model):,}")

model = model.to(device)


# =======================
# Optim
# =======================
criterion = nn.SmoothL1Loss()
optimizer = optim.AdamW(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, args.epochs
)


# =======================
# Checkpoint
# =======================
best_metrics_checkpoint = MetricsCheckpoint(loss=float("+inf"), epoch=-1)
start_epoch = 0

last_checkpoint_filename = os.path.join(args.save, "last_checkpoint.pth")
best_checkpoint_filename = os.path.join(args.save, "best_checkpoint.pth")

if args.resume:
    logger.info("==> Resuming from checkpoint.")
    checkpoint = torch.load(last_checkpoint_filename)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = checkpoint["epoch"] + 1
    best_metrics_checkpoint = torch.load(best_checkpoint_filename)["metrics"]


# =======================
# Loss
# =======================
def margin_loss(u_eff, spike_target, threshold=1.0, margin=None):
    #print('u_eff',u_eff)
    #print("1",spike_target)
    # 使用 Softplus 代替 ReLU
    # beta=5 让曲线在转角处更陡峭，更接近 ReLU，但保留微弱梯度
    loss_pos = F.relu((threshold + margin) - u_eff)
    #print('2',loss_pos)
    loss_neg = F.relu(u_eff + (threshold + margin))
    loss_zero = F.relu(torch.abs(u_eff) - (threshold - margin))
    # 将线性推力改为平方推力
    #loss_zero = torch.relu(u_eff - (threshold - margin))
    #print('2',loss_zero)
    # 掩码掩掉无关位置
    m_pos = (spike_target == 1).float()
    #print('m_pos',m_pos)
    m_neg = (spike_target == -1).float()
    m_zero = (spike_target == 0).float()
    #print('m_zero',m_zero)

    # 计算loss
    l_pos = (loss_pos * m_pos).sum() / (m_pos.sum() + 1e-6)
    ##l_pos = loss_pos * m_pos
    #print('l_pos',l_pos)
    #print('l_pos',(loss_pos * m_pos).sum())
    l_neg = (loss_neg * m_neg).sum() / (m_neg.sum() + 1e-6)
    ##l_neg = loss_neg * m_neg
    #print('l_neg',l_neg)
    #print('l_neg',l_neg)
    l_zero = (loss_zero * m_zero).sum() / (m_zero.sum() + 1e-6)
    ##l_zero = loss_zero * m_zero
    #print('l_zero',l_zero)

    ##element_wise_loss = (loss_pos * m_pos) + (loss_zero * m_zero)
    ##element_wise_loss = (loss_pos * m_pos) + (loss_neg * m_neg) + (loss_zero * m_zero)
    #print('element_wise_loss',element_wise_loss)

    ##final_loss = element_wise_loss.mean()
    ##final_loss = l_pos+l_neg+l_zero
    #print(f"Total elements: {element_wise_loss.numel()}")
    #print('final_loss',final_loss)
    #print("1",final_loss.grad_fn)
    total_loss = l_pos + l_neg +  l_zero
    #print(f"Weighted: L_pulse={100*l_pos:.6f}, L_neg={100*l_neg:.6f}, L_zero={200*l_zero:.6f}")

    return total_loss

# Training
def train(trainloader):
    model.train()
    train_loss = Metrics("Loss")
    abs_error = []
    acc1 = Metrics("Acc@1", scale=100, format=".2f", suffix="%")
    for inputs, targets, spikes in trainloader:
        inputs, targets, spikes = (
            inputs.to(device),
            targets.to(device),
            spikes.to(device),
        )
        optimizer.zero_grad()
        outputs = model(inputs)
        loss_1 = criterion(outputs, targets)
        loss_2 = margin_loss(outputs, spikes, threshold=1.0, margin=args.margin)
        loss = 0.0 * loss_1 + 1.0 *loss_2

        #print('loss',loss)
        #pred_s = (outputs + inputs.squeeze() >= 1).float()
        pred_s = (
            (outputs >= 1).float()
            - (outputs <= -1).float()
        )
        #print(outputs.detach() + inputs.squeeze())
        #print(pred_s)
        acc = (pred_s == spikes).float().mean()
        acc1.update(acc.item(), inputs.size(0))

        loss.backward()
        #print(f"Negative Mask Active Count: {m_neg.sum().item()}")
        #if loss > 0:
        # 检查梯度是否传到了 outputs
            #print(f"Outputs Gradient Norm: {model.decoder.weight.grad.norm().item()}")
        optimizer.step()

        abs_error.append((outputs - targets).abs())
        train_loss.update(loss.item())
    abs_error = torch.cat(abs_error)
    std, mean = torch.std_mean(abs_error)
    return MetricsCheckpoint(
        loss=train_loss.avg,
        mae_max=abs_error.max().item(),
        mae_mean=mean.item(),
        mae_std=std.item(),
        acc1=acc1.avg,
    )

@torch.no_grad()
def eval(dataloader):
    model.eval()
    eval_loss = Metrics("Loss")
    abs_error = []
    acc1 = Metrics("Acc@1", scale=100, format=".2f", suffix="%")
    for inputs, targets, spikes in dataloader:
        inputs, targets, spikes = (
            inputs.to(device),
            targets.to(device),
            spikes.to(device),
        )
        outputs = model(inputs)
        #print(f"Outputs range: [min: {outputs.min().item():.4f}, max: {outputs.max().item():.4f}]")

        loss_1 = criterion(outputs, targets)
        loss_2 = margin_loss(outputs, spikes, threshold=1.0, margin=args.margin)
        loss = 0.0 * loss_1 + 1.0 *loss_2
        #print('loss_2',loss_2)
        #pred_s = (outputs + inputs.squeeze() >= 1).float()
        pred_s = (
            (outputs >= 1).float()
            - (outputs <= -1).float()
        )

        acc = (pred_s == spikes).float().mean()
        acc1.update(acc.item(), inputs.size(0))

        abs_error.append((outputs - targets).abs())
        eval_loss.update(loss.item())

    abs_error = torch.cat(abs_error)
    std, mean = torch.std_mean(abs_error)

    return MetricsCheckpoint(
        loss=eval_loss.avg,
        mae_max=abs_error.max().item(),
        mae_mean=mean.item(),
        mae_std=std.item(),
        acc1=acc1.avg,
    )


# =======================
# Main Loop
# =======================
logger.info("==> Training")

for epoch in range(start_epoch, args.epochs):
    logger.info(f"==> Epoch {epoch}")

    train_metrics = train(trainloader)
    val_metrics = eval(valloader)
    test_metrics = eval(testloader)

    logger.info(f"Train: {train_metrics}")
    logger.info(f"Val:   {val_metrics}")
    logger.info(f"Test:  {test_metrics}")

    scheduler.step()

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metrics": val_metrics,
        "epoch": epoch,
    }

    torch.save(checkpoint, last_checkpoint_filename)

    if val_metrics < best_metrics_checkpoint:
        best_metrics_checkpoint = val_metrics
        checkpoint["training_metrics"] = train_metrics
        checkpoint["test_metrics"] = test_metrics
        torch.save(checkpoint, best_checkpoint_filename)


best_checkpoint = torch.load(best_checkpoint_filename)
logger.info("=================================")
logger.info("Best Performance:")
logger.info(f"Training:   {best_checkpoint['training_metrics']}")
logger.info(f"Validation: {best_checkpoint['metrics']}")
logger.info(f"Test:       {best_checkpoint['test_metrics']}")
logger.info("=================================")
logger.info("==> Finished.")
