import logging
import time
import os
import json
import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch

from graphgps.loss.subtoken_prediction_loss import subtoken_cross_entropy
from graphgps.utils import cfg_to_dict, flatten_dict, make_wandb_name

from deepspeed.profiling.flops_profiler import FlopsProfiler
#from torch.autograd import profiler
from torch.profiler import profile, record_function, ProfilerActivity

def subsample_batch_index(batch, min_k = 1, ratio = 0.1):
    torch.manual_seed(0)
    unique_batches = torch.unique(batch.batch)
    # Initialize list to store permuted indices
    permuted_indices = []
    for batch_index in unique_batches:
        # Extract indices for the current batch
        indices_in_batch = (batch.batch == batch_index).nonzero().squeeze()
        # See how many nodes in the graphs
        # And how many left after subsetting
        k = int(indices_in_batch.size(0)*ratio)
        # If subsetting gives more than 1, do subsetting
        if k > min_k:
            perm = torch.randperm(indices_in_batch.size(0))
            idx = perm[:k]
            idx = indices_in_batch[idx]
            idx, _ = torch.sort(idx)
        # Otherwise retain entire graph
        else:
            idx = indices_in_batch
        permuted_indices.append(idx)
    idx = torch.cat(permuted_indices)
    return idx


def arxiv_cross_entropy(pred, true, split_idx):
    true = true.squeeze(-1)
    if pred.ndim > 1 and true.ndim == 1:
        pred_score = F.log_softmax(pred[split_idx], dim=-1)
        loss =  F.nll_loss(pred_score, true[split_idx])
    else:
        raise ValueError("In ogbn cross_entropy calculation dimensions did not match.")
    return loss, pred_score

# @profiler.profile
# def profile_mem_forward(model, batch):
#     pred, true = model(batch)
#     return pred, true

def train_epoch(logger, loader, model, optimizer, scheduler, batch_accumulation):
    # flop related
    if_mem = False
    if_flop = False
    if_select = False
    if if_flop:
        prof = FlopsProfiler(model, None)
        #profile_step = 0
        total_flop_s = 0.
        sample_count = 0
        if if_select:
            total_node = 0

    model.train()
    optimizer.zero_grad()
    time_start = time.time()
    for iter, batch in enumerate(loader):

        # 确保 batch.batch 存在
        if not hasattr(batch, 'batch') or batch.batch is None:
            batch.batch = torch.full((batch.num_nodes,), iter, dtype=torch.long)

        if if_select:
            ratio = 1.0
            idx = subsample_batch_index(batch, min_k = 1, ratio = ratio)
            batch = batch.subgraph(idx)

        # flop related
        if if_flop: # and iter == profile_step:
            prof.start_profile()
        batch.split = 'train'
        batch.to(torch.device(cfg.device))

        pred, true = model(batch)
        if cfg.dataset.name == 'ogbg-code2':
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
        elif cfg.dataset.name == 'ogbn-arxiv':
            split_idx = loader.dataset.split_idx['train'].to(torch.device(cfg.device))
            loss, pred_score = arxiv_cross_entropy(pred, true, split_idx)
            _true = true[split_idx].detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)
        else:
            loss, pred_score = compute_loss(pred, true)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)

        # ============= 进阶版 Spike Loss (单边惩罚 + 动态压力) =============
        target_gcn = 0.25   
        target_mamba = 0.22
        
        # 策略：增加基础权重。由于使用了 relu，只有超标才罚，所以权重可以大胆设大一点
        # 你可以尝试在训练函数传入 cur_epoch，实现 spike_loss_weight = 1e-2 * (1.1 ** cur_epoch)
        #spike_loss_weight = 1e-1 
        spike_loss_weight = 0
         
        current_fire_rates = model.get_layer_fire_rate_avg()
        spike_loss = torch.tensor(0.0, device=loss.device)
        pair_count = 0
        
        num_layers = len(current_fire_rates["pos_gatedgcn_fire_rate"])
        
        for i in range(num_layers):
            # --- 约束 GatedGCN 组 ---
            p_gcn = current_fire_rates["pos_gatedgcn_fire_rate"][i]
            n_gcn = current_fire_rates["neg_gatedgcn_fire_rate"][i]
            if isinstance(p_gcn, torch.Tensor) and p_gcn.requires_grad:
                total_gcn = p_gcn + n_gcn
                # 【修改点】：使用 relu。只有 total > 0.25 时才有损失，低于 0.25 损失为 0
                diff_gcn = torch.relu(total_gcn - target_gcn)
                spike_loss = spike_loss + torch.pow(diff_gcn, 2) 
                pair_count += 1

            # --- 约束 Mamba 组 ---
            p_mam = current_fire_rates["pos_train_mamba_fire_rate"][i]
            n_mam = current_fire_rates["neg_train_mamba_fire_rate"][i]
            if isinstance(p_mam, torch.Tensor) and p_mam.requires_grad:
                total_mamba = p_mam + n_mam
                # 【修改点】：同上。只有总和超标才罚，鼓励模型在保证精度的前提下尽可能稀疏
                diff_mamba = torch.relu(total_mamba - target_mamba)
                spike_loss = spike_loss + torch.pow(diff_mamba, 2)
                pair_count += 1

        if pair_count > 0:
            # 合并损失
            # 注意：这里的 spike_loss / pair_count 如果是 0，说明全部达标，非常理想
            actual_spike_penalty = spike_loss / pair_count
            loss = loss + spike_loss_weight * actual_spike_penalty
            
            # 打印调试信息
            if iter % 10 == 0: # 每 10 个 batch 打印一次即可，避免刷屏
                print(f"Task_Loss: {loss.item():.4f} | Spike_Penalty: {actual_spike_penalty.item():.6f}")
        # ============================================================
        
        # ============= 加入 Spike Loss (方案 B: MSE 约束) =============
#        target_firing_rate = 0.25  # 设定你期望的目标发放率，例如 10%
#        spike_loss_weight = 1e-3  # 权重系数，需要根据实验调整
#
#        # 获取当前 batch 所有层的平均发放率
#        # 假设 model.get_layer_fire_rate_avg() 返回的是 { 'type1': [layer0_rate, layer1_rate, ...] }
#        current_fire_rates = model.get_layer_fire_rate_avg()
#
#        spike_loss = 0
#        count = 0
#        for key in current_fire_rates:
#            rates = current_fire_rates[key]
#            for r in rates:
#                if isinstance(r, torch.Tensor) and r.requires_grad:
#                    spike_loss += F.mse_loss(r, torch.full_like(r, target_firing_rate))
#                    #print('spike_loss',spike_loss)
#                    count += 1
#                else:
#                    if not isinstance(r, torch.Tensor):
#                        print(f"DEBUG: r is NOT a tensor, it is {type(r)}")
#                    elif not r.requires_grad:
#                        print(f"DEBUG: r IS a tensor, but requires_grad is FALSE. grad_fn is {r.grad_fn}")
#        if count > 0:
#            loss = loss + spike_loss_weight * (spike_loss / count)
        # ============================================================
        if if_flop:
            prof.stop_profile()
            flops = prof.get_total_flops()
            flops_s = flops/1000000000.
            total_flop_s+=flops_s
            sample_count+=len(torch.unique(batch.batch))
            params = prof.get_total_params()
            prof.end_profile()
            if if_select:
                total_node += batch.x.size(0)

        loss.backward()
        # Parameters update after accumulating gradients for given num. batches.
        if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader)):
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        for layer in model.layers:
                if hasattr(layer, 'fire_rate_list'):
                    for key in layer.fire_rate_list:
                        layer.fire_rate_list[key].clear()
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name)
        time_start = time.time()
    if if_flop:
        print('################ Print flop')
        print(total_flop_s/sample_count, params)
        print('################ End print flop')
    if if_mem:
        print('################ Print mem')
        print(torch.cuda.max_memory_allocated() / (1024 ** 2))
        print(torch.cuda.max_memory_reserved() / (1024 ** 2))
        #print(prof_mem.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
        print('################ End print mem')
    if if_select:
        print('################ Print avg nodes')
        print(total_node/sample_count)



@torch.no_grad()
def eval_epoch(logger, loader, model, split='val'):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.device))
        if cfg.gnn.head == 'inductive_edge':
            pred, true, extra_stats = model(batch)
        else:
            pred, true = model(batch)
            extra_stats = {}
        if cfg.dataset.name == 'ogbg-code2':
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
        elif cfg.dataset.name == 'ogbn-arxiv':
            index_split = loader.dataset.split_idx[split].to(torch.device(cfg.device))
            loss, pred_score = arxiv_cross_entropy(pred, true, index_split)
            _true = true[index_split].detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)
        else:
            loss, pred_score = compute_loss(pred, true)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=0, time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name,
                            **extra_stats)
        time_start = time.time()


@register_train('custom')
def custom_train(loggers, loaders, model, optimizer, scheduler):
    """
    Customized training pipeline.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler,
                                cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch %s', start_epoch)

    if cfg.wandb.use:
        try:
            import wandb
        except:
            raise ImportError('WandB is not installed.')
        if cfg.wandb.name == '':
            wandb_name = make_wandb_name(cfg)
        else:
            wandb_name = cfg.wandb.name
        run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project,
                         name=wandb_name)
        run.config.update(cfg_to_dict(cfg))

    num_splits = len(loggers)
    split_names = ['val', 'test']
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]

    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        start_time = time.perf_counter()
        # 每个 epoch 开始前，清空 fire_rate_list
        for layer in model.layers:
            for key in layer.fire_rate_list:
                layer.fire_rate_list[key].clear()

        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler,
                    cfg.optim.batch_accumulation)
        perf[0].append(loggers[0].write_epoch(cur_epoch))
        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model,
                           split=split_names[i - 1])
                perf[i].append(loggers[i].write_epoch(cur_epoch))
        else:
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1])

        val_perf = perf[1]
        if cfg.optim.scheduler == 'reduce_on_plateau':
            scheduler.step(val_perf[-1]['loss'])
        else:
            scheduler.step()
        full_epoch_times.append(time.perf_counter() - start_time)
        # Checkpoint with regular frequency (if enabled).
        if cfg.train.enable_ckpt and not cfg.train.ckpt_best \
                and is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

        if cfg.wandb.use:
            run.log(flatten_dict(perf), step=cur_epoch)

        # Log current best stats on eval epoch.
        if is_eval_epoch(cur_epoch):
            best_epoch = np.array([vp['loss'] for vp in val_perf]).argmin()
            best_train = best_val = best_test = ""
            if cfg.metric_best != 'auto':
                # Select again based on val perf of `cfg.metric_best`.
                m = cfg.metric_best
                best_epoch = getattr(np.array([vp[m] for vp in val_perf]),
                                     cfg.metric_agg)()
                if m in perf[0][best_epoch]:
                    best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
                else:
                    # Note: For some datasets it is too expensive to compute
                    # the main metric on the training set.
                    best_train = f"train_{m}: {0:.4f}"
                best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
                best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

                if cfg.wandb.use:
                    bstats = {"best/epoch": best_epoch}
                    for i, s in enumerate(['train', 'val', 'test']):
                        bstats[f"best/{s}_loss"] = perf[i][best_epoch]['loss']
                        if m in perf[i][best_epoch]:
                            bstats[f"best/{s}_{m}"] = perf[i][best_epoch][m]
                            run.summary[f"best_{s}_perf"] = \
                                perf[i][best_epoch][m]
                        for x in ['hits@1', 'hits@3', 'hits@10', 'mrr']:
                            if x in perf[i][best_epoch]:
                                bstats[f"best/{s}_{x}"] = perf[i][best_epoch][x]
                    run.log(bstats, step=cur_epoch)
                    run.summary["full_epoch_time_avg"] = np.mean(full_epoch_times)
                    run.summary["full_epoch_time_sum"] = np.sum(full_epoch_times)
            # Checkpoint the best epoch params (if enabled).
            if cfg.train.enable_ckpt and cfg.train.ckpt_best and \
                    best_epoch == cur_epoch:
                save_ckpt(model, optimizer, scheduler, cur_epoch)
                if cfg.train.ckpt_clean:  # Delete old ckpt each time.
                    clean_ckpt()

            # ===== 在这一段之后加上 fire_rate 保存逻辑 =====
            avg_fire_rates = model.get_layer_fire_rate_avg()
            logging.info(f"Epoch {cur_epoch} average fire rates:")
            epoch_dict = {}

            for i in range(len(model.layers)):
                # 在生成字符串展示时，调用 .item() 转为 float
                layer_info_list = []
                current_layer_data = {}
                for k in avg_fire_rates:
                    val = avg_fire_rates[k][i]
                    # 安全转换：如果是 Tensor 就 item，否则直接用
                    float_val = val.item() if isinstance(val, torch.Tensor) else val
                    layer_info_list.append(f"{k}: {float_val:.4f}")
                    current_layer_data[k] = float_val
                
                logging.info(f"  Layer {i}: {', '.join(layer_info_list)}")
                epoch_dict[f"Layer_{i}"] = current_layer_data
                
#            for i in range(len(model.layers)):
#                layer_info = ", ".join([f"{k}: {avg_fire_rates[k][i]:.4f}" for k in avg_fire_rates])
#                logging.info(f"  Layer {i}: {layer_info}")
#                epoch_dict[f"Layer_{i}"] = {k: avg_fire_rates[k][i] for k in avg_fire_rates}

            # === 保存为 JSON ===
            save_path = os.path.join(cfg.run_dir, "fire_rate_log.json")

            if os.path.exists(save_path):
                with open(save_path, "r") as f:
                    fire_rate_record = json.load(f)
            else:
                fire_rate_record = {}

            fire_rate_record[f"Epoch_{cur_epoch}"] = epoch_dict

            with open(save_path, "w") as f:
                json.dump(fire_rate_record, f, indent=4)

            logging.info(f"🔥 Fire rate data saved to {save_path}")
            # ===============================================

            logging.info(
                f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s "
                f"(avg {np.mean(full_epoch_times):.1f}s) | "
                f"Best so far: epoch {best_epoch}\t"
                f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
                f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
                f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
            )
            torch.cuda.empty_cache()
            if hasattr(model, 'trf_layers'):
                # Log SAN's gamma parameter values if they are trainable.
                for li, gtl in enumerate(model.trf_layers):
                    if torch.is_tensor(gtl.attention.gamma) and \
                            gtl.attention.gamma.requires_grad:
                        logging.info(f"    {gtl.__class__.__name__} {li}: "
                                     f"gamma={gtl.attention.gamma.item()}")
    logging.info(f"Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
    logging.info(f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h")
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()
    # close wandb
    if cfg.wandb.use:
        run.finish()
        run = None

    logging.info('Task done, results saved in %s', cfg.run_dir)


@register_train('inference-only')
def inference_only(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to run inference only.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    num_splits = len(loggers)
    split_names = ['train', 'val', 'test']
    perf = [[] for _ in range(num_splits)]
    cur_epoch = 0
    start_time = time.perf_counter()

    for i in range(0, num_splits):
        eval_epoch(loggers[i], loaders[i], model,
                   split=split_names[i])
        perf[i].append(loggers[i].write_epoch(cur_epoch))
    val_perf = perf[1]

    best_epoch = np.array([vp['loss'] for vp in val_perf]).argmin()
    best_train = best_val = best_test = ""
    if cfg.metric_best != 'auto':
        # Select again based on val perf of `cfg.metric_best`.
        m = cfg.metric_best
        best_epoch = getattr(np.array([vp[m] for vp in val_perf]),
                             cfg.metric_agg)()
        if m in perf[0][best_epoch]:
            best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
        else:
            # Note: For some datasets it is too expensive to compute
            # the main metric on the training set.
            best_train = f"train_{m}: {0:.4f}"
        best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
        best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

    logging.info(
        f"> Inference | "
        f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
        f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
        f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
    )
    logging.info(f'Done! took: {time.perf_counter() - start_time:.2f}s')
    for logger in loggers:
        logger.close()
