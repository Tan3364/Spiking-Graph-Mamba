import json
import matplotlib.pyplot as plt
import os


# ==== 1. 读取JSON文件 ====
json_path = "./mamba_norm_fire_rate_log.json"  # 你的文件路径
with open(json_path, 'r') as f:
    data = json.load(f)

# ==== 2. 提取 epoch 和 layer 信息 ====
epochs = sorted(data.keys(), key=lambda x: int(x.split('_')[1]))  # 按数字排序
num_epochs = len(epochs)

# 获取层数和 fire rate 键名
sample_epoch = data[epochs[0]]
num_layers = len(sample_epoch)
fire_keys = list(next(iter(sample_epoch.values())).keys())

print(f"共 {num_epochs} 个 epoch，{num_layers} 层，每层包含 {len(fire_keys)} 个指标")

# ==== 3. 创建输出文件夹 ====
output_dir = "layer_fire_rate_plots"
os.makedirs(output_dir, exist_ok=True)

# ==== 4. 绘制每层的 fire rate 曲线 ====
for layer_idx in range(num_layers):
    layer_name = f"Layer_{layer_idx}"
    
    # 为每个 fire_rate 类型收集该层随 epoch 的数值
    layer_metrics = {key: [] for key in fire_keys}
    for epoch in epochs:
        epoch_data = data[epoch][layer_name]
        for key in fire_keys:
            layer_metrics[key].append(epoch_data[key])
    
    # === 绘图：每个指标单独一张图 ===
    for key in fire_keys:
        plt.figure(figsize=(6, 4))
        plt.plot(range(num_epochs), layer_metrics[key], marker='o', linewidth=2)
        plt.title(f"{layer_name} - {key}")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # 自动调整 x 轴标签显示密度
        step = max(1, num_epochs // 10)  # 最多显示10个标签
        plt.xticks(
            range(0, num_epochs, step),
            [e.split('_')[1] for e in epochs[::step]],
            rotation=45,
            ha='right',
            fontsize=8
        )
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{layer_name}_{key}.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

print(f"✅ 所有图已保存到文件夹：{output_dir}")
