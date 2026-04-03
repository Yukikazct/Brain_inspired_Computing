import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def ttfs_encode(input_values, T):
    spikes = np.zeros((len(input_values), T), dtype=int)
    first_spike_times = np.full(len(input_values), np.nan)

    for i, value in enumerate(input_values):
        if value <= 0:
            continue

        spike_time = np.round((1 - value) * (T - 1))  # 线性映射公式
        spike_time = np.clip(spike_time, 0, T - 1)  # 限制范围[0, T-1]
        spike_time = int(spike_time)  # 转整数时间步
        spikes[i, spike_time] = 1  # 标记脉冲
        first_spike_times[i] = spike_time  # 记录首次脉冲时间
    return spikes, first_spike_times


def plot_spike_times(ax, spikes, color="tab:red"):
    neuron_ids, time_ids = np.where(spikes == 1)
    ax.scatter(time_ids, neuron_ids, s=25, color=color)
    ax.set_ylabel("神经元编号")
    ax.grid(alpha=0.3)


# 本实验要求修改的参数
input_values = np.array([0.05, 0.18, 0.35, 0.52, 0.74, 0.93])
T = 30

spikes, first_spike_times = ttfs_encode(input_values, T=T)
neuron_ids = np.arange(len(input_values))
valid_mask = ~np.isnan(first_spike_times)

fig, axes = plt.subplots(3, 1, figsize=(10, 8))

axes[0].bar(neuron_ids, input_values, color="tab:gray")
axes[0].set_title("输入强度")
axes[0].set_ylabel("value")
axes[0].grid(alpha=0.3)

plot_spike_times(axes[1], spikes)
axes[1].set_title("首次脉冲发放时间编码的脉冲发放时刻图")
axes[1].set_xlabel("time step")

axes[2].scatter(input_values[valid_mask], first_spike_times[valid_mask], s=60, color="tab:purple")
axes[2].set_title("输入强度与首次脉冲时间")
axes[2].set_xlabel("input intensity")
axes[2].set_ylabel("first spike time")
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()
