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


def poisson_encode(input_values, T, c=1.0, seed=7):
    rng = np.random.default_rng(seed)
    spikes = np.zeros((len(input_values), T), dtype=int)

    for i, value in enumerate(input_values):
        for t in range(T):
            rand_num = rng.random()  # 生成[0,1)随机数
            if rand_num < c * value:
                spikes[i, t] = 1
    return spikes


def plot_spike_times(ax, spikes, color="tab:blue"):
    neuron_ids, time_ids = np.where(spikes == 1)
    ax.scatter(time_ids, neuron_ids, s=20, color=color)
    ax.set_ylabel("神经元编号")
    ax.grid(alpha=0.3)


# 本实验要求修改的参数
input_values = np.array([0.05, 0.18, 0.35, 0.52, 0.88, 0.98])
T = 40

# 其余默认参数（一般不需要修改）
c = 1.0
seed = 7  # 固定随机种子，便于复现实验结果

spikes = poisson_encode(input_values, T=T, c=c, seed=seed)
spike_counts = spikes.sum(axis=1)
time = np.arange(T)
neuron_ids = np.arange(len(input_values))

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)

axes[0].bar(neuron_ids, input_values, color="tab:gray")
axes[0].set_title("输入强度")
axes[0].set_ylabel("value")
axes[0].grid(alpha=0.3)

plot_spike_times(axes[1], spikes)
axes[1].set_title("泊松编码的脉冲发放时刻图")
axes[1].set_xlabel("time step")

axes[2].bar(neuron_ids, spike_counts, color="tab:orange")
axes[2].set_title("时间窗内的总发放次数")
axes[2].set_xlabel("neuron id")
axes[2].set_ylabel("count")
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()
