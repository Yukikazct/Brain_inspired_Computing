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
            if rng.random() < c * value:
                spikes[i, t] = 1

    return spikes


def uniform_encode(input_values, T):
    spikes = np.zeros((len(input_values), T), dtype=int)

    for i, value in enumerate(input_values):
        # 1. 计算发放次数
        count = int(np.round(value * T))
        if count <= 0:
            continue
        # 2. 均匀分配脉冲时刻
        spike_times = np.linspace(0, T - 1, count, dtype=int)
        # 去重，避免重复赋值
        spike_times = np.unique(spike_times)
        # 3. 赋值脉冲
        spikes[i, spike_times] = 1
    return spikes


def ttfs_encode(input_values, T):
    spikes = np.zeros((len(input_values), T), dtype=int)

    for i, value in enumerate(input_values):
        if value <= 0:
            continue
        spike_time = int(np.round((1 - value) * (T - 1)))
        spike_time = int(np.clip(spike_time, 0, T - 1))
        spikes[i, spike_time] = 1

    return spikes


def rank_order_encode(input_values, T, active_threshold=0.05):
    spikes = np.zeros((len(input_values), T), dtype=int)

    # 第一步：先筛出真正参与等级排序编码的神经元
    active_idx = []
    for i, value in enumerate(input_values):
        if value > active_threshold:
            active_idx.append(i)

    if len(active_idx) == 0:
        return spikes

    # 第二步：按输入强度从大到小排序
    sorted_idx = sorted(active_idx, key=lambda idx: input_values[idx], reverse=True)

    # 第三步：根据排序结果依次分配脉冲时刻
    step = max(1, T // (len(sorted_idx) + 1))
    for rank, idx in enumerate(sorted_idx):
        spike_time = min(rank * step, T - 1)
        spikes[idx, spike_time] = 1


    return spikes


def plot_spike_times(ax, spikes, title):
    neuron_ids, time_ids = np.where(spikes == 1)
    ax.scatter(time_ids, neuron_ids, s=18, color="tab:blue")
    ax.set_title(title)
    ax.set_ylabel("神经元编号")
    ax.grid(alpha=0.3)


def summarize(spikes):
    counts = spikes.sum(axis=1)
    active_mask = counts > 0

    if np.any(active_mask):
        first_times = np.argmax(spikes[active_mask], axis=1)
        mean_first_time = float(first_times.mean())
    else:
        mean_first_time = np.nan

    return {
        "total_spikes": int(counts.sum()),
        "active_neurons": int(active_mask.sum()),
        "mean_first_time": mean_first_time,
        "single_spike": bool(np.all(counts[active_mask] <= 1)) if np.any(active_mask) else True,
    }


# 本实验要求修改的参数
input_values = np.array([0.05, 0.18, 0.35, 0.52, 0.74, 0.93])
T = 30

# 其余默认参数（一般不需要修改）
c = 1.0
seed = 7  # 固定随机种子，便于复现实验结果
active_threshold = 0.05

poisson_spikes = poisson_encode(input_values, T=T, c=c, seed=seed)
uniform_spikes = uniform_encode(input_values, T=T)
ttfs_spikes = ttfs_encode(input_values, T=T)
rank_spikes = rank_order_encode(input_values, T=T, active_threshold=active_threshold)

all_spikes = {
    "泊松编码": poisson_spikes,
    "均匀编码": uniform_spikes,
    "首次脉冲发放时间编码": ttfs_spikes,
    "等级排序编码": rank_spikes,
}

summaries = {name: summarize(spikes) for name, spikes in all_spikes.items()}
random_flags = {"泊松编码": "是", "均匀编码": "否", "首次脉冲发放时间编码": "否", "等级排序编码": "否"}

fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
for ax, (name, spikes) in zip(axes, all_spikes.items()):
    plot_spike_times(ax, spikes, f"{name}的脉冲发放时刻图")
axes[-1].set_xlabel("time step")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

names = list(all_spikes.keys())
total_spikes = [summaries[name]["total_spikes"] for name in names]

axes[0].bar(names, total_spikes, color=["tab:blue", "tab:orange", "tab:red", "tab:green"])
axes[0].set_title("不同编码方式的总发放次数")
axes[0].set_ylabel("count")
axes[0].grid(alpha=0.3)

table_data = []
for name in names:
    info = summaries[name]
    mean_first_time = "N/A" if np.isnan(info["mean_first_time"]) else f"{info['mean_first_time']:.2f}"
    table_data.append([
        str(info["active_neurons"]),
        mean_first_time,
        random_flags[name],
        "是" if info["single_spike"] else "否",
    ])

axes[1].axis("off")
table = axes[1].table(
    cellText=table_data,
    rowLabels=names,
    colLabels=["活跃神经元数", "平均首次脉冲时间", "是否随机", "是否单次发放"],
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)
axes[1].set_title("编码方式特性比较", pad=12)

plt.tight_layout()
plt.show()
