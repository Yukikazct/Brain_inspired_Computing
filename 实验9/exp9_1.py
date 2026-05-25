import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "SimHei",
    "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# 本实验要求修改的参数
THRESHOLD = 1.0        # 发放阈值；越高越容易漏掉正样本，越低越容易让负样本误发放
ETA = 0.025            # Tempotron 学习率；控制每次误分类后的权重更新幅度
PATTERN_JITTER = 2.0   # 同一模式内共同发放的时间抖动；越大表示共同发放越不精确

# 其余默认参数（一般不需要修改）
N_INPUTS = 24
T_MAX = 80
TAU_M = 15.0
TAU_S = 3.0
KEY_NEURONS = np.arange(0, 8)
NEGATIVE_PATTERN_NEURONS = np.arange(8, 16)
DISTRACTOR_NEURONS = np.arange(16, N_INPUTS)
NON_KEY_NEURONS = np.arange(8, N_INPUTS)
POS_CENTER_RANGE = (15, 65)


def psp_kernel(delta_t, tau_m=TAU_M, tau_s=TAU_S):
    # TODO 1：
    # 请补全 Tempotron 使用的双指数 PSP 核函数。
    # 要求：
    # 1. 当 delta_t < 0 时，核函数值应为 0；
    # 2. 当 delta_t >= 0 时，计算 exp(-delta_t/tau_m)-exp(-delta_t/tau_s)；
    # 3. 将核函数除以其最大值，使峰值约为 1。
    if np.isscalar(delta_t):
        if delta_t < 0:
            return 0.0
        val = np.exp(-delta_t / tau_m) - np.exp(-delta_t / tau_s)
        return val
    else:
        vals = np.zeros_like(delta_t, dtype=float)
        mask = delta_t >= 0
        vals[mask] = np.exp(-delta_t[mask] / tau_m) - np.exp(-delta_t[mask] / tau_s)
        return vals


def generate_temporal_patterns(num_per_class=60, seed=7):
    rng = np.random.default_rng(seed)
    samples = []
    labels = []

    for _ in range(num_per_class):
        spikes = np.zeros((N_INPUTS, T_MAX), dtype=float)
        center_t = int(rng.integers(POS_CENTER_RANGE[0], POS_CENTER_RANGE[1] + 1))

        for neuron in KEY_NEURONS:
            t = int(np.clip(rng.normal(loc=center_t, scale=PATTERN_JITTER), 5, T_MAX - 5))
            spikes[neuron, t] = 1.0
        for _ in range(10):
            neuron = int(rng.choice(NON_KEY_NEURONS))
            t = int(rng.integers(5, T_MAX - 5))
            spikes[neuron, t] = 1.0
        samples.append(spikes)
        labels.append(1)

    for _ in range(num_per_class):
        spikes = np.zeros((N_INPUTS, T_MAX), dtype=float)
        center_t = int(rng.integers(POS_CENTER_RANGE[0], POS_CENTER_RANGE[1] + 1))
        for neuron in NEGATIVE_PATTERN_NEURONS:
            t = int(np.clip(rng.normal(loc=center_t, scale=PATTERN_JITTER), 5, T_MAX - 5))
            spikes[neuron, t] = 1.0
        for _ in range(10):
            neuron = int(rng.choice(np.concatenate([KEY_NEURONS, DISTRACTOR_NEURONS])))
            t = int(rng.integers(5, T_MAX - 5))
            spikes[neuron, t] = 1.0
        samples.append(spikes)
        labels.append(0)

    samples = np.array(samples)
    labels = np.array(labels, dtype=int)
    order = rng.permutation(len(labels))
    return samples[order], labels[order]


# Precompute normalized PSP kernel lookup table
_PSP_TABLE = None


def _get_psp_table():
    global _PSP_TABLE
    if _PSP_TABLE is None:
        raw = psp_kernel(np.arange(T_MAX, dtype=float))
        _PSP_TABLE = raw / np.max(raw)
    return _PSP_TABLE


def compute_voltage(spikes, weights, v_rest=0.0):
    # TODO 2：
    # 请根据 V(t)=V_rest+sum_i w_i sum_{t_i^f} K(t-t_i^f) 计算整段膜电位曲线。
    # 提示：
    # 1. voltage 的形状应为 (T_MAX,)；
    # 2. 需要遍历每个输入神经元；
    # 3. 对同一个输入神经元的多个脉冲也要逐个累加 PSP；
    # 4. 不同输入神经元的贡献需要按权重加权后相加。
    kernel_table = _get_psp_table()
    voltage = np.full(T_MAX, v_rest, dtype=float)
    for i in range(spikes.shape[0]):
        spike_times = np.where(spikes[i] > 0)[0]
        if len(spike_times) == 0:
            continue
        for t_f in spike_times:
            duration = T_MAX - t_f
            voltage[t_f:] += weights[i] * kernel_table[:duration]
    return voltage


def classify_by_peak(voltage, threshold):
    # TODO 3：
    # 请找出最大膜电位时刻 t_max、最大膜电位 v_max，并判断是否越过阈值。
    # 返回 predicted_spike, t_max, v_max。
    t_max = int(np.argmax(voltage))
    v_max = float(voltage[t_max])
    predicted_spike = int(v_max >= threshold)
    return predicted_spike, t_max, v_max


def tempotron_weight_update(spikes, label, predicted_spike, t_max, eta):
    # TODO 4：
    # 请补全 Tempotron 误分类更新规则。
    # 说明：
    # 1. label=1 表示正样本，应该发放；
    # 2. label=0 表示负样本，不应该发放；
    # 3. 正样本未发放时，direction=+1；
    # 4. 负样本错误发放时，direction=-1；
    # 5. 分类正确时，返回全 0 的 delta_w；
    # 6. 只累加 t_i^f < t_max 的输入脉冲贡献。
    delta_w = np.zeros(spikes.shape[0], dtype=float)

    if label == 1 and predicted_spike == 0:
        direction = 1.0
    elif label == 0 and predicted_spike == 1:
        direction = -1.0
    else:
        return delta_w

    for i in range(spikes.shape[0]):
        spike_times = np.where(spikes[i] > 0)[0]
        spike_times = spike_times[spike_times < t_max]
        if len(spike_times) == 0:
            continue
        contributions = np.sum(psp_kernel(t_max - spike_times))
        delta_w[i] = eta * direction * contributions

    return delta_w


def evaluate(samples, labels, weights, threshold):
    preds = []
    for spikes in samples:
        voltage = compute_voltage(spikes, weights)
        pred, _, _ = classify_by_peak(voltage, threshold)
        preds.append(pred)

    preds = np.array(preds, dtype=int)
    positive_mask = labels == 1
    negative_mask = labels == 0

    error_rate = float(np.mean(preds != labels))
    positive_hit_rate = float(np.mean(preds[positive_mask] == 1))
    false_alarm_rate = float(np.mean(preds[negative_mask] == 1))

    return {
        "error_rate": error_rate,
        "positive_hit_rate": positive_hit_rate,
        "false_alarm_rate": false_alarm_rate,
    }


def train_tempotron(samples, labels, epochs=35, eta=ETA, threshold=THRESHOLD, seed=11):
    rng = np.random.default_rng(seed)
    # 初始权重设置为未学会状态：目标模式输入偏低，非目标模式输入偏高。
    weights = rng.uniform(0.03, 0.08, size=N_INPUTS)
    weights[KEY_NEURONS] = rng.uniform(0.01, 0.04, size=len(KEY_NEURONS))
    weights[NEGATIVE_PATTERN_NEURONS] = rng.uniform(0.12, 0.20, size=len(NEGATIVE_PATTERN_NEURONS))
    initial_weights = weights.copy()
    history = {"error_rate": [], "positive_hit_rate": [], "false_alarm_rate": []}

    for _ in range(epochs):
        order = rng.permutation(len(labels))
        for sample_idx in order:
            spikes = samples[sample_idx]
            label = int(labels[sample_idx])
            voltage = compute_voltage(spikes, weights)
            pred, t_max, _ = classify_by_peak(voltage, threshold)
            delta_w = tempotron_weight_update(spikes, label, pred, t_max, eta)
            weights = np.clip(weights + delta_w, 0.0, 1.2)

        metrics = evaluate(samples, labels, weights, threshold)
        history["error_rate"].append(metrics["error_rate"])
        history["positive_hit_rate"].append(metrics["positive_hit_rate"])
        history["false_alarm_rate"].append(metrics["false_alarm_rate"])

    return weights, initial_weights, history


def plot_input_spike_sequence(ax, spikes, title):
    event_times = [np.where(spikes[i] > 0)[0] for i in range(spikes.shape[0])]
    colors = []
    for i in range(spikes.shape[0]):
        if i in KEY_NEURONS:
            colors.append("#2563eb")
        elif i in NEGATIVE_PATTERN_NEURONS:
            colors.append("#f97316")
        else:
            colors.append("#666666")
    ax.eventplot(event_times, lineoffsets=np.arange(spikes.shape[0]), linelengths=0.75, colors=colors)
    ax.set_xlim(0, T_MAX)
    ax.set_title(title)
    ax.set_xlabel("时间 (ms)")
    ax.set_ylabel("输入神经元")
    ax.grid(axis="x", alpha=0.18)
    ax.text(
        0.01,
        0.98,
        "蓝：目标模式输入   橙：非目标模式输入   灰：其他输入",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor="#dddddd", alpha=0.85),
    )


def plot_voltage(ax, spikes, weights_before, weights_after, threshold, title):
    before = compute_voltage(spikes, weights_before)
    after = compute_voltage(spikes, weights_after)
    pred_b, t_b, v_b = classify_by_peak(before, threshold)
    pred_a, t_a, v_a = classify_by_peak(after, threshold)
    ax.plot(before, label=f"训练前 Vmax={v_b:.2f}, pred={pred_b}", color="#94a3b8", linewidth=2.0)
    ax.plot(after, label=f"训练后 Vmax={v_a:.2f}, pred={pred_a}", color="#c1121f", linewidth=2.2)
    ax.axhline(threshold, color="#333333", linestyle="--", linewidth=1.2, label="阈值")
    ax.scatter([t_b], [v_b], color="#64748b", s=42, zorder=3)
    ax.scatter([t_a], [v_a], color="#c1121f", s=52, zorder=3)
    ax.set_xlim(0, T_MAX)
    ax.set_title(title)
    ax.set_xlabel("时间 (ms)")
    ax.set_ylabel("膜电位")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9, loc="upper right")


def main():
    samples, labels = generate_temporal_patterns(num_per_class=70, seed=7)
    threshold = THRESHOLD
    weights, initial_weights, history = train_tempotron(samples, labels, threshold=threshold)
    metrics = evaluate(samples, labels, weights, threshold)

    print("final error rate:", metrics["error_rate"])
    print("positive hit rate:", metrics["positive_hit_rate"])
    print("false alarm rate:", metrics["false_alarm_rate"])

    pos_idx = int(np.where(labels == 1)[0][0])
    neg_idx = int(np.where(labels == 0)[0][0])

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.05, 0.95])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])

    plot_input_spike_sequence(ax1, samples[pos_idx], "正样本输入脉冲序列图：目标模式输入共同发放")
    plot_input_spike_sequence(ax2, samples[neg_idx], "负样本输入脉冲序列图：非目标模式输入共同发放")
    plot_voltage(ax3, samples[pos_idx], initial_weights, weights, threshold, "正样本膜电位曲线")
    plot_voltage(ax4, samples[neg_idx], initial_weights, weights, threshold, "负样本膜电位曲线")

    ax5.bar(np.arange(N_INPUTS) - 0.18, initial_weights, width=0.36, label="训练前", color="#94a3b8")
    ax5.bar(np.arange(N_INPUTS) + 0.18, weights, width=0.36, label="训练后", color="#15803d")
    ax5.axvspan(KEY_NEURONS[0] - 0.5, KEY_NEURONS[-1] + 0.5, color="#dbeafe", alpha=0.7, label="目标模式输入")
    ax5.axvspan(
        NEGATIVE_PATTERN_NEURONS[0] - 0.5,
        NEGATIVE_PATTERN_NEURONS[-1] + 0.5,
        color="#ffedd5",
        alpha=0.7,
        label="非目标模式输入",
    )
    ax5.set_title("训练前后的突触权重")
    ax5.set_xlabel("输入神经元")
    ax5.set_ylabel("突触权重")
    ax5.grid(axis="y", alpha=0.2)
    ax5.legend(fontsize=9)

    fig.suptitle("Tempotron 二分类时空脉冲模式识别", fontsize=18, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs("outputs", exist_ok=True)
    fig.savefig("outputs/exp9_1_tempotron_result.png", dpi=180)
    plt.show()


if __name__ == "__main__":
    main()
