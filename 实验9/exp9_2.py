import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "SimHei",
    "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# 本实验要求修改的参数
ETA_TARGET = 0.012     # 目标脉冲驱动的增强幅度；越大越容易补上缺失目标脉冲
ETA_OUTPUT = 0.012     # 实际输出脉冲驱动的抑制幅度；越大越容易压制多余或错位输出
TAU_TRACE = 6.0        # 输入脉冲迹变量时间常数；越大表示更久以前的输入也会被归因

# 其余默认参数（一般不需要修改）
THRESHOLD = 0.95
ALPHA = 0.84
N_INPUTS = 32
T_MAX = 100
TARGET_TIMES = np.array([20, 32, 44, 56, 68, 80])
TARGET_INPUT_GROUPS = [np.arange(i * 3, (i + 1) * 3) for i in range(len(TARGET_TIMES))]
TARGET_INPUTS = np.concatenate(TARGET_INPUT_GROUPS)
NON_TARGET_TIMES = np.array([26, 38, 50, 62, 74])
NON_TARGET_INPUTS = np.arange(18, 30)
OTHER_INPUTS = np.arange(30, N_INPUTS)


def generate_input_spikes(seed=13):
    rng = np.random.default_rng(seed)
    spikes = np.zeros((N_INPUTS, T_MAX), dtype=float)

    for group, target_t in zip(TARGET_INPUT_GROUPS, TARGET_TIMES):
        for neuron in group:
            for loc in [target_t - 2, target_t]:
                t = int(np.clip(rng.normal(loc=loc, scale=0.45), 5, T_MAX - 5))
                spikes[neuron, t] = 1.0

    for idx, neuron in enumerate(NON_TARGET_INPUTS):
        non_target_t = NON_TARGET_TIMES[idx % len(NON_TARGET_TIMES)]
        for loc in [non_target_t - 2, non_target_t]:
            t = int(np.clip(rng.normal(loc=loc, scale=0.5), 5, T_MAX - 5))
            spikes[neuron, t] = 1.0

    for neuron in OTHER_INPUTS:
        for _ in range(2):
            t = int(rng.integers(8, T_MAX - 8))
            spikes[neuron, t] = 1.0

    return spikes


def generate_target_spikes():
    target = np.zeros(T_MAX, dtype=float)
    target[TARGET_TIMES] = 1.0
    return target


def compute_pre_trace(input_spikes, tau_trace=TAU_TRACE):
    # TODO 5：
    # 请补全输入脉冲迹变量。
    # 要求：
    # 1. trace 的形状与 input_spikes 相同；
    # 2. 每个时间步先让上一时刻迹变量按 exp(-1/tau_trace) 衰减；
    # 3. 再加上当前时间步的输入脉冲；
    # 4. 返回每个输入神经元在每个时刻的迹变量。
    trace = np.zeros_like(input_spikes, dtype=float)
    decay = np.exp(-1.0 / tau_trace)
    for t in range(1, T_MAX):
        trace[:, t] = decay * trace[:, t - 1] + input_spikes[:, t]
    return trace


def simulate_output_neuron(input_spikes, weights, alpha=ALPHA, threshold=THRESHOLD, reset=0.0):
    # TODO 6：
    # 请补全输出神经元仿真。
    # 要求：
    # 1. 使用 u(t)=alpha*u(t-1)+sum_i w_i*x_i(t)；
    # 2. 当 u(t)>=threshold 时，output[t]=1；
    # 3. 发放后将膜电位复位为 reset；
    # 4. 返回 membrane 和 output，二者形状均为 (T_MAX,)。
    membrane = np.zeros(T_MAX, dtype=float)
    output = np.zeros(T_MAX, dtype=float)
    for t in range(1, T_MAX):
        current = np.dot(weights, input_spikes[:, t])
        membrane[t] = alpha * membrane[t - 1] + current
        if membrane[t] >= threshold:
            output[t] = 1.0
            membrane[t] = reset
    return membrane, output


def resume_update(input_spikes, target_spikes, output_spikes, eta_target=ETA_TARGET, eta_output=ETA_OUTPUT, tau_trace=TAU_TRACE):
    pre_trace = compute_pre_trace(input_spikes, tau_trace=tau_trace)
    delta_w = np.zeros(input_spikes.shape[0], dtype=float)

    for t in range(T_MAX):
        if target_spikes[t] > 0 and output_spikes[t] == 0:
            # TODO 7：
            # 目标脉冲出现而实际输出没有出现时，根据输入脉冲迹变量增强对应突触。
            # 近期活跃输入应更容易推动目标时刻发放。
            delta_w += eta_target * pre_trace[:, t]

        if output_spikes[t] > 0 and target_spikes[t] == 0:
            # TODO 8：
            # 实际输出脉冲出现而目标脉冲没有出现时，根据输入脉冲迹变量执行 anti-STDP 抑制。
            # 实际输出出现时，削弱近期活跃输入。
            delta_w -= eta_output * pre_trace[:, t]

    return delta_w


def nearest_spike_distance(target_spikes, output_spikes, tolerance=2):
    target_times = np.where(target_spikes > 0)[0]
    output_times = np.where(output_spikes > 0)[0]

    if len(output_times) == 0:
        return {"missed": len(target_times), "extra": 0, "mean_distance": float(T_MAX)}

    distances = []
    missed = 0
    for target_t in target_times:
        nearest_idx = int(np.argmin(np.abs(output_times - target_t)))
        nearest_t = int(output_times[nearest_idx])
        distance = abs(nearest_t - target_t)
        distances.append(distance)
        if distance > tolerance:
            missed += 1

    extra = 0
    for out_t in output_times:
        if np.min(np.abs(target_times - out_t)) > tolerance:
            extra += 1

    return {"missed": int(missed), "extra": int(extra), "mean_distance": float(np.mean(distances))}


def train_resume(input_spikes, target_spikes, epochs=120, seed=5):
    rng = np.random.default_rng(seed)
    # 初始权重设置为未学会状态：目标脉冲前输入偏低，非目标时刻输入偏高。
    weights = rng.uniform(0.01, 0.03, size=N_INPUTS)
    for group in TARGET_INPUT_GROUPS:
        weights[group] = rng.uniform(0.015, 0.035, size=len(group))
    weights[NON_TARGET_INPUTS] = rng.uniform(0.22, 0.28, size=len(NON_TARGET_INPUTS))
    initial_weights = weights.copy()
    snapshots = {}

    for epoch in range(epochs + 1):
        membrane, output = simulate_output_neuron(input_spikes, weights)
        if epoch in {0, epochs}:
            snapshots[epoch] = {"weights": weights.copy(), "membrane": membrane.copy(), "output": output.copy()}

        if epoch == epochs:
            break

        delta_w = resume_update(input_spikes, target_spikes, output)
        weights = np.clip(weights + delta_w, 0.0, 1.2)

    return weights, initial_weights, snapshots


def main():
    input_spikes = generate_input_spikes(seed=13)
    target_spikes = generate_target_spikes()
    final_weights, initial_weights, snapshots = train_resume(input_spikes, target_spikes)
    final_membrane, final_output = simulate_output_neuron(input_spikes, final_weights)
    metrics = nearest_spike_distance(target_spikes, final_output)

    print("target spike times:", np.where(target_spikes > 0)[0].tolist())
    print("initial output spike times:", np.where(snapshots[0]["output"] > 0)[0].tolist())
    print("final output spike times:", np.where(final_output > 0)[0].tolist())
    print("missed target spikes:", metrics["missed"])
    print("extra output spikes:", metrics["extra"])
    print("mean target-to-output distance:", metrics["mean_distance"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    events = [np.where(input_spikes[i] > 0)[0] for i in range(N_INPUTS)]
    colors = []
    for i in range(N_INPUTS):
        if i in TARGET_INPUTS:
            colors.append("#2563eb")
        elif i in NON_TARGET_INPUTS:
            colors.append("#f97316")
        else:
            colors.append("#666666")
    axes[0, 0].eventplot(events, lineoffsets=np.arange(N_INPUTS), linelengths=0.72, colors=colors)
    axes[0, 0].set_title("输入脉冲序列图：目标脉冲前输入与非目标时刻输入")
    axes[0, 0].set_xlim(0, T_MAX)
    axes[0, 0].set_xlabel("时间 (ms)")
    axes[0, 0].set_ylabel("输入神经元")
    axes[0, 0].grid(axis="x", alpha=0.18)
    axes[0, 0].text(0.01, 0.96, "蓝：目标脉冲前输入   橙：非目标时刻输入   灰：其他输入", transform=axes[0, 0].transAxes, va="top", fontsize=9)

    before = snapshots[0]
    after = snapshots[max(snapshots.keys())]
    rows = [np.where(target_spikes > 0)[0], np.where(before["output"] > 0)[0], np.where(after["output"] > 0)[0]]
    axes[0, 1].eventplot(rows, lineoffsets=[2, 1, 0], linelengths=0.65, colors=["#c1121f", "#94a3b8", "#2563eb"])
    axes[0, 1].set_yticks([2, 1, 0])
    axes[0, 1].set_yticklabels(["目标输出", "训练前实际输出", "训练后实际输出"])
    axes[0, 1].set_xlim(0, T_MAX)
    axes[0, 1].set_title("目标脉冲与实际输出脉冲对比")
    axes[0, 1].set_xlabel("时间 (ms)")
    axes[0, 1].grid(axis="x", alpha=0.2)

    axes[1, 0].plot(before["membrane"], color="#94a3b8", linewidth=2.0, label="训练前膜电位")
    axes[1, 0].plot(after["membrane"], color="#2563eb", linewidth=2.2, label="训练后膜电位")
    axes[1, 0].axhline(THRESHOLD, color="#333333", linestyle="--", linewidth=1.2, label="阈值")
    for target_t in np.where(target_spikes > 0)[0]:
        axes[1, 0].axvline(target_t, color="#c1121f", linestyle=":", linewidth=1.2)
    axes[1, 0].set_xlim(0, T_MAX)
    axes[1, 0].set_title("训练前后膜电位曲线")
    axes[1, 0].set_xlabel("时间 (ms)")
    axes[1, 0].set_ylabel("膜电位")
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].legend(fontsize=9)

    x = np.arange(N_INPUTS)
    axes[1, 1].bar(x - 0.18, initial_weights, width=0.36, color="#94a3b8", label="训练前")
    axes[1, 1].bar(x + 0.18, final_weights, width=0.36, color="#15803d", label="训练后")
    axes[1, 1].axvspan(TARGET_INPUTS[0] - 0.5, TARGET_INPUTS[-1] + 0.5, color="#dbeafe", alpha=0.65, label="目标脉冲前输入")
    axes[1, 1].axvspan(NON_TARGET_INPUTS[0] - 0.5, NON_TARGET_INPUTS[-1] + 0.5, color="#ffedd5", alpha=0.7, label="非目标时刻输入")
    axes[1, 1].set_title("训练前后的突触权重")
    axes[1, 1].set_xlabel("输入神经元")
    axes[1, 1].set_ylabel("突触权重")
    axes[1, 1].grid(axis="y", alpha=0.2)
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("ReSuMe 目标脉冲序列学习", fontsize=18, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs("outputs", exist_ok=True)
    fig.savefig("outputs/exp9_2_resume_result.png", dpi=180)
    plt.show()


if __name__ == "__main__":
    main()
