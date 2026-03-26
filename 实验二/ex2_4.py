import numpy as np
import matplotlib.pyplot as plt

# ========== 核心：macOS 中文显示配置 ==========
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ========== 脉冲神经元仿真核心函数 ==========
def simulate_neuron(input_current, threshold, decay=0.85, T=50, reset_value=0.0):
    voltage = np.zeros(T)
    spikes = np.zeros(T)
    for t in range(1, T):
        # 上一步发放则重置电位，否则继承上一步电位
        prev_v = reset_value if spikes[t - 1] == 1 else voltage[t - 1]
        # 膜电位积分与衰减
        voltage[t] = decay * prev_v + input_current
        # 阈值判断触发脉冲
        if voltage[t] >= threshold:
            spikes[t] = 1
    return voltage, spikes


def main():
    T = 50
    time = np.arange(T)


    # 情况1：相同阈值下，不同输入电流（修改后参数）
    threshold_same = 0.9  # 原1.0 → 调整为0.9，现象更明显
    current_small = 0.10  # 原0.12 → 更小，对比更突出
    current_large = 0.30  # 原0.28 → 更大，发放频率更高
    decay_base = 0.85  # 基准衰减系数

    # 情况2：相同输入电流下，不同阈值（修改后参数）
    input_current_same = 0.20  # 原0.22 → 调整为0.2，适配新阈值
    threshold_low = 0.70  # 原0.75 → 更低，易发放
    threshold_high = 1.10  # 原1.30 → 调整为1.1，避免完全不发放

    # 情况3：新增decay变化对比（回答第三个实验问题）
    decay_small = 0.70  # 小衰减系数（膜电位衰减快）
    decay_large = 0.95  # 大衰减系数（膜电位衰减慢）
    current_decay = 0.20  # 固定电流，仅变decay
    threshold_decay = 0.9  # 固定阈值

    # ===================== 计算所有实验组数据 =====================
    # 情况1：相同阈值，不同电流
    voltage_small, spikes_small = simulate_neuron(current_small, threshold_same, decay_base, T)
    voltage_large, spikes_large = simulate_neuron(current_large, threshold_same, decay_base, T)

    # 情况2：相同电流，不同阈值
    voltage_low, spikes_low = simulate_neuron(input_current_same, threshold_low, decay_base, T)
    voltage_high, spikes_high = simulate_neuron(input_current_same, threshold_high, decay_base, T)

    # 情况3：相同电流+阈值，不同decay
    voltage_decay_small, spikes_decay_small = simulate_neuron(current_decay, threshold_decay, decay_small, T)
    voltage_decay_large, spikes_decay_large = simulate_neuron(current_decay, threshold_decay, decay_large, T)


    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

    # 子图1：相同阈值下的膜电位变化（修改后参数）
    axes[0, 0].plot(time, voltage_small, label=f"输入电流={current_small}", color="tab:blue")
    axes[0, 0].plot(time, voltage_large, label=f"输入电流={current_large}", color="tab:orange")
    axes[0, 0].axhline(threshold_same, linestyle="--", color="tab:green", label=f"阈值={threshold_same}")
    axes[0, 0].set_title("相同阈值下，不同输入电流的膜电位变化", fontweight='bold')
    axes[0, 0].set_ylabel("膜电位")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    # 子图2：相同阈值下的脉冲输出
    axes[1, 0].step(time, spikes_small, where="mid", label=f"输入电流={current_small}", color="tab:blue")
    axes[1, 0].step(time, spikes_large, where="mid", label=f"输入电流={current_large}", color="tab:orange")
    axes[1, 0].set_title("相同阈值下，不同输入电流的脉冲输出", fontweight='bold')
    axes[1, 0].set_xlabel("时间步")
    axes[1, 0].set_ylabel("脉冲（1=发放，0=未发放）")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    # 子图3：相同电流下的膜电位变化（修改后参数）
    axes[0, 1].plot(time, voltage_low, label=f"阈值={threshold_low}", color="tab:red")
    axes[0, 1].plot(time, voltage_high, label=f"阈值={threshold_high}", color="tab:purple")
    axes[0, 1].axhline(threshold_low, linestyle="--", color="tab:red", alpha=0.6, label=f"低阈值={threshold_low}")
    axes[0, 1].axhline(threshold_high, linestyle="--", color="tab:purple", alpha=0.6, label=f"高阈值={threshold_high}")
    axes[0, 1].set_title("相同输入电流下，不同阈值的膜电位变化", fontweight='bold')
    axes[0, 1].set_ylabel("膜电位")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    # 子图4：不同decay的膜电位+脉冲（回答第三个问题）
    axes[1, 1].plot(time, voltage_decay_small, label=f"decay={decay_small}", color="tab:brown")
    axes[1, 1].plot(time, voltage_decay_large, label=f"decay={decay_large}", color="tab:cyan")
    axes[1, 1].axhline(threshold_decay, linestyle="--", color="gray", label=f"阈值={threshold_decay}")
    # 叠加脉冲输出（虚线）
    axes[1, 1].step(time, spikes_decay_small, where="mid", color="tab:brown", linestyle="--", alpha=0.8)
    axes[1, 1].step(time, spikes_decay_large, where="mid", color="tab:cyan", linestyle="--", alpha=0.8)
    axes[1, 1].set_title("相同电流+阈值下，不同decay的膜电位/脉冲变化", fontweight='bold')
    axes[1, 1].set_xlabel("时间步")
    axes[1, 1].set_ylabel("膜电位（实线）/ 脉冲（虚线）")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    # 全局标题+布局优化
    fig.suptitle("简化脉冲神经元实验（修改参数后）", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



if __name__ == "__main__":
    # 主实验（修改参数后）
    main()