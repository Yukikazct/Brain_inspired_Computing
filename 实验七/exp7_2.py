import numpy as np
import matplotlib.pyplot as plt


# 配置中文字体，避免绘图时标题或坐标轴出现乱码
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


# pair-based STDP：根据 pre 和 post 的相对时间差计算一次权重变化
def pair_based_stdp(delta_t, tau_plus=10.0, tau_minus=10.0, a_plus=0.08, a_minus=0.08):
    # delta_t = t_post - t_pre
    if delta_t > 0:
        # pre 先于 post 发放 → 权重增强（LTP）
        return a_plus * np.exp(-delta_t / tau_plus)
    elif delta_t < 0:
        # post 先于 pre 发放 → 权重减弱（LTD）
        return -a_minus * np.exp(delta_t / tau_minus)
    else:
        # 时间同步 → 权重不变
        return 0.0


# 权重依赖 STDP：根据当前权重计算增强或减弱的幅度
def weight_dependent_amplitude(weight, direction, mode, eta_plus, eta_minus, w_min=0.0, w_max=1.0):
    if mode == "classic":
        # 经典模式：无权重依赖，直接返回固定幅度
        return eta_plus if direction == "ltp" else eta_minus
    elif mode == "hard":
        # 硬边界模式：达到边界后更新幅度为0
        if direction == "ltp":
            return eta_plus if weight < w_max else 0.0
        else:
            return eta_minus if weight > w_min else 0.0
    elif mode == "soft":
        # 软边界模式：权重越接近边界，更新幅度越小
        if direction == "ltp":
            return (w_max - weight) * eta_plus
        else:
            return (weight - w_min) * eta_minus
    else:
        raise ValueError("mode必须是 classic/hard/soft")


# 使用权重依赖幅度计算一次 STDP 更新
def weight_dependent_stdp(delta_t, weight, mode, tau_plus=10.0, tau_minus=10.0, eta_plus=0.05, eta_minus=0.05):
    if delta_t > 0:
        amp = weight_dependent_amplitude(weight, "ltp", mode, eta_plus, eta_minus)
        return amp * np.exp(-delta_t / tau_plus)
    if delta_t < 0:
        amp = weight_dependent_amplitude(weight, "ltd", mode, eta_plus, eta_minus)
        return -amp * np.exp(delta_t / tau_minus)
    return 0.0


# 重复相同 pre-post 配对，观察权重演化
def simulate_repeated_pair(
    delta_t,
    mode,
    repeat_count=40,
    initial_weight=0.5,
    tau_plus=10.0,
    tau_minus=10.0,
    eta_plus=0.05,
    eta_minus=0.05,
    w_min=0.0,
    w_max=1.0,
):
    weight = float(initial_weight)
    history = [weight]

    for _ in range(repeat_count):
        delta_w = weight_dependent_stdp(
            delta_t,
            weight,
            mode=mode,
            tau_plus=tau_plus,
            tau_minus=tau_minus,
            eta_plus=eta_plus,
            eta_minus=eta_minus,
        )
        if mode == "classic":
            weight = weight + delta_w
        else:
            weight = np.clip(weight + delta_w, w_min, w_max)
        history.append(weight)

    return np.array(history)


# 绘制任务一：STDP 时间窗口
def draw_stdp_window(ax, tau_plus, tau_minus, a_plus, a_minus):
    delta_range = np.linspace(-40, 40, 401)
    window = np.array(
        [
            pair_based_stdp(
                dt,
                tau_plus=tau_plus,
                tau_minus=tau_minus,
                a_plus=a_plus,
                a_minus=a_minus,
            )
            for dt in delta_range
        ]
    )

    ax.plot(delta_range, window, color="#333333", linewidth=2.2)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"$\Delta t = t_{post} - t_{pre}$")
    ax.set_ylabel(r"$\Delta w$")
    ax.set_title("STDP 时间窗口：pre 先于 post 增强，post 先于 pre 减弱")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# 绘制任务二：权重依赖 STDP 的时间窗口形状
def draw_weight_dependent_window_panel(ax, tau_plus, tau_minus, a_plus, a_minus, w_demo, eta_plus, eta_minus, w_min=0.0, w_max=1.0):
    delta_range = np.linspace(-40, 40, 401)
    classic_window = np.array(
        [
            pair_based_stdp(dt, tau_plus=tau_plus, tau_minus=tau_minus, a_plus=a_plus, a_minus=a_minus)
            for dt in delta_range
        ]
    )
    soft_window = np.array(
        [
            weight_dependent_stdp(
                dt,
                weight=w_demo,
                mode="soft",
                tau_plus=tau_plus,
                tau_minus=tau_minus,
                eta_plus=eta_plus,
                eta_minus=eta_minus,
            )
            for dt in delta_range
        ]
    )

    ax.plot(delta_range, classic_window, label="经典 STDP", color="#333333", linewidth=2.1)
    ax.plot(delta_range, soft_window, label=f"软边界 STDP（当前权重 w_ij={w_demo:.1f}）", color="#2563eb", linewidth=2.1)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"$\Delta t = t_{post} - t_{pre}$")
    ax.set_ylabel(r"$\Delta w$")
    ax.set_title(r"权重依赖 STDP 时间窗口（本实验约定 $\Delta t=t_{post}-t_{pre}$）")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    # 本实验要求修改的参数
    tau_plus = 10.0       # LTP 时间常数；越大，pre 先于 post 的增强窗口越宽
    tau_minus = 10.0      # LTD 时间常数；越大，post 先于 pre 的减弱窗口越宽
    a_plus = 0.08         # pair-based STDP 中 LTP 的最大权重变化幅度
    a_minus = 0.08        # pair-based STDP 中 LTD 的最大权重变化幅度
    w_demo = 0.80         # 权重依赖 STDP 时间窗口中用于演示的当前权重
    eta_plus = 0.08       # 权重依赖 STDP 中 LTP 的基础更新幅度
    eta_minus = 0.08      # 权重依赖 STDP 中 LTD 的基础更新幅度
    fig = plt.figure(figsize=(13, 8.8))
    fig.patch.set_facecolor("#eef2f7")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05])
    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08, hspace=0.48)

    fig.text(0.08, 0.97, "STDP 学习规则及其变体", fontsize=20, weight="bold", color="#333333")

    ax_window = fig.add_subplot(gs[0])
    ax_weight = fig.add_subplot(gs[1])

    draw_stdp_window(ax_window, tau_plus=tau_plus, tau_minus=tau_minus, a_plus=a_plus, a_minus=a_minus)
    draw_weight_dependent_window_panel(
        ax_weight,
        tau_plus=tau_plus,
        tau_minus=tau_minus,
        a_plus=a_plus,
        a_minus=a_minus,
        w_demo=w_demo,
        eta_plus=eta_plus,
        eta_minus=eta_minus,
    )

    plt.show()


if __name__ == "__main__":
    main()