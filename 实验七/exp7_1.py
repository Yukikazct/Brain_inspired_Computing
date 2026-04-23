import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


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


# 将权重变化量转成便于观察的方向标记
def delta_to_sign(delta, eps=1e-10):
    if delta > eps:
        return 1
    if delta < -eps:
        return -1
    return 0


# 标准 Hebb 学习：突触前和突触后共同活跃时增强
def standard_hebb_delta(v_i, v_j, eta):
    return eta * v_i * v_j


# 基于衰减的 Hebb 学习：共同活动增强，无刺激时权重衰减
def decay_hebb_delta(w, v_i, v_j, gamma2, gamma0):
    return gamma2 * (1 - w) * v_i * v_j - gamma0 * w


# 门控 Hebb 学习：比较突触前门控和突触后门控
def gated_hebb_delta(v_i, v_j, eta, v_theta, gate_type):
    if gate_type == "pre":
        return eta * (v_i - v_theta) * v_j
    elif gate_type == "post":
        return eta * v_i * (v_j - v_theta)
    else:
        raise ValueError("gate_type必须是pre或post")


# 协变规则：关注活动相对平均活动水平的共同变化
def covariance_hebb_delta(v_i, v_j, v_i_bar, v_j_bar, eta):
    return eta * (v_i - v_i_bar) * (v_j - v_j_bar)


# 标准 Hebb 的有界更新量：按 PPT 中 c2corr(w_ij) 的软/硬边界形式实现
def bounded_hebb_delta(w, v_i, v_j, bound_type, eta0=0.06, gamma2=0.06, w_max=1.0):
    if bound_type == "none":
        return eta0 * v_i * v_j
    elif bound_type == "hard":
        # 阶跃函数：w < w_max 时为1，否则为0
        H = 1 if w < w_max else 0
        return eta0 * H * v_i * v_j
    elif bound_type == "soft":
        return gamma2 * (w_max - w) * v_i * v_j
    else:
        raise ValueError("bound_type必须是none/hard/soft")


# BCM 规则：突触后活动超过阈值时增强，低于阈值时减弱
def bcm_delta(v_i, v_j, theta, eta):
    return eta * v_i * (v_i - theta) * v_j


# 任务一：生成不同 Hebb 变体的更新方向表
def build_hebb_rule_table(eta, gamma2, gamma0, v_theta):
    w0 = 0.5
    v_i_bar = 0.5
    v_j_bar = 0.5

    activity_cases = [
        ("后ON\n前ON", 1.0, 1.0),
        ("后ON\n前OFF", 1.0, 0.0),
        ("后OFF\n前ON", 0.0, 1.0),
        ("后OFF\n前OFF", 0.0, 0.0),
    ]

    rule_names = [
        "标准Hebb",
        "衰减Hebb",
        "突触前门控",
        "突触后门控",
        "协变规则",
    ]

    table = np.zeros((len(rule_names), len(activity_cases)), dtype=int)

    for col, (_, v_i, v_j) in enumerate(activity_cases):
        deltas = [
            standard_hebb_delta(v_i, v_j, eta),
            decay_hebb_delta(w0, v_i, v_j, gamma2, gamma0),
            gated_hebb_delta(v_i, v_j, eta, v_theta, gate_type="pre"),
            gated_hebb_delta(v_i, v_j, eta, v_theta, gate_type="post"),
            covariance_hebb_delta(v_i, v_j, v_i_bar, v_j_bar, eta),
        ]
        table[:, col] = [delta_to_sign(delta) for delta in deltas]

    return table, rule_names, [case[0] for case in activity_cases]


# 任务二：比较无边界、硬边界、软边界下的权重演化
def simulate_boundary_effect(bound_type, steps=60, eta0=0.06, gamma2=0.06, w0=0.15, w_max=1.0):
    weights = [w0]
    w = w0
    v_i = 1.0
    v_j = 1.0

    for _ in range(steps):
        delta = bounded_hebb_delta(w, v_i, v_j, bound_type=bound_type, eta0=eta0, gamma2=gamma2, w_max=w_max)
        w = w + delta
        if bound_type in {"hard", "soft"}:
            w = min(w, w_max)
        weights.append(w)

    return np.array(weights)


# 任务三：比较固定阈值和滑动阈值下的 BCM 特异化过程
def simulate_bcm_specialization(use_sliding_threshold, epochs=200, eta=0.08, theta_fixed=0.10, alpha_theta=1.0, avg_decay=0.03):
    # Group 1 和 Group 2 表示两组不同的突触前输入模式。
    # Group 1 输入强度更高，Group 2 输入强度较低，二者交替出现。
    input_strengths = np.array([1.0, 0.55], dtype=float)
    weights = np.array([0.35, 0.35], dtype=float)
    avg_output = 0.10

    history = [weights.copy()]
    theta_history = []
    output_history = []

    for epoch in range(epochs):
        active_group = 0 if (epoch // 5) % 2 == 0 else 1
        pre_activity = np.zeros(2, dtype=float)
        pre_activity[active_group] = input_strengths[active_group]
        post_activity = float(np.dot(weights, pre_activity))

        avg_output = (1.0 - avg_decay) * avg_output + avg_decay * post_activity
        if use_sliding_threshold:
            theta = alpha_theta * avg_output
        else:
            theta = theta_fixed

        for j in range(len(weights)):
            weights[j] += bcm_delta(post_activity, pre_activity[j], theta, eta)
        weights = np.clip(weights, 0.0, 1.5)
        history.append(weights.copy())
        theta_history.append(theta)
        output_history.append(post_activity)

    return {
        "weights": np.array(history),
        "theta": np.array(theta_history),
        "output": np.array(output_history),
    }


# 绘制任务一：更新方向表
def draw_rule_table(ax, table, rule_names, activity_labels):
    colors = {
        1: "#2f7d4f",
        0: "#fff7b2",
        -1: "#b91c3d",
    }
    text_map = {-1: "减弱", 0: "不变", 1: "增强"}

    n_rows, n_cols = table.shape
    ax.set_xlim(-1.7, n_cols)
    ax.set_ylim(0, n_rows + 1.15)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title("Hebb 基础变体更新方向（BCM 见下方动态实验）", fontsize=17, pad=14)

    ax.text(-0.15, 0.55, "规则", ha="right", va="center", fontsize=12, weight="bold")
    for j, label in enumerate(activity_labels):
        ax.text(j + 0.5, 0.55, label, ha="center", va="center", fontsize=11, weight="bold")

    for i in range(table.shape[0]):
        y = i + 1
        ax.text(-0.15, y + 0.5, rule_names[i], ha="right", va="center", fontsize=12)
        for j in range(table.shape[1]):
            value = int(table[i, j])
            rect = Rectangle((j, y), 1, 1, facecolor=colors[value], edgecolor="white", linewidth=1.5)
            ax.add_patch(rect)
            text_color = "white" if value in {-1, 1} else "#222222"
            ax.text(j + 0.5, y + 0.5, text_map[value], ha="center", va="center", color=text_color, fontsize=12, weight="bold")


# 绘制任务二：边界约束曲线
def draw_boundary_panel(ax, eta0, gamma2, w_max):
    steps = 60
    time = np.arange(steps + 1)

    none_curve = simulate_boundary_effect("none", steps=steps, eta0=eta0, gamma2=gamma2, w_max=w_max)
    hard_curve = simulate_boundary_effect("hard", steps=steps, eta0=eta0, gamma2=gamma2, w_max=w_max)
    soft_curve = simulate_boundary_effect("soft", steps=steps, eta0=eta0, gamma2=gamma2, w_max=w_max)

    ax.plot(time, none_curve, label="无边界", linewidth=2.0, color="#555555")
    ax.plot(time, hard_curve, label="硬边界", linewidth=2.0, color="#c1121f")
    ax.plot(time, soft_curve, label="软边界", linewidth=2.0, color="#2563eb")
    ax.axhline(w_max, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel("时间步")
    ax.set_ylabel("权重值")
    ax.set_title("有界性：无边界、硬边界与软边界")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# 绘制任务三：BCM 固定阈值与滑动阈值对比
def draw_bcm_panel(ax, eta, theta_fixed, alpha_theta, avg_decay):
    fixed = simulate_bcm_specialization(
        use_sliding_threshold=False,
        eta=eta,
        theta_fixed=theta_fixed,
        alpha_theta=alpha_theta,
        avg_decay=avg_decay,
    )
    sliding = simulate_bcm_specialization(
        use_sliding_threshold=True,
        eta=eta,
        theta_fixed=theta_fixed,
        alpha_theta=alpha_theta,
        avg_decay=avg_decay,
    )

    time = np.arange(fixed["weights"].shape[0])
    ax.plot(time, fixed["weights"][:, 0], linestyle="--", color="#2563eb", linewidth=2.0, label="Group 1 固定阈值（对照）")
    ax.plot(time, fixed["weights"][:, 1], linestyle="--", color="#c1121f", linewidth=2.0, label="Group 2 固定阈值（对照）")
    ax.plot(time, sliding["weights"][:, 0], linestyle="-", color="#2563eb", linewidth=2.4, label="Group 1 滑动阈值（BCM）")
    ax.plot(time, sliding["weights"][:, 1], linestyle="-", color="#c1121f", linewidth=2.4, label="Group 2 滑动阈值（BCM）")

    ax.set_xlabel("训练轮次")
    ax.set_ylabel("平均权重")
    ax.set_title("BCM 特异化：滑动阈值让神经元偏向强输入组")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    # 本实验要求修改的参数
    eta = 0.10        # 标准 Hebb、门控 Hebb 和协变规则中的学习率
    eta0 = 0.06       # 硬边界 Hebb 中未达到上界时的固定增强系数
    gamma2 = 0.12     # 衰减 Hebb 和软边界 Hebb 中控制增强速度的系数
    gamma0 = 0.04     # 衰减 Hebb 中控制无刺激衰减速度的系数
    v_theta = 0.50    # 门控 Hebb 中判断活动是否足够强的阈值
    w_max = 3.00      # 有界性实验中允许达到的最大突触权重
    theta_fixed = 0.10 # BCM 对照实验中的固定阈值
    alpha_theta = 1.00 # BCM 滑动阈值系数：theta(t)=alpha_theta * <v_i>
    avg_decay = 0.03   # 输出活动运行平均值的更新速度；越大表示阈值跟随当前输出变化越快

    table, rule_names, activity_labels = build_hebb_rule_table(
        eta=eta,
        gamma2=gamma2,
        gamma0=gamma0,
        v_theta=v_theta,
    )

    fig = plt.figure(figsize=(13, 10.5))
    fig.patch.set_facecolor("#eef2f7")
    gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 1.0, 1.2])
    plt.subplots_adjust(left=0.12, right=0.95, top=0.91, bottom=0.07, hspace=0.46)

    fig.text(0.08, 0.965, "Hebb 学习规则及其变体", fontsize=20, weight="bold", color="#333333")

    ax_table = fig.add_subplot(gs[0])
    ax_boundary = fig.add_subplot(gs[1])
    ax_bcm = fig.add_subplot(gs[2])

    draw_rule_table(ax_table, table, rule_names, activity_labels)
    draw_boundary_panel(ax_boundary, eta0=eta0, gamma2=gamma2, w_max=w_max)
    draw_bcm_panel(ax_bcm, eta=eta, theta_fixed=theta_fixed, alpha_theta=alpha_theta, avg_decay=avg_decay)

    plt.show()


if __name__ == "__main__":
    main()