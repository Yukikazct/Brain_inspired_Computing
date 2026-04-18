import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

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



# 计算单次pre-post配对的STDP权重变化
def pair_based_stdp(delta_t, eta, tau_plus=12.0, tau_minus=12.0, a_plus=1.0, a_minus=1.0):
    if delta_t > 0:
        delta_w = eta * a_plus * np.exp(-delta_t / tau_plus)
    elif delta_t < 0:
        delta_w = -eta * a_minus * np.exp(delta_t / tau_minus)
    else:
        delta_w = 0.0
    return delta_w


# 模拟权重演化
def simulate_weight_history(initial_weight, delta_w, repeat_count, w_min=0.0, w_max=1.0):
    weight = float(initial_weight)
    history = [weight]
    for _ in range(repeat_count):
        weight = np.clip(weight + delta_w, w_min, w_max)
        history.append(weight)
    return np.array(history)


# 绘制脉冲面板
def draw_spike_panel(ax, pre_time, post_time, total_time):
    ax.clear()
    ax.eventplot(
        [[pre_time], [post_time]],
        lineoffsets=[1, 0], linelengths=0.6, colors=["#2563eb", "#c1121f"], linewidths=2.5
    )
    ax.set_xlim(-0.5, total_time - 0.5)
    ax.set_ylim(-0.7, 1.7)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["突触前pre", "突触后post"])
    ax.set_xticks(np.arange(0, total_time + 1, 5))
    ax.set_title("pre/post脉冲发放时刻")
    ax.grid(axis="x", alpha=0.2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


# 绘制STDP窗口
def draw_window_panel(ax, current_delta_t, current_delta_w, eta):
    ax.clear()
    delta_range = np.linspace(-40, 40, 400)
    window = np.array([pair_based_stdp(dt, eta=eta) for dt in delta_range])
    ax.plot(delta_range, window, color="#333", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--")
    ax.axvline(0, color="gray", linestyle="--")
    ax.scatter(current_delta_t, current_delta_w, s=80, color="#ff7f0e", zorder=3)

    if current_delta_t > 0:
        text = "pre先于post：权重增强"
    elif current_delta_t < 0:
        text = "post先于pre：权重减弱"
    else:
        text = "同时：无变化"
    ax.text(0.02, 0.95, text, transform=ax.transAxes, fontsize=12, va="top", bbox=dict(boxstyle="round", fc="#f6f6f6"))
    ax.set_xlabel(r"$\Delta t = t_{post}-t_{pre}$")
    ax.set_ylabel(r"$\Delta w$")
    ax.set_title("STDP窗口曲线")
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)


# 绘制权重演化
def draw_weight_panel(ax, weight_history):
    ax.clear()
    ax.plot(range(len(weight_history)), weight_history, marker="o", color="#15803d", linewidth=2)
    ax.set_xlim(0, len(weight_history) - 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("配对次数")
    ax.set_ylabel("权重值")
    ax.set_title("重复配对下的权重演化")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    total_time = 60
    pre_time_init = 18
    post_time_init = 28
    eta_init = 0.08
    repeat_count_init = 20
    initial_weight = 0.5

    fig = plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor("#eef2f7")
    gs = fig.add_gridspec(3, 1, height_ratios=[0.9, 1, 1])
    plt.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.18, hspace=0.38)

    ax_spikes = fig.add_subplot(gs[0])
    ax_window = fig.add_subplot(gs[1])
    ax_weight = fig.add_subplot(gs[2])

    fig.text(0.08, 0.93, "STDP学习规则：时序因果决定突触增强或减弱", fontsize=18, weight="bold")
    stats_text = fig.text(0.67, 0.905, "", fontsize=13)

    # 初始绘制
    delta_t = post_time_init - pre_time_init
    delta_w = pair_based_stdp(delta_t, eta_init)
    history = simulate_weight_history(initial_weight, delta_w, repeat_count_init)
    draw_spike_panel(ax_spikes, pre_time_init, post_time_init, total_time)
    draw_window_panel(ax_window, delta_t, delta_w, eta_init)
    draw_weight_panel(ax_weight, history)
    stats_text.set_text(f"Δt {delta_t:>4.1f}  单次Δw {delta_w:+.4f}  最终权重 {history[-1]:.4f}")

    # 滑块
    slider_y1, slider_y2 = 0.10, 0.05
    ax_pre = fig.add_axes([0.18, slider_y1, 0.22, 0.025], fc="#f5f5f5")
    ax_post = fig.add_axes([0.68, slider_y1, 0.22, 0.025], fc="#f5f5f5")
    ax_eta = fig.add_axes([0.18, slider_y2, 0.22, 0.025], fc="#f5f5f5")
    ax_repeat = fig.add_axes([0.68, slider_y2, 0.22, 0.025], fc="#f5f5f5")

    fig.text(0.06, slider_y1, "pre时间", fontsize=14)
    fig.text(0.56, slider_y1, "post时间", fontsize=14)
    fig.text(0.06, slider_y2, "学习率(eta)", fontsize=14)
    fig.text(0.53, slider_y2, "重复配对次数", fontsize=14)


    slider_pre = Slider(ax=ax_pre, label="", valmin=0, valmax=total_time - 1, valinit=pre_time_init, valstep=1,
                        color="#2563eb")
    slider_post = Slider(ax=ax_post, label="", valmin=0, valmax=total_time - 1, valinit=post_time_init, valstep=1,
                         color="#c1121f")
    slider_eta = Slider(ax=ax_eta, label="", valmin=0.01, valmax=0.20, valinit=eta_init, valstep=0.01, color="#2563eb")
    slider_repeat = Slider(ax=ax_repeat, label="", valmin=1, valmax=50, valinit=repeat_count_init, valstep=1,
                           color="#15803d")


    def update(_):
        pre = int(slider_pre.val)
        post = int(slider_post.val)
        eta = slider_eta.val
        repeat = int(slider_repeat.val)
        dt = post - pre
        dw = pair_based_stdp(dt, eta)
        h = simulate_weight_history(initial_weight, dw, repeat)
        draw_spike_panel(ax_spikes, pre, post, total_time)
        draw_window_panel(ax_window, dt, dw, eta)
        draw_weight_panel(ax_weight, h)
        stats_text.set_text(f"Δt {dt:>4.1f}  单次Δw {dw:+.4f}  最终权重 {h[-1]:.4f}")
        fig.canvas.draw_idle()

    slider_pre.on_changed(update)
    slider_post.on_changed(update)
    slider_eta.on_changed(update)
    slider_repeat.on_changed(update)
    plt.show()


if __name__ == "__main__":
    main()