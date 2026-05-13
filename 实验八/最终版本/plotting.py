"""
plotting.py
绘图与结果分析模块

功能：
    - 图1：输入脉冲栅格图 + 群体/单神经元发放率
    - 图2：膜电位与阈值（初期、中期、后期对比）
    - 图3：潜伏期随发放次数下降曲线
    - 图4：参数敏感性扫描结果（成功率 vs 参数值）

所有图表均保存为 300 dpi 的 PNG 文件。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from config import N, T_sim, T_thresh, T_pattern


def plot_fig1(all_times, all_neurons, all_types, neuron_spikes, pattern_neurons,
              save_path='fig1_input_pattern.png'):
    """
    图1：输入脉冲栅格图 + 发放率统计。

    展示前 600 ms 内随机选取的 50 个模式神经元与 50 个非模式神经元的脉冲，
    以及对应的群体发放率和单神经元平均发放率。
    """
    np.random.seed(123)
    pat_ids = np.random.choice(pattern_neurons, 50, replace=False)
    non_ids_all = np.array([i for i in range(N) if i not in pattern_neurons])
    non_ids = np.random.choice(non_ids_all, 50, replace=False)
    display_neurons = np.sort(np.concatenate([pat_ids, non_ids]))
    disp_map = {nid: i for i, nid in enumerate(display_neurons)}
    is_pat = np.isin(display_neurons, pattern_neurons)

    t_max_ms = 600
    mask = np.isin(all_neurons, display_neurons) & (all_times < t_max_ms)
    t_show = all_times[mask]
    n_show = np.array([disp_map[n] for n in all_neurons[mask]], dtype=np.int32)
    typ_show = all_types[mask]

    # 群体发放率（仅这 100 个神经元，10 ms 分箱）
    bins = np.arange(0, t_max_ms + 1, 10)
    counts_100, _ = np.histogram(t_show, bins=bins)
    rate_100 = counts_100 / (100 * 0.01)  # Hz

    # 单神经元平均发放率（整段时间）
    avg_rates = np.array([len(neuron_spikes[nid]) / (T_sim / 1000.0)
                        for nid in display_neurons])

    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(3, 2, width_ratios=[5, 1], height_ratios=[4, 1, 0],
                  hspace=0.05, wspace=0.08)

    # --- 栅格图 ---
    ax1 = fig.add_subplot(gs[0, 0])
    mask_bg = typ_show == 0
    ax1.scatter(t_show[mask_bg] / 1000.0, n_show[mask_bg] + 1,
                s=1.5, c='blue', alpha=0.5, label='Background', zorder=1)
    mask_pt = typ_show == 1
    ax1.scatter(t_show[mask_pt] / 1000.0, n_show[mask_pt] + 1,
                s=2.0, c='red', alpha=0.8, label='Pattern', zorder=2)
    ax1.set_ylabel('# afferent', fontsize=11)
    ax1.set_xlim(0, 0.6)
    ax1.set_ylim(0.5, 100.5)
    ax1.set_yticks([1, 100])
    ax1.set_xticks([])
    ax1.legend(loc='upper right', fontsize=8, frameon=False)

    # --- 群体平均发放率 ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(bins[:-1] / 1000.0, rate_100, width=0.01, color='blue',
            edgecolor='black', linewidth=0.3)
    ax2.set_xlabel('t (s)', fontsize=11)
    ax2.set_ylabel('Firing rate (Hz)', fontsize=11)
    ax2.set_xlim(0, 0.6)
    ax2.set_ylim(0, 100)
    ax2.set_xticks(np.arange(0, 0.7, 0.1))
    ax2.set_yticks([0, 50, 100])

    # --- 单神经元平均发放率 ---
    ax3 = fig.add_subplot(gs[0:2, 1])
    colors = ['red' if is_pat[i] else 'blue' for i in range(100)]
    ax3.barh(np.arange(100) + 1, avg_rates, height=1, color=colors,
             edgecolor='black', linewidth=0.3)
    ax3.set_xlabel('Firing rate (Hz)', fontsize=11)
    ax3.set_xlim(0, 100)
    ax3.set_ylim(0.5, 100.5)
    ax3.set_yticks([])
    ax3.set_xticks([0, 50, 100])

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[保存] {save_path}")


def plot_fig2(mem_recs, pattern_intervals, save_path='fig2_membrane_potential.png'):
    """
    图2：膜电位与阈值（初期、中期、后期）。

    展示三个典型时段内输出神经元的膜电位轨迹，
    并用灰色阴影标注模式出现区间。
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    labels = ['a', 'b', 'c']
    for ax, (t_arr, p_arr), lab in zip(axes, mem_recs, labels):
        if len(t_arr) == 0:
            continue
        r0, r1 = t_arr[0], t_arr[-1]
        ax.plot(t_arr, p_arr, 'b-', linewidth=0.9, label='potential')
        ax.axhline(T_thresh, c='red', ls='--', linewidth=1.2, label='threshold')
        ax.axhline(0, c='black', ls=':', linewidth=0.8, label='resting pot.')

        for s, e in pattern_intervals:
            s_s, e_s = s / 1000.0, e / 1000.0
            if e_s >= r0 and s_s <= r1:
                ax.axvspan(max(s_s, r0), min(e_s, r1), color='gray', alpha=0.3)

        ax.set_xlim(r0, r1)
        ax.set_ylim(-200, 1200)
        ax.set_ylabel('Potential (a. u.)', fontsize=11)
        ax.text(0.02, 0.88, lab, transform=ax.transAxes, fontsize=14, fontweight='bold')
        if ax == axes[0]:
            ax.legend(loc='upper right', fontsize=8)
        if ax == axes[-1]:
            ax.set_xlabel('t (s)', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[保存] {save_path}")


def plot_fig3(latencies, save_path='fig3_latency.png'):
    """
    图3：潜伏期随发放次数下降。

    展示输出神经元每次发放时相对于最近模式起始的潜伏期，
    反映 STDP 学习过程中响应逐渐提前的趋势。
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(np.arange(len(latencies)), latencies, s=2, c='blue', alpha=0.6)
    ax.set_xlabel('# discharges', fontsize=11)
    ax.set_ylabel('Postsynaptic spike latency (ms)', fontsize=11)
    ax.set_xlim(0, max(3000, len(latencies)))
    ax.set_ylim(0, 50)
    ax.set_xticks(np.arange(0, max(3000, len(latencies)) + 1, 500))
    ax.set_yticks(np.arange(0, 51, 5))
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[保存] {save_path}")


def plot_fig4_two_params(name1, vals1, succ1, name2, vals2, succ2,
                         save_path='fig4_sensitivity.png'):
    """
    图4：两个参数的敏感性扫描结果。

    分别绘制不同参数值下的成功率（百分比），
    用于分析网络对模式频率和初始权重的鲁棒性。
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, name, vals, succ, tag in [
        (ax1, name1, vals1, succ1, 'a'),
        (ax2, name2, vals2, succ2, 'b')
    ]:
        ax.plot(vals, np.array(succ) * 100, 'o-', c='black',
                markerfacecolor='white', markeredgecolor='black',
                markersize=7, linewidth=1.2)
        ax.set_xlabel(name, fontsize=11)
        ax.set_ylabel('% of success', fontsize=11)
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 50, 100])
        ax.text(0.05, 0.9, tag, transform=ax.transAxes, fontsize=14, fontweight='bold')
        ax.grid(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[保存] {save_path}")
