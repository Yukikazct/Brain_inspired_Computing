import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time

# =============================================================================
# 全局参数（与实验文档/论文一致）
# =============================================================================
N = 2000
T_sim = 450 * 1000          # 主仿真 450 s
dt = 1
T_pattern = 50

# 输入脉冲参数
r_bg = 52.0
r_spont = 10.0
pattern_freq = 0.25
N_pattern_neurons = N // 2
jitter_std = 1.0

# SRM 神经元参数
tau_m = 10.0
tau_s = 2.5
T_thresh = 500
K1 = 2.0
K2 = 4.0
refractory = 1
w_init = 0.475
w_min, w_max = 0.0, 1.0

# STDP 参数
a_plus = 0.03125
a_minus = 0.85 * a_plus
tau_plus = 16.8
tau_minus = 33.7
LTP_WINDOW = 7 * tau_plus      # ~117.6 ms
LTD_WINDOW = 7 * tau_minus     # ~235.9 ms

# EPSP 核归一化
t_max_epsp = tau_m * tau_s / (tau_m - tau_s) * np.log(tau_m / tau_s)
max_epsp = np.exp(-t_max_epsp / tau_m) - np.exp(-t_max_epsp / tau_s)
K_epsp = 1.0 / max_epsp
decay_m = np.exp(-dt / tau_m)
decay_s = np.exp(-dt / tau_s)

np.random.seed(42)


# =============================================================================
# 输入脉冲生成
# =============================================================================
def generate_input_spikes(T_steps=T_sim, freq=pattern_freq, jitter=jitter_std,
                          n_pattern=N_pattern_neurons):
    T_sec = T_steps / 1000.0
    bg_spikes = []
    for j in range(N):
        n_spikes = np.random.poisson(r_bg * T_sec)
        n_spikes = min(n_spikes, T_steps - 1)
        times = np.sort(np.random.choice(T_steps, size=n_spikes, replace=False))
        bg_spikes.append(times.astype(np.int32))

    pattern_neurons = np.arange(n_pattern)

    # 截取模板
    t0_template = np.random.randint(0, T_steps - T_pattern)
    template = {}
    for j in pattern_neurons:
        times = bg_spikes[j]
        mask = (times >= t0_template) & (times < t0_template + T_pattern)
        template[j] = (times[mask] - t0_template).astype(np.float64)

    # 模式插入位置（避免相邻）
    total_blocks = T_steps // T_pattern
    n_patterns = max(1, int(total_blocks * freq))
    order = np.random.permutation(total_blocks)
    selected_blocks = []
    for b in order:
        if all(abs(b - s) >= 2 for s in selected_blocks):
            selected_blocks.append(b)
        if len(selected_blocks) == n_patterns:
            break
    pattern_starts = sorted([b * T_pattern for b in selected_blocks])
    pattern_intervals = [(s, s + T_pattern) for s in pattern_starts]

    neuron_spikes = []
    neuron_types = []

    for j in range(N):
        bg = bg_spikes[j].astype(np.float64)
        if j in pattern_neurons:
            valid = np.ones(len(bg), dtype=bool)
            for t_start in pattern_starts:
                valid &= ~((bg >= t_start) & (bg < t_start + T_pattern))
            final_bg = bg[valid]

            pat_list = []
            for t_start in pattern_starts:
                rel = template[j].copy()
                noisy_rel = rel + np.random.randn(len(rel)) * jitter
                noisy_rel = np.round(noisy_rel).astype(np.int32)
                abs_t = t_start + noisy_rel
                abs_t = abs_t[(abs_t >= 0) & (abs_t < T_steps)]
                pat_list.append(abs_t)
            pattern_arr = np.concatenate(pat_list) if pat_list else np.array([], dtype=np.int32)
        else:
            final_bg, pattern_arr = bg, np.array([], dtype=np.int32)

        # 自发活动
        n_spont = np.random.poisson(r_spont * T_sec)
        n_spont = min(n_spont, T_steps - 1)
        spont = np.sort(np.random.choice(T_steps, size=n_spont, replace=False)).astype(np.int32)

        # 合并 → 确保全部为 int32
        all_t = np.concatenate([final_bg.astype(np.int32), pattern_arr, spont])
        all_typ = np.concatenate([
            np.zeros(len(final_bg), dtype=np.int8),
            np.ones(len(pattern_arr), dtype=np.int8),
            np.zeros(len(spont), dtype=np.int8)
        ])
        order_j = np.argsort(all_t)
        neuron_spikes.append(all_t[order_j])
        neuron_types.append(all_typ[order_j])

    # 构建全局时间索引（用于快速按时间步检索）
    counts = np.zeros(T_steps, dtype=np.int64)
    for spk in neuron_spikes:
        if len(spk):
            np.add.at(counts, spk, 1)

    time_start = np.zeros(T_steps + 1, dtype=np.int64)
    time_start[1:] = np.cumsum(counts)

    total_spikes = time_start[-1]
    all_times = np.empty(total_spikes, dtype=np.int32)
    all_neurons = np.empty(total_spikes, dtype=np.int16)
    all_types = np.empty(total_spikes, dtype=np.int8)

    pos = time_start[:-1].copy()
    for j, (spk, typ) in enumerate(zip(neuron_spikes, neuron_types)):
        for k, t in enumerate(spk):
            idx = pos[t]
            all_times[idx] = t
            all_neurons[idx] = j
            all_types[idx] = typ[k]
            pos[t] += 1

    return neuron_spikes, neuron_types, all_times, all_neurons, all_types, time_start, pattern_intervals


# =============================================================================
# SRM + STDP 仿真（优化版）
# =============================================================================
def simulate(neuron_spikes, all_times, all_neurons, all_types, time_start,
             pattern_intervals, record_windows, w_init_val=w_init,
             T_steps=T_sim):
    w = np.full(N, w_init_val, dtype=np.float64)
    u, v = 0.0, 0.0
    last_spike = -1
    refr_until = -1

    output_spikes = []
    latencies = []
    membrane_records = [[] for _ in record_windows]

    spike_ptr = np.zeros(N, dtype=np.int32)
    spike_list = neuron_spikes

    pattern_starts_arr = np.array([s for s, _ in pattern_intervals], dtype=np.int32)

    # 预分配 STDP 缓冲区
    tj_pre = np.empty(N, dtype=np.int32)
    tj_post = np.empty(N, dtype=np.int32)
    has_pre = np.empty(N, dtype=bool)
    has_post = np.empty(N, dtype=bool)

    for t in range(T_steps):
        idx0, idx1 = time_start[t], time_start[t + 1]

        if idx0 < idx1:
            js = all_neurons[idx0:idx1]
            np.add.at(spike_ptr, js, 1)
            s = np.sum(w[js])
        else:
            s = 0.0

        # EPSP 差分更新
        u = decay_m * u + s
        v = decay_s * v + s
        epsp = K_epsp * (u - v)

        # 发放后电位
        eta = 0.0
        if last_spike >= 0:
            dt_eta = t - last_spike
            eta = T_thresh * (
                K1 * np.exp(-dt_eta / tau_m)
                - K2 * (np.exp(-dt_eta / tau_m) - np.exp(-dt_eta / tau_s))
            )

        p = eta + epsp

        # 记录膜电位
        for ri, (t1, t2) in enumerate(record_windows):
            if t1 <= t < t2:
                membrane_records[ri].append((t, p))

        # 发放判定
        if t >= refr_until and p >= T_thresh:
            output_spikes.append(t)

            # 潜伏期（相对最近模式起点）
            latency = 0.0
            ip = np.searchsorted(pattern_starts_arr, t)
            if ip > 0:
                s_p = pattern_starts_arr[ip - 1]
                if t < s_p + T_pattern:
                    latency = t - s_p
            latencies.append(latency)

            # 重置
            last_spike = t
            refr_until = t + refractory
            u, v = 0.0, 0.0

            # ====== STDP（向量化 + 窗口限制）======
            for j in range(N):
                ptr = spike_ptr[j]
                has_pre[j] = ptr > 0
                has_post[j] = ptr < len(spike_list[j])
                if has_pre[j]:
                    tj_pre[j] = spike_list[j][ptr - 1]
                if has_post[j]:
                    tj_post[j] = spike_list[j][ptr]

            # LTP
            dt_pre = t - tj_pre[has_pre]
            mask_ltp = dt_pre <= LTP_WINDOW
            idx_ltp = np.where(has_pre)[0][mask_ltp]
            if len(idx_ltp):
                w[idx_ltp] += a_plus * np.exp(-dt_pre[mask_ltp] / tau_plus)

            # LTD
            dt_post = tj_post[has_post] - t
            mask_ltd = (dt_post > 0) & (dt_post <= LTD_WINDOW)
            idx_ltd = np.where(has_post)[0][mask_ltd]
            if len(idx_ltd):
                w[idx_ltd] -= a_minus * np.exp(-dt_post[mask_ltd] / tau_minus)

            # 裁剪
            w = np.clip(w, w_min, w_max)

    rec_arrays = [
        (np.array([x[0] for x in rec]) / 1000.0,
         np.array([x[1] for x in rec]))
        for rec in membrane_records
    ]
    return np.array(output_spikes), np.array(latencies), rec_arrays, w


# =============================================================================
# 评估函数（论文成功率标准）
# =============================================================================
def evaluate_trial(output_spikes, pattern_intervals, T_steps=T_sim, eval_last_s=150):
    t_eval_start = T_steps - eval_last_s * 1000
    patterns_eval = [(s, e) for s, e in pattern_intervals if s >= t_eval_start]
    n_pat = len(patterns_eval)
    if n_pat == 0:
        return False, 0.0, 999.0, 0

    out_eval = output_spikes[output_spikes >= t_eval_start]

    hits = 0
    hit_latencies = []
    for s, e in patterns_eval:
        mask = (out_eval >= s) & (out_eval < e)
        if np.any(mask):
            hits += 1
            hit_latencies.append(out_eval[mask][0] - s)

    hit_rate = hits / n_pat
    avg_latency = np.mean(hit_latencies) if hit_latencies else 999.0

    false_alarms = 0
    for t in out_eval:
        in_pat = any(s <= t < e for s, e in patterns_eval)
        if not in_pat:
            false_alarms += 1

    success = (avg_latency < 10) and (hit_rate >= 0.98) and (false_alarms == 0)
    return success, hit_rate, avg_latency, false_alarms


# =============================================================================
# 绘图模块
# =============================================================================
def plot_fig1(all_times, all_neurons, all_types, neuron_spikes, save_path='fig1_input_pattern.png'):
    """从2000个中抽取100个（50 pattern + 50 non-pattern），匹配参考图样式。"""
    np.random.seed(123)
    pat_ids = np.random.choice(N_pattern_neurons, 50, replace=False)
    non_ids = np.random.choice(np.arange(N_pattern_neurons, N), 50, replace=False)
    display_neurons = np.sort(np.concatenate([pat_ids, non_ids]))
    disp_map = {nid: i for i, nid in enumerate(display_neurons)}
    is_pat = (display_neurons < N_pattern_neurons)

    t_max_ms = 600
    mask = np.isin(all_neurons, display_neurons) & (all_times < t_max_ms)
    t_show = all_times[mask]
    n_show = np.array([disp_map[n] for n in all_neurons[mask]], dtype=np.int32)
    typ_show = all_types[mask]

    # 群体发放率（仅这100个神经元）
    bins = np.arange(0, t_max_ms + 1, 10)
    counts_100, _ = np.histogram(t_show, bins=bins)
    rate_100 = counts_100 / (100 * 0.01)  # Hz

    # 单神经元平均发放率
    avg_rates = np.array([len(neuron_spikes[nid]) / (T_sim / 1000.0) for nid in display_neurons])

    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(3, 2, width_ratios=[5, 1], height_ratios=[4, 1, 0], hspace=0.05, wspace=0.08)

    # 上方：栅格图
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

    # 下方：群体平均发放率
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(bins[:-1] / 1000.0, rate_100, width=0.01, color='blue',
            edgecolor='black', linewidth=0.3)
    ax2.set_xlabel('t (s)', fontsize=11)
    ax2.set_ylabel('Firing rate (Hz)', fontsize=11)
    ax2.set_xlim(0, 0.6)
    ax2.set_ylim(0, 100)
    ax2.set_xticks(np.arange(0, 0.7, 0.1))
    ax2.set_yticks([0, 50, 100])

    # 右侧：单神经元平均发放率
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
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    labels = ['a', 'b', 'c']
    ranges = [(0, 1), (13.3, 14.2), (449, 450)]

    for ax, (t_arr, p_arr), lab, (r0, r1) in zip(axes, mem_recs, labels, ranges):
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


def plot_fig4(name1, vals1, succ1, name2, vals2, succ2, save_path='fig4_sensitivity.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, name, vals, succ, tag in [
        (ax1, name1, vals1, succ1, 'a'),
        (ax2, name2, vals2, succ2, 'b')
    ]:
        ax.plot(vals, np.array(succ) * 100, 'o-', c='black',
                markerfacecolor='white', markeredgecolor='black', markersize=7, linewidth=1.2)
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


# =============================================================================
# 参数扫描（真实仿真统计成功率）
# =============================================================================
def scan_parameter(param_name, param_values, n_trials=5, T_steps=100 * 1000):
    results = []
    print(f"\n>>> 扫描: {param_name} (每点 {n_trials} 次, 单次 {T_steps // 1000}s)")
    for val in param_values:
        n_succ = 0
        for trial in range(n_trials):
            # 准备参数
            gen_kwargs = {}
            w_init_val = w_init
            if param_name == 'Pattern frequency':
                gen_kwargs['freq'] = val
            elif param_name == 'Jitter (ms)':
                gen_kwargs['jitter'] = val
            elif param_name == 'Prop. of aff. in pattern':
                gen_kwargs['n_pattern'] = int(N * val)
            elif param_name == 'Initial weight':
                w_init_val = val
            else:
                raise ValueError(f"未知参数: {param_name}")

            neuron_spikes, neuron_types, all_times, all_neurons, all_types, time_start, pattern_intervals = \
                generate_input_spikes(T_steps=T_steps, **gen_kwargs)

            out_spikes, _, _, _ = simulate(
                neuron_spikes, all_times, all_neurons, all_types, time_start,
                pattern_intervals, record_windows=[],
                w_init_val=w_init_val, T_steps=T_steps
            )
            success, _, _, _ = evaluate_trial(out_spikes, pattern_intervals, T_steps=T_steps, eval_last_s=50)
            if success:
                n_succ += 1
        rate = n_succ / n_trials
        results.append(rate)
        print(f"    {param_name}={val:.3f} => 成功率 {rate * 100:.0f}%")
    return results


# =============================================================================
# 主程序
# =============================================================================
if __name__ == "__main__":
    # -------------------- 阶段1：主仿真 450s（图1/2/3） --------------------
    print("=" * 60)
    print("阶段1：主仿真 (450s) —— 生成图1、图2、图3")
    print("=" * 60)

    t0 = time.time()
    neuron_spikes, neuron_types, all_times, all_neurons, all_types, time_start, pattern_intervals = \
        generate_input_spikes()
    print(f"[1/4] 输入脉冲生成完成，耗时 {time.time() - t0:.2f} s")

    record_windows = [
        (0, 1000),
        (13300, 14200),
        (449000, 450000)
    ]

    t0 = time.time()
    out_spikes, latencies, mem_recs, final_w = simulate(
        neuron_spikes, all_times, all_neurons, all_types, time_start,
        pattern_intervals, record_windows
    )
    print(f"[2/4] 仿真完成，输出脉冲 {len(out_spikes)} 个，耗时 {time.time() - t0:.2f} s")

    success, hit_rate, avg_lat, fa = evaluate_trial(out_spikes, pattern_intervals)
    print(f"[评估] 成功={success} | 命中率={hit_rate * 100:.1f}% | "
          f"平均潜伏期={avg_lat:.2f}ms | 误报={fa}")

    t0 = time.time()
    plot_fig1(all_times, all_neurons, all_types, neuron_spikes)
    plot_fig2(mem_recs, pattern_intervals)
    plot_fig3(latencies)
    print(f"[3/4] 图1/2/3 绘制完成，耗时 {time.time() - t0:.2f} s")

    # -------------------- 阶段2：参数扫描（图4） --------------------
    print("\n" + "=" * 60)
    print("阶段2：参数敏感性扫描 —— 生成图4")
    print("提示：若电脑较慢，请下调 T_SCAN 和 N_TRIALS")
    print("=" * 60)

    # 可调：扫描时长与次数（越大越准、越慢）
    T_SCAN = 100 * 1000      # 每次仿真 100 秒（论文用 450s，扫描时可缩短）
    N_TRIALS = 3           # 每个参数值重复 5 次

    # 参数1：模式出现频率
    freq_vals = [0.20, 0.25, 0.30, 0.40, 0.50]

    t0 = time.time()
    succ_freq = scan_parameter('Pattern frequency', freq_vals, n_trials=N_TRIALS, T_steps=T_SCAN)
    print(f"频率扫描耗时 {time.time() - t0:.1f} s")

    # 参数2：初始权重
    weight_vals = [0.40, 0.43, 0.46, 0.48, 0.50]
    t0 = time.time()
    succ_weight = scan_parameter('Initial weight', weight_vals, n_trials=N_TRIALS, T_steps=T_SCAN)
    print(f"权重扫描耗时 {time.time() - t0:.1f} s")

    plot_fig4('Pattern frequency', freq_vals, succ_freq,
              'Initial weight', weight_vals, succ_weight)

    print("\n[完成] 所有图表已生成！")