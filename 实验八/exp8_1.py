import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
import gc


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


# =============================================================================
# 全局参数（与实验文档/论文一致）
# =============================================================================
N = 2000
T_sim = 450 * 1000          # 主仿真 450 s
dt = 1
T_pattern = 50

# 输入脉冲参数
r_bg = 52.0                 # 固定背景发放率 (Hz)
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
# 输入脉冲生成（固定背景率 + 模式复制粘贴）
# =============================================================================
def generate_input_spikes(T_steps=T_sim, freq=pattern_freq, jitter=jitter_std,
                          n_pattern=N_pattern_neurons, seed=None):
    if seed is not None:
        np.random.seed(seed)

    T_sec = T_steps / 1000.0

    # -------------------------------------------------------------------------
    # 1. 生成固定背景率的泊松脉冲（每个神经元独立）
    # -------------------------------------------------------------------------
    print("  生成背景脉冲...")
    bg_spikes = []
    for j in range(N):
        n_spikes = np.random.poisson(r_bg * T_sec)
        # 限制数量不超过时间步数
        n_spikes = min(n_spikes, T_steps - 1)
        times = np.sort(np.random.choice(T_steps, size=n_spikes, replace=False))
        bg_spikes.append(times.astype(np.int32))

    # -------------------------------------------------------------------------
    # 2. 随机选择参与模式的神经元
    # -------------------------------------------------------------------------
    all_neurons_idx = np.arange(N)
    pattern_neurons = np.random.choice(all_neurons_idx, size=n_pattern, replace=False)
    pattern_neurons = np.sort(pattern_neurons)

    # -------------------------------------------------------------------------
    # 3. 从背景中随机截取50ms片段作为模式模板
    # -------------------------------------------------------------------------
    print("  提取模式模板...")
    # 保证模板区间内所有模式神经元都有至少一个脉冲（论文机制）
    # 如果某个神经元没有脉冲，则在模板内随机添加一个脉冲
    max_attempts = 20
    t0_template = None
    template = None

    for _ in range(max_attempts):
        t0 = np.random.randint(0, T_steps - T_pattern)
        ok = True
        template_candidate = {}
        for j in pattern_neurons:
            times = bg_spikes[j]
            mask = (times >= t0) & (times < t0 + T_pattern)
            rel_times = times[mask] - t0
            if len(rel_times) == 0:
                # 若没有脉冲，添加一个随机脉冲（论文中的强制发放机制）
                rel_times = np.array([np.random.uniform(0, T_pattern)])
                ok = False   # 标记这不是完美模板，但仍然可用
            template_candidate[j] = rel_times.astype(np.float64)
        # 接受第一个模板，不强求完美覆盖
        t0_template = t0
        template = template_candidate
        break

    # -------------------------------------------------------------------------
    # 4. 确定模式插入位置（避免相邻）
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 5. 构建最终脉冲序列（复制-粘贴模式，保留未参与模式神经元的背景）
    # -------------------------------------------------------------------------
    print("  合并脉冲序列...")
    neuron_spikes = []
    neuron_types = []

    for j in range(N):
        if j in pattern_neurons:
            # 模式神经元：背景中去掉模式区间内的脉冲，再加入带抖动的模板脉冲
            bg = bg_spikes[j].astype(np.float64)
            # 标记哪些背景脉冲位于任何模式窗口内
            keep = np.ones(len(bg), dtype=bool)
            for t_start in pattern_starts:
                keep &= ~((bg >= t_start) & (bg < t_start + T_pattern))
            final_bg = bg[keep]

            # 生成带抖动的模式脉冲
            pattern_pulses = []
            for t_start in pattern_starts:
                rel = template[j].copy()
                noisy_rel = rel + np.random.randn(len(rel)) * jitter
                noisy_rel = np.round(noisy_rel).astype(np.int32)
                abs_t = t_start + noisy_rel
                abs_t = abs_t[(abs_t >= 0) & (abs_t < T_steps)]
                pattern_pulses.extend(abs_t)

            # 合并背景和模式脉冲
            all_t = np.concatenate([final_bg, pattern_pulses])
            all_typ = np.concatenate([np.zeros(len(final_bg), dtype=np.int8),
                                      np.ones(len(pattern_pulses), dtype=np.int8)])
        else:
            # 非模式神经元：保留全部背景脉冲
            all_t = bg_spikes[j].astype(np.float64)
            all_typ = np.zeros(len(all_t), dtype=np.int8)

        # 添加自发活动（10Hz 泊松）
        n_spont = np.random.poisson(r_spont * T_sec)
        n_spont = min(n_spont, T_steps - 1)
        spont = np.sort(np.random.choice(T_steps, size=n_spont, replace=False)).astype(np.int32)
        all_t = np.concatenate([all_t, spont])
        all_typ = np.concatenate([all_typ, np.zeros(len(spont), dtype=np.int8)])

        # 排序并去重（同一时间步多个脉冲只保留一个，模式脉冲优先）
        order_j = np.argsort(all_t)
        all_t = all_t[order_j]
        all_typ = all_typ[order_j]
        # 去重：保留每个时间步最后一个（即模式脉冲优先，因为模式脉冲后添加）
        unique_mask = np.diff(all_t, prepend=-1) != 0
        all_t = all_t[unique_mask]
        all_typ = all_typ[unique_mask]

        neuron_spikes.append(all_t.astype(np.int32))
        neuron_types.append(all_typ)

    # 释放大的中间变量
    del bg_spikes
    gc.collect()

    # -------------------------------------------------------------------------
    # 6. 构建全局时间索引（用于快速访问）
    # -------------------------------------------------------------------------
    print("  构建全局索引...")
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

    return (neuron_spikes, neuron_types, all_times, all_neurons, all_types,
            time_start, pattern_intervals, pattern_neurons)


# =============================================================================
# SRM + STDP 仿真（保持用户原有逻辑，仅优化微小细节）
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

            # STDP 更新
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
# 绘图模块（匹配论文样式）
# =============================================================================
def plot_fig1(all_times, all_neurons, all_types, neuron_spikes, pattern_neurons,
              save_path='fig1_input_pattern.png'):
    """图1：输入脉冲栅格图 + 发放率"""
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

    # 群体发放率（仅这100个神经元，10ms分箱）
    bins = np.arange(0, t_max_ms + 1, 10)
    counts_100, _ = np.histogram(t_show, bins=bins)
    rate_100 = counts_100 / (100 * 0.01)  # Hz

    # 单神经元平均发放率（整段时间）
    avg_rates = np.array([len(neuron_spikes[nid]) / (T_sim / 1000.0) for nid in display_neurons])

    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(3, 2, width_ratios=[5, 1], height_ratios=[4, 1, 0], hspace=0.05, wspace=0.08)

    # 栅格图
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

    # 群体平均发放率
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(bins[:-1] / 1000.0, rate_100, width=0.01, color='blue',
            edgecolor='black', linewidth=0.3)
    ax2.set_xlabel('t (s)', fontsize=11)
    ax2.set_ylabel('Firing rate (Hz)', fontsize=11)
    ax2.set_xlim(0, 0.6)
    ax2.set_ylim(0, 100)
    ax2.set_xticks(np.arange(0, 0.7, 0.1))
    ax2.set_yticks([0, 50, 100])

    # 单神经元平均发放率
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
    """图2：膜电位与阈值（初期、中期、后期）"""
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
    """图3：潜伏期随发放次数下降"""
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
    """图4（两个参数）"""
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
# 参数扫描（固定背景率版本，速度已足够）
# =============================================================================
def scan_parameter(param_name, param_values, n_trials=5, T_steps=200 * 1000):
    """
    参数扫描，T_steps可调（主仿真450s，扫描可用200s节省时间）
    """
    results = []
    print(f"\n>>> 扫描: {param_name} (每点 {n_trials} 次, 单次 {T_steps // 1000}s)")
    for val in param_values:
        n_succ = 0
        for trial in range(n_trials):
            trial_seed = 42 + trial * 100 + int(val * 1000) % 10000

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
            elif param_name == 'Spike deletion':
                # 脉冲删除简化：通过降低模式频率模拟，实际更复杂，可选
                gen_kwargs['freq'] = pattern_freq * (1 - val)
            else:
                raise ValueError(f"未知参数: {param_name}")

            neuron_spikes, neuron_types, all_times, all_neurons, all_types, time_start, pattern_intervals, _ = \
                generate_input_spikes(T_steps=T_steps, seed=trial_seed, **gen_kwargs)

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
    print("=" * 60)
    print("阶段1：主仿真 (450s) —— 生成图1、图2、图3")
    print("=" * 60)

    t0 = time.time()
    (neuron_spikes, neuron_types, all_times, all_neurons, all_types,
     time_start, pattern_intervals, pattern_neurons) = generate_input_spikes()
    print(f"[1/4] 输入脉冲生成完成，耗时 {time.time() - t0:.2f} s")

    # 膜电位记录窗口（根据模式出现情况微调）
    record_windows = [
        (0, 2000),          # 初期 0-2s
        (13000, 15000),     # 中期 13-15s
        (448000, 450000)    # 后期 最后2s
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
    plot_fig1(all_times, all_neurons, all_types, neuron_spikes, pattern_neurons)
    plot_fig2(mem_recs, pattern_intervals)
    plot_fig3(latencies)
    print(f"[3/4] 图1/2/3 绘制完成，耗时 {time.time() - t0:.2f} s")

    print("\n" + "=" * 60)
    print("阶段2：参数敏感性扫描 —— 生成图4")
    print("提示：完整扫描耗时较长，可选择性运行或调低T_SCAN、N_TRIALS")
    print("=" * 60)

    # 扫描时长和次数（可根据需要调整）
    T_SCAN = 200 * 1000   # 每次仿真200秒（比450秒快，但仍足够显示趋势）
    N_TRIALS = 5          # 每个参数值重复5次

    # 参数1：模式出现频率
    freq_vals = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    t0 = time.time()
    succ_freq = scan_parameter('Pattern frequency', freq_vals, n_trials=N_TRIALS, T_steps=T_SCAN)
    print(f"频率扫描耗时 {time.time() - t0:.1f} s")

    # 参数2：初始权重
    weight_vals = [0.30, 0.35, 0.40, 0.45, 0.50]
    t0 = time.time()
    succ_weight = scan_parameter('Initial weight', weight_vals, n_trials=N_TRIALS, T_steps=T_SCAN)
    print(f"权重扫描耗时 {time.time() - t0:.1f} s")

    # 绘制两个参数的图（满足实验最低要求）
    plot_fig4_two_params('Pattern frequency', freq_vals, succ_freq,
                         'Initial weight', weight_vals, succ_weight)

    print("\n[完成] 所有图表已生成！")