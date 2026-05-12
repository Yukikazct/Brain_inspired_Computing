import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time

# ============================================================
# 全局参数 —— 恢复 N=2000，但缩短仿真至 150s
# ============================================================
N = 2000
T_sim = 150 * 1000          # <--- 由 450s 改为 150s
dt = 1
T_pattern = 50

r_bg = 52.0
r_spont = 10.0
pattern_freq = 0.25
N_pattern_neurons = N // 2
jitter_std = 1.0

tau_m = 10.0
tau_s = 2.5
T_thresh_base = 500
K1 = 2.0
K2 = 4.0
refractory = 1
w_init = 0.475
w_min, w_max = 0.0, 1.0

a_plus = 0.03125
a_minus = 0.85 * a_plus
tau_plus = 16.8
tau_minus = 33.7
LTP_WINDOW = 7 * tau_plus
LTD_WINDOW = 7 * tau_minus

t_max_epsp = tau_m * tau_s / (tau_m - tau_s) * np.log(tau_m / tau_s)
max_epsp = np.exp(-t_max_epsp / tau_m) - np.exp(-t_max_epsp / tau_s)
K_epsp = 1.0 / max_epsp
decay_m = np.exp(-dt / tau_m)
decay_s = np.exp(-dt / tau_s)

np.random.seed(42)

# ============================================================
# 【关键修改】generate_input_spikes：让模式神经元发放率天然更高
# ============================================================
def generate_input_spikes(T_steps=T_sim, freq=pattern_freq, jitter=jitter_std,
                          n_pattern=N_pattern_neurons):
    T_sec = T_steps / 1000.0
    bg_spikes = []
    # --- 【修改1：模式神经元背景发放率+10Hz，制造天然差异】 ---
    for j in range(N):
        if j < N_pattern_neurons:
            rate = r_bg + 10.0  # 模式神经元背景率更高
        else:
            rate = r_bg
        n_spikes = np.random.poisson(rate * T_sec)
        n_spikes = min(n_spikes, T_steps - 1)
        times = np.sort(np.random.choice(T_steps, size=n_spikes, replace=False))
        bg_spikes.append(times.astype(np.int32))

    pattern_neurons = np.arange(n_pattern)
    t0_template = np.random.randint(0, T_steps - T_pattern)
    template = {}
    for j in pattern_neurons:
        times = bg_spikes[j]
        mask = (times >= t0_template) & (times < t0_template + T_pattern)
        template[j] = (times[mask] - t0_template).astype(np.float64)

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

    neuron_spikes, neuron_types = [], []
    for j in range(N):
        bg = bg_spikes[j].astype(np.float64)
        if j in pattern_neurons:
            # --- 【修改2：不删除背景脉冲，直接叠加模式脉冲】 ---
            final_bg = bg  # 保留所有背景脉冲，不再删除模式区间内的
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

        n_spont = np.random.poisson(r_spont * T_sec)
        n_spont = min(n_spont, T_steps - 1)
        spont = np.sort(np.random.choice(T_steps, size=n_spont, replace=False)).astype(np.int32)
        all_t = np.concatenate([final_bg.astype(np.int32), pattern_arr, spont])
        all_typ = np.concatenate([
            np.zeros(len(final_bg), dtype=np.int8),
            np.ones(len(pattern_arr), dtype=np.int8),
            np.zeros(len(spont), dtype=np.int8)
        ])
        order_j = np.argsort(all_t)
        neuron_spikes.append(all_t[order_j])
        neuron_types.append(all_typ[order_j])

    counts = np.zeros(T_steps, dtype=np.int64)
    for spk in neuron_spikes:
        if len(spk): np.add.at(counts, spk, 1)
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

def simulate(neuron_spikes, all_times, all_neurons, all_types, time_start,
             pattern_intervals, record_windows, w_init_val=w_init,
             T_steps=T_sim, T_thresh=T_thresh_base):
    w = np.full(N, w_init_val, dtype=np.float64)
    u, v = 0.0, 0.0
    last_spike = -1
    refr_until = -1
    output_spikes, latencies = [], []
    membrane_records = [[] for _ in record_windows]
    spike_ptr = np.zeros(N, dtype=np.int32)
    spike_list = neuron_spikes
    pattern_starts_arr = np.array([s for s, _ in pattern_intervals], dtype=np.int32)
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
        u = decay_m * u + s
        v = decay_s * v + s
        epsp = K_epsp * (u - v)
        eta = 0.0
        if last_spike >= 0:
            dt_eta = t - last_spike
            eta = T_thresh * (K1 * np.exp(-dt_eta / tau_m) -
                              K2 * (np.exp(-dt_eta / tau_m) - np.exp(-dt_eta / tau_s)))
        p = eta + epsp
        for ri, (t1, t2) in enumerate(record_windows):
            if t1 <= t < t2:
                membrane_records[ri].append((t, p))
        if t >= refr_until and p >= T_thresh:
            output_spikes.append(t)
            ip = np.searchsorted(pattern_starts_arr, t)
            latency = 0.0
            if ip > 0 and t < pattern_starts_arr[ip - 1] + T_pattern:
                latency = t - pattern_starts_arr[ip - 1]
            latencies.append(latency)
            last_spike = t
            refr_until = t + refractory
            u, v = 0.0, 0.0
            for j in range(N):
                ptr = spike_ptr[j]
                has_pre[j] = ptr > 0
                has_post[j] = ptr < len(spike_list[j])
                if has_pre[j]: tj_pre[j] = spike_list[j][ptr - 1]
                if has_post[j]: tj_post[j] = spike_list[j][ptr]
            dt_pre = t - tj_pre[has_pre]
            mask_ltp = dt_pre <= LTP_WINDOW
            idx_ltp = np.where(has_pre)[0][mask_ltp]
            if len(idx_ltp): w[idx_ltp] += a_plus * np.exp(-dt_pre[mask_ltp] / tau_plus)
            dt_post = tj_post[has_post] - t
            mask_ltd = (dt_post > 0) & (dt_post <= LTD_WINDOW)
            idx_ltd = np.where(has_post)[0][mask_ltd]
            if len(idx_ltd): w[idx_ltd] -= a_minus * np.exp(-dt_post[mask_ltd] / tau_minus)
            w = np.clip(w, w_min, w_max)

    rec_arrays = [(np.array([x[0] for x in rec]) / 1000.0,
                   np.array([x[1] for x in rec])) for rec in membrane_records]
    return np.array(output_spikes), np.array(latencies), rec_arrays, w


def evaluate_trial(output_spikes, pattern_intervals, T_steps=T_sim, eval_last_s=50):
    t_start = T_steps - eval_last_s * 1000
    patterns_eval = [(s, e) for s, e in pattern_intervals if s >= t_start]
    n_pat = len(patterns_eval)
    if n_pat == 0: return False, 0.0, 999.0, 0
    out_eval = output_spikes[output_spikes >= t_start]
    hits = 0
    hit_latencies = []
    for s, e in patterns_eval:
        mask = (out_eval >= s) & (out_eval < e)
        if np.any(mask):
            hits += 1
            hit_latencies.append(out_eval[mask][0] - s)
    hit_rate = hits / n_pat
    avg_latency = np.mean(hit_latencies) if hit_latencies else 999.0
    false_alarms = sum(1 for t in out_eval if not any(s <= t < e for s, e in patterns_eval))
    success = (avg_latency < 10) and (hit_rate > 0.98) and (false_alarms == 0)
    return success, hit_rate, avg_latency, false_alarms


# 绘图函数（略作窗口调整）
def plot_fig1(all_times, all_neurons, all_types, neuron_spikes, save_path='fig1_input_pattern.png'):
    np.random.seed(123)
    pat_ids = np.sort(np.random.choice(N_pattern_neurons, 50, replace=False))
    non_ids = np.sort(np.random.choice(np.arange(N_pattern_neurons, N), 50, replace=False))
    display_neurons = np.empty(100, dtype=np.int32)
    display_neurons[0::2] = pat_ids[:50]
    display_neurons[1::2] = non_ids[:50]
    disp_map = {nid: i for i, nid in enumerate(display_neurons)}
    is_pat = (display_neurons < N_pattern_neurons)
    t_max_ms = 600
    mask = np.isin(all_neurons, display_neurons) & (all_times < t_max_ms)
    t_show = all_times[mask]
    n_show = np.array([disp_map[n] for n in all_neurons[mask]], dtype=np.int32)
    typ_show = all_types[mask]
    bins = np.arange(0, t_max_ms + 1, 10)
    counts_100, _ = np.histogram(t_show, bins=bins)
    rate_100 = counts_100 / (100 * 0.01)
    avg_rates = np.array([len(neuron_spikes[nid]) / (T_sim / 1000.0) for nid in display_neurons])
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(3, 2, width_ratios=[5, 1], height_ratios=[4, 1, 0], hspace=0.05, wspace=0.08)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(t_show[typ_show == 0] / 1000.0, n_show[typ_show == 0] + 1,
                s=1.5, c='blue', alpha=0.5, label='Background', zorder=1)
    ax1.scatter(t_show[typ_show == 1] / 1000.0, n_show[typ_show == 1] + 1,
                s=2.0, c='red', alpha=0.8, label='Pattern', zorder=2)
    ax1.set_ylabel('# afferent', fontsize=11)
    ax1.set_xlim(0, 0.6); ax1.set_ylim(0.5, 100.5)
    ax1.set_yticks([1, 100]); ax1.set_xticks([])
    ax1.legend(loc='upper right', fontsize=8, frameon=False)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(bins[:-1] / 1000.0, rate_100, width=0.01, color='blue', edgecolor='black', linewidth=0.3)
    ax2.set_xlabel('t (s)', fontsize=11); ax2.set_ylabel('Firing rate (Hz)', fontsize=11)
    ax2.set_xlim(0, 0.6); ax2.set_ylim(0, 100)
    ax2.set_xticks(np.arange(0, 0.7, 0.1)); ax2.set_yticks([0, 50, 100])
    ax3 = fig.add_subplot(gs[0:2, 1])
    colors = ['red' if is_pat[i] else 'blue' for i in range(100)]
    ax3.barh(np.arange(100) + 1, avg_rates, height=1, color=colors, edgecolor='black', linewidth=0.3)
    ax3.set_xlabel('Firing rate (Hz)', fontsize=11)
    ax3.set_xlim(0, 100); ax3.set_ylim(0.5, 100.5); ax3.set_yticks([]); ax3.set_xticks([0, 50, 100])
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"[保存] {save_path}")


def plot_fig2(mem_recs, pattern_intervals, save_path='fig2_membrane_potential.png'):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    labels = ['a', 'b', 'c']
    # 窗口针对150s调整：初期0-1s，中期13-14s（选择性出现），后期149-150s
    ranges = [(0, 1), (13, 14), (149, 150)]
    for ax, (t_arr, p_arr), lab, (r0, r1) in zip(axes, mem_recs, labels, ranges):
        ax.plot(t_arr, p_arr, 'b-', linewidth=0.9, label='potential')
        ax.axhline(T_thresh_base, c='red', ls='--', linewidth=1.2, label='threshold')
        ax.axhline(0, c='black', ls=':', linewidth=0.8, label='resting pot.')
        for s, e in pattern_intervals:
            s_s, e_s = s / 1000.0, e / 1000.0
            if e_s >= r0 and s_s <= r1:
                ax.axvspan(max(s_s, r0), min(e_s, r1), color='gray', alpha=0.3)
        ax.set_xlim(r0, r1); ax.set_ylim(-200, 1200)
        ax.set_ylabel('Potential (a. u.)', fontsize=11)
        ax.text(0.02, 0.88, lab, transform=ax.transAxes, fontsize=14, fontweight='bold')
        if ax == axes[0]: ax.legend(loc='upper right', fontsize=8)
        if ax == axes[-1]: ax.set_xlabel('t (s)', fontsize=11)
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"[保存] {save_path}")


def plot_fig3(latencies, save_path='fig3_latency.png'):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(np.arange(len(latencies)), latencies, s=2, c='blue', alpha=0.6)
    ax.set_xlabel('# discharges', fontsize=11)
    ax.set_ylabel('Postsynaptic spike latency (ms)', fontsize=11)
    ax.set_xlim(0, max(1500, len(latencies)))
    ax.set_ylim(0, 50)
    ax.set_xticks(np.arange(0, max(1500, len(latencies)) + 1, 250))
    ax.set_yticks(np.arange(0, 51, 5))
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"[保存] {save_path}")


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("折中参数仿真 (N=2000, T=150s) —— 预计 5~10 分钟")
    print("=" * 70)

    t0 = time.time()
    neuron_spikes, neuron_types, all_times, all_neurons, all_types, time_start, pattern_intervals = \
        generate_input_spikes()
    print(f"[1/3] 脉冲生成完成, 耗时 {time.time() - t0:.1f}s")

    record_windows = [
        (0, 1000),          # 初期 (0~1s)
        (13000, 14000),     # 中期 (选择性出现附近)
        (149000, 150000)    # 后期 (最后1s)
    ]

    t0 = time.time()
    out_spikes, latencies, mem_recs, final_w = simulate(
        neuron_spikes, all_times, all_neurons, all_types, time_start,
        pattern_intervals, record_windows)
    print(f"[2/3] 仿真完成, 放电 {len(out_spikes)} 次, 耗时 {time.time() - t0:.1f}s")

    success, hit_rate, avg_lat, fa = evaluate_trial(out_spikes, pattern_intervals, eval_last_s=50)
    print(f"[评估] 成功={success}, 命中率={hit_rate*100:.1f}%, "
          f"平均潜伏期={avg_lat:.2f}ms, 误报={fa}")

    plot_fig1(all_times, all_neurons, all_types, neuron_spikes)
    plot_fig2(mem_recs, pattern_intervals)
    plot_fig3(latencies)
    print("\n[完成] 所有图表已生成。")