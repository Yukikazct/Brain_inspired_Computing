"""
neuron_stdp.py
输出神经元与 STDP 学习模块

功能：
    - 基于 SRM（Spike Response Model）实现单输出神经元的膜电位动力学
    - 采用指数型 EPSP 核与发放后电位（AHP）
    - 实现基于脉冲时序的 STDP（Spike-Timing-Dependent Plasticity）
    - 提供仿真主循环与评估指标计算

核心函数
--------
simulate(...)      : 运行完整仿真，返回输出脉冲、潜伏期、膜电位记录、最终权重
evaluate_trial(...) : 根据输出脉冲评估模式识别性能（命中率、潜伏期、误报）
"""

import numpy as np
from config import (
    N, T_sim, dt,
    tau_m, tau_s, T_thresh, K1, K2, refractory, w_init, w_min, w_max,
    a_plus, a_minus, tau_plus, tau_minus, LTP_WINDOW, LTD_WINDOW,
    K_epsp, decay_m, decay_s, T_pattern
)


def simulate(neuron_spikes, all_times, all_neurons, all_types, time_start,
             pattern_intervals, record_windows, w_init_val=w_init,
             T_steps=T_sim):
    """
    运行 SRM + STDP 仿真。

    参数
    ----
    neuron_spikes : list[np.ndarray]
        每个输入神经元的脉冲时间列表。
    all_times, all_neurons, all_types : np.ndarray
        全局展平的脉冲时间与神经元编号、类型。
    time_start : np.ndarray
        每个时间步的全局起始索引。
    pattern_intervals : list[tuple]
        模式出现的时间区间。
    record_windows : list[tuple]
        需要记录膜电位的 (start_ms, end_ms) 窗口列表。
    w_init_val : float
        初始权重值。
    T_steps : int
        仿真总时长（ms）。

    返回
    ----
    tuple : (output_spikes, latencies, membrane_records, final_weights)
        output_spikes  : np.ndarray  输出神经元发放时间（ms）
        latencies      : np.ndarray  每次发放相对于最近模式起始的潜伏期（ms）
        membrane_records : list[tuple]  每个记录窗口的 (time_s, potential) 数组对
        final_weights  : np.ndarray  仿真结束后的 N 个突触权重
    """
    w = np.full(N, w_init_val, dtype=np.float64)
    u, v = 0.0, 0.0               # EPSP 双指数状态的内部变量
    last_spike = -1               # 上次发放时间
    refr_until = -1               # 不应期结束时间

    output_spikes = []
    latencies = []
    membrane_records = [[] for _ in record_windows]

    spike_ptr = np.zeros(N, dtype=np.int32)   # 每个神经元当前读到的脉冲索引
    spike_list = neuron_spikes

    # 将模式起始时间转为数组，便于快速搜索
    pattern_starts_arr = np.array([s for s, _ in pattern_intervals], dtype=np.int32)

    # 预分配 STDP 辅助数组
    tj_pre = np.empty(N, dtype=np.int32)
    tj_post = np.empty(N, dtype=np.int32)
    has_pre = np.empty(N, dtype=bool)
    has_post = np.empty(N, dtype=bool)

    # ================================================================
    # 主循环：逐 ms 推进
    # ================================================================
    for t in range(T_steps):
        idx0, idx1 = time_start[t], time_start[t + 1]

        if idx0 < idx1:
            # 当前时刻有输入脉冲到达
            js = all_neurons[idx0:idx1]
            np.add.at(spike_ptr, js, 1)          # 推进对应神经元的指针
            s = np.sum(w[js])                     # 总突触电流（权重和）
        else:
            s = 0.0

        # EPSP 差分更新（双指数衰减）
        u = decay_m * u + s
        v = decay_s * v + s
        epsp = K_epsp * (u - v)

        # 发放后电位（After-Hyperpolarization Potential）
        eta = 0.0
        if last_spike >= 0:
            dt_eta = t - last_spike
            eta = T_thresh * (
                K1 * np.exp(-dt_eta / tau_m)
                - K2 * (np.exp(-dt_eta / tau_m) - np.exp(-dt_eta / tau_s))
            )

        # 膜电位 = 发放后电位 + EPSP
        p = eta + epsp

        # ================================================================
        # 发放判定与记录
        # ================================================================
        if t >= refr_until and p >= T_thresh:
            # ----- 发放！ -----
            output_spikes.append(t)

            # ----- 潜伏期计算：距离最近模式起始的时间 -----
            latency = 0.0
            ip = np.searchsorted(pattern_starts_arr, t)
            if ip > 0:
                s_p = pattern_starts_arr[ip - 1]
                if t < s_p + T_pattern:
                    latency = t - s_p
            latencies.append(latency)

            # ----- 重置神经元状态 -----
            last_spike = t
            refr_until = t + refractory
            u, v = 0.0, 0.0

            # 发放后电位在 t=0 时的峰值（用于记录）
            eta_spike = T_thresh * K1   # = 1000 a.u.
            p_record = eta_spike

            # 记录膜电位峰值（仅在指定窗口内）
            for ri, (t1, t2) in enumerate(record_windows):
                if t1 <= t < t2:
                    membrane_records[ri].append((t, p_record))

            # ============================================================
            # STDP 权重更新（仅在输出神经元发放时触发）
            # ============================================================
            for j in range(N):
                ptr = spike_ptr[j]
                has_pre[j] = ptr > 0
                has_post[j] = ptr < len(spike_list[j])
                if has_pre[j]:
                    tj_pre[j] = spike_list[j][ptr - 1]   # 最近一次输入脉冲
                if has_post[j]:
                    tj_post[j] = spike_list[j][ptr]       # 下一次输入脉冲

            # LTP：pre -> post，输入脉冲在输出脉冲之前
            dt_pre = t - tj_pre[has_pre]
            mask_ltp = dt_pre <= LTP_WINDOW
            idx_ltp = np.where(has_pre)[0][mask_ltp]
            if len(idx_ltp):
                w[idx_ltp] += a_plus * np.exp(-dt_pre[mask_ltp] / tau_plus)

            # LTD：post -> pre，输入脉冲在输出脉冲之后
            dt_post = tj_post[has_post] - t
            mask_ltd = (dt_post > 0) & (dt_post <= LTD_WINDOW)
            idx_ltd = np.where(has_post)[0][mask_ltd]
            if len(idx_ltd):
                w[idx_ltd] -= a_minus * np.exp(-dt_post[mask_ltd] / tau_minus)

            # 权重裁剪到 [w_min, w_max]
            w = np.clip(w, w_min, w_max)

        else:
            # 未发放，正常记录膜电位轨迹
            for ri, (t1, t2) in enumerate(record_windows):
                if t1 <= t < t2:
                    membrane_records[ri].append((t, p))
            # 未发放时不做 STDP 更新

    # 将记录列表转为 numpy 数组，时间单位转为秒
    rec_arrays = [
        (np.array([x[0] for x in rec]) / 1000.0,
         np.array([x[1] for x in rec]))
        for rec in membrane_records
    ]
    return np.array(output_spikes), np.array(latencies), rec_arrays, w


def evaluate_trial(output_spikes, pattern_intervals, T_steps=T_sim, eval_last_s=150):
    """
    评估单次仿真的模式识别性能。

    成功标准（与实验文档一致）：
        - 平均潜伏期 < 10 ms
        - 模式命中率 >= 98%
        - 在评估窗口内无误报（非模式时段无输出脉冲）

    参数
    ----
    output_spikes : np.ndarray
        输出神经元的所有发放时间（ms）。
    pattern_intervals : list[tuple]
        模式出现的时间区间。
    T_steps : int
        仿真总时长（ms）。
    eval_last_s : int
        评估时段：取最后 eval_last_s 秒进行统计。

    返回
    ----
    tuple : (success, hit_rate, avg_latency, false_alarms)
        success      : bool   是否满足成功标准
        hit_rate     : float  模式命中率
        avg_latency  : float  平均潜伏期（ms）
        false_alarms : int    误报次数
    """
    t_eval_start = T_steps - eval_last_s * 1000
    patterns_eval = [(s, e) for s, e in pattern_intervals if s >= t_eval_start]
    n_pat = len(patterns_eval)
    if n_pat == 0:
        return False, 0.0, 999.0, 0

    out_eval = output_spikes[output_spikes >= t_eval_start]

    # 统计命中次数与命中潜伏期
    hits = 0
    hit_latencies = []
    for s, e in patterns_eval:
        mask = (out_eval >= s) & (out_eval < e)
        if np.any(mask):
            hits += 1
            hit_latencies.append(out_eval[mask][0] - s)

    hit_rate = hits / n_pat
    avg_latency = np.mean(hit_latencies) if hit_latencies else 999.0

    # 统计误报：评估窗口内、非模式时段的输出脉冲
    false_alarms = 0
    for t in out_eval:
        in_pat = any(s <= t < e for s, e in patterns_eval)
        if not in_pat:
            false_alarms += 1

    success = (avg_latency < 10) and (hit_rate >= 0.98) and (false_alarms == 0)
    return success, hit_rate, avg_latency, false_alarms
