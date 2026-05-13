"""
input_generator.py
输入脉冲生成模块

功能：
    - 为 N 个输入神经元生成背景泊松脉冲
    - 随机选取部分神经元构成模式神经元群
    - 从背景脉冲中截取 50ms 片段作为模式模板
    - 按指定频率将带抖动的模式模板插入全局时间轴
    - 添加自发活动噪声
    - 构建全局时间索引，便于仿真时快速访问

输出数据结构：
    neuron_spikes : list[np.ndarray]   每个神经元的脉冲时间（ms）
    neuron_types  : list[np.ndarray]   每个脉冲的类型（0=背景/自发, 1=模式）
    all_times     : np.ndarray         全局脉冲时间（展平）
    all_neurons   : np.ndarray         全局脉冲对应的神经元编号
    all_types     : np.ndarray         全局脉冲类型
    time_start    : np.ndarray         每个时间步的全局起始索引（cumsum）
    pattern_intervals : list[tuple]    模式插入区间 [(start_ms, end_ms), ...]
    pattern_neurons   : np.ndarray     参与模式的神经元编号
"""

import numpy as np
import gc
from config import (
    N, T_sim, T_pattern,
    r_bg, r_spont, pattern_freq, N_pattern_neurons, jitter_std
)


def generate_input_spikes(T_steps=T_sim, freq=pattern_freq, jitter=jitter_std,
                          n_pattern=N_pattern_neurons, seed=None):
    """
    生成输入脉冲序列。

    参数
    ----
    T_steps : int
        仿真总时长（ms）。
    freq : float
        模式出现频率（占空比）。
    jitter : float
        模式脉冲抖动标准差（ms）。
    n_pattern : int
        参与模式的神经元数量。
    seed : int, optional
        随机种子，用于复现。

    返回
    ----
    tuple : (neuron_spikes, neuron_types, all_times, all_neurons, all_types,
             time_start, pattern_intervals, pattern_neurons)
    """
    if seed is not None:
        np.random.seed(seed)

    T_sec = T_steps / 1000.0

    # -----------------------------------------------------------------
    # 1. 生成固定背景率的泊松脉冲（每个神经元独立）
    # -----------------------------------------------------------------
    print("  [输入生成] 生成背景脉冲...")
    bg_spikes = []
    for j in range(N):
        n_spikes = np.random.poisson(r_bg * T_sec)
        # 限制数量不超过时间步数，避免越界
        n_spikes = min(n_spikes, T_steps - 1)
        times = np.sort(np.random.choice(T_steps, size=n_spikes, replace=False))
        bg_spikes.append(times.astype(np.int32))

    # -----------------------------------------------------------------
    # 2. 随机选择参与模式的神经元
    # -----------------------------------------------------------------
    all_neurons_idx = np.arange(N)
    pattern_neurons = np.random.choice(all_neurons_idx, size=n_pattern, replace=False)
    pattern_neurons = np.sort(pattern_neurons)

    # -----------------------------------------------------------------
    # 3. 从背景中随机截取 50ms 片段作为模式模板
    # -----------------------------------------------------------------
    print("  [输入生成] 提取模式模板...")
    t0_template = None
    template = None

    # 随机选取起始点，提取各模式神经元在该窗口内的背景脉冲作为模板
    t0 = np.random.randint(0, T_steps - T_pattern)
    template_candidate = {}
    for j in pattern_neurons:
        times = bg_spikes[j]
        mask = (times >= t0) & (times < t0 + T_pattern)
        rel_times = times[mask] - t0
        if len(rel_times) == 0:
            # 若该神经元在窗口内无脉冲，则添加一个随机脉冲，保证模板非空
            rel_times = np.array([np.random.uniform(0, T_pattern)])
        template_candidate[j] = rel_times.astype(np.float64)
    t0_template = t0
    template = template_candidate

    # -----------------------------------------------------------------
    # 4. 确定模式插入位置（均匀分布，且相邻模式间隔 >= 2 个模板长度）
    # -----------------------------------------------------------------
    total_blocks = T_steps // T_pattern
    n_patterns = max(1, int(total_blocks * freq))
    order = np.random.permutation(total_blocks)
    selected_blocks = []
    for b in order:
        # 确保相邻模式块之间至少间隔 2 个 T_pattern，避免重叠
        if all(abs(b - s) >= 2 for s in selected_blocks):
            selected_blocks.append(b)
        if len(selected_blocks) == n_patterns:
            break
    pattern_starts = sorted([b * T_pattern for b in selected_blocks])
    pattern_intervals = [(s, s + T_pattern) for s in pattern_starts]

    # -----------------------------------------------------------------
    # 5. 构建最终脉冲序列（背景 + 模式 + 自发活动）
    # -----------------------------------------------------------------
    print("  [输入生成] 合并脉冲序列...")
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
            all_typ = np.concatenate([
                np.zeros(len(final_bg), dtype=np.int8),
                np.ones(len(pattern_pulses), dtype=np.int8)
            ])
        else:
            # 非模式神经元：保留全部背景脉冲
            all_t = bg_spikes[j].astype(np.float64)
            all_typ = np.zeros(len(all_t), dtype=np.int8)

        # 添加自发活动（10 Hz 泊松）
        n_spont = np.random.poisson(r_spont * T_sec)
        n_spont = min(n_spont, T_steps - 1)
        spont = np.sort(np.random.choice(T_steps, size=n_spont, replace=False)).astype(np.int32)
        all_t = np.concatenate([all_t, spont])
        all_typ = np.concatenate([all_typ, np.zeros(len(spont), dtype=np.int8)])

        # 排序并去重（同一时间步多个脉冲只保留一个，模式脉冲优先）
        order_j = np.argsort(all_t)
        all_t = all_t[order_j]
        all_typ = all_typ[order_j]
        # 去重：保留每个时间步最后一个（模式脉冲后添加，因此优先保留）
        unique_mask = np.diff(all_t, prepend=-1) != 0
        all_t = all_t[unique_mask]
        all_typ = all_typ[unique_mask]

        neuron_spikes.append(all_t.astype(np.int32))
        neuron_types.append(all_typ)

    # 释放大的中间变量，降低内存峰值
    del bg_spikes
    gc.collect()

    # -----------------------------------------------------------------
    # 6. 构建全局时间索引（用于仿真时的快速向量化访问）
    # -----------------------------------------------------------------
    print("  [输入生成] 构建全局索引...")
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
