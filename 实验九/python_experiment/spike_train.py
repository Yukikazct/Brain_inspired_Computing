"""脉冲序列生成，内嵌重复模式。

基于 Masquelier et al. (2009) 的 copy-paste 方法：
1. 为所有输入神经元生成背景泊松脉冲
2. 一次性嵌入模式（全局向量化处理）
3. 仅修改参与模式的输入神经元的脉冲
"""

import numpy as np


def generate_spike_train(params):
    """生成带嵌入模式的完整脉冲序列。

    返回
    -------
    spikeList : ndarray  排序后的脉冲时间
    afferentList : ndarray  脉冲对应的输入神经元索引（0起始）
    """
    rng = np.random.RandomState(params.randomState)

    nAfferent = params.nAfferent
    T_total = params.T * params.nRun
    cpDuration = params.copyPasteDuration
    jitter = params.jitter
    max_total_rate = params.maxFiringRate + params.spontaneousActivity

    # ── 第1步：生成背景脉冲 ──
    print(f"  生成背景脉冲 (最大发放率={max_total_rate:.0f}Hz, "
          f"T={T_total:.1f}s, N={nAfferent})...")

    afferent_rates = rng.uniform(0, params.maxFiringRate, nAfferent) + params.spontaneousActivity
    mean_rate = np.mean(afferent_rates)
    print(f"  平均发放率: {mean_rate:.1f} Hz")

    n_candidates = int(max_total_rate * T_total * nAfferent * 1.05)
    cand_times = rng.uniform(0, T_total, n_candidates)
    cand_afferents = rng.randint(0, nAfferent, n_candidates).astype(np.uint16)

    accept_prob = afferent_rates[cand_afferents] / max_total_rate
    mask = rng.uniform(0, 1, n_candidates) < accept_prob

    spikeList = cand_times[mask]
    afferentList = cand_afferents[mask]

    print(f"  生成了 {len(spikeList)} 个背景脉冲 "
          f"(预期约 {mean_rate * T_total * nAfferent:.0f})")

    # ── 第2步：嵌入模式 ──
    print(f"  嵌入 {params.nPattern} 个模式...")

    # 构建每个模式的参与神经元集合
    participating_sets = {}
    for pat in range(params.nPattern):
        if params.nPattern == 1:
            participating_sets[pat] = set(range(params.nCopyPasteAfferent))
        else:
            p_set = set()
            for a in range(nAfferent):
                if rng.rand() < params.nCopyPasteAfferent / nAfferent:
                    p_set.add(a)
            participating_sets[pat] = p_set

    # 从每个模式的首次出现中提取模板
    pattern_templates = {}
    for pat in range(params.nPattern):
        posList = params.posCopyPaste.get(pat, [])
        if len(posList) == 0:
            pattern_templates[pat] = {}
            continue
        participating = participating_sets[pat]
        first_pos = posList[0]
        win_start = first_pos * cpDuration
        win_end = win_start + cpDuration

        templates = {}
        for a in participating:
            in_window = ((spikeList >= win_start) & (spikeList < win_end) &
                         (afferentList == a))
            rel_times = spikeList[in_window] - win_start
            if len(rel_times) > 0:
                templates[a] = rel_times.copy()
        pattern_templates[pat] = templates

    # 执行 copy-delete-paste：只删除参与神经元的脉冲
    all_new_spikes = []
    all_new_afferents = []
    keep_mask = np.ones(len(spikeList), dtype=bool)

    for run_idx in range(params.nRun):
        run_offset = run_idx * params.T

        for pat in range(params.nPattern):
            posList = params.posCopyPaste.get(pat, [])
            templates = pattern_templates.get(pat, {})
            participating = participating_sets[pat]
            participating_list = list(participating)

            for p_idx in posList:
                win_start = run_offset + p_idx * cpDuration
                win_end = win_start + cpDuration

                # 仅删除该窗口中参与神经元的脉冲
                if len(participating_list) > 0:
                    in_win = ((spikeList >= win_start) & (spikeList < win_end) &
                              np.isin(afferentList, participating_list))
                    keep_mask[in_win] = False

                # 粘贴模板副本
                for a, rel_times in templates.items():
                    for rt in rel_times:
                        jittered = rt + jitter * rng.randn()
                        jittered = np.clip(jittered, 0, cpDuration)
                        all_new_spikes.append(win_start + jittered)
                        all_new_afferents.append(a)

    # 合并
    if len(all_new_spikes) > 0:
        new_spikes = np.array(all_new_spikes)
        new_afferents = np.array(all_new_afferents, dtype=np.uint16)
        kept_spikes = spikeList[keep_mask]
        kept_afferents = afferentList[keep_mask]
        spikeList = np.concatenate([kept_spikes, new_spikes])
        afferentList = np.concatenate([kept_afferents, new_afferents])

    # ── 第3步：排序 ──
    sort_idx = np.argsort(spikeList)
    spikeList = spikeList[sort_idx]
    afferentList = afferentList[sort_idx].astype(np.uint16)

    print(f"  最终: {len(spikeList)} 个脉冲")

    return spikeList, afferentList
