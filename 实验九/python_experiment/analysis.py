"""分析工具：潜伏期计算、命中率、误报率。"""

import numpy as np


def compute_latencies(neurons, params):
    """计算每个（模式，神经元）对的响应潜伏期。

    参数
    ----------
    neurons : Neuron列表
    params : Params

    返回
    -------
    latency : dict  {(pat, neur): ndarray}
        每次发放的潜伏期值（0 = 误报）。
    HR : ndarray  (nPattern, nNeuron)
        命中率。
    FA : ndarray  (nPattern, nNeuron)
        误报率 (Hz)。
    final_latency : ndarray  (nPattern, nNeuron)
        仿真最后三分之一阶段的平均潜伏期（非成功神经元为NaN）。
    """
    nPattern = params.nPattern
    nNeuron = len(neurons)
    T = params.T
    nRun = params.nRun

    # 评估范围：最后一个run
    eval_start = (nRun - 1) * T
    eval_end = nRun * T

    latency = {}
    HR = np.zeros((nPattern, nNeuron))
    FA = np.zeros((nPattern, nNeuron))
    final_latency = np.full((nPattern, nNeuron), np.nan)

    for neur_idx, neuron in enumerate(neurons):
        nF = int(neuron.nFiring)
        if nF == 0:
            for pat in range(nPattern):
                latency[(pat, neur_idx)] = np.array([])
            continue

        firingTimes = neuron.firingTime[:nF]

        for pat in range(nPattern):
            lat_vals = []
            posCopyPaste = params.posCopyPaste[pat]

            for ft in firingTimes:
                # 确定此次发放属于哪个run
                run_idx = int(np.ceil(ft / T)) - 1
                if run_idx < 0:
                    run_idx = 0

                # 确定它落在哪个模式窗口中
                period = int(np.ceil((ft - run_idx * T) / params.copyPasteDuration))

                if period - 1 in posCopyPaste:
                    # 命中：潜伏期相对于模式起始
                    lat = ft - run_idx * T - (period - 1) * params.copyPasteDuration
                    lat_vals.append(lat)
                else:
                    # 误报
                    lat_vals.append(0.0)

            latency[(pat, neur_idx)] = np.array(lat_vals)

            # 在评估窗口中计算HR和FA
            in_window = (firingTimes >= eval_start) & (firingTimes < eval_end)
            if np.any(in_window):
                lat_arr = np.array(lat_vals)
                hits = np.sum((lat_arr > 0) & in_window)

                # 统计评估窗口中的模式出现次数
                n_pattern_occurrences = np.sum(
                    (np.array(posCopyPaste) * params.copyPasteDuration +
                     (nRun - 1) * T >= eval_start) &
                    (np.array(posCopyPaste) * params.copyPasteDuration +
                     (nRun - 1) * T < eval_end)
                )

                if n_pattern_occurrences > 0:
                    HR[pat, neur_idx] = hits / n_pattern_occurrences

                total_firings = np.sum(in_window)
                FA[pat, neur_idx] = (total_firings - hits) / (eval_end - eval_start)

                # 最终潜伏期：评估窗口最后三分之一
                final_third_start = eval_start + 2.0 / 3.0 * (eval_end - eval_start)
                in_final = (firingTimes >= final_third_start) & (firingTimes < eval_end)
                if np.any(in_final):
                    final_lats = lat_arr[in_final]
                    final_lats = final_lats[final_lats > 0]
                    if len(final_lats) > 0:
                        final_latency[pat, neur_idx] = np.mean(final_lats)

    return latency, HR, FA, final_latency


def identify_successful_neurons(HR, FA, hr_threshold=0.9, fa_threshold=1.0):
    """识别成功学习了模式的神经元。

    返回
    -------
    success : ndarray (nPattern, nNeuron)
        成功（模式，神经元）对的布尔掩码。
    """
    return (HR > hr_threshold) & (FA < fa_threshold)


def compute_latency_differences(final_latency, success_mask):
    """计算成功神经元之间的相邻潜伏期差异。

    返回
    -------
    diffs : float列表
        相邻成功神经元之间的潜伏期差异（按潜伏期排序）。
    """
    nPattern = final_latency.shape[0]
    diffs = []

    for pat in range(nPattern):
        latencies = final_latency[pat, :]
        successful = success_mask[pat, :]
        success_lats = latencies[successful]
        success_lats = success_lats[~np.isnan(success_lats)]
        success_lats = np.sort(success_lats)

        for i in range(1, len(success_lats)):
            diffs.append(success_lats[i] - success_lats[i - 1])

    return diffs
