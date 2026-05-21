"""实验3：侧向抑制强度分析。

扫描不同抑制强度，测量成功神经元之间的潜伏期差异。
脉冲序列预生成后在不同抑制值之间复用以提高效率。
"""

import numpy as np
import pickle
import os
from spike_train import generate_spike_train
from simulation import Simulation
from analysis import compute_latencies
from plotting import plot_inhibition_strength_analysis
from utils import get_params_single_pattern, ensure_figure_dir, FIGURE_DIR


def run_single_simulation(params, spikeList, afferentList, rng_seed_offset):
    """使用预生成的脉冲序列运行一次仿真。"""
    sim = Simulation(params)
    sim.initialize(spikeList, rng_seed_offset=rng_seed_offset)
    sim.run(spikeList, afferentList)
    latency, HR, FA, final_latency = compute_latencies(sim.neurons, params)
    return final_latency, HR, FA


def run_experiment3():
    print("=" * 60)
    print("实验3：侧向抑制强度分析")
    print("=" * 60)

    ensure_figure_dir()

    # 待测试的抑制值（阈值比例）
    inhib_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    n_repeats = 5
    hr_threshold = 0.9
    fa_threshold = 1.0
    T_sim = 75.0  # 与实验1和2一致（每run）

    # ── 预生成所有重复的脉冲序列 ──
    print("\n预生成脉冲序列...")
    spike_trains = []  # (spikeList, afferentList)列表
    for rep in range(n_repeats):
        print(f"  重复 {rep + 1}/{n_repeats}...", end=" ")
        rs = 100 + rep * 100  # 此重复的随机状态
        params_base = get_params_single_pattern(inhib_strength=0.0, random_state=rs)
        params_base.T = T_sim
        sl, al = generate_spike_train(params_base)
        spike_trains.append((sl, al, params_base))
        print(f"{len(sl)} 个脉冲")

    all_mean_diffs = []
    all_std_diffs = []

    for inhib in inhib_values:
        print(f"\n{'─' * 50}")
        print(f"测试抑制强度 = {inhib}")
        print(f"{'─' * 50}")

        all_diffs_for_this_inhib = []

        for rep in range(n_repeats):
            print(f"  重复 {rep + 1}/{n_repeats}...", end=" ")

            sl, al, params_base = spike_trains[rep]
            # 使用相同的模式位置创建此抑制强度的新参数
            params = get_params_single_pattern(inhib_strength=inhib,
                                                random_state=params_base.randomState)
            params.T = T_sim
            params.posCopyPaste = params_base.posCopyPaste
            params._posCopyPasteLists = params_base._posCopyPasteLists

            try:
                final_latency, HR, FA = run_single_simulation(
                    params, sl, al, rng_seed_offset=params_base.randomState + 1
                )
            except Exception as e:
                print(f"错误: {e}")
                continue

            # 识别成功神经元
            success_mask = (HR > hr_threshold) & (FA < fa_threshold)
            success_lats = final_latency[0, success_mask[0, :]]
            success_lats = success_lats[~np.isnan(success_lats)]
            success_lats = np.sort(success_lats)

            if len(success_lats) >= 2:
                diffs = np.diff(success_lats)
                all_diffs_for_this_inhib.extend(diffs)
                print(f"成功: {len(success_lats)} 个神经元, "
                      f"潜伏期={success_lats * 1000} ms, "
                      f"差异={diffs * 1000} ms")
            else:
                print(f"仅 {len(success_lats)} 个成功神经元")

        if len(all_diffs_for_this_inhib) > 0:
            mean_diff = np.mean(all_diffs_for_this_inhib)
            std_diff = np.std(all_diffs_for_this_inhib)
        else:
            mean_diff = 0.0
            std_diff = 0.0

        all_mean_diffs.append(mean_diff)
        all_std_diffs.append(std_diff)
        print(f"  => 平均差异: {mean_diff * 1000:.2f} ms, 标准差: {std_diff * 1000:.2f} ms")

    # 绘图
    plot_inhibition_strength_analysis(
        inhib_values, all_mean_diffs, all_std_diffs, None,
        filename=os.path.join(FIGURE_DIR, "exp3_inhib_analysis.png")
    )

    # 保存结果
    results = {
        'inhib_values': inhib_values,
        'mean_diffs': all_mean_diffs,
        'std_diffs': all_std_diffs,
    }
    save_path = os.path.join(FIGURE_DIR, "exp3_results.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)

    print("\n实验3 完成。")
    return results


if __name__ == "__main__":
    run_experiment3()
