"""实验2：单模式、3个竞争输出神经元（有侧向抑制）。

测试侧向抑制是否能使神经元形成多样化的时间表征。
"""

import numpy as np
import pickle
import os
from spike_train import generate_spike_train
from simulation import Simulation
from analysis import compute_latencies
from plotting import plot_latency_subplots
from utils import get_params_single_pattern, ensure_figure_dir, FIGURE_DIR


def run_experiment2():
    print("=" * 60)
    print("实验2：单模式、3个竞争神经元（有抑制）")
    print("=" * 60)

    ensure_figure_dir()

    # 参数：有侧向抑制
    params = get_params_single_pattern(inhib_strength=0.25, random_state=42)

    print(f"\n参数:")
    print(f"  输入神经元={params.nAfferent}, 模式数={params.nPattern}")
    print(f"  输出神经元={params.nNeuron}, 抑制强度={params.inhibStrength}")
    print(f"  T={params.T:.1f}s, 阈值={params.threshold:.1f}")

    # 生成脉冲序列（与实验1相同）
    print("\n[1/4] 生成脉冲序列...")
    spikeList, afferentList = generate_spike_train(params)

    # 初始化仿真
    print("\n[2/4] 初始化仿真...")
    sim = Simulation(params)
    sim.initialize(spikeList)

    # 运行仿真
    print("\n[3/4] 运行仿真...")
    sim.run(spikeList, afferentList)

    # 报告发放次数
    print("\n发放次数:")
    for i, neuron in enumerate(sim.neurons):
        print(f"  神经元 {i + 1}: {int(neuron.nFiring)} 次发放")

    # 分析
    print("\n[4/4] 分析和绘图...")
    latency, HR, FA, final_latency = compute_latencies(sim.neurons, params)

    print("\n命中率和误报率:")
    for i in range(params.nNeuron):
        print(f"  神经元 {i + 1}: HR={HR[0, i]:.3f}, FA={FA[0, i]:.2f} Hz, "
              f"最终潜伏期={final_latency[0, i] * 1000 if not np.isnan(final_latency[0, i]) else 'N/A'} ms")

    # 计算成功神经元之间的潜伏期差异
    success_mask = (HR > 0.9) & (FA < 1.0)
    if np.sum(success_mask[0, :]) >= 2:
        success_lats = final_latency[0, success_mask[0, :]]
        success_lats = success_lats[~np.isnan(success_lats)]
        success_lats = np.sort(success_lats)
        diffs = np.diff(success_lats) * 1000  # ms
        print(f"\n相邻潜伏期差异 (ms): {diffs}")

    # 绘图
    plot_latency_subplots(latency, params,
                          title_prefix="实验2：有侧向抑制",
                          filename=os.path.join(FIGURE_DIR, "exp2_competitive.png"),
                          nNeuron=params.nNeuron, nPattern=1,
                          HR=HR, FA=FA)

    # 保存结果
    results = {
        'params': params,
        'latency': latency,
        'HR': HR,
        'FA': FA,
        'final_latency': final_latency,
        'firing_counts': [int(n.nFiring) for n in sim.neurons],
        'weights': [n.weight.copy() for n in sim.neurons],
    }

    save_path = os.path.join(FIGURE_DIR, "exp2_results.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)

    print("\n实验2 完成。")
    return results


if __name__ == "__main__":
    run_experiment2()
