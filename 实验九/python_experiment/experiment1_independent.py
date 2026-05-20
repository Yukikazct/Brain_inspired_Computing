"""实验1：单模式、3个独立输出神经元（无侧向抑制）。

测试仅增加输出神经元数量是否会产生多样化的表征。
"""

import numpy as np
import pickle
import os
from parameters import Params
from spike_train import generate_spike_train
from simulation import Simulation
from analysis import compute_latencies
from plotting import plot_latency_subplots
from utils import get_params_single_pattern, ensure_figure_dir, FIGURE_DIR


def run_experiment1():
    print("=" * 60)
    print("实验1：单模式、3个独立神经元（无抑制）")
    print("=" * 60)

    ensure_figure_dir()

    # 参数：无侧向抑制
    params = get_params_single_pattern(inhib_strength=0.0, random_state=42)

    print(f"\n参数:")
    print(f"  输入神经元={params.nAfferent}, 模式数={params.nPattern}")
    print(f"  输出神经元={params.nNeuron}, 抑制强度={params.inhibStrength}")
    print(f"  T={params.T:.1f}s, 阈值={params.threshold:.1f}")

    # 生成脉冲序列
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

    # 绘图
    plot_latency_subplots(latency, params,
                          title_prefix="实验1：无侧向抑制",
                          filename=os.path.join(FIGURE_DIR, "exp1_independent.png"),
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

    save_path = os.path.join(FIGURE_DIR, "exp1_results.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)

    print("\n实验1 完成。")
    return results


if __name__ == "__main__":
    run_experiment1()
