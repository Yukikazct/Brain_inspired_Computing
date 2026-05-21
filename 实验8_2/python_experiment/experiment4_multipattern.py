"""实验4：多模式、多输出神经元。

测试竞争STDP神经元群体是否能在无监督条件下学习表示多个重复模式。
采用MATLAB方式：生成一个T-block脉冲序列，将其平移后重复运行nRun次。
"""

import numpy as np
import pickle
import os
from spike_train import generate_spike_train
from simulation import Simulation
from analysis import compute_latencies
from plotting import plot_latency_matrix
from utils import get_params_multi_pattern, ensure_figure_dir, FIGURE_DIR


def run_experiment4():
    print("=" * 60)
    print("实验4：多模式、多竞争神经元")
    print("=" * 60)

    ensure_figure_dir()

    # 参数：多模式、有侧向抑制
    params = get_params_multi_pattern(inhib_strength=0.25, random_state=42)

    # 仅生成一个T-block的脉冲序列
    params_one = get_params_multi_pattern(inhib_strength=0.25, random_state=42)
    params_one.nRun = 1  # 仅生成一个block

    print(f"\n参数:")
    print(f"  输入神经元={params_one.nAfferent}, 模式数={params_one.nPattern}")
    print(f"  输出神经元={params_one.nNeuron}, 抑制强度={params_one.inhibStrength}")
    print(f"  T={params_one.T:.1f}s, nRun={params.nRun} (多run模式)")
    print(f"  总仿真时间: {params_one.T * params.nRun:.1f}s")
    print(f"  阈值={params_one.threshold:.1f}")

    # 生成一个T-block的脉冲序列
    print("\n[1/5] 生成脉冲序列（一个T-block）...")
    spikeList, afferentList = generate_spike_train(params_one)

    # 初始化仿真
    print("\n[2/5] 初始化仿真...")
    sim = Simulation(params)
    sim.initialize(spikeList)

    # 运行nRun个block的仿真
    print(f"\n[3/5] 运行仿真 ({params.nRun} runs)...")
    for run_idx in range(params.nRun):
        shift = run_idx * params_one.T
        shifted_spikes = spikeList + shift
        print(f"  Run {run_idx + 1}/{params.nRun}: {len(shifted_spikes)} 个脉冲, "
              f"平移={shift:.1f}s")
        sim.run(shifted_spikes, afferentList)

    # 报告发放次数
    print("\n发放次数:")
    for i, neuron in enumerate(sim.neurons):
        status = "沉默" if int(neuron.nFiring) == 0 else "活跃"
        print(f"  神经元 {i + 1}: {int(neuron.nFiring)} 次发放 [{status}]")

    # 分析
    print("\n[4/5] 分析...")
    latency, HR, FA, final_latency = compute_latencies(sim.neurons, params)

    print("\n命中率（行=模式，列=神经元）:")
    for pat in range(params.nPattern):
        hr_row = ", ".join(f"{HR[pat, n]:.3f}" for n in range(params.nNeuron))
        print(f"  模式 {pat + 1}: [{hr_row}]")

    print("\n误报率 (Hz):")
    for pat in range(params.nPattern):
        fa_row = ", ".join(f"{FA[pat, n]:.2f}" for n in range(params.nNeuron))
        print(f"  模式 {pat + 1}: [{fa_row}]")

    print("\n最终潜伏期 (ms):")
    for pat in range(params.nPattern):
        lat_row = ", ".join(
            f"{final_latency[pat, n] * 1000:.1f}" if not np.isnan(final_latency[pat, n])
            else "N/A"
            for n in range(params.nNeuron)
        )
        print(f"  模式 {pat + 1}: [{lat_row}]")

    # 识别成功的（模式，神经元）对
    success_mask = (HR > 0.9) & (FA < 1.0)
    print("\n成功的（模式，神经元）对 (HR>0.9, FA<1Hz):")
    for pat in range(params.nPattern):
        for n in range(params.nNeuron):
            if success_mask[pat, n]:
                print(f"  模式 {pat + 1}, 神经元 {n + 1}: "
                      f"HR={HR[pat, n]:.3f}, FA={FA[pat, n]:.2f}, "
                      f"潜伏期={final_latency[pat, n] * 1000:.1f}ms")

    # ── 绘图 ──
    print("\n[5/5] 绘图...")
    plot_latency_matrix(latency, HR, FA, params,
                        filename=os.path.join(FIGURE_DIR, "exp4_multipattern.png"))

    # 保存结果
    results = {
        'params': params,
        'latency': latency,
        'HR': HR,
        'FA': FA,
        'final_latency': final_latency,
        'success_mask': success_mask,
        'firing_counts': [int(n.nFiring) for n in sim.neurons],
        'weights': [n.weight.copy() for n in sim.neurons],
    }

    save_path = os.path.join(FIGURE_DIR, "exp4_results.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)

    print("\n实验4 完成。")
    return results


if __name__ == "__main__":
    run_experiment4()
