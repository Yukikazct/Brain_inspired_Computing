"""
main.py
主程序入口

运行流程：
    1. 阶段1：主仿真（450 s）
       - 生成输入脉冲
       - 运行 SRM + STDP 仿真
       - 评估模式识别性能
       - 绘制图1（输入脉冲）、图2（膜电位）、图3（潜伏期）

    2. 阶段2：参数敏感性扫描
       - 扫描模式频率与初始权重
       - 每个参数值重复多次试验，统计成功率
       - 绘制图4（敏感性分析）

使用方式：
    python main.py

依赖：
    numpy, matplotlib
    config.py, input_generator.py, neuron_stdp.py, plotting.py
"""

import time
from config import T_sim, w_init, pattern_freq, N
from input_generator import generate_input_spikes
from neuron_stdp import simulate, evaluate_trial
from plotting import plot_fig1, plot_fig2, plot_fig3, plot_fig4_two_params


# =============================================================================
# 参数扫描辅助函数
# =============================================================================
def scan_parameter(param_name, param_values, n_trials=5, T_steps=200 * 1000):
    """
    对指定参数进行扫描，每个参数值运行 n_trials 次，统计成功率。

    参数
    ----
    param_name : str
        参数名称，用于控制生成逻辑。
    param_values : list[float]
        待扫描的参数值列表。
    n_trials : int
        每个参数值的重复试验次数。
    T_steps : int
        单次仿真时长（ms），扫描时通常缩短以节省计算时间。

    返回
    ----
    list[float] : 每个参数值对应的成功率（0~1）。
    """
    results = []
    print(f"\n>>> 扫描: {param_name} (每点 {n_trials} 次, 单次 {T_steps // 1000}s)")
    for val in param_values:
        n_succ = 0
        for trial in range(n_trials):
            # 构造与参数值相关的随机种子，保证可复现
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
                gen_kwargs['freq'] = pattern_freq * (1 - val)
            else:
                raise ValueError(f"未知参数: {param_name}")

            # 生成输入并运行仿真
            neuron_spikes, neuron_types, all_times, all_neurons, all_types, \
                time_start, pattern_intervals, _ = \
                generate_input_spikes(T_steps=T_steps, seed=trial_seed, **gen_kwargs)

            out_spikes, _, _, _ = simulate(
                neuron_spikes, all_times, all_neurons, all_types, time_start,
                pattern_intervals, record_windows=[],
                w_init_val=w_init_val, T_steps=T_steps
            )
            success, _, _, _ = evaluate_trial(
                out_spikes, pattern_intervals, T_steps=T_steps, eval_last_s=50
            )
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

    # 膜电位记录窗口（初期、中期、后期各 2 秒）
    record_windows = [
        (0, 2000),          # 初期 0–2 s
        (13000, 15000),     # 中期 13–15 s
        (448000, 450000)    # 后期 最后 2 s
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
    print("=" * 60)

    # 扫描时长和次数（缩短仿真时间以加速扫描）
    T_SCAN = 200 * 1000   # 每次仿真 200 秒
    N_TRIALS = 10         # 每个参数值重复 10 次

    # 参数1：模式出现频率
    freq_vals = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    t0 = time.time()
    succ_freq = scan_parameter('Pattern frequency', freq_vals,
                               n_trials=N_TRIALS, T_steps=T_SCAN)
    print(f"频率扫描耗时 {time.time() - t0:.1f} s")

    # 参数2：初始权重
    weight_vals = [0.30, 0.35, 0.40, 0.45, 0.50]
    t0 = time.time()
    succ_weight = scan_parameter('Initial weight', weight_vals,
                                 n_trials=N_TRIALS, T_steps=T_SCAN)
    print(f"权重扫描耗时 {time.time() - t0:.1f} s")

    # 绘制两个参数的敏感性图
    plot_fig4_two_params('Pattern frequency', freq_vals, succ_freq,
                         'Initial weight', weight_vals, succ_weight)

    print("\n[完成] 所有图表已生成！")
