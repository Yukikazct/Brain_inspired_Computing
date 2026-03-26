import numpy as np
import matplotlib.pyplot as plt

# 根据 LIF 动力学方程更新一个时间步的膜电位
def lif_step(v_prev, input_current, tau_m, dt, R=1.0, v_rest=0.0):
    return v_prev + dt / tau_m * (-(v_prev - v_rest) + R * input_current)

# 在恒定电流输入下仿真 LIF 神经元，并返回平均发放频率
def simulate_lif_constant(
    current_value,
    tau_m=10.0,
    v_rest=0.0,
    v_reset=0.0,
    v_th=15.0,
    R=1.0,
    dt=0.02,
    time_window=1000.0,
):
    time = np.arange(0, time_window + dt, dt)
    spikes = np.zeros_like(time)
    v = v_rest
    for i in range(1, len(time)):
        v = lif_step(v, current_value, tau_m, dt, R, v_rest)
        if v >= v_th:
            spikes[i] = 1
            v = v_reset
    # 这里把时间窗从 ms 转换为 s，再计算平均发放频率
    firing_rate = spikes.sum() / (time_window / 1000.0)
    return firing_rate

# 根据课上理论公式计算 LIF 神经元的理论发放频率
def theoretical_firing_rate(I, tau_m=10.0, v_th=15.0, R=1.0):
    # TODO 3：
    RI = R * I
    # 1. 判断是否达到发放阈值
    if RI <= v_th:
        return 0.0
    # 2. 计算发放周期 T
    T = tau_m * np.log(RI / (RI - v_th))
    # 3. 计算发放频率 f (Hz)
    f = 1000 / T
    return f

# 本实验要求修改的参数
tau_m = 10.0    # 膜时间常数
V_th = 15.0    # 发放阈值
R = 1.0        # 膜电阻
dt = 0.02      # 仿真时间步长（ms）
I_values = np.arange(10, 31, 1)  # 依次测试的一组输入电流值

# 其余默认参数
time_window = 1000.0  # 用于统计平均发放频率的总时长（ms）

theory_rates = []
sim_rates = []

for I in I_values:
    theory_rates.append(theoretical_firing_rate(I, tau_m=tau_m, v_th=V_th, R=R))
    sim_rates.append(
        simulate_lif_constant(
            current_value=I,
            tau_m=tau_m,
            v_th=V_th,
            R=R,
            dt=dt,
            time_window=time_window,
        )
    )

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(I_values, theory_rates, marker="o", label="theory")
plt.plot(I_values, sim_rates, marker="s", label="simulation")
plt.xlabel("Input Current I")
plt.ylabel("Firing Rate (Hz)")
plt.title("Theoretical vs Simulated Firing Rate of LIF")
plt.grid(alpha=0.3)
plt.legend()
plt.show()