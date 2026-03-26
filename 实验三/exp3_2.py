import numpy as np
import matplotlib.pyplot as plt
# TODO1：补全LIF膜电位离散更新公式
def lif_step(v_prev, input_current, tau_m, dt, R=1.0, v_rest=0.0):
    v_next = v_prev + dt / tau_m * (-(v_prev - v_rest) + R * input_current)
    return v_next

# TODO2：补全阈值判断、脉冲发放与膜电位复位逻辑
def simulate_lif_constant(current_value, tau_m=10.0, v_rest=0.0, v_reset=0.0,
                          v_th=15.0, R=1.0, dt=0.05, time_window=100.0):
    time = np.arange(0, time_window + dt, dt)
    current = np.full_like(time, current_value)
    voltage = np.zeros_like(time)
    spikes = np.zeros_like(time)
    v = v_rest
    for i in range(1, len(time)):
        v = lif_step(v, current[i], tau_m, dt, R, v_rest)
        if v >= v_th:
            spikes[i] = 1  # 标记脉冲发放
            voltage[i] = v_th  # 发放时显示为阈值电位
            v = v_reset  # 膜电位立即复位
        else:
            voltage[i] = v  # 未达阈值，保留更新后膜电位
    return time, current, voltage, spikes

# 本实验要求修改的参数
I_const = 20.0 # 恒定输⼊电流⼤⼩
tau_m = 20.0 # 膜时间常数，决定膜电位变化快慢
V_th = 15.0 # 发放阈值
R = 1.0 # 膜电阻
dt = 0.05 # 数值仿真的时间步长（ ms）
# 其余默认参数（⼀般不需要修改）
V_rest = 0.0 # 静息电位
V_reset = 0.0 # 发放后的复位电位
time_window = 100.0 # 总仿真时长（ ms）
time, current, voltage, spikes = simulate_lif_constant(
current_value=I_const,
tau_m=tau_m,
v_rest=V_rest,
v_reset=V_reset,
v_th=V_th,
R=R,
dt=dt,
time_window=time_window,
)
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
axes[0].plot(time, current, color="tab:orange")
axes[0].set_title("Constant Input Current")
axes[0].set_ylabel("I(t)")
axes[0].grid(alpha=0.3)
axes[1].plot(time, voltage, color="tab:blue", label="membrane potential")
axes[1].axhline(V_th, color="tab:red", linestyle="--", label="threshold")
axes[1].set_title("LIF Membrane Potential under Constant Input")
axes[1].set_xlabel("Time (ms)")
axes[1].set_ylabel("V_m (mV)")
axes[1].grid(alpha=0.3)
axes[1].legend()
plt.tight_layout()
plt.show()