import numpy as np
import matplotlib.pyplot as plt
# HH 模型常⽤参数（单位与经典模型⼀致）
C_m = 1.0
g_Na = 120.0
g_K = 36.0
g_L = 0.3
E_Na = 50.0
E_K = -77.0
E_L = -54.387
V_rest = -65.0
# n 门控变量的开启速率
def alpha_n(V):
    x = V + 55.0
# 当分母接近 0 时，直接使⽤极限值，避免数值不稳定
    return 0.01 * x / (1 - np.exp(-x / 10)) if abs(x) > 1e-6 else 0.1
# n 门控变量的关闭速率
def beta_n(V):
    return 0.125 * np.exp(-(V + 65.0) / 80)
# m 门控变量的开启速率
def alpha_m(V):
    x = V + 40.0
    return 0.1 * x / (1 - np.exp(-x / 10)) if abs(x) > 1e-6 else 1.0
# m 门控变量的关闭速率
def beta_m(V):
    return 4.0 * np.exp(-(V + 65.0) / 18)
# h 门控变量的开启速率
def alpha_h(V):
    return 0.07 * np.exp(-(V + 65.0) / 20)
# h 门控变量的关闭速率
def beta_h(V):
    return 1.0 / (1 + np.exp(-(V + 35.0) / 10))
# 定义外加电流随时间的变化⽅式
def external_current(t, amplitude=10.0, start=10.0, end=40.0):
# 在指定时间窗内施加⼀个矩形电流脉冲
    return amplitude if start <= t <= end else 0.0
# 本实验要求修改的参数
stim_amplitude = 5.0 # 外加电流幅值，决定刺激强弱
stim_start = 10.0 # 外加电流开始时刻（ ms）
stim_end = 40.0 # 外加电流结束时刻（ ms）
dt = 0.01 # 数值积分时间步长（ ms）
# 其余默认参数（⼀般不需要修改）
time_window = 60.0 # 总仿真时长（ ms）
# 在静息电位处，使⽤稳态值作为初始门控变量
m0 = alpha_m(V_rest) / (alpha_m(V_rest) + beta_m(V_rest))
h0 = alpha_h(V_rest) / (alpha_h(V_rest) + beta_h(V_rest))
n0 = alpha_n(V_rest) / (alpha_n(V_rest) + beta_n(V_rest))
time = np.arange(0, time_window + dt, dt)
current = np.array([external_current(t, stim_amplitude, stim_start, stim_end) for t in time])
voltage = np.zeros_like(time)
m_values = np.zeros_like(time)
h_values = np.zeros_like(time)
n_values = np.zeros_like(time)
# 设置初始状态
voltage[0] = V_rest
m_values[0] = m0
h_values[0] = h0
n_values[0] = n0
for i in range(1, len(time)):
    V_prev = voltage[i - 1]
m_prev = m_values[i - 1]
h_prev = h_values[i - 1]
n_prev = n_values[i - 1]
I_ext = current[i - 1]
I_Na = g_Na * (m_prev ** 3) * h_prev * (V_prev - E_Na)
I_K = g_K * (n_prev ** 4) * (V_prev - E_K)
I_L = g_L * (V_prev - E_L)
# 先计算各变量的导数
dVdt = (I_ext - I_Na - I_K - I_L) / C_m
dmdt = alpha_m(V_prev) * (1 - m_prev) - beta_m(V_prev) * m_prev
dhdt = alpha_h(V_prev) * (1 - h_prev) - beta_h(V_prev) * h_prev
dndt = alpha_n(V_prev) * (1 - n_prev) - beta_n(V_prev) * n_prev
# 再⽤欧拉法更新到下⼀时刻
voltage[i] = V_prev + dt * dVdt
m_values[i] = m_prev + dt * dmdt
h_values[i] = h_prev + dt * dhdt
n_values[i] = n_prev + dt * dndt
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
axes[0].plot(time, current, color="tab:orange")
axes[0].set_title("Input Current of HH Model")
axes[0].set_ylabel("I_ext")
axes[0].grid(alpha=0.3)
axes[1].plot(time, voltage, color="tab:blue")
axes[1].axhline(0.0, color="gray", linestyle="--", alpha=0.7, label="0 mV reference")
axes[1].set_title("Membrane Potential of HH Model")
axes[1].set_xlabel("Time (ms)")
axes[1].set_ylabel("V_m (mV)")
axes[1].grid(alpha=0.3)
axes[1].legend()
plt.tight_layout()
plt.show()