import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def add_wave(response, wave, start_time):
    """叠加单个EPSP/IPSP波形到总响应"""
    for k, value in enumerate(wave):
        if start_time + k < len(response):
            response[start_time + k] += value


def calculate_response(inh_times, epsp, ipsp, threshold, T=20, exc_times=[2, 6, 7, 12, 13]):
    """通用计算函数：输入实验参数，返回时间轴、总响应、输出、兴奋/抑制时刻"""
    time = np.arange(T)
    response = np.zeros(T)

    # 叠加所有兴奋输入
    for t0 in exc_times:
        add_wave(response, epsp, t0)
    # 叠加所有抑制输入
    for t0 in inh_times:
        add_wave(response, ipsp, t0)

    exc_idx = np.array(exc_times)
    inh_idx = np.array(inh_times)
    output = (response >= threshold).astype(int)

    return time, response, output, exc_idx, inh_idx


# ===================== 定义4组实验条件 =====================
# 条件1：原始参数（基准组）
cond1_params = {
    "inh_times": [13], "epsp": np.array([0.8, 0.4, 0.2]),
    "ipsp": np.array([-0.7, -0.4, -0.2]), "threshold": 1.2,
    "title": "原始条件"
}

# 条件2：删除抑制输入
cond2_params = {
    "inh_times": [], "epsp": np.array([0.8, 0.4, 0.2]),
    "ipsp": np.array([-0.7, -0.4, -0.2]), "threshold": 1.2,
    "title": "删除抑制输入"
}

# 条件3：增大EPSP幅值（兴奋作用增强）
cond3_params = {
    "inh_times": [13], "epsp": np.array([1.2, 0.6, 0.3]),  # 幅值提升50%
    "ipsp": np.array([-0.7, -0.4, -0.2]), "threshold": 1.2,
    "title": "增大EPSP幅值"
}

# 条件4：增大IPSP绝对值（抑制作用增强）
cond4_params = {
    "inh_times": [13], "epsp": np.array([0.8, 0.4, 0.2]),
    "ipsp": np.array([-1.0, -0.6, -0.3]),  # 绝对值提升~40%
    "threshold": 1.2,
    "title": "增大IPSP绝对值"
}

# 汇总所有实验条件
all_conditions = [cond1_params, cond2_params, cond3_params, cond4_params]

# ===================== 绘制整合对比图 =====================
# 创建2行2列的子图，统一设置大图尺寸
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
axes = axes.flatten()  # 展平轴数组，方便循环

# 循环绘制每个实验条件的结果
for idx, cond in enumerate(all_conditions):
    # 计算当前条件的响应
    time, response, output, exc_idx, inh_idx = calculate_response(
        cond["inh_times"], cond["epsp"], cond["ipsp"], cond["threshold"]
    )

    # 绘制子图核心内容
    ax = axes[idx]
    # 1. 总响应曲线
    ax.plot(time, response, marker='o', color='tab:purple', label='总响应')
    # 2. 阈值线
    ax.axhline(cond["threshold"], color='tab:green', linestyle='--', label=f'阈值={cond["threshold"]}')
    # 3. 二值输出
    ax.step(time, output, where='mid', color='tab:orange', label='二值输出')

    # 统一标注所有组的兴奋/抑制时刻（无特殊强化）
    for t in exc_idx:
        ax.text(t, response[t] + 0.1, '兴奋', fontsize=8, color='tab:blue', ha='center')
    for t in inh_idx:
        ax.text(t, response[t] - 0.1, '抑制', fontsize=8, color='tab:red', ha='center')

    # 通用格式设置
    ax.set_title(cond["title"], fontsize=12, fontweight='bold')
    ax.set_ylabel('膜电位/响应值' if idx in [0, 1] else '', fontsize=10)
    ax.set_xlabel('时间步 (time step)' if idx in [2, 3] else '', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

# 全局优化
plt.ylim(-0.5, 1.8)
fig.suptitle('EPSP/IPSP', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()