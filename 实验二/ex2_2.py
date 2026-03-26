import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS 全版本自带的通用中文字体
plt.rcParams['axes.unicode_minus'] = False              # 解决负号显示为方块

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def relu(x):
    return np.maximum(0.0, x)

def tanh(x):
    return np.tanh(x)

# 可自行修改的参数
weights = np.array([0.9, -0.5, 0.7])
inputs = np.array([1.0, 0.6, 1.2])
bias = -0.2

# 计算加权和
z = np.dot(inputs, weights) + bias

# 打印结果
print("------ 人工神经元实验 ------")
print("输入:", inputs)
print("权重:", weights)
print("偏置:", bias)
print("加权和 z =", round(z, 3))
print("sigmoid(z) =", round(sigmoid(z), 3))  # 简化写法，无需转数组
print("ReLU(z) =", round(relu(z), 3))
print("tanh(z) =", round(tanh(z), 3))

# 绘制激活函数曲线
x = np.linspace(-5, 5, 400)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Sigmoid 子图
axes[0].plot(x, sigmoid(x), color="tab:orange")
axes[0].set_title("Sigmoid 激活函数")  # 测试中文显示
axes[0].set_xlabel("输入值 z")
axes[0].set_ylabel("输出值")
axes[0].grid(alpha=0.3)

# ReLU 子图
axes[1].plot(x, relu(x), color="tab:green")
axes[1].set_title("ReLU 激活函数")
axes[1].set_xlabel("输入值 z")
axes[1].set_ylabel("输出值")
axes[1].grid(alpha=0.3)

# Tanh 子图
axes[2].plot(x, tanh(x), color="tab:red")
axes[2].set_title("Tanh 激活函数")
axes[2].set_xlabel("输入值 z")
axes[2].set_ylabel("输出值")
axes[2].grid(alpha=0.3)

# 总标题
fig.suptitle("不同激活函数的输出曲线（中文测试）", fontsize=14)

# 调整布局（避免标题重叠）
plt.tight_layout()
# 显示图像
plt.show()