"""
STDP完整训练 — 基于 Diehl & Cook (2015) 的无监督 STDP 学习, 180K 样本

分三阶段:
  1. train() — STDP 无监督训练, 学习输入模式的聚类中心 (感受野)
  2. observe() — 对5000个训练样本做前向推理, 投票为每个神经元分配数字标签
  3. test() — 对10000个测试样本推理, 生成混淆矩阵 confusion.npy

输出文件 (保存在 data/stdp_full/):
  weights.npy  — (1600, 784) 输入→兴奋权重矩阵 (即感受野)
  assign.npy   — (1600,) 每个神经元的类别标签
  theta.npy    — (1600,) 自适应阈值
  confusion.npy — (10, 10) 混淆矩阵
"""
import os, sys, time, numpy as np
os.environ['CFLAGS'] = '-O3'

# ---- 路径配置 ----
# 切换到 stdp-mnist 目录, 因为 Diehl_Cook_2015_brian2 使用相对路径
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'stdp-mnist')
sys.path.insert(0, BASE)
os.chdir(BASE)

# ---- Brian2 配置: C++代码生成 + O3优化 ----
from brian2 import prefs, seed as bseed
prefs.codegen.cpp.extra_compile_args_gcc = ['-O3', '-ffast-math']
prefs.codegen.target = 'cython'

import Diehl_Cook_2015_brian2 as dc

# ---- 超参数配置 ----
dc.MNIST_PATH = dc.Path('../data')          # MNIST原始数据路径
dc.DATA_PATH = dc.Path('data/stdp_full')    # 训练结果输出路径
dc.DATA_PATH.mkdir(parents=True, exist_ok=True)
dc.N_NEURONS = 1600                        # 兴奋神经元总数 (每类约160个)
dc.N_TRAIN = 180000                        # 训练样本数 (60K × 3轮)
dc.N_OBSERVE = 5000                        # 标签分配用样本数
dc.N_TEST = 10000                          # 测试样本数
dc.SEED = 42                               # 全局随机种子
bseed(42); np.random.seed(42)

print("=" * 50)
print("STDP完整训练: 1600神经元 × 180K样本")
print("输出:", dc.DATA_PATH)
print("=" * 50)

# ---- 阶段1: STDP无监督训练 ----
# 每个输入样本: 泊松编码 → LIF仿真 → STDP权重更新
# 输出: weights.npy (1600个28×28感受野)
t0 = time.time()
dc.train()
print(f"训练: {(time.time()-t0)/60:.0f}min")

# ---- 阶段2: 标签分配 (observe) ----
# 对N_OBSERVE个训练样本做前向推理, 统计每个神经元对各类别的响应,
# 投票决定每个神经元的类别归属, 输出 assign.npy
t0 = time.time()
dc.observe()
print(f"标签: {(time.time()-t0)/60:.0f}min")

# ---- 阶段3: 测试 ----
# 对N_TEST个测试样本推理, 生成混淆矩阵 confusion.npy
t0 = time.time()
dc.test()
print(f"测试: {(time.time()-t0)/60:.0f}min")
print("完成!")
