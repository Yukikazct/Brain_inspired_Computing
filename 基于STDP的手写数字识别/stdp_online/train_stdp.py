"""
STDP完整训练 — Diehl & Cook (2015) 无监督 STDP 学习

三阶段:
  1. train()   — STDP 无监督训练
  2. observe() — 标签分配
  3. test()    — 推理 + 混淆矩阵

输出 (data/stdp_full/):
  weights.npy / assign.npy / theta.npy / confusion.npy
"""
import os, time, numpy as np
from pathlib import Path
os.environ['CFLAGS'] = '-O3'

# ---- 路径 ----
HERE = Path(__file__).resolve().parent

# ---- Brian2 ----
from brian2 import prefs, seed as bseed
prefs.codegen.cpp.extra_compile_args_gcc = ['-O3', '-ffast-math']
prefs.codegen.target = 'cython'

import stdp_model as dc

# ---- 超参数 ----
dc.MNIST_PATH = HERE / '..' / 'data'
dc.DATA_PATH = HERE / 'data' / 'stdp_full'
dc.DATA_PATH.mkdir(parents=True, exist_ok=True)
dc.N_NEURONS = 1200
dc.N_TRAIN = 180000
dc.N_OBSERVE = 5000
dc.N_TEST = 10000
dc.SEED = 42
bseed(42); np.random.seed(42)

print("=" * 50)
print("STDP: 1200神经元 × 180K样本")
print("=" * 50)

# ---- 阶段1: 训练 ----
t0 = time.time()
dc.train()
print(f"训练: {(time.time() - t0) / 60:.0f}min")

# ---- 阶段2: 标签分配 ----
t0 = time.time()
dc.observe()
print(f"标签: {(time.time() - t0) / 60:.0f}min")

# ---- 阶段3: 测试 ----
t0 = time.time()
dc.test()
print(f"测试: {(time.time() - t0) / 60:.0f}min")
print("完成!")
