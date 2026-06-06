"""STDP完整训练 — 追清晰感受野 (180K样本)"""
import os, sys, time, numpy as np
os.environ['CFLAGS'] = '-O3'

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'stdp-mnist')
sys.path.insert(0, BASE)
os.chdir(BASE)

from brian2 import prefs, seed as bseed
prefs.codegen.cpp.extra_compile_args_gcc = ['-O3', '-ffast-math']
prefs.codegen.target = 'cython'

import Diehl_Cook_2015_brian2 as dc

# 配置
dc.MNIST_PATH = dc.Path('../data')
dc.DATA_PATH = dc.Path('data/stdp_full')
dc.DATA_PATH.mkdir(parents=True, exist_ok=True)
dc.N_NEURONS = 1600
dc.N_TRAIN = 180000       # 3轮 × 60K = 180K
dc.N_OBSERVE = 5000
dc.N_TEST = 10000
dc.SEED = 42
bseed(42); np.random.seed(42)

print("=" * 50)
print("STDP完整训练: 1600神经元 × 180K样本")
print("输出:", dc.DATA_PATH)
print("=" * 50)

# 训练
t0 = time.time()
dc.train()
print(f"训练: {(time.time()-t0)/60:.0f}min")

# 标签
t0 = time.time()
dc.observe()
print(f"标签: {(time.time()-t0)/60:.0f}min")

# 测试
t0 = time.time()
dc.test()
print(f"测试: {(time.time()-t0)/60:.0f}min")
print("完成!")
