"""
Brian2 STDP训练 — Diehl & Cook (2015)
Apple Silicon优化版 (M5 Pro / 15核 / Cython编译)
用法: python run_brian.py
"""

import sys, os, time, numpy as np
from brian2 import prefs

# ============================================================
# 🚀 优化1: Cython编译后端 (比numpy快50-100倍)
# ============================================================
prefs.codegen.target = 'numpy'  # M5 Pro Cython有兼容问题

# 🚀 优化2: 持久化编译缓存 (避免每次重新编译)
brian_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'stdp-mnist')
cache_dir = os.path.join(brian_dir, '.brian_cython_cache')
os.makedirs(cache_dir, exist_ok=True)
prefs.codegen.runtime.cython.cache_dir = cache_dir

# 🚀 优化3: 减少Brian2内部日志开销
prefs.logging.std_redirection = False

# 配置路径
os.chdir(brian_dir)
sys.path.insert(0, brian_dir)

import Diehl_Cook_2015_brian2 as dc
from brian2 import device, second

# ============================================================
# 参数配置 (与论文一致)
# ============================================================
dc.MNIST_PATH = dc.Path('../data')
dc.N_NEURONS = 400        # 兴奋性神经元数
dc.N_TRAIN = 60_000       # 训练样本数
dc.N_OBSERVE = 2_000      # 标签分配样本数
dc.N_TEST = 10_000        # 测试样本数

out_dir = os.path.join(brian_dir, 'data/brian_output')
os.makedirs(out_dir, exist_ok=True)
dc.DATA_PATH = dc.Path(out_dir)

print("=" * 60)
print("Brian2 STDP — Diehl & Cook (2015)")
print(f"400神经元 × 60K样本 | Cython编译 | Apple M5 Pro")
print(f"缓存: {cache_dir}")
print(f"输出: {out_dir}")
print("=" * 60)

# ============================================================
# 🚀 优化4: 预热编译 + 基准测试
# ============================================================
print("\n[预热] 编译Cython模块 (仅首次需要)...")
t_warm = time.time()
X_train, Y_train = dc.read_mnist(True)
net = dc.build_network(True)
net.run(0 * second)  # 触编译
print(f"  编译耗时: {time.time()-t_warm:.1f}s")

# 基准测试: 跑3个样本取平均
print("[基准] 测量单样本仿真时间...")
sample_times = []
for i in range(3):
    t0 = time.time()
    dc.normalize_plastic_weights(net['inp_exc'])
    dc.show_sample(net, X_train[i % len(X_train)], dc.INTENSITY)
    sample_times.append(time.time() - t0)

avg_sample = np.mean(sample_times)
train_est = avg_sample * dc.N_TRAIN
observe_est = avg_sample * dc.N_OBSERVE
test_est = avg_sample * dc.N_TEST * 0.6  # 测试更快 (无STDP)

print(f"  单样本: {avg_sample:.3f}s")
print(f"  预计训练: {train_est/3600:.1f}h | 标签分配: {observe_est/60:.0f}min | 测试: {test_est/60:.0f}min")
print(f"  总计约: {(train_est + observe_est + test_est)/3600:.1f}小时")

# ============================================================
# 阶段1: 训练 (无监督STDP)
# ============================================================
print("\n" + "=" * 60)
print("[阶段1] 训练 (无监督STDP, Cython加速)...")
print("=" * 60)

# 重新构建网络 (预热时的网络状态已被show_sample改变)
net = dc.build_network(True)
n_samples = X_train.shape[0]
rows = [dc.stats(net) + [-1]]
w_hist = [np.array(net['inp_exc'].w)]

t0 = time.time()
ratio = max(dc.N_TRAIN // dc.N_SAVE_POINTS, 1)

for i in range(dc.N_TRAIN):
    ix = i % n_samples
    dc.normalize_plastic_weights(net['inp_exc'])
    dc.show_sample(net, X_train[ix], dc.INTENSITY)
    rows.append(dc.stats(net) + [Y_train[ix]])
    if i % ratio == 0:
        w_hist.append(np.array(net['inp_exc'].w))

    # 进度显示 (每500步或最后一步)
    if i % 500 == 0 or i == dc.N_TRAIN - 1:
        elapsed = time.time() - t0
        pct = (i + 1) / dc.N_TRAIN * 100
        eta = elapsed / (i + 1) * (dc.N_TRAIN - i - 1)
        sps = (i + 1) / elapsed  # samples per second
        print(f"  [{i+1:6d}/{dc.N_TRAIN}] {pct:5.1f}% | "
              f"耗时: {elapsed/60:.0f}min | 预计剩余: {eta/60:.0f}min | "
              f"速度: {sps:.1f}样本/s")

train_time = time.time() - t0
print(f"\n  训练完成! 总耗时: {train_time/60:.1f}min ({train_time/3600:.1f}h)")

dc.save_npy(rows, dc.DATA_PATH / 'train_stats.npy')
dc.save_npy(w_hist, dc.DATA_PATH / 'train_w_hist.npy')
dc.save_npy(net['inp_exc'].w, dc.DATA_PATH / 'weights.npy')
dc.save_npy(net['exc'].theta, dc.DATA_PATH / 'theta.npy')

# ============================================================
# 阶段2: 标签分配
# ============================================================
print("\n" + "=" * 60)
print("[阶段2] 标签分配...")
print("=" * 60)

t0 = time.time()
dc.observe()
print(f"  耗时: {(time.time()-t0)/60:.1f}min")

# ============================================================
# 阶段3: 测试
# ============================================================
print("\n" + "=" * 60)
print("[阶段3] 测试...")
print("=" * 60)

t0 = time.time()
dc.test()
test_time = time.time() - t0
print(f"  耗时: {test_time/60:.1f}min")

print(f"\n全部完成! 结果保存在: {out_dir}")
print(f"总耗时: {(train_time + test_time)/3600:.1f}小时")
