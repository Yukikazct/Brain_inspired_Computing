"""STDP 训练脚本 — 在线三重STDP + WTA 无监督学习。"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Params, Network

# ── 超参数 ──
p = Params(
    n_exc=400, n_train=60000, n_epochs=3, n_label=30000, seed=42,
    duration=350.0, max_rate=63.75,
    nu_pre=0.0001, nu_post=0.01,
    tau_pre=20.0, tau_post1=20.0, tau_post2=40.0,
    theta_plus=0.05, tc_theta=1e7, theta_offset=20.0,
    inh_strength=17.0, target_sum=78.0,
)

# ── 加载数据 ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stdp_hebbian'))
from utils import load_mnist

print("加载 MNIST...")
train_img, train_lbl, test_img, test_lbl = load_mnist()
net = Network(p)

# ── 训练 ──
print(f"训练: {p.n_exc}神经元 × {p.n_train}样本 × {p.n_epochs}轮")
t0 = time.time(); total_spikes = 0
for epoch in range(p.n_epochs):
    for i in range(p.n_train):
        for retry in range(5):
            counts = net.train_one(train_img[i], seed=(epoch*100000+i)*100+retry)
            if counts.sum() >= p.min_spikes or retry == 4: break
            net.intensity = min(net.intensity + 1.0, 10.0)
        if counts.sum() >= p.min_spikes:
            net.intensity = max(p.start_intensity, net.intensity - 0.5)
        total_spikes += counts.sum()
        if (i+1) % 5000 == 0:
            e = time.time() - t0
            print(f"  [E{epoch+1}][{i+1}/{p.n_train}] {e:.0f}s")
print(f"训练完成: {(time.time()-t0)/60:.1f}min")

# ── 标签分配 ──
print("标签分配...")
n_label = min(p.n_label, len(train_img))
sc = np.zeros((n_label, p.n_exc))
for i in range(n_label):
    sc[i] = net.forward(train_img[i], train=False, seed=500000+i)
net.assign_labels(sc, train_lbl[:n_label])
for c in range(10): print(f"  类别{c}: {(net.labels==c).sum()}个")

# ── 测试 ──
print("测试...")
correct = 0; conf = np.zeros((10,10), dtype=int)
for i in range(len(test_img)):
    pred, _ = net.predict(test_img[i], top_k=10, seed=600000+i)
    if pred == test_lbl[i]: correct += 1
    conf[test_lbl[i], pred] += 1
acc = correct / len(test_img) * 100
print(f"★ 准确率: {acc:.2f}%")

# ── 保存 ──
DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
np.save(os.path.join(DATA, 'confusion_stdp.npy'), conf)
net.save(os.path.join(DATA, 'model_stdp.npz'))
print(f"已保存: {DATA}/model_stdp.npz, confusion_stdp.npy")
