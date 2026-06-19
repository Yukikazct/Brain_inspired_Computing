"""STDP训练 + MLP读出层 — 三阶段训练。

Phase 1: STDP无监督特征学习
Phase 2: 提取firing rate特征
Phase 3: MLP监督分类训练
"""
import sys, os, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Params, Network
from readout import MLP, train_mlp, evaluate_mlp

# ── 超参数 (对齐94.37%仓库) ──
p = Params(
    n_exc=400, n_train=60000, n_epochs=1, n_label=30000, seed=42,
    duration=350.0, max_rate=200.0,
    nu_pre=0.005, nu_post=0.00525,          # 大学习率, 快速收敛
    tau_pre=20.0, tau_post1=20.0, tau_post2=40.0,
    theta_plus=0.05, tc_theta=1e7, theta_offset=20.0,
    inh_strength=17.0, target_sum=78.0,
)

np.random.seed(42); torch.manual_seed(42)
DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stdp_hebbian'))
from utils import load_mnist

print("加载 MNIST...")
train_img, train_lbl, test_img, test_lbl = load_mnist()

# ═══════════════════════════════════════════════════
# Phase 1: STDP 无监督训练
# ═══════════════════════════════════════════════════
net = Network(p)
print(f"\nPhase 1: STDP ({p.n_exc}神经元 × {p.n_train}样本)")
t0 = time.time()
for i in range(p.n_train):
    for retry in range(5):
        counts = net.train_one(train_img[i], seed=i*100+retry)
        if counts.sum() >= p.min_spikes or retry == 4: break
        net.intensity = min(net.intensity + 1.0, 10.0)
    if counts.sum() >= p.min_spikes:
        net.intensity = max(p.start_intensity, net.intensity - 0.5)
    if (i+1) % 5000 == 0:
        print(f"  [{i+1}/{p.n_train}] {time.time()-t0:.0f}s")
print(f"Phase 1完成: {(time.time()-t0)/60:.1f}min")

# ═══════════════════════════════════════════════════
# Phase 2: 提取firing rate特征
# ═══════════════════════════════════════════════════
print(f"\nPhase 2: 提取特征...")
def extract(images, labels, indices):
    feats, labs = [], []
    for idx in indices:
        c = net.forward(images[idx], train=False, seed=900000+idx)
        feats.append(torch.from_numpy(c.astype(np.float32)))
        labs.append(int(labels[idx]))
    return torch.stack(feats), torch.tensor(labs)

n_feat = min(p.n_train, len(train_img))
train_idx = np.random.choice(len(train_img), n_feat, replace=False)
t0 = time.time()
X_train, y_train = extract(train_img, train_lbl, train_idx)
print(f"  训练特征: {X_train.shape} ({(time.time()-t0)/60:.1f}min)")

t0 = time.time()
test_idx = np.arange(len(test_img))
X_test, y_test = extract(test_img, test_lbl, test_idx)
print(f"  测试特征: {X_test.shape} ({(time.time()-t0)/60:.1f}min)")

# ═══════════════════════════════════════════════════
# Phase 3: MLP 监督训练
# ═══════════════════════════════════════════════════
print(f"\nPhase 3: MLP (400 → 256 → ReLU → Dropout → 10)")
mlp = train_mlp(X_train, y_train, n_epochs=300, lr=0.001, wd=1e-4)

# ═══════════════════════════════════════════════════
# 评估
# ═══════════════════════════════════════════════════
train_acc = evaluate_mlp(mlp, X_train, y_train)
test_acc = evaluate_mlp(mlp, X_test, y_test)
print(f"\n★ 训练准确率: {train_acc:.2%}")
print(f"★ 测试准确率: {test_acc:.2%}")

# 各类别
with torch.no_grad():
    preds = mlp(X_test).argmax(dim=1)
print("各类别:")
for c in range(10):
    mask = y_test == c
    if mask.sum() > 0:
        a = (preds[mask] == c).float().mean().item()
        print(f"  {c}: {a:.1%}")

# 保存
torch.save(mlp.state_dict(), os.path.join(DATA, 'mlp_stdp.pth'))
net.save(os.path.join(DATA, 'model_stdp.npz'))
print(f"\n保存: {DATA}/model_stdp.npz, mlp_stdp.pth")
