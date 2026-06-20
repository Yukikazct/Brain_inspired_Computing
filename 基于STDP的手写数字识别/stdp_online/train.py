"""STDP训练 + MLP读出层 — Diehl & Cook (2015) 三阶段流水线。

Phase 1: STDP无监督训练 (Brian2仿真)
Phase 2: 标签分配 (赢家频率法)
Phase 3: MLP监督训练 (STDP特征 → 类别)

用法:
  python train.py              # 训练 + 测试
  python train.py --test       # 仅测试
"""

import os, sys, time, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from collections import defaultdict
from random import randrange, seed as rseed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from parameters import Params
from network import (build_network, read_mnist, show_sample,
                     normalize_plastic_weights, save_npy)


# ═══════════════════════════════════════════════════════════════
# MLP读出层 (监督学习)
# ═══════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """STDP特征 → 类别概率: 400 → 256 → ReLU → Dropout → 10。"""
    def __init__(self, n_in=400, n_hidden=256, n_out=10, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(n_hidden, n_out))

    def forward(self, x): return self.net(x)


def train_mlp(features, labels, n_epochs=300, lr=0.001, wd=1e-4):
    """训练MLP读出层。features: (N,400) firing rate特征。"""
    model = MLP(n_in=features.shape[1])
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(n_epochs):
        opt.zero_grad()
        loss = loss_fn(model(features), labels); loss.backward(); opt.step()
        if (epoch+1) % 50 == 0:
            acc = (model(features).argmax(1) == labels).float().mean()
            print(f"  epoch {epoch+1}: loss={loss.item():.4f} acc={acc:.2%}")
    return model


# ═══════════════════════════════════════════════════════════════
# 三阶段训练流水线
# ═══════════════════════════════════════════════════════════════

def train(p):
    """Phase 1: STDP无监督训练。全程不使用标签。"""
    X, Y = read_mnist(True, p.mnist_path)
    net = build_network(p, training=True)
    t0 = time.time()
    print(f"训练 {p.n_train} 样本...")
    for i in range(p.n_train):
        normalize_plastic_weights(net['inp_exc'])
        show_sample(net, X[i % len(X)], p.intensity)
        if (i+1) % 5000 == 0:
            e = time.time()-t0
            print(f"  [{i+1}/{p.n_train}] {e/60:.0f}min ETA:{e/(i+1)*(p.n_train-i-1)/60:.0f}min")
    print(f"训练完成: {(time.time()-t0)/60:.0f}min")
    save_npy(net['inp_exc'].w, p.data_path/'weights.npy')
    save_npy(np.array(net['exc'].theta), p.data_path/'theta.npy')


def observe(p):
    """Phase 2: 标签分配 (赢家频率法)。训练后唯一使用标签的阶段。"""
    X, Y = read_mnist(True, p.mnist_path)
    net = build_network(p, training=False)
    responses = defaultdict(list)
    for i in range(p.n_observe):
        exc = show_sample(net, X[i % len(X)], p.intensity)
        responses[Y[i % len(Y)]].append(exc)
    res = np.zeros((10, p.n_exc))
    for cls, vals in responses.items():
        res[cls] = np.array(vals).mean(axis=0)
    assign = np.argmax(res, axis=0)
    save_npy(assign, p.data_path/'assign.npy')
    for c in range(10): print(f"  类别{c}: {(assign==c).sum()}个神经元")


def test(p):
    """Phase 3a: 纯STDP无监督投票测试。输出混淆矩阵。"""
    assign = np.load(p.data_path/'assign.npy')
    groups = [np.where(assign == i)[0] for i in range(10)]
    X, Y = read_mnist(False, p.mnist_path)
    net = build_network(p, training=False)
    conf = np.zeros((10,10))
    for i in range(p.n_test):
        exc = show_sample(net, X[i % len(X)], p.intensity)
        guess = np.argmax([exc[grp].mean() for grp in groups])
        conf[Y[i % len(Y)], guess] += 1
    save_npy(conf, p.data_path/'confusion.npy')
    acc = np.trace(conf)/np.sum(conf)
    print(f"\n★ 纯STDP (无监督投票): {acc:.2%}")
    for c in range(10):
        a = conf[c,c]/conf[c].sum() if conf[c].sum()>0 else 0
        print(f"  {c}: {a:.1%}")


def extract_features(net, X, p):
    """对每张图做前向推理, 收集各神经元发放计数作为STDP特征。"""
    feats = []
    for i in range(len(X)):
        pat = show_sample(net, X[i], p.intensity)
        feats.append(torch.from_numpy(pat.astype(np.float32)))
    return torch.stack(feats)


# ═══════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.chdir(HERE)
    p = Params()
    p.data_path.mkdir(parents=True, exist_ok=True)
    from brian2 import seed as bseed
    bseed(p.seed); rseed(p.seed); np.random.seed(p.seed); torch.manual_seed(p.seed)

    print("=" * 50)
    print("Brian2 STDP + MLP — Diehl & Cook (2015)")
    print(f"{p.n_exc}神经元 × {p.n_train}样本")
    print("=" * 50)

    if "--test" in sys.argv:
        test(p)
    else:
        # Phase 1-2: STDP + 标签分配
        train(p); observe(p)

        # Phase 3a: 纯STDP投票
        test(p)

        # Phase 3b: MLP读出层
        print("\nPhase 3: STDP特征 + MLP训练...")
        Xt, Yt = read_mnist(True, p.mnist_path)
        Xv, Yv = read_mnist(False, p.mnist_path)
        net = build_network(p, training=False)

        N_FEAT = 10000
        print(f"提取训练特征({N_FEAT}样本)...")
        F_train = extract_features(net, Xt[:N_FEAT], p)
        L_train = torch.tensor(Yt[:N_FEAT], dtype=torch.long)

        N_TEST = 2000
        print(f"提取测试特征({N_TEST}样本)...")
        F_test = extract_features(net, Xv[:N_TEST], p)
        L_test = torch.tensor(Yv[:N_TEST], dtype=torch.long)

        mlp = train_mlp(F_train, L_train)
        torch.save(mlp.state_dict(), p.data_path/'mlp.pth')

        with torch.no_grad():
            test_preds = mlp(F_test).argmax(1)
            test_acc = (test_preds == L_test).float().mean()

        # 保存MLP混淆矩阵
        mlp_conf = np.zeros((10,10), dtype=int)
        for i in range(len(L_test)):
            mlp_conf[int(L_test[i]), int(test_preds[i])] += 1
        np.save(p.data_path/'mlp_confusion.npy', mlp_conf)

        print(f"\n★ STDP+MLP: {test_acc:.2%} ({N_TEST}样本)")
        for c in range(10):
            mask = L_test == c
            if mask.sum()>0:
                a = (test_preds[mask]==c).float().mean()
                print(f"  {c}: {a:.1%}")
