"""STDP测试 — 加载MLP模型，秒出结果。"""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from readout import MLP

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
model_path = os.path.join(DATA, 'mlp_stdp.pth')

if not os.path.exists(model_path):
    print("❌ 先运行 train.py 训练"); sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stdp_hebbian'))
from utils import load_mnist, visualize_weights, visualize_label_responses
from model import Network, Params

# 加载MLP
mlp = MLP()
mlp.load_state_dict(torch.load(model_path, map_location='cpu'))
mlp.eval()

# 加载STDP模型(用于可视化)
net_path = os.path.join(DATA, 'model_stdp.npz')
if os.path.exists(net_path):
    d = np.load(net_path, allow_pickle=True)
    net = Network.load(net_path)

    # 提取测试特征并评估
    _, _, X_t, Y_t = load_mnist()
    feats = []
    for i in range(len(X_t)):
        c = net.forward(X_t[i], train=False, seed=900000+i)
        feats.append(torch.from_numpy(c.astype(np.float32)))
    X = torch.stack(feats)
    Y = torch.tensor(Y_t)

    with torch.no_grad():
        preds = mlp(X).argmax(dim=1)
    acc = (preds == Y).float().mean().item()
    print(f"★ STDP+MLP 准确率: {acc:.2%}")

    # 可视化
    assign = net.labels
    idxs = []
    for c in range(10):
        c_n = np.where(assign == c)[0]
        if len(c_n) >= 2: idxs.extend(c_n[:2])
    if idxs:
        visualize_weights(net.w, idxs, save_path=os.path.join(os.path.dirname(__file__), 'receptive_fields_stdp.png'))
    spc = net.per_class if net.per_class is not None else np.zeros((10, 400))
    visualize_label_responses(spc, assign, save_path=os.path.join(os.path.dirname(__file__), 'neuron_assignment_stdp.png'))
    print("图片已更新")
