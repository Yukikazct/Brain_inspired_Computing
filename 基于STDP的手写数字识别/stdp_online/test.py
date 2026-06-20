"""STDP 测试 — 秒出结果 (纯STDP投票 + MLP读出层)"""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import MLP

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data', 'stdp_full')

# ── 纯STDP无监督投票 ──
conf = np.load(os.path.join(DATA, 'confusion.npy'))
acc = np.trace(conf) / np.sum(conf) * 100
print(f"★ 纯STDP (无监督投票): {acc:.2f}%")
for i in range(10):
    a = conf[i,i] / conf[i].sum() * 100 if conf[i].sum() > 0 else 0
    print(f"  {i}: {a:.1f}%")

# ── STDP+MLP读出层 ──
mlp_path = os.path.join(DATA, 'mlp.pth')
if os.path.exists(mlp_path):
    mlp = MLP(); mlp.load_state_dict(torch.load(mlp_path, map_location='cpu')); mlp.eval()
    print(f"\n★ STDP+MLP: 模型已加载 ({mlp_path})")
    print("  (运行 train.py 查看完整MLP准确率)")
else:
    print(f"\n⚠️ MLP模型未训练, 运行 train.py 生成")

# ── 可视化 ──
assign_path = os.path.join(DATA, 'assign.npy')
weights_path = os.path.join(DATA, 'weights.npy')
if os.path.exists(assign_path) and os.path.exists(weights_path):
    assign = np.load(assign_path)
    w = np.load(weights_path).reshape(len(assign), -1)
    sys.path.insert(0, os.path.join(HERE, '..', 'stdp_hebbian'))
    from utils import visualize_weights, visualize_label_responses
    idxs = []
    for c in range(10):
        cn = np.where(assign == c)[0]
        if len(cn) >= 2: idxs.extend(cn[:2])
    if idxs: visualize_weights(w, idxs, save_path=os.path.join(HERE, 'receptive_fields_stdp.png'))
    spc = np.zeros((10, len(assign)))
    for c in range(10): spc[c, assign == c] = 1
    visualize_label_responses(spc, assign, save_path=os.path.join(HERE, 'neuron_assignment_stdp.png'))
    print("图片已更新")
