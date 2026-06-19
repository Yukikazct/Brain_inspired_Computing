"""Hebbian模型测试 — 直接输出结果"""
import sys, os, numpy as np
from utils import visualize_weights, visualize_label_responses

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
conf_path = os.path.join(MODEL_DIR, "confusion_hebbian.npy")

if not os.path.exists(conf_path):
    print("❌ 混淆矩阵不存在, 先运行 main.py 训练"); sys.exit(1)

conf = np.load(conf_path)
acc = np.trace(conf) / np.sum(conf) * 100
print(f"★ Hebbian准确率: {acc:.2f}%")
for i in range(10):
    a = conf[i,i] / conf[i].sum() * 100 if conf[i].sum() > 0 else 0
    print(f"  {i}: {a:.1f}%")

# 可视化
npz_path = os.path.join(MODEL_DIR, "model_hebbian.npz")
if os.path.exists(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    assigned = data['assigned_labels']
    spc = data['spike_counts_per_class']
    idxs = []
    for c in range(10):
        c_neurons = np.where(assigned == c)[0]
        if len(c_neurons) >= 2 and spc is not None:
            best = c_neurons[np.argsort(spc[c, c_neurons])[-2:]]
            idxs.extend(best)
    if idxs: visualize_weights(data['w_ie'], idxs, save_path=os.path.join(MODEL_DIR, "receptive_fields_hebbian.png"))
    if spc is not None: visualize_label_responses(spc, assigned, save_path=os.path.join(MODEL_DIR, "neuron_assignment_hebbian.png"))
    print("图片已更新")
