"""STDP 测试 — 加载模型，秒出结果。"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Params, Network

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
conf_path = os.path.join(DATA, 'confusion_stdp.npy')

if not os.path.exists(conf_path):
    print("❌ 先运行 train.py 训练"); sys.exit(1)

conf = np.load(conf_path)
acc = np.trace(conf) / np.sum(conf) * 100
print(f"★ STDP准确率: {acc:.2f}%")
for i in range(10):
    a = conf[i,i] / conf[i].sum() * 100 if conf[i].sum() > 0 else 0
    print(f"  {i}: {a:.1f}%")

# 可视化
model_path = os.path.join(DATA, 'model_stdp.npz')
if os.path.exists(model_path):
    d = np.load(model_path, allow_pickle=True)
    assign = d['labels']; w_ie = d['w']
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stdp_hebbian'))
    from utils import visualize_weights, visualize_label_responses
    idxs = []
    for c in range(10):
        c_n = np.where(assign == c)[0]
        if len(c_n) >= 2: idxs.extend(c_n[:2])
    if idxs:
        visualize_weights(w_ie, idxs, save_path=os.path.join(os.path.dirname(__file__), 'receptive_fields_stdp.png'))
    spc = d['per_class']
    visualize_label_responses(spc, assign, save_path=os.path.join(os.path.dirname(__file__), 'neuron_assignment_stdp.png'))
    print("图片已更新")
