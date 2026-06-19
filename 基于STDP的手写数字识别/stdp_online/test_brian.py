"""Brian2 STDP 测试"""
import sys, os, numpy as np
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, '..', 'data', 'stdp_full')

conf = np.load(os.path.join(DATA_DIR, 'confusion.npy'))
acc = np.trace(conf) / np.sum(conf) * 100
print(f"★ STDP准确率: {acc:.2f}%")
for i in range(10):
    a = conf[i,i] / conf[i].sum() * 100 if conf[i].sum() > 0 else 0
    print(f"  {i}: {a:.1f}%")

# 可视化
assign = np.load(os.path.join(DATA_DIR, 'assign.npy'))
w_ie = np.load(os.path.join(DATA_DIR, 'weights.npy')).reshape(len(assign), -1)

sys.path.insert(0, os.path.join(BASE, '..', 'stdp_hebbian'))
from utils import visualize_weights, visualize_label_responses
idxs = []
for c in range(10):
    c_neurons = np.where(assign == c)[0]
    if len(c_neurons) >= 2: idxs.extend(c_neurons[:2])
if idxs: visualize_weights(w_ie, idxs, save_path=os.path.join(BASE, 'receptive_fields_stdp.png'))
expanded = np.zeros((10, len(assign)))
for c in range(10): expanded[c, assign == c] = 1
visualize_label_responses(expanded, assign, save_path=os.path.join(BASE, 'neuron_assignment_stdp.png'))
print("图片已更新")
