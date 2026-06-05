"""Brian2 STDP 测试 — 秒出结果 + 生成可视化"""
import sys, os, numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'stdp-mnist', 'data', 'stdp_cython')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- 准确率 (秒出, 读混淆矩阵) ----
conf_path = os.path.join(DATA_DIR, 'confusion.npy')
if not os.path.exists(conf_path):
    print(f"❌ {conf_path} 不存在")
    sys.exit(1)

conf = np.load(conf_path)
acc = np.trace(conf) / np.sum(conf) * 100
print(f"★ STDP准确率: {acc:.2f}%")
for i in range(10):
    row = conf[i]
    a = row[i] / row.sum() * 100 if row.sum() > 0 else 0
    print(f"  {i}: {a:.1f}%")

# ---- 可视化 (加载权重 + 标签分配) ----
weights_path = os.path.join(DATA_DIR, 'weights.npy')
assign_path = os.path.join(DATA_DIR, 'assign.npy')

if not os.path.exists(weights_path) or not os.path.exists(assign_path):
    print("⚠️ 权重/标签文件缺失, 跳过可视化")
    sys.exit(0)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stdp_hebbian'))
from utils import visualize_weights, visualize_label_responses

weights = np.load(weights_path)  # shape: (N_INP * N_NEURONS,)
assign = np.load(assign_path)    # shape: (N_NEURONS,)
n_neurons = len(assign)
w_ie = weights.reshape(n_neurons, -1)

# 各类别Top-2神经元
idxs = []
for c in range(10):
    c_neurons = np.where(assign == c)[0]
    if len(c_neurons) >= 2:
        best = c_neurons[np.argsort(-np.ones(len(c_neurons)))[:2]]
        idxs.extend(best)
    elif len(c_neurons) == 1:
        idxs.extend(c_neurons)

if idxs:
    path = os.path.join(OUT_DIR, 'receptive_fields_stdp.png')
    visualize_weights(w_ie, idxs, save_path=path)
    print(f"  → {path}")

# 标签分配统计（用混淆矩阵的行和近似各类响应）
spike_counts_per_class = conf  # (10, 10) 混淆矩阵替代
if spike_counts_per_class is not None:
    path = os.path.join(OUT_DIR, 'neuron_assignment_stdp.png')
    # 统计各类别分配了多少神经元
    n_per_class = np.array([(assign == c).sum() for c in range(10)])
    # 扩展到 n_neurons 维度来适配 visualize 函数
    expanded = np.zeros((10, n_neurons))
    for c in range(10):
        expanded[c, assign == c] = 1
    visualize_label_responses(expanded, assign, save_path=path)
    print(f"  → {path}")
