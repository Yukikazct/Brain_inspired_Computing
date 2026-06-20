"""STDP 测试 — 加载模型，秒出结果。"""
import sys, os, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, '..', 'data', 'stdp_full')

conf_path = os.path.join(DATA_DIR, 'confusion.npy')
if not os.path.exists(conf_path):
    print(f"❌ {conf_path} 不存在, 先运行 train.py 训练")
    sys.exit(1)

conf = np.load(conf_path)
acc = np.trace(conf) / np.sum(conf) * 100
print(f"★ STDP准确率: {acc:.2f}%")
for i in range(10):
    a = conf[i,i] / conf[i].sum() * 100 if conf[i].sum() > 0 else 0
    print(f"  {i}: {a:.1f}%")

# 可视化
assign_path = os.path.join(DATA_DIR, 'assign.npy')
weights_path = os.path.join(DATA_DIR, 'weights.npy')
if os.path.exists(assign_path) and os.path.exists(weights_path):
    assign = np.load(assign_path)
    w_ie = np.load(weights_path).reshape(len(assign), -1)

    sys.path.insert(0, os.path.join(HERE, '..', 'stdp_hebbian'))
    from utils import visualize_weights, visualize_label_responses
    idxs = []
    for c in range(10):
        c_n = np.where(assign == c)[0]
        if len(c_n) >= 2: idxs.extend(c_n[:2])
    if idxs: visualize_weights(w_ie, idxs, save_path=os.path.join(HERE, 'receptive_fields_stdp.png'))

    expanded = np.zeros((10, len(assign)))
    for c in range(10): expanded[c, assign == c] = 1
    visualize_label_responses(expanded, assign, save_path=os.path.join(HERE, 'neuron_assignment_stdp.png'))
    print("图片已更新")
