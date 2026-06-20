"""STDP 测试 """
import sys, os, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data', 'stdp_full')

# ── 纯STDP无监督投票 (confusion.npy) ──
conf = np.load(os.path.join(DATA, 'confusion.npy'))
acc = np.trace(conf) / np.sum(conf) * 100
print(f"★ 纯STDP (无监督投票): {acc:.2f}% ({int(np.sum(conf))}样本)")

# ── STDP+MLP (mlp_confusion.npy) ──
mlp_conf_path = os.path.join(DATA, 'mlp_confusion.npy')
if os.path.exists(mlp_conf_path):
    mlp_conf = np.load(mlp_conf_path)
    mlp_acc = np.trace(mlp_conf) / np.sum(mlp_conf) * 100
    print(f"★ STDP+MLP: {mlp_acc:.2f}% ({int(np.sum(mlp_conf))}样本)")
    for i in range(10):
        a = mlp_conf[i,i] / mlp_conf[i].sum() * 100 if mlp_conf[i].sum() > 0 else 0
        print(f"  {i}: {a:.1f}%")
else:
    print("⚠️ MLP结果未生成, 运行 train.py 训练")

# ── 可视化 ──
assign_path = os.path.join(DATA, 'assign.npy')
weights_path = os.path.join(DATA, 'weights.npy')
if os.path.exists(assign_path) and os.path.exists(weights_path):
    assign = np.load(assign_path)
    w = np.load(weights_path).reshape(len(assign), -1)
    sys.path.insert(0, os.path.join(HERE, '..', 'stdp_hebbian'))
    from utils import visualize_weights, visualize_label_responses
    idxs = []; [idxs.extend(np.where(assign==c)[0][:2]) for c in range(10) if (assign==c).sum()>=2]
    if idxs: visualize_weights(w, idxs, save_path=os.path.join(HERE, 'receptive_fields_stdp.png'))
    spc = np.zeros((10, len(assign)))
    for c in range(10): spc[c, assign == c] = 1
    visualize_label_responses(spc, assign, save_path=os.path.join(HERE, 'neuron_assignment_stdp.png'))
    print("图片已更新")
