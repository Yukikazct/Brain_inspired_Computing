"""
STDP模型独立测试脚本
用法: python test.py [模型路径]

默认加载 stdp_spike/model_online.npz
"""

import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_mnist, visualize_weights, visualize_label_responses
from model import SNN

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_online.npz")
TOP_K_VOTE = 10


def test(model_path=None):
    mpath = model_path or MODEL_PATH
    if not os.path.exists(mpath):
        print(f"❌ 模型文件不存在: {mpath}")
        return

    print("=" * 60)
    print("STDP模型测试 — Diehl & Cook (2015)")
    print("=" * 60)

    # 加载
    print(f"\n📂 加载: {mpath}")
    _, _, test_imgs, test_labels = load_mnist()
    snn = SNN.load(mpath)

    n_labeled = (snn.assigned_labels >= 0).sum()
    print(f"   权重: {snn.w_ie.shape}")
    print(f"   已标记神经元: {n_labeled}/{snn.n_excitatory}")
    for c in range(10):
        n = (snn.assigned_labels == c).sum()
        print(f"   类别 {c}: {n} 个")

    # 测试
    n_test = len(test_imgs)
    correct = 0
    pc_correct = np.zeros(10, dtype=int)
    pc_total = np.zeros(10, dtype=int)
    t0 = time.time()

    print(f"\n🧪 测试 {n_test} 样本...")
    for i in range(n_test):
        pred, _ = snn.predict(test_imgs[i], top_k=TOP_K_VOTE, seed=700000 + i)
        if pred == test_labels[i]:
            correct += 1
            pc_correct[test_labels[i]] += 1
        pc_total[test_labels[i]] += 1
        if (i + 1) % 2000 == 0:
            print(f"  [{i+1}/{n_test}] {correct/(i+1)*100:.1f}% ({time.time()-t0:.0f}s)")

    # 结果
    accuracy = correct / n_test * 100
    print(f"\n{'='*60}")
    print(f"  ★ 准确率: {accuracy:.2f}% ({correct}/{n_test})")
    print(f"  耗时: {time.time()-t0:.1f}s")
    print(f"{'='*60}")

    print("\n📊 各类别:")
    for c in range(10):
        a = pc_correct[c] / pc_total[c] * 100 if pc_total[c] > 0 else 0
        bar = "█" * int(a / 2)
        print(f"  {c}: {a:5.1f}% {bar}")

    # 可视化
    print("\n🖼️  生成可视化...")
    idxs = []
    for c in range(10):
        c_neurons = np.where(snn.assigned_labels == c)[0]
        if len(c_neurons) > 0:
            if snn.spike_counts_per_class is not None:
                best = c_neurons[np.argsort(snn.spike_counts_per_class[c, c_neurons])[-2:]]
            else:
                best = c_neurons[:2]
            idxs.extend(best)

    if idxs:
        visualize_weights(snn.w_ie, idxs, save_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "receptive_fields_stdp.png"))
        print("  → receptive_fields_stdp.png")

    if snn.spike_counts_per_class is not None:
        visualize_label_responses(snn.spike_counts_per_class, snn.assigned_labels,
                                  save_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "neuron_assignment_stdp.png"))
        print("  → neuron_assignment_stdp.png")

    return accuracy


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else MODEL_PATH
    test(path)
