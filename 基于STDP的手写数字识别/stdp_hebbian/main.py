"""
STDP速率近似 — Hebbian LTP + 权重归一化 (竞争学习)
电导型LIF + 泊松编码 + 侧向抑制 + 内在可塑性 + 不应期轮换

用法:
  python main.py             训练+测试
  python main.py --test      仅测试
"""

import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_mnist, visualize_weights, visualize_label_responses
from model import SNN

RANDOM_SEED = 42
N_INPUT = 784
N_EXCITATORY = 3000
DT_MS = 1.0
DURATION_MS = 50.0
MAX_RATE_HZ = 400.0
V_REST_E = -65.0; V_THRESH_E = -52.0; TAU_M_E = 100.0; REFRAC_E = 5.0
TAU_GE = 1.0; TAU_GI = 2.0
LR = 0.02; W_MAX = 1.0
THETA_PLUS = 0.05; TC_THETA = 1e7; THETA_OFFSET = 20.0
TARGET_WEIGHT_SUM = 78.0
REF_PERIOD_SAMPLES = 50; INH_STRENGTH = 17.0
N_TRAIN_SAMPLES = 60000; N_EPOCHS = 4
N_LABEL_SAMPLES = 30000
TOP_K_VOTE = 10
MIN_SPIKE_COUNT = 5; MAX_INTENSITY = 10.0; START_INTENSITY = 2.0
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_hebbian.npz")


def train_pipeline():
    """
    完整训练流程: 初始化 → Hebbian竞争学习 → 标签分配 → 保存模型。

    流程:
      1. 加载 MNIST 数据集
      2. 创建 SNN 并基于训练样本初始化权重 (每类均匀采样)
      3. 多轮训练循环 (N_EPOCHS × N_TRAIN_SAMPLES):
         - 每个样本送入 train_on_sample() 执行 Hebbian 竞争学习
         - 自适应输入强度: 发放不足时提高, 正常时降低, 确保稳定发放
         - 每5000步输出进度和统计
      4. 标签分配: 对 N_LABEL_SAMPLES 个训练样本做前向推理,
         统计赢家归属, 为每个神经元分配数字类别
      5. 保存模型到 MODEL_PATH

    返回:
        (snn, train_imgs, train_labels, test_imgs, test_labels, train_time)
    """
    print("=" * 60)
    print("STDP速率近似 — Hebbian LTP + 权重归一化")
    print(f"网络: {N_INPUT}输入 → {N_EXCITATORY} LIF神经元")
    print(f"仿真: {DURATION_MS}ms/dt={DT_MS}ms | 泊松{MAX_RATE_HZ}Hz")
    print(f"学习: Hebbian LTP (lr={LR}) + 权重归一化 (target={TARGET_WEIGHT_SUM})")
    print(f"训练: {N_TRAIN_SAMPLES}×{N_EPOCHS}轮 = {N_TRAIN_SAMPLES*N_EPOCHS}")
    print("=" * 60)

    np.random.seed(RANDOM_SEED)
    train_imgs, train_labels, test_imgs, test_labels = load_mnist()

    snn = SNN(n_input=N_INPUT, n_excitatory=N_EXCITATORY,
              dt_ms=DT_MS, duration_ms=DURATION_MS, max_rate_hz=MAX_RATE_HZ,
              v_rest_e=V_REST_E, v_thresh_e=V_THRESH_E, tau_m_e=TAU_M_E, refrac_e=REFRAC_E,
              tau_ge=TAU_GE, tau_gi=TAU_GI, lr=LR, w_max=W_MAX,
              theta_plus=THETA_PLUS, tc_theta=TC_THETA, theta_offset=THETA_OFFSET,
              target_weight_sum=TARGET_WEIGHT_SUM,
              ref_period_samples=REF_PERIOD_SAMPLES, inh_strength=INH_STRENGTH)
    snn.initialize_from_exemplars(train_imgs, train_labels, n_per_class=N_EXCITATORY // 10)

    print(f"\n[训练] Hebbian竞争学习 (无监督)...")
    t0 = time.time()
    train_n = min(N_TRAIN_SAMPLES, len(train_imgs))
    snn.input_intensity = START_INTENSITY
    total_spikes, skipped, global_step = 0, 0, 0
    total_steps = train_n * N_EPOCHS

    for epoch in range(N_EPOCHS):
        for i in range(train_n):
            for retry in range(5):
                counts = snn.train_on_sample(train_imgs[i], seed=(epoch*100000+i)*100+retry)
                sc = counts.sum()
                if sc >= MIN_SPIKE_COUNT or retry == 4:
                    if sc < MIN_SPIKE_COUNT: skipped += 1
                    break
                snn.input_intensity = min(snn.input_intensity + 1.0, MAX_INTENSITY)
            if sc >= MIN_SPIKE_COUNT:
                snn.input_intensity = max(START_INTENSITY, snn.input_intensity - 0.5)
            total_spikes += sc; global_step += 1
            if global_step % 5000 == 0 or global_step == 1:
                elapsed = time.time() - t0
                print(f"  [E{epoch+1}][{global_step}/{total_steps}] {elapsed:.0f}s "
                      f"ETA={elapsed/global_step*total_steps-elapsed:.0f}s "
                      f"发放/样本={total_spikes/global_step:.0f} 强度={snn.input_intensity:.1f}", flush=True)

    train_time = time.time() - t0
    print(f"  训练完成: {train_time:.1f}s ({train_time/60:.1f}min)")

    print(f"\n[标签分配] {N_LABEL_SAMPLES}样本...")
    n_label = min(N_LABEL_SAMPLES, len(train_imgs))
    sc_label = np.zeros((n_label, N_EXCITATORY))
    for i in range(n_label):
        sc_label[i] = snn.forward(train_imgs[i], seed=500000 + i)
        if (i+1) % 5000 == 0: print(f"  {i+1}/{n_label}")
    snn.assign_labels(sc_label, train_labels[:n_label])
    for c in range(10): print(f"  类别{c}: {(snn.assigned_labels==c).sum()}个")

    snn.save(MODEL_PATH)
    print(f"  保存: {MODEL_PATH}")
    return snn, train_imgs, train_labels, test_imgs, test_labels, train_time


def test_pipeline(snn=None, test_imgs=None, test_labels=None, mpath=None):
    """
    测试流程: 加载模型 → 遍历测试集 → Top-K投票预测 → 输出准确率 → 保存混淆矩阵 → 可视化。

    可通过参数传入已训练的 SNN 和数据, 也可通过 mpath 自动加载。
    测试完成后自动生成:
      - confusion_hebbian.npy: 10×10 混淆矩阵 (供 test.py 秒出准确率)
      - receptive_fields_hebbian.png: 感受野可视化
      - neuron_assignment_hebbian.png: 神经元标签分配图

    参数:
        snn: 已训练的 SNN 实例 (可选)
        test_imgs: 测试图像 (可选)
        test_labels: 测试标签 (可选)
        mpath: 模型文件路径 (可选, 默认 MODEL_PATH)

    返回:
        acc: float 测试准确率 (%)
    """
    print("=" * 60)
    print("Hebbian SNN 测试")
    print("=" * 60)
    if snn is None:
        p = mpath or MODEL_PATH
        if not os.path.exists(p): print(f"❌ {p}"); return
        _, _, test_imgs, test_labels = load_mnist()
        snn = SNN.load(p)
    n_test = len(test_imgs)
    correct = 0; pc = np.zeros(10, int); pt = np.zeros(10, int)
    t0 = time.time()
    for i in range(n_test):
        pred, _ = snn.predict(test_imgs[i], top_k=TOP_K_VOTE, seed=600000+i)
        if pred == test_labels[i]: correct += 1; pc[test_labels[i]] += 1
        pt[test_labels[i]] += 1
        if (i+1) % 2000 == 0: print(f"  [{i+1}/{n_test}] {correct/(i+1)*100:.1f}%")
    acc = correct/n_test*100
    print(f"\n  ★ 准确率: {acc:.2f}% ({correct}/{n_test})")
    for c in range(10): print(f"    {c}: {pc[c]/pt[c]*100:.1f}%")
    # 保存混淆矩阵 (供test.py秒出)
    conf = np.zeros((10,10), dtype=int)
    for i in range(n_test):
        pred,_ = snn.predict(test_imgs[i], top_k=TOP_K_VOTE, seed=600000+i)
        conf[test_labels[i], pred] += 1
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "confusion_hebbian.npy"), conf)
    # 可视化
    idxs = []
    for c in range(10):
        c_neurons = np.where(snn.assigned_labels == c)[0]
        if len(c_neurons) > 0 and snn.spike_counts_per_class is not None:
            best = c_neurons[np.argsort(snn.spike_counts_per_class[c, c_neurons])[-2:]]
            idxs.extend(best)
    if idxs: visualize_weights(snn.w_ie, idxs, save_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "receptive_fields_hebbian.png"))
    if snn.spike_counts_per_class is not None:
        visualize_label_responses(snn.spike_counts_per_class, snn.assigned_labels,
                                  save_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "neuron_assignment_hebbian.png"))
    return acc


def main():
    """
    CLI 入口: 解析命令行参数, 执行训练+测试或仅测试。

    用法:
      python main.py                训练+测试 (默认)
      python main.py --test         仅测试 (加载已有模型)
      python main.py --test --model-path <路径>  从指定路径加载模型测试

    关键超参数在文件顶部定义 (N_EXCITATORY, N_EPOCHS, LR 等)。
    """
    mpath = None
    for i, a in enumerate(sys.argv):
        if a == '--model-path' and i+1 < len(sys.argv): mpath = sys.argv[i+1]
    if "--test" in sys.argv: test_pipeline(mpath=mpath)
    else:
        snn, ti, tl, tst_i, tst_l, tt = train_pipeline()
        test_pipeline(snn, tst_i, tst_l)
        print(f"\n训练:{tt:.0f}s | 模型:{MODEL_PATH}")

if __name__ == "__main__": main()
