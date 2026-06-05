"""
STDP手写数字识别 — Diehl & Cook (2015) 在线三重STDP
电导型LIF + 泊松编码 + 侧向抑制 + 内在可塑性 + 在线STDP + 权重归一化

无监督学习:
  1. 电导型LIF脉冲仿真(泊松编码) → 逐时间步Top-1 WTA
  2. 在线STDP权重更新:
     LTD: 输入脉冲 → w -= nu_pre × post1_trace   (nu_pre=0.0001)
     LTP: 神经元发放 → w += nu_post × pre_trace × post2_trace  (nu_post=0.01)
  3. 每样本权重归一化 (LTD等效稳态)
  4. 事后标签分配(仅用于评估, 不参与训练)

用法:
  python main.py                             训练 + 保存 + 测试 (CPU)
  python main.py --gpu                       训练 + 保存 + 测试 (Apple GPU MPS)
  python main.py --test                      加载默认路径模型直接测试
  python main.py --test --model-path <path>  加载指定路径模型直接测试
  python main.py --train                     仅训练并保存
"""

import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_mnist, visualize_weights, visualize_label_responses

USE_GPU = "--gpu" in sys.argv
if USE_GPU:
    from model_torch import SNN
else:
    from model import SNN


# ==================== 超参数 (Diehl & Cook 2015) ====================
RANDOM_SEED = 42

# 网络
N_INPUT = 784
N_EXCITATORY = 1600          # 400=论文基线 | 更多→训练不足(每个神经元机会少)

# 仿真
DT_MS = 1.0
DURATION_MS = 350.0         # 刺激期 350ms (论文标准)
REST_MS = 150.0             # 静息期 150ms (trace衰减)
MAX_RATE_HZ = 63.75         # 最大泊松发放率 (对应像素值=1.0)

# 电导型LIF
V_REST_E = -65.0
V_THRESH_E = -52.0
TAU_M_E = 100.0
REFRAC_E = 5.0
TAU_GE = 1.0
TAU_GI = 2.0

# ======== 在线三重STDP (Diehl & Cook 2015) ========
NU_PRE = 0.0001             # LTD学习率
NU_POST = 0.01              # LTP学习率 (LTP:LTD ≈ 100:1)
W_MAX = 1.0                 # 权重上限
TAU_PRE = 20.0              # pre trace 时间常数 (ms)
TAU_POST1 = 20.0            # post1 trace 时间常数 (ms) — 用于LTD
TAU_POST2 = 40.0            # post2 trace 时间常数 (ms) — 用于LTP

# 内在可塑性 (自适应阈值)
THETA_PLUS = 0.05
TC_THETA = 1e7
THETA_OFFSET = 20.0

# 稳态
TARGET_WEIGHT_SUM = 78.0

# 侧向抑制
INH_STRENGTH = 17.0

# 训练 (无监督)
N_TRAIN_SAMPLES = 60000
N_EPOCHS = 3                # 400神经元需要多轮才能充分训练

# 事后标签分配
N_LABEL_SAMPLES = 30000

# 推理
TOP_K_VOTE = 10

# 输入强度自适应 (防止发放过少)
MIN_SPIKE_COUNT = 5
MAX_INTENSITY = 10.0
START_INTENSITY = 2.0

# 模型保存路径
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_online.npz")


def train_pipeline():
    """训练 + 保存"""
    print("=" * 60)
    print("无监督脉冲神经网络 — Diehl & Cook (2015) 在线三重STDP")
    print(f"网络: {N_INPUT}输入 → {N_EXCITATORY} LIF神经元")
    print(f"仿真: {DURATION_MS}ms刺激 + {REST_MS}ms静息 | dt={DT_MS}ms | 泊松{MAX_RATE_HZ}Hz")
    print(f"STDP: nu_pre={NU_PRE} | nu_post={NU_POST} | tau_pre={TAU_PRE}ms")
    print(f"      tau_post1={TAU_POST1}ms | tau_post2={TAU_POST2}ms")
    print(f"机制: 电导型LIF + 侧向抑制 + 内在可塑性 + 在线STDP")
    print(f"加速: {'Apple GPU (MPS)' if USE_GPU else 'Numba CPU (多核)'}")
    print(f"训练: {N_TRAIN_SAMPLES}样本×{N_EPOCHS}轮 = {N_TRAIN_SAMPLES * N_EPOCHS} (无监督)")
    print("=" * 60)

    np.random.seed(RANDOM_SEED)
    train_imgs, train_labels, test_imgs, test_labels = load_mnist()

    kwargs = dict(n_input=N_INPUT, n_excitatory=N_EXCITATORY,
                  dt_ms=DT_MS, duration_ms=DURATION_MS, rest_ms=REST_MS,
                  max_rate_hz=MAX_RATE_HZ,
                  v_rest_e=V_REST_E, v_thresh_e=V_THRESH_E,
                  tau_m_e=TAU_M_E, refrac_e=REFRAC_E,
                  tau_ge=TAU_GE, tau_gi=TAU_GI,
                  nu_pre=NU_PRE, nu_post=NU_POST, w_max=W_MAX,
                  tau_pre=TAU_PRE, tau_post1=TAU_POST1, tau_post2=TAU_POST2,
                  theta_plus=THETA_PLUS, tc_theta=TC_THETA, theta_offset=THETA_OFFSET,
                  target_weight_sum=TARGET_WEIGHT_SUM,
                  inh_strength=INH_STRENGTH)
    if USE_GPU:
        kwargs['gpu'] = True
    snn = SNN(**kwargs)

    # ---- 无监督STDP训练 ----
    print(f"\n[训练] 无监督在线STDP (不看标签)...")
    t0 = time.time()
    train_n = min(N_TRAIN_SAMPLES, len(train_imgs))
    snn.input_intensity = START_INTENSITY
    total_spikes = 0
    skipped = 0
    global_step = 0
    total_steps = train_n * N_EPOCHS
    total_steps_float = float(total_steps)

    for epoch in range(N_EPOCHS):
        for i in range(train_n):
            # 输入强度自适应: 若发放<5则增量重试
            for retry in range(5):
                counts = snn.train_on_sample(train_imgs[i],
                                             seed=(epoch * 100000 + i) * 100 + retry)
                sc = counts.sum()
                if sc >= MIN_SPIKE_COUNT or retry == 4:
                    if sc < MIN_SPIKE_COUNT:
                        skipped += 1
                    break
                snn.input_intensity = min(snn.input_intensity + 1.0, MAX_INTENSITY)
            if sc >= MIN_SPIKE_COUNT:
                snn.input_intensity = max(START_INTENSITY, snn.input_intensity - 0.5)
            total_spikes += sc
            global_step += 1

            if global_step % 5000 == 0 or global_step == 1:
                elapsed = time.time() - t0
                eta = elapsed / global_step * total_steps_float - elapsed
                print(f"  [E{epoch+1}][{global_step}/{total_steps}] {elapsed:.0f}s "
                      f"ETA={eta:.0f}s 发放/样本={total_spikes/global_step:.1f} "
                      f"强度={snn.input_intensity:.1f}")

    train_time = time.time() - t0
    print(f"\n  训练完成: {train_time:.1f}s ({train_time/60:.1f}min)")
    print(f"  平均发放/样本: {total_spikes/global_step:.1f} | 跳过: {skipped}")

    # ---- 事后标签分配 ----
    print(f"\n[标签分配] 用{N_LABEL_SAMPLES}样本...")
    n_label = min(N_LABEL_SAMPLES, len(train_imgs))
    sc_label = np.zeros((n_label, N_EXCITATORY))
    t0 = time.time()
    for i in range(n_label):
        sc_label[i] = snn.forward(train_imgs[i], train_mode=False, seed=500000 + i)
        if (i + 1) % 5000 == 0:
            print(f"  收集 {i+1}/{n_label} ({time.time()-t0:.0f}s)")
    snn.assign_labels(sc_label, train_labels[:n_label])
    for c in range(10):
        print(f"  类别 {c}: {(snn.assigned_labels == c).sum()} 个神经元")

    # ---- 保存 ----
    snn.save(MODEL_PATH)
    print(f"\n  模型已保存至: {MODEL_PATH}")

    return snn, train_imgs, train_labels, test_imgs, test_labels, train_time


def test_pipeline(snn=None, test_imgs=None, test_labels=None, model_path=None):
    """加载 + 测试"""
    print("=" * 60)
    print("STDP脉冲神经网络 — 测试模式")
    print("=" * 60)

    if snn is None:
        mpath = model_path or MODEL_PATH
        if not os.path.exists(mpath):
            print(f"错误: 模型文件不存在 ({mpath})")
            print("请先运行: python main.py --train  进行训练并保存模型")
            return
        print(f"\n加载模型: {mpath}")
        _, _, test_imgs, test_labels = load_mnist()
        if USE_GPU:
            snn = SNN.load(mpath, gpu=True)
        else:
            snn = SNN.load(mpath)
        n_labeled = (snn.assigned_labels >= 0).sum()
        print(f"  权重形状: {snn.w_ie.shape}")
        print(f"  已标记神经元: {n_labeled}/{snn.n_excitatory}")
        print(f"  输入强度: {snn.input_intensity:.1f}")
        for c in range(10):
            n_c = (snn.assigned_labels == c).sum()
            print(f"    类别 {c}: {n_c} 个神经元")

    n_test = len(test_imgs)
    correct = 0
    pc_correct = np.zeros(10, dtype=int)
    pc_total = np.zeros(10, dtype=int)
    t0 = time.time()

    for i in range(n_test):
        pred, _ = snn.predict(test_imgs[i], top_k=TOP_K_VOTE, seed=600000 + i)
        if pred == test_labels[i]:
            correct += 1
            pc_correct[test_labels[i]] += 1
        pc_total[test_labels[i]] += 1
        if (i + 1) % 2000 == 0:
            print(f"  测试 {i+1}/{n_test}: {correct/(i+1)*100:.1f}% ({time.time()-t0:.0f}s)")

    accuracy = correct / n_test * 100
    print(f"\n  ★ 准确率: {accuracy:.2f}% ({correct}/{n_test})")
    for c in range(10):
        a = pc_correct[c] / pc_total[c] * 100 if pc_total[c] > 0 else 0
        print(f"    {c}: {a:5.1f}%")
    print(f"  测试耗时: {time.time()-t0:.1f}s")

    # 可视化
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
    if snn.spike_counts_per_class is not None:
        visualize_label_responses(snn.spike_counts_per_class, snn.assigned_labels,
                                  save_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "neuron_assignment_stdp.png"))


def main():
    # 解析 --model-path 参数
    model_path = None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--model-path' and i + 1 < len(args):
            model_path = args[i + 1]
            i += 2
        elif args[i] == '--model-path':
            print("错误: --model-path 需要指定路径")
            print("用法: python main.py --test --model-path <path>")
            return
        else:
            i += 1

    if "--test" in sys.argv:
        test_pipeline(model_path=model_path)
    elif "--train" in sys.argv:
        train_pipeline()
    else:
        snn, train_imgs, train_labels, test_imgs, test_labels, train_time = train_pipeline()
        # 训练后自动测试
        print(f"\n{'='*60}")
        test_pipeline(snn, test_imgs, test_labels)
        print(f"\n{'='*60}")
        print(f"训练耗时: {train_time:.1f}s | 模型已保存: {MODEL_PATH}")
        print(f"下次可直接: python main.py --test")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
