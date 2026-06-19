"""基于STDP的手写数字识别 — Brian2在线三重STDP + 无监督学习。

参考:
  Diehl PU, Cook M (2015). Unsupervised learning of digit recognition
  using spike-timing-dependent plasticity. Front. Comput. Neurosci. 9:99.

用法:
  python train.py              # 训练 + 测试
  python train.py --test       # 仅测试(加载已保存模型)
"""

import os, sys, time, numpy as np
from pathlib import Path
from collections import defaultdict
from random import randrange, seed as rseed
from struct import unpack

os.environ['CFLAGS'] = '-O3'
from brian2 import (prefs, seed as bseed, NeuronGroup, Synapses,
                     PoissonGroup, Network, SpikeMonitor, Equations,
                     ms, mV, Hz, second, defaultclock)
from brian2.units import volt

prefs.codegen.cpp.extra_compile_args_gcc = ['-O3', '-ffast-math']
prefs.codegen.target = 'cython'

HERE = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════
# 参数配置
# ═══════════════════════════════════════════════════════════════

class Params:
    """Diehl & Cook (2015) STDP 仿真参数。"""

    def __init__(self, **kwargs):
        # ── 网络结构 ──
        self.n_input    = kwargs.get("n_input", 784)
        self.n_exc      = kwargs.get("n_exc", 1600)
        self.n_inh      = self.n_exc

        # ── 仿真 ──
        self.dt         = kwargs.get("dt", 0.5)             # 时间步长 (ms)
        self.duration   = kwargs.get("duration", 350.0)     # 刺激时长 (ms)
        self.rest       = kwargs.get("rest", 150.0)         # 静息时长 (ms)
        self.intensity  = kwargs.get("intensity", 2.0)      # 输入强度

        # ── LIF 神经元 ──
        self.v_rest_e  = -65.0   # 兴奋静息电位 (mV)
        self.v_rest_i  = -60.0   # 抑制静息电位 (mV)
        self.v_thresh_e = -52.0  # 兴奋发放阈值 (mV)
        self.v_thresh_i = -40.0  # 抑制发放阈值 (mV)
        self.v_reset_e  = -65.0  # 兴奋重置电位 (mV)
        self.v_reset_i  = -45.0  # 抑制重置电位 (mV)
        self.refrac_e  = 5.0     # 兴奋不应期 (ms)
        self.refrac_i  = 2.0     # 抑制不应期 (ms)
        self.tau_m_e   = 100.0   # 兴奋膜时间常数 (ms)
        self.tau_m_i   = 10.0    # 抑制膜时间常数 (ms)
        self.tau_ge    = 1.0     # 兴奋电导时间常数 (ms)
        self.tau_gi    = 2.0     # 抑制电导时间常数 (ms)

        # ── 在线三重 STDP ──
        self.nu_pre     = kwargs.get("nu_pre", 0.0001)      # LTD 学习率
        self.nu_post    = kwargs.get("nu_post", 0.01)       # LTP 学习率
        self.w_max      = kwargs.get("w_max", 1.0)          # 权重上限
        self.tau_pre    = kwargs.get("tau_pre", 20.0)       # pre trace (ms)
        self.tau_post1  = kwargs.get("tau_post1", 20.0)     # post1 trace (ms)
        self.tau_post2  = kwargs.get("tau_post2", 40.0)     # post2 trace (ms)

        # ── 内在可塑性 ──
        self.theta_plus   = kwargs.get("theta_plus", 0.05)
        self.tc_theta     = kwargs.get("tc_theta", 1e7)
        self.theta_offset = kwargs.get("theta_offset", 20.0)

        # ── E→I→E 侧向抑制 ──
        self.w_exc_inh  = kwargs.get("w_exc_inh", 10.4)
        self.w_inh_exc  = kwargs.get("w_inh_exc", 17.0)
        self.target_sum = kwargs.get("target_sum", 78.0)

        # ── 训练 ──
        self.n_train    = kwargs.get("n_train", 180000)
        self.n_observe  = kwargs.get("n_observe", 5000)
        self.n_test     = kwargs.get("n_test", 10000)
        self.n_save     = kwargs.get("n_save", 20)
        self.seed       = kwargs.get("seed", 42)

        # ── 路径 ──
        self.mnist_path = kwargs.get("mnist_path", Path('../data'))
        self.data_path  = kwargs.get("data_path", Path('../data/stdp_full'))


# ═══════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════

def read_mnist(training, mnist_path):
    """加载MNIST数据集。返回 (images, labels), images除以8归一化。"""
    tag = 'train' if training else 't10k'
    with open(mnist_path / f'{tag}-images-idx3-ubyte', 'rb') as f:
        f.read(4); n = unpack('>I', f.read(4))[0]
        f.read(8)
        x = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, -1) / 8.0
    with open(mnist_path / f'{tag}-labels-idx1-ubyte', 'rb') as f:
        f.read(8)
        y = np.frombuffer(f.read(), dtype=np.uint8)
    return x, y


# ═══════════════════════════════════════════════════════════════
# 网络构建 (Brian2)
# ═══════════════════════════════════════════════════════════════

def build_network(p, training=True):
    """构建 Diehl & Cook (2015) 脉冲神经网络。

    架构: 输入(Poisson) → 兴奋(LIF) ↔ 抑制(LIF)
          E→I一对一, I→E全连接(除自己)
          STDP在输入→兴奋突触上
    """
    # ── 兴奋神经元方程 ──
    exc_eqs = Equations('''
    dv/dt = (v_rest - v + i_exc + i_inh) / tau_mem  : volt (unless refractory)
    i_exc = ge * -v                         : volt
    i_inh = gi * (v_inh_base - v)           : volt
    dge/dt = -ge/(1 * ms)                   : 1
    dgi/dt = -gi/(2 * ms)                   : 1
    dtimer/dt = 1                           : second
    ''',
        tau_mem=p.tau_m_e * ms, v_rest=p.v_rest_e * mV, v_inh_base=-100 * mV)

    if training:
        exc_eqs += Equations('dtheta/dt = -theta / (1e7 * ms) : volt')
        reset = f'v = {p.v_rest_e} * mV; timer = 0 * ms; theta += 0.05 * mV'
        arr_theta = np.ones(p.n_exc) * 20 * mV
    else:
        exc_eqs += Equations('theta : volt')
        reset = f'v = {p.v_rest_e} * mV; timer = 0 * ms'
        arr_theta = np.load(p.data_path / 'theta.npy') * volt

    # 阈值: v > (theta - 72mV) and timer > 50ms (不应期延长)
    ng_exc = NeuronGroup(p.n_exc, exc_eqs,
                         threshold=f'v > (theta - 72 * mV) and (timer > 50 * ms)',
                         refractory=5 * ms, reset=reset, method='euler', name='exc')
    ng_exc.v = p.v_rest_e * mV; ng_exc.theta = arr_theta

    # ── 抑制神经元 ──
    inh_eqs = Equations('''
    dv/dt = (v_rest - v + i_exc + i_inh) / tau_mem  : volt (unless refractory)
    i_exc = ge * -v                         : volt
    i_inh = gi * (v_inh_base - v)           : volt
    dge/dt = -ge/(1 * ms)                   : 1
    dgi/dt = -gi/(2 * ms)                   : 1
    dtimer/dt = 1                           : second
    ''',
        tau_mem=p.tau_m_i * ms, v_rest=p.v_rest_i * mV, v_inh_base=-85 * mV)

    ng_inh = NeuronGroup(p.n_inh, inh_eqs, threshold='v > -40 * mV',
                         refractory=2 * ms, reset=f'v = {p.v_reset_i} * mV',
                         method='euler', name='inh')
    ng_inh.v = p.v_rest_i * mV

    # ── E→I (一对一) ──
    syns_exc_inh = Synapses(ng_exc, ng_inh, on_pre=f'ge_post += {p.w_exc_inh}')
    syns_exc_inh.connect(j='i')

    # ── I→E (全连接, 除自己) ──
    syns_inh_exc = Synapses(ng_inh, ng_exc, on_pre=f'gi_post += {p.w_inh_exc}')
    syns_inh_exc.connect("i != j")

    # ── 输入层 ──
    pg_inp = PoissonGroup(p.n_input, 0 * Hz, name='inp')

    # ── 输入→兴奋 (STDP) ──
    on_pre = 'ge_post += w'
    on_post = ''
    model = 'w : 1'
    if training:
        on_pre += '; pre = 1.; w = clip(w - 0.0001 * post1, 0, 1.0)'
        on_post += ('post2bef = post2; '
                    'w = clip(w + 0.01 * pre * post2bef, 0, 1.0); '
                    'post1 = 1.; post2 = 1.')
        model += '''
        post2bef                        : 1
        dpre/dt   = -pre/(20 * ms)      : 1 (event-driven)
        dpost1/dt = -post1/(20 * ms)    : 1 (event-driven)
        dpost2/dt = -post2/(40 * ms)    : 1 (event-driven)
        '''
        weights = (np.random.random(p.n_input * p.n_exc) + 0.01) * 0.3
    else:
        weights = np.load(p.data_path / 'weights.npy')

    syns_inp_exc = Synapses(pg_inp, ng_exc, model=model,
                            on_pre=on_pre, on_post=on_post, name='inp_exc')
    syns_inp_exc.connect(True)
    syns_inp_exc.delay = 'rand() * 10 * ms'
    syns_inp_exc.w = weights

    exc_mon = SpikeMonitor(ng_exc, name='sp_exc')
    net = Network([pg_inp, ng_exc, ng_inh,
                   syns_inp_exc, syns_exc_inh, syns_inh_exc, exc_mon])
    net.run(0 * ms)
    return net


# ═══════════════════════════════════════════════════════════════
# 训练/推理核心
# ═══════════════════════════════════════════════════════════════

def show_sample(net, sample, intensity):
    """展示一个样本: 350ms刺激 + 150ms静息。返回各神经元发放计数。"""
    exc_mon = net['sp_exc']
    prev = exc_mon.count[:]
    net['inp'].rates = sample * intensity * Hz
    net.run(350 * ms)
    net['inp'].rates = 0 * Hz
    net.run(150 * ms)
    pat = np.array(exc_mon.count[:] - prev)
    if np.sum(pat) < 5:
        return show_sample(net, sample, intensity + 1)
    return pat


def normalize_plastic_weights(syns):
    """权重归一化: 每个神经元输入权重总和 = target_sum。"""
    conns = np.reshape(syns.w, (784, -1))
    col_sums = np.sum(conns, axis=0)
    col_sums[col_sums == 0] = 1.0
    conns *= 78.0 / col_sums
    syns.w = conns.reshape(-1)


def save_npy(arr, path):
    """保存numpy数组。"""
    arr = np.array(arr)
    np.save(path, arr)


def load_npy(path):
    """加载numpy数组。"""
    return np.load(path)


# ═══════════════════════════════════════════════════════════════
# 三阶段训练流水线
# ═══════════════════════════════════════════════════════════════

def train(p):
    """Phase 1: STDP无监督训练。"""
    X, Y = read_mnist(True, p.mnist_path)
    n_samples = X.shape[0]
    net = build_network(p, training=True)
    w_hist = [np.array(net['inp_exc'].w)]

    ratio = max(p.n_train // p.n_save, 1)
    t0 = time.time()
    print(f"训练 {p.n_train} 样本...")
    for i in range(p.n_train):
        ix = i % n_samples
        normalize_plastic_weights(net['inp_exc'])
        show_sample(net, X[ix], p.intensity)
        if i % ratio == 0:
            w_hist.append(np.array(net['inp_exc'].w))
        if (i + 1) % 20000 == 0:
            e = time.time() - t0
            print(f"  [{i+1}/{p.n_train}] {e/60:.0f}min ETA:{e/(i+1)*(p.n_train-i-1)/60:.0f}min")

    print(f"训练完成: {(time.time()-t0)/60:.0f}min")
    save_npy(net['inp_exc'].w, p.data_path / 'weights.npy')
    save_npy(np.array(net['exc'].theta), p.data_path / 'theta.npy')
    return net


def observe(p):
    """Phase 2: 标签分配。"""
    X, Y = read_mnist(True, p.mnist_path)
    net = build_network(p, training=False)
    responses = defaultdict(list)

    print(f"标签分配 {p.n_observe} 样本...")
    t0 = time.time()
    for i in range(p.n_observe):
        exc = show_sample(net, X[i % len(X)], p.intensity)
        responses[Y[i % len(Y)]].append(exc)
        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{p.n_observe}] {time.time()-t0:.0f}s")

    res = np.zeros((10, p.n_exc))
    for cls, vals in responses.items():
        res[cls] = np.array(vals).mean(axis=0)
    assign = np.argmax(res, axis=0)
    save_npy(assign, p.data_path / 'assign.npy')

    for c in range(10):
        print(f"  类别{c}: {(assign==c).sum()}个神经元")
    return assign


def test(p):
    """Phase 3: 测试评估。"""
    assign = np.load(p.data_path / 'assign.npy')
    groups = [np.where(assign == i)[0] for i in range(10)]
    X, Y = read_mnist(False, p.mnist_path)
    net = build_network(p, training=False)

    conf = np.zeros((10, 10))
    print(f"测试 {p.n_test} 样本...")
    t0 = time.time()
    for i in range(p.n_test):
        exc = show_sample(net, X[i % len(X)], p.intensity)
        guess = np.argmax([exc[grp].mean() for grp in groups])
        conf[Y[i % len(Y)], guess] += 1
        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{p.n_test}] {time.time()-t0:.0f}s")

    save_npy(conf, p.data_path / 'confusion.npy')
    acc = np.trace(conf) / np.sum(conf)
    print(f"\n★ 准确率: {acc:.2%}")
    for c in range(10):
        a = conf[c, c] / conf[c].sum() if conf[c].sum() > 0 else 0
        print(f"  {c}: {a:.1%}")
    return acc


# ═══════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = Params()
    p.data_path.mkdir(parents=True, exist_ok=True)
    bseed(p.seed); rseed(p.seed); np.random.seed(p.seed)

    print("=" * 50)
    print("Brian2 STDP — Diehl & Cook (2015)")
    print(f"{p.n_exc}神经元 × {p.n_train}样本")
    print(f"输出: {p.data_path}")
    print("=" * 50)

    if "--test" in sys.argv:
        test(p)
    else:
        train(p)
        observe(p)
        test(p)
