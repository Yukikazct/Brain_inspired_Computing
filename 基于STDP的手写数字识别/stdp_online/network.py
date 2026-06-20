"""Brian2脉冲神经网络构建 — Diehl & Cook (2015) E→I→E架构。

包含: 数据加载, 网络构建, 单样本仿真, 权重管理。
"""

import os, numpy as np
from struct import unpack

# Brian2仿真引擎 (Cython C代码生成加速)
os.environ['CFLAGS'] = '-O3'
from brian2 import (prefs, NeuronGroup, Synapses, PoissonGroup, Network,
                     SpikeMonitor, Equations, ms, mV, Hz)
from brian2.units import volt

prefs.codegen.cpp.extra_compile_args_gcc = ['-O3', '-ffast-math']
prefs.codegen.target = 'numpy'


# ═══════════════════════════════════════════════════════════════
# MNIST数据加载
# ═══════════════════════════════════════════════════════════════

def read_mnist(training, mnist_path):
    """加载MNIST数据集。training=True→训练集(60K), False→测试集(10K)。
    images除以8.0归一化 (与论文一致, 像素值范围[0, 31.875])。"""
    tag = 'train' if training else 't10k'
    with open(mnist_path / f'{tag}-images-idx3-ubyte', 'rb') as f:
        f.read(4); n = unpack('>I', f.read(4))[0]; f.read(8)
        x = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, -1) / 8.0
    with open(mnist_path / f'{tag}-labels-idx1-ubyte', 'rb') as f:
        f.read(8)
        y = np.frombuffer(f.read(), dtype=np.uint8)
    return x, y


# ═══════════════════════════════════════════════════════════════
# 网络构建
# ═══════════════════════════════════════════════════════════════

def build_network(p, training=True):
    """构建Diehl & Cook (2015)脉冲神经网络。

    架构: 输入(Poisson,784) → 兴奋(LIF,N) ↔ 抑制(LIF,N)
    E→I一对一, I→E全连接(除自己), 输入→兴奋全连接+STDP可塑性。
    """
    # ── 兴奋神经元 (电导型LIF) ──
    exc_eqs = Equations('''
    dv/dt = (v_rest - v + i_exc + i_inh) / tau_mem  : volt (unless refractory)
    i_exc = ge * -v                         : volt
    i_inh = gi * (v_inh_base - v)           : volt
    dge/dt = -ge/(1 * ms)                   : 1
    dgi/dt = -gi/(2 * ms)                   : 1
    dtimer/dt = 1                           : second
    ''', tau_mem=p.tau_m_e * ms, v_rest=p.v_rest_e * mV, v_inh_base=-100 * mV)

    if training:
        exc_eqs += Equations('dtheta/dt = -theta / (1e7 * ms) : volt')
        reset = f'v = {p.v_rest_e} * mV; timer = 0 * ms; theta += 0.05 * mV'
        arr_theta = np.ones(p.n_exc) * 20 * mV
    else:
        exc_eqs += Equations('theta : volt')
        reset = f'v = {p.v_rest_e} * mV; timer = 0 * ms'
        arr_theta = np.load(p.data_path / 'theta.npy') * volt

    ng_exc = NeuronGroup(p.n_exc, exc_eqs,
                         threshold=f'v > (theta - 72 * mV) and (timer > 50 * ms)',
                         refractory=5 * ms, reset=reset, method='euler', name='exc')
    ng_exc.v = p.v_rest_e * mV; ng_exc.theta = arr_theta

    # ── 抑制神经元 (快速LIF) ──
    inh_eqs = Equations('''
    dv/dt = (v_rest - v + i_exc + i_inh) / tau_mem  : volt (unless refractory)
    i_exc = ge * -v                         : volt
    i_inh = gi * (v_inh_base - v)           : volt
    dge/dt = -ge/(1 * ms)                   : 1
    dgi/dt = -gi/(2 * ms)                   : 1
    dtimer/dt = 1                           : second
    ''', tau_mem=p.tau_m_i * ms, v_rest=p.v_rest_i * mV, v_inh_base=-85 * mV)

    ng_inh = NeuronGroup(p.n_inh, inh_eqs, threshold='v > -40 * mV',
                         refractory=2 * ms, reset=f'v = {p.v_reset_i} * mV',
                         method='euler', name='inh')
    ng_inh.v = p.v_rest_i * mV

    # ── E→I 一对一 ──
    syns_exc_inh = Synapses(ng_exc, ng_inh, on_pre=f'ge_post += {p.w_exc_inh}')
    syns_exc_inh.connect(j='i')

    # ── I→E 全连接(除自己), 实现侧向抑制 (WTA) ──
    syns_inh_exc = Synapses(ng_inh, ng_exc, on_pre=f'gi_post += {p.w_inh_exc}')
    syns_inh_exc.connect("i != j")

    # ── 输入层 (泊松脉冲) ──
    pg_inp = PoissonGroup(p.n_input, 0 * Hz, name='inp')

    # ── 输入→兴奋 (STDP可塑性) ──
    on_pre = 'ge_post += w'; on_post = ''
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
    syns_inp_exc.connect(True); syns_inp_exc.delay = 'rand() * 10 * ms'
    syns_inp_exc.w = weights

    exc_mon = SpikeMonitor(ng_exc, name='sp_exc')
    net = Network([pg_inp, ng_exc, ng_inh,
                   syns_inp_exc, syns_exc_inh, syns_inh_exc, exc_mon])
    net.run(0 * ms)
    return net


# ═══════════════════════════════════════════════════════════════
# 单样本仿真
# ═══════════════════════════════════════════════════════════════

def show_sample(net, sample, intensity):
    """展示单样本: 350ms刺激 + 150ms静息。输入强度自适应。"""
    exc_mon = net['sp_exc']
    prev = exc_mon.count[:]
    net['inp'].rates = sample * intensity * Hz; net.run(350 * ms)
    net['inp'].rates = 0 * Hz; net.run(150 * ms)
    pat = np.array(exc_mon.count[:] - prev)
    if np.sum(pat) < 5:
        return show_sample(net, sample, intensity + 1)
    return pat


# ═══════════════════════════════════════════════════════════════
# 权重管理
# ═══════════════════════════════════════════════════════════════

def normalize_plastic_weights(syns):
    """权重归一化: 每神经元输入权重总和缩放到78.0 (LTD等效稳态)。"""
    conns = np.reshape(syns.w, (784, -1))
    col_sums = np.sum(conns, axis=0); col_sums[col_sums == 0] = 1.0
    conns *= 78.0 / col_sums; syns.w = conns.reshape(-1)


def save_npy(arr, path): np.save(path, np.array(arr))
