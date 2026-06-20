"""基于STDP的手写数字识别 — Brian2在线三重STDP + 无监督学习。

参考:
  Diehl PU, Cook M (2015). Unsupervised learning of digit recognition
  using spike-timing-dependent plasticity. Front. Comput. Neurosci. 9:99.

用法:
  python train.py              # 训练 + 测试 (1600神经元 × 180K样本)
  python train.py --test       # 仅测试 (加载已保存模型)
"""

import os, sys, time, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from pathlib import Path
from collections import defaultdict
from random import randrange, seed as rseed
from struct import unpack

# Brian2仿真引擎 (Cython C代码生成加速)
os.environ['CFLAGS'] = '-O3'
from brian2 import (prefs, seed as bseed, NeuronGroup, Synapses,
                     PoissonGroup, Network, SpikeMonitor, Equations,
                     ms, mV, Hz, second, defaultclock)
from brian2.units import volt

prefs.codegen.cpp.extra_compile_args_gcc = ['-O3', '-ffast-math']
prefs.codegen.target = 'numpy'  # M5 Pro Cython编译兼容问题

HERE = os.path.dirname(os.path.abspath(__file__))  # 脚本所在目录, 用于路径解析


# ═══════════════════════════════════════════════════════════════
# 参数配置
# ═══════════════════════════════════════════════════════════════

class Params:
    """Diehl & Cook (2015) STDP 仿真参数容器。

    所有超参数集中管理, 可通过kwargs覆盖默认值。
    训练用: p = Params(n_exc=400, n_train=60000)
    测试用: p = Params()  # 用默认值
    """

    def __init__(self, **kwargs):
        # ── 网络结构 ──
        self.n_input    = kwargs.get("n_input", 784)   # MNIST像素数 (28×28)
        self.n_exc      = kwargs.get("n_exc", 400)     # 兴奋神经元数
        self.n_inh      = self.n_exc                   # 抑制神经元数 (=兴奋数)

        # ── 仿真时间 ──
        self.dt         = kwargs.get("dt", 0.5)        # 时间步长 (ms)
        self.duration   = kwargs.get("duration", 350.0) # 刺激时长 (ms)
        self.rest       = kwargs.get("rest", 150.0)    # 静息时长 (ms)
        self.intensity  = kwargs.get("intensity", 2.0)  # 输入强度 (发放<5则自适应增加)

        # ── 兴奋神经元LIF参数 ──
        self.v_rest_e   = -65.0   # 静息电位 (mV)
        self.v_thresh_e = -52.0   # 发放阈值 (mV)
        self.v_reset_e  = -65.0   # 重置电位 (mV)
        self.refrac_e   = 5.0     # 不应期 (ms)
        self.tau_m_e    = 100.0   # 膜时间常数 (ms, 增大以增强发放率估计稳定性)
        self.tau_ge     = 1.0     # 兴奋电导时间常数 (ms)
        self.tau_gi     = 2.0     # 抑制电导时间常数 (ms)

        # ── 抑制神经元LIF参数 ──
        self.v_rest_i   = -60.0   # 静息电位 (mV)
        self.v_thresh_i = -40.0   # 发放阈值 (mV)
        self.v_reset_i  = -45.0   # 重置电位 (mV)
        self.refrac_i   = 2.0     # 不应期 (ms)
        self.tau_m_i    = 10.0    # 膜时间常数 (ms, 比兴奋快10倍)

        # ── 在线三重trace STDP ──
        # pre trace (τ=20ms): 输入脉冲到达 → pre=1
        # post1 trace (τ=20ms): 神经元发放 → post1=1, 用于LTD
        # post2 trace (τ=40ms): 神经元发放 → post2=1, 用于LTP
        self.nu_pre     = kwargs.get("nu_pre", 0.0001)   # LTD学习率
        self.nu_post    = kwargs.get("nu_post", 0.01)     # LTP学习率 (LTP:LTD≈100:1)
        self.w_max      = kwargs.get("w_max", 1.0)        # 权重上限
        self.tau_pre    = kwargs.get("tau_pre", 20.0)     # pre trace时间常数 (ms)
        self.tau_post1  = kwargs.get("tau_post1", 20.0)   # post1 trace时间常数 (ms)
        self.tau_post2  = kwargs.get("tau_post2", 40.0)   # post2 trace时间常数 (ms)

        # ── 内在可塑性 (自适应阈值, 防止单神经元主导) ──
        self.theta_plus   = kwargs.get("theta_plus", 0.05)   # 每次发放阈值增量
        self.tc_theta     = kwargs.get("tc_theta", 1e7)      # 阈值衰减时间常数 (ms, ≈2.8h)
        self.theta_offset = kwargs.get("theta_offset", 20.0)  # 阈值偏移量

        # ── E→I→E侧向抑制 (实现涌现式WTA竞争) ──
        self.w_exc_inh  = kwargs.get("w_exc_inh", 10.4)   # E→I权重
        self.w_inh_exc  = kwargs.get("w_inh_exc", 17.0)   # I→E权重
        self.target_sum = kwargs.get("target_sum", 78.0)   # 每神经元权重总和目标

        # ── 训练控制 ──
        self.n_train    = kwargs.get("n_train", 10000)    # 训练样本总数
        self.n_observe  = kwargs.get("n_observe", 5000)   # 标签分配样本数
        self.n_test     = kwargs.get("n_test", 10000)     # 测试样本数
        self.n_save     = kwargs.get("n_save", 20)         # 权重复制保存间隔
        self.seed       = kwargs.get("seed", 42)           # 随机种子 (可复现)

        # ── 数据路径 ──
        self.mnist_path = kwargs.get("mnist_path", Path('../data'))
        self.data_path  = kwargs.get("data_path", Path('../data/stdp_full'))


# ═══════════════════════════════════════════════════════════════
# MNIST数据加载
# ═══════════════════════════════════════════════════════════════

def read_mnist(training, mnist_path):
    """加载MNIST数据集, 返回 (images, labels)。

    MNIST原始格式: 大端序4字节魔数+数量+行+列, 然后像素/标签字节流。
    images除以8.0归一化 (与论文一致, 使像素值在[0, 31.875]范围内)。

    参数:
        training: True=训练集(60K), False=测试集(10K)
        mnist_path: 数据目录路径
    返回:
        x: (N, 784) float64, 归一化像素值
        y: (N,) uint8, 标签0-9
    """
    tag = 'train' if training else 't10k'
    with open(mnist_path / f'{tag}-images-idx3-ubyte', 'rb') as f:
        f.read(4)  # 跳过魔数
        n = unpack('>I', f.read(4))[0]  # 图像数量
        f.read(8)  # 跳过行数+列数
        x = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, -1) / 8.0
    with open(mnist_path / f'{tag}-labels-idx1-ubyte', 'rb') as f:
        f.read(8)  # 跳过魔数+数量
        y = np.frombuffer(f.read(), dtype=np.uint8)
    return x, y


# ═══════════════════════════════════════════════════════════════
# 网络构建 (Brian2仿真引擎)
# ═══════════════════════════════════════════════════════════════

def build_network(p, training=True):
    """构建Diehl & Cook (2015)脉冲神经网络。

    架构:
        输入层(Poisson, 784) → 兴奋层(LIF, N) ↔ 抑制层(LIF, N)
        - E→I: 一对一固定权重 (w_exc_inh=10.4)
        - I→E: 全连接, 排除自己 (w_inh_exc=17.0), 实现侧向抑制
        - 输入→兴奋: 全连接, STDP可塑性

    神经元模型:
        电导型LIF: dv/dt = (v_rest-v + ge*(-v) + gi*(v_rev-v)) / tau_m
        突触: 脉冲到达时电导瞬时增加w, 否则指数衰减
        自适应阈值: 发放后theta增加0.05mV, 缓慢衰减(tc=1e7ms)

    STDP规则 (在线三重trace):
        LTD (输入脉冲):  w -= 0.0001 * post1
        LTP (神经元发放): w += 0.01 * pre * post2_before

    参数:
        p: Params对象
        training: True=训练模式(STDP+自适应阈值), False=推理模式
    返回:
        net: Brian2 Network对象, 已初始化
    """
    # ── 兴奋神经元 ──
    # 电导型LIF方程: 膜电位 + 兴奋/抑制电导 + 不应期计时器
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
        # 训练时theta动态调整 (内在可塑性)
        exc_eqs += Equations('dtheta/dt = -theta / (1e7 * ms) : volt')
        reset = f'v = {p.v_rest_e} * mV; timer = 0 * ms; theta += 0.05 * mV'
        arr_theta = np.ones(p.n_exc) * 20 * mV
    else:
        # 推理时theta固定 (加载训练结果)
        exc_eqs += Equations('theta : volt')
        reset = f'v = {p.v_rest_e} * mV; timer = 0 * ms'
        arr_theta = np.load(p.data_path / 'theta.npy') * volt

    # 发放条件: v > (theta-72mV) AND 不应期结束(timer>50ms)
    ng_exc = NeuronGroup(p.n_exc, exc_eqs,
                         threshold=f'v > (theta - 72 * mV) and (timer > 50 * ms)',
                         refractory=5 * ms, reset=reset, method='euler', name='exc')
    ng_exc.v = p.v_rest_e * mV; ng_exc.theta = arr_theta

    # ── 抑制神经元 ──
    # 与兴奋相同方程, 但时间常数更快 (tau_m=10ms vs 100ms)
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

    # ── E→I 突触: 兴奋→抑制, 一对一连接 ──
    # 兴奋神经元发放 → 激活对应抑制神经元
    syns_exc_inh = Synapses(ng_exc, ng_inh, on_pre=f'ge_post += {p.w_exc_inh}')
    syns_exc_inh.connect(j='i')

    # ── I→E 突触: 抑制→兴奋, 全连接("i != j"排除自己) ──
    # 抑制神经元发放 → 抑制所有其他兴奋神经元 (侧向抑制)
    # 这是实现WTA的关键: 获胜神经元通过E→I→E回路抑制竞争对手
    syns_inh_exc = Synapses(ng_inh, ng_exc, on_pre=f'gi_post += {p.w_inh_exc}')
    syns_inh_exc.connect("i != j")

    # ── 输入层: 泊松脉冲发生器 ──
    # 每个MNIST像素对应一个泊松神经元, 发放率=像素值×强度
    pg_inp = PoissonGroup(p.n_input, 0 * Hz, name='inp')

    # ── 输入→兴奋突触: STDP可塑性 ──
    on_pre = 'ge_post += w'
    on_post = ''
    model = 'w : 1'
    if training:
        # LTD (输入脉冲触发): w -= nu_pre * post1_trace
        on_pre += '; pre = 1.; w = clip(w - 0.0001 * post1, 0, 1.0)'
        # LTP (神经元发放触发): w += nu_post * pre_trace * post2_before
        on_post += ('post2bef = post2; '
                    'w = clip(w + 0.01 * pre * post2bef, 0, 1.0); '
                    'post1 = 1.; post2 = 1.')
        # 三重trace: pre(20ms), post1(20ms), post2(40ms)
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
    syns_inp_exc.connect(True)                    # 全连接
    syns_inp_exc.delay = 'rand() * 10 * ms'       # 随机延迟0-10ms
    syns_inp_exc.w = weights

    # 脉冲监视器 (记录兴奋神经元发放)
    exc_mon = SpikeMonitor(ng_exc, name='sp_exc')

    # 组装网络并初始化
    net = Network([pg_inp, ng_exc, ng_inh,
                   syns_inp_exc, syns_exc_inh, syns_inh_exc, exc_mon])
    net.run(0 * ms)  # 触发代码生成和编译
    return net


# ═══════════════════════════════════════════════════════════════
# 单样本仿真与权重管理
# ═══════════════════════════════════════════════════════════════

def show_sample(net, sample, intensity):
    """对单张MNIST图像进行脉冲仿真, 统计各兴奋神经元发放次数。

    流程:
        1. 将图像像素值转换为泊松脉冲发放率 (sample × intensity Hz)
        2. 运行350ms刺激期 (泊松输入 + STDP在线更新)
        3. 运行150ms静息期 (零输入, 所有变量自然衰减)
        4. 统计静息期前后脉冲计数差值

    输入强度自适应: 如果总发放<5, 增加intensity递归重试。

    参数:
        net: Brian2网络
        sample: (784,) 归一化像素值
        intensity: 输入强度缩放因子
    返回:
        pat: (n_exc,) 各神经元发放计数
    """
    exc_mon = net['sp_exc']
    prev = exc_mon.count[:]                     # 静息期前的计数
    net['inp'].rates = sample * intensity * Hz  # 设置泊松率
    net.run(350 * ms)                           # 刺激期仿真
    net['inp'].rates = 0 * Hz                   # 清零输入
    net.run(150 * ms)                           # 静息期仿真
    pat = np.array(exc_mon.count[:] - prev)     # 差值=刺激期发放数

    # 输入强度自适应: 总发放太少说明输入太弱, 增加强度重试
    if np.sum(pat) < 5:
        return show_sample(net, sample, intensity + 1)
    return pat


def normalize_plastic_weights(syns):
    """权重归一化: 每个后突触神经元的输入权重总和缩放至78.0。

    这是LTD的全局等效机制: 权重归一化后, 弱连接被相对削弱,
    强连接被相对增强, 形成"富人愈富"的竞争效应。

    每样本前调用一次, 维持权重稳态。
    """
    conns = np.reshape(syns.w, (784, -1))        # (n_input, n_exc)
    col_sums = np.sum(conns, axis=0)              # 每神经元权重总和
    col_sums[col_sums == 0] = 1.0                 # 避免除零
    conns *= 78.0 / col_sums                      # 缩放到目标值
    syns.w = conns.reshape(-1)


def save_npy(arr, path):
    """保存numpy数组到指定路径, 自动创建父目录。"""
    np.save(path, np.array(arr))


# ═══════════════════════════════════════════════════════════════
# 三阶段训练流水线
# ═══════════════════════════════════════════════════════════════

def train(p):
    """Phase 1: 无监督STDP训练。

    对p.n_train个样本逐一进行:
        1. 权重归一化 (维持稳态)
        2. 泊松编码 + LIF仿真 (350ms刺激 + 150ms静息)
        3. 在线STDP权重更新 (LTD + LTP)

    训练全程不使用标签 (无监督学习)。
    输出: weights.npy (输入→兴奋权重矩阵), theta.npy (自适应阈值)

    参数:
        p: Params对象
    返回:
        net: 训练后的Brian2网络
    """
    X, Y = read_mnist(True, p.mnist_path)         # 加载训练集
    n_samples = X.shape[0]
    net = build_network(p, training=True)          # 构建训练网络
    w_hist = [np.array(net['inp_exc'].w)]          # 权重复制记录

    ratio = max(p.n_train // p.n_save, 1)          # 权重复制间隔
    t0 = time.time()
    print(f"训练 {p.n_train} 样本...")
    for i in range(p.n_train):
        ix = i % n_samples                          # 循环使用训练集
        normalize_plastic_weights(net['inp_exc'])   # 每样本前归一化
        show_sample(net, X[ix], p.intensity)        # 仿真+STDP

        if i % ratio == 0:                          # 定期保存权重复制
            w_hist.append(np.array(net['inp_exc'].w))
        if (i + 1) % 20000 == 0:                    # 进度显示
            e = time.time() - t0
            print(f"  [{i+1}/{p.n_train}] {e/60:.0f}min "
                  f"ETA:{e/(i+1)*(p.n_train-i-1)/60:.0f}min")

    print(f"训练完成: {(time.time()-t0)/60:.0f}min")
    save_npy(net['inp_exc'].w, p.data_path / 'weights.npy')
    save_npy(np.array(net['exc'].theta), p.data_path / 'theta.npy')
    return net


def observe(p):
    """Phase 2: 事后标签分配 (赢家频率法)。

    对p.n_observe个训练样本做前向推理, 统计每个神经元对各类别的
    平均发放数。每个神经元被分配给响应最强的类别。

    这是唯一使用标签的阶段, 但不参与权重更新 (不修改网络)。

    参数:
        p: Params对象
    返回:
        assign: (n_exc,) 每个神经元的类别标签, -1表示未分配
    """
    X, Y = read_mnist(True, p.mnist_path)
    net = build_network(p, training=False)         # 推理模式(无STDP)
    responses = defaultdict(list)

    print(f"标签分配 {p.n_observe} 样本...")
    t0 = time.time()
    for i in range(p.n_observe):
        exc = show_sample(net, X[i % len(X)], p.intensity)
        responses[Y[i % len(Y)]].append(exc)        # 按真实类别分组
        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{p.n_observe}] {time.time()-t0:.0f}s")

    # 计算每个神经元对各类别的平均发放数
    res = np.zeros((10, p.n_exc))
    for cls, vals in responses.items():
        res[cls] = np.array(vals).mean(axis=0)

    # 赢家频率: 每个神经元分配给平均响应最强的类别
    assign = np.argmax(res, axis=0)
    save_npy(assign, p.data_path / 'assign.npy')

    for c in range(10):
        print(f"  类别{c}: {(assign==c).sum()}个神经元")
    return assign


def test(p):
    """Phase 3: 测试评估。

    对p.n_test个测试样本做前向推理, 按各类别神经元平均发放率投票。
    生成10×10混淆矩阵并计算准确率。

    推理方式: Top-1平均发放率 (各类别神经元平均发放数最大的获胜)

    参数:
        p: Params对象
    返回:
        acc: 总准确率 (0~1)
    """
    assign = np.load(p.data_path / 'assign.npy')
    groups = [np.where(assign == i)[0] for i in range(10)]  # 各类别神经元索引
    X, Y = read_mnist(False, p.mnist_path)                  # 加载测试集
    net = build_network(p, training=False)                   # 推理模式

    conf = np.zeros((10, 10))              # 混淆矩阵
    print(f"测试 {p.n_test} 样本...")
    t0 = time.time()
    for i in range(p.n_test):
        exc = show_sample(net, X[i % len(X)], p.intensity)
        # 预测: 各类别神经元平均发放数最大的类别
        guess = np.argmax([exc[grp].mean() for grp in groups])
        conf[Y[i % len(Y)], guess] += 1
        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{p.n_test}] {time.time()-t0:.0f}s")

    save_npy(conf, p.data_path / 'confusion.npy')

    # 输出结果
    acc = np.trace(conf) / np.sum(conf)                    # 对角线/总和
    print(f"\n★ 准确率: {acc:.2%}")
    for c in range(10):
        a = conf[c, c] / conf[c].sum() if conf[c].sum() > 0 else 0
        print(f"  {c}: {a:.1%}")
    return acc


# ═══════════════════════════════════════════════════════════════
# MLP读出层 (监督学习, 提升分类准确率)
# ═══════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """STDP特征 → 类别概率: 400 → 256 → ReLU → Dropout → 10。"""
    def __init__(self, n_in=400, n_hidden=256, n_out=10, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(n_hidden, n_out))

    def forward(self, x): return self.net(x)


def train_mlp(features, labels, n_epochs=300, lr=0.001, wd=1e-4):
    """训练MLP读出层。features: (N, 400) firing rate特征, labels: (N,)"""
    model = MLP(n_in=features.shape[1])
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(n_epochs):
        opt.zero_grad()
        loss = loss_fn(model(features), labels)
        loss.backward(); opt.step()
        if (epoch+1) % 50 == 0:
            acc = (model(features).argmax(1) == labels).float().mean()
            print(f"  epoch {epoch+1}: loss={loss.item():.4f} acc={acc:.2%}")
    return model


def extract_features(net, X, p):
    """提取STDP特征: 对每张图做前向推理, 收集各神经元发放计数。"""
    feats, labs = [], []
    for i in range(len(X)):
        pat = show_sample(net, X[i], p.intensity)
        feats.append(torch.from_numpy(pat.astype(np.float32)))
        labs.append(0)  # placeholder
    return torch.stack(feats)


# ═══════════════════════════════════════════════════════════════
# 主程序入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.chdir(HERE)
    p = Params()
    p.data_path.mkdir(parents=True, exist_ok=True)
    bseed(p.seed); rseed(p.seed); np.random.seed(p.seed); torch.manual_seed(p.seed)

    print("=" * 50)
    print("Brian2 STDP — Diehl & Cook (2015)")
    print(f"{p.n_exc}神经元 × {p.n_train}样本")
    print(f"输出: {p.data_path}")
    print("=" * 50)

    if "--test" in sys.argv:
        test(p)
    else:
        # Phase 1: STDP无监督训练
        train(p)

        # Phase 2: 标签分配
        observe(p)

        # Phase 3: 提取STDP特征 + 训练MLP读出层
        print("\nPhase 3: 提取特征 + MLP训练...")
        Xt, Yt = read_mnist(True, p.mnist_path)
        Xv, Yv = read_mnist(False, p.mnist_path)

        # 无监督投票基线
        test(p)

        # 提取特征 (用部分样本加速)
        net = build_network(p, training=False)
        N_MLP_TRAIN = 10000  # MLP训练用样本数
        N_MLP_TEST = 2000
        print(f"提取训练特征 ({N_MLP_TRAIN}样本)...")
        F_train = extract_features(net, Xt[:N_MLP_TRAIN], p)
        L_train = torch.tensor(Yt[:N_MLP_TRAIN], dtype=torch.long)
        print(f"提取测试特征 ({N_MLP_TEST}样本)...")
        F_test = extract_features(net, Xv[:N_MLP_TEST], p)
        L_test = torch.tensor(Yv[:N_MLP_TEST], dtype=torch.long)

        # 训练MLP
        print(f"\nMLP训练: {F_train.shape[1]} → 256 → ReLU → Dropout → 10")
        mlp = train_mlp(F_train, L_train, n_epochs=300)

        # 评估
        with torch.no_grad():
            train_preds = mlp(F_train).argmax(1)
            train_acc = (train_preds == L_train).float().mean()
            test_preds = mlp(F_test).argmax(1)
            test_acc = (test_preds == L_test).float().mean()
        print(f"\n★ MLP训练准确率: {train_acc:.2%}")
        print(f"★ MLP测试准确率: {test_acc:.2%} ({N_MLP_TEST}样本)")

        for c in range(10):
            mask = L_test == c
            if mask.sum() > 0:
                a = (test_preds[mask] == c).float().mean()
                print(f"  {c}: {a:.1%}")

        torch.save(mlp.state_dict(), p.data_path / 'mlp.pth')
