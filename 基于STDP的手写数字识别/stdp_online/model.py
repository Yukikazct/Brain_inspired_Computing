"""基于STDP的手写数字识别 — 在线三重STDP + WTA竞争学习。

参考:
  Diehl PU, Cook M (2015). Unsupervised learning of digit recognition
  using spike-timing-dependent plasticity. Front. Comput. Neurosci. 9:99.
"""

import numpy as np
from numba import njit, prange


# ═══════════════════════════════════════════════════════════════
# 参数配置
# ═══════════════════════════════════════════════════════════════

class Params:
    """Diehl & Cook (2015) STDP 仿真参数。"""

    def __init__(self, **kwargs):
        # ── 网络结构 ──
        self.n_input   = kwargs.get("n_input", 784)
        self.n_exc     = kwargs.get("n_exc", 400)

        # ── 仿真 ──
        self.dt        = kwargs.get("dt", 1.0)            # 时间步长 (ms)
        self.duration  = kwargs.get("duration", 350.0)    # 刺激时长 (ms)
        self.max_rate  = kwargs.get("max_rate", 63.75)    # 最大泊松率 (Hz)

        # ── LIF 神经元 ──
        self.v_rest    = kwargs.get("v_rest", -65.0)      # 静息电位 (mV)
        self.v_thresh  = kwargs.get("v_thresh", -52.0)    # 发放阈值 (mV)
        self.tau_m     = kwargs.get("tau_m", 100.0)       # 膜时间常数 (ms)
        self.refrac    = kwargs.get("refrac", 5.0)        # 不应期 (ms)
        self.tau_ge    = kwargs.get("tau_ge", 1.0)        # 兴奋电导时间常数 (ms)
        self.tau_gi    = kwargs.get("tau_gi", 2.0)        # 抑制电导时间常数 (ms)

        # ── 在线三重 STDP ──
        self.nu_pre    = kwargs.get("nu_pre", 0.0001)     # LTD 学习率
        self.nu_post   = kwargs.get("nu_post", 0.01)      # LTP 学习率
        self.w_max     = kwargs.get("w_max", 1.0)         # 权重上限
        self.tau_pre   = kwargs.get("tau_pre", 20.0)      # pre trace 时间常数 (ms)
        self.tau_post1 = kwargs.get("tau_post1", 20.0)    # post1 trace 时间常数 (ms)
        self.tau_post2 = kwargs.get("tau_post2", 40.0)    # post2 trace 时间常数 (ms)

        # ── 内在可塑性 ──
        self.theta_plus   = kwargs.get("theta_plus", 0.05)
        self.tc_theta     = kwargs.get("tc_theta", 1e7)
        self.theta_offset = kwargs.get("theta_offset", 20.0)

        # ── WTA 侧向抑制 ──
        self.inh_strength = kwargs.get("inh_strength", 17.0)
        self.target_sum   = kwargs.get("target_sum", 78.0)

        # ── 训练 ──
        self.n_train       = kwargs.get("n_train", 60000)
        self.n_label       = kwargs.get("n_label", 30000)
        self.n_epochs      = kwargs.get("n_epochs", 1)
        self.start_intensity = kwargs.get("start_intensity", 2.0)
        self.min_spikes     = kwargs.get("min_spikes", 5)
        self.seed           = kwargs.get("seed", 42)


# ═══════════════════════════════════════════════════════════════
# Numba 加速仿真核心
# ═══════════════════════════════════════════════════════════════

@njit(cache=True)
def _simulate_step_stdp(
    w, v, ge, gi, theta, timer, gi_global,
    pre_trace, post1, post2,
    inp_sp, is_stim,
    ge_decay, gi_decay, theta_decay,
    pre_decay, post1_decay, post2_decay,
    dt, v_rest, v_thresh, tau_m, refrac,
    theta_plus, theta_offset, inh_strength,
    nu_pre, nu_post, w_max, n_exc, n_inp,
):
    """单个时间步: 衰减 + 突触输入 + STDP + 膜电位 + WTA。
    所有状态原地修改。返回本步发放的神经元索引列表。
    """
    # ── 衰减 ──
    for i in range(n_exc):
        ge[i] *= ge_decay
        gi[i] *= gi_decay
    gi_global[0] *= gi_decay
    for i in range(n_exc):
        theta[i] *= theta_decay
        timer[i] += dt
    for j in range(n_inp):
        pre_trace[j] *= pre_decay
    for i in range(n_exc):
        post1[i] *= post1_decay
        post2[i] *= post2_decay

    fired = []

    if is_stim:
        # ── 突触输入 (np.dot → BLAS 多核加速) ──
        ge += np.dot(w, inp_sp)

        # ── STDP LTD: 输入脉冲触发 ──
        for j in range(n_inp):
            if inp_sp[j] > 0.5:
                for i in range(n_exc):
                    w[i, j] -= nu_pre * post1[i]
                pre_trace[j] = 1.0

    # ── 膜电位更新 (多核并行) ──
    for i in prange(n_exc):
        i_syn_e = ge[i] * (-v[i])
        i_syn_i = (gi[i] + gi_global[0]) * (-100.0 - v[i])
        v[i] += dt * ((v_rest - v[i]) + i_syn_e + i_syn_i) / tau_m

    # ── WTA 侧向抑制: 跨阈值神经元依次发放 ──
    thresh = theta - theta_offset + v_thresh
    for _ in range(50):
        best = -1; best_v = -1e9
        for i in range(n_exc):
            if v[i] > thresh[i] and timer[i] >= refrac:
                if v[i] > best_v:
                    best_v = v[i]; best = i
        if best < 0:
            break

        v[best] = v_rest
        theta[best] += theta_plus
        timer[best] = 0.0
        gi_global[0] += inh_strength
        fired.append(best)

        # ── STDP LTP: 神经元发放触发 ──
        if is_stim:
            p2b = post2[best]
            for j in range(n_inp):
                w[best, j] += nu_post * pre_trace[j] * p2b
        post1[best] = 1.0
        post2[best] = 1.0

    # ── 权重裁剪 ──
    for i in range(n_exc):
        for j in range(n_inp):
            if w[i, j] < 0.0:
                w[i, j] = 0.0
            elif w[i, j] > w_max:
                w[i, j] = w_max

    return fired


class Network:
    """Diehl & Cook (2015) 在线三重STDP脉冲神经网络。"""

    def __init__(self, p: Params = None):
        p = p or Params()
        self.p = p; self.n_exc = p.n_exc; self.n_inp = p.n_input
        self.n_steps = int(p.duration / p.dt)

        # ── 预计算衰减因子 ──
        self.ge_decay = np.exp(-p.dt / p.tau_ge)
        self.gi_decay = np.exp(-p.dt / p.tau_gi)
        self.theta_decay = np.exp(-p.dt / p.tc_theta)
        self.pre_decay = np.exp(-p.dt / p.tau_pre)
        self.post1_decay = np.exp(-p.dt / p.tau_post1)
        self.post2_decay = np.exp(-p.dt / p.tau_post2)

        # ── 权重初始化 ──
        rng = np.random.RandomState(p.seed)
        self.w = (rng.rand(p.n_exc, p.n_input).astype(np.float64) + 0.01) * 0.3
        self._normalize()

        # ── 持久神经元状态 ──
        self.v       = np.full(p.n_exc, p.v_rest, dtype=np.float64)
        self.ge      = np.zeros(p.n_exc, dtype=np.float64)
        self.gi      = np.zeros(p.n_exc, dtype=np.float64)
        self.theta   = np.full(p.n_exc, 20.0, dtype=np.float64)
        self.timer   = np.full(p.n_exc, p.refrac + 1.0, dtype=np.float64)
        self.gi_global = np.zeros(1, dtype=np.float64)
        self.pre_trace  = np.zeros(p.n_input, dtype=np.float64)
        self.post1      = np.zeros(p.n_exc, dtype=np.float64)
        self.post2      = np.zeros(p.n_exc, dtype=np.float64)

        # ── 标签 ──
        self.labels  = np.full(p.n_exc, -1, dtype=np.int32)
        self.per_class = None
        self.intensity = p.start_intensity

    def _normalize(self):
        s = self.w.sum(axis=1)
        for j in range(self.n_exc):
            if s[j] > 0:
                self.w[j] *= self.p.target_sum / s[j]

    def forward(self, image, train=False, seed=None):
        """前向仿真。train=True 时在线STDP更新权重。"""
        rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        rates = image * self.p.max_rate * (self.intensity / 2.0)
        prob = rates * (self.p.dt / 1000.0)
        spikes = (rng.rand(self.n_inp, self.n_steps)
                  < prob[:, np.newaxis]).astype(np.float64)
        spikes = np.ascontiguousarray(spikes)

        if not train:
            self.v.fill(self.p.v_rest); self.ge.fill(0.0); self.gi.fill(0.0)
            self.timer.fill(self.p.refrac + 1.0); self.gi_global.fill(0.0)

        counts = np.zeros(self.n_exc, dtype=np.int32)
        for t in range(self.n_steps):
            inp = spikes[:, t]
            fired = _simulate_step_stdp(
                self.w, self.v, self.ge, self.gi, self.theta, self.timer, self.gi_global,
                self.pre_trace, self.post1, self.post2,
                inp, train,
                self.ge_decay, self.gi_decay, self.theta_decay,
                self.pre_decay, self.post1_decay, self.post2_decay,
                self.p.dt, self.p.v_rest, self.p.v_thresh, self.p.tau_m, self.p.refrac,
                self.p.theta_plus, self.p.theta_offset, self.p.inh_strength,
                self.p.nu_pre, self.p.nu_post, self.p.w_max, self.n_exc, self.n_inp,
            )
            for idx in fired:
                counts[idx] += 1
        return counts

    def train_one(self, image, seed=None):
        self._normalize()
        return self.forward(image, train=True, seed=seed)

    def assign_labels(self, spike_counts, labels):
        n_cls = 10
        self.per_class = np.zeros((n_cls, self.n_exc))
        winners = np.argmax(spike_counts, axis=1)
        for c in range(n_cls):
            mask = labels == c
            for w in winners[mask]:
                self.per_class[c, w] += 1
        self.labels = np.argmax(self.per_class, axis=0)
        self.labels[self.per_class.sum(axis=0) == 0] = -1

    def predict(self, image, top_k=10, seed=None):
        counts = self.forward(image, train=False, seed=seed)
        if counts.sum() == 0:
            return -1, counts
        top = np.argsort(counts)[-top_k:]
        votes = np.zeros(10)
        for w in top:
            if self.labels[w] >= 0:
                votes[self.labels[w]] += counts[w]
        return (np.argmax(votes), counts) if votes.sum() > 0 else (-1, counts)

    def save(self, path):
        import os; os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        spc = self.per_class if self.per_class is not None else np.zeros((10, self.n_exc))
        np.savez(path, w=self.w, theta=self.theta, labels=self.labels,
                 per_class=spc, intensity=np.array(self.intensity))

    @staticmethod
    def load(path, **kw):
        d = np.load(path, allow_pickle=True)
        p = Params(n_exc=d['w'].shape[0], **kw)
        net = Network(p)
        net.w = d['w']; net.labels = d['labels']; net.per_class = d['per_class']
        if 'theta' in d: net.theta = d['theta']
        if 'intensity' in d: net.intensity = float(d['intensity'])
        return net
