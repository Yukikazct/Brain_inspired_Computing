"""
脉冲神经网络 — Diehl & Cook (2015) 在线三重STDP
电导型LIF + 泊松脉冲编码 + 侧向抑制 + 内在可塑性 + 在线STDP + 权重归一化

无监督学习流程:
  1. 电导型LIF仿真(泊松脉冲输入) → 在线STDP权重更新
  2. 逐时间步多赢家WTA + 全局抑制 → 涌现式竞争
  3. 内在可塑性(自适应阈值) → 跨样本持久 → 防止单神经元主导
  4. 每样本权重归一化 → 稳态维持

STDP规则 (Diehl & Cook 2015 三重trace):
  - pre trace  (tau=20ms): 输入脉冲发放 → pre=1
  - post1 trace (tau=20ms): 神经元发放 → post1=1
  - post2 trace (tau=40ms): 神经元发放 → post2=1
  - LTD: 输入脉冲时, w[j,i] -= nu_pre × post1[i]      (nu_pre=0.0001)
  - LTP: 神经元发放时, w[j,i] += nu_post × pre[j] × post2_before[i]  (nu_post=0.01)

关键设计 (与论文严格对齐):
  - 所有神经元状态(v, ge, gi, theta, timer, traces)跨样本持久化
  - theta永不重置 → 频繁发放的神经元阈值升高 → 自然轮换
  - 150ms静息期让快速变量(v, ge, gi)自然衰减回静息值
  - 权重每样本前归一化(total=78)

参考: Diehl & Cook (2015) Front. Comput. Neurosci. 9, 99.
"""

import os
import numpy as np
from numba import jit, prange


# ==================== Numba JIT 仿真核心 ====================

@jit(nopython=True, cache=True, parallel=True)
def _simulate_lif_stdp(
    # -- 输入 --
    input_spikes,            # (n_input, n_stim_steps) float64 泊松脉冲
    n_stim_steps,            # 刺激期步数 (350ms)
    n_rest_steps,            # 静息期步数 (150ms)
    # -- 突触权重 (原地修改) --
    w_ie,                    # (n_exc, n_input)
    # -- 持久状态 (全部原地修改) --
    v_e,                     # (n_exc,) 膜电位
    ge_e,                    # (n_exc,) 兴奋电导
    gi_e,                    # (n_exc,) 抑制电导
    theta,                   # (n_exc,) 自适应阈值 ⭐跨样本持久
    timer_e,                 # (n_exc,) 不应期计时器
    gi_global,               # (1,) 全局抑制电导
    pre_trace,               # (n_input,) STDP pre trace
    post1_trace,             # (n_exc,) STDP post1 trace
    post2_trace,             # (n_exc,) STDP post2 trace
    # -- 参数 --
    v_rest_e, v_thresh_e, tau_m_e, refrac_e,
    tau_ge, tau_gi,
    theta_plus, tc_theta, theta_offset,
    inh_strength, dt,
    nu_pre, nu_post,
    tau_pre, tau_post1, tau_post2,
    w_max,
):
    """
    电导型LIF + 在线三重STDP + 逐时间步Top-1 WTA + 全局抑制。

    ⭐ 所有神经元状态在调用间持久 — theta积累跨样本生效。
    刺激期 (前n_stim_steps): 输入脉冲 + 在线STDP
    静息期 (后n_rest_steps): 零输入, 快速变量衰减, theta几乎不变
    """
    n_exc = w_ie.shape[0]
    n_input = w_ie.shape[1]

    # ---- 衰减因子 ----
    ge_decay = np.exp(-dt / tau_ge)
    gi_decay = np.exp(-dt / tau_gi)
    theta_decay = np.exp(-dt / tc_theta)
    pre_decay = np.exp(-dt / tau_pre)
    post1_decay = np.exp(-dt / tau_post1)
    post2_decay = np.exp(-dt / tau_post2)

    counts = np.zeros(n_exc, dtype=np.int32)

    for t in range(n_stim_steps + n_rest_steps):
        is_stim = t < n_stim_steps

        # ---- 电导衰减 ----
        ge_e *= ge_decay
        gi_e *= gi_decay
        gi_global[0] *= gi_decay

        # ---- theta衰减 (极慢: tc_theta=1e7ms ≈ 2.8小时) ----
        theta *= theta_decay

        # ---- 不应期计时 ----
        timer_e += dt

        # ---- STDP trace衰减 ----
        pre_trace *= pre_decay
        post1_trace *= post1_decay
        post2_trace *= post2_decay

        # ---- 突触输入 + STDP LTD ----
        if is_stim:
            inp_sp = np.ascontiguousarray(input_spikes[:, t])
            ge_e += np.dot(w_ie, inp_sp)

            # LTD: prange多核并行 (无内存分配, 比broadcast更高效)
            for j in prange(n_input):
                if inp_sp[j] > 0.5:
                    for i in range(n_exc):
                        w_ie[i, j] -= nu_pre * post1_trace[i]
                    pre_trace[j] = 1.0

        # ---- 膜电位更新 (多核并行) ----
        for i in prange(n_exc):
            i_syn_e = ge_e[i] * (-v_e[i])
            i_syn_i = (gi_e[i] + gi_global[0]) * (-100.0 - v_e[i])
            v_e[i] += dt * ((v_rest_e - v_e[i]) + i_syn_e + i_syn_i) / tau_m_e

        # ---- Top-K WTA: 每步最多3个赢家 (论文E→I→E涌现式竞争的近似) ----
        thresh = theta - theta_offset + v_thresh_e
        for _ in range(3):  # 最多3个同时发放
            best_j = -1
            best_v = -1e9
            for i in range(n_exc):
                if v_e[i] > thresh[i] and timer_e[i] >= refrac_e:
                    if v_e[i] > best_v:
                        best_v = v_e[i]
                        best_j = i
            if best_j < 0:
                break

            v_e[best_j] = v_rest_e
            theta[best_j] += theta_plus       # ⭐ 跨样本累积
            timer_e[best_j] = 0.0
            gi_global[0] += inh_strength

            if is_stim:
                counts[best_j] += 1

                post2_before = post2_trace[best_j]
                for j in range(n_input):
                    w_ie[best_j, j] += nu_post * pre_trace[j] * post2_before

            post1_trace[best_j] = 1.0
            post2_trace[best_j] = 1.0

    # ---- 权重裁剪 ----
    for i in prange(n_exc):
        for j in range(n_input):
            if w_ie[i, j] < 0.0:
                w_ie[i, j] = 0.0
            elif w_ie[i, j] > w_max:
                w_ie[i, j] = w_max

    return counts


@jit(nopython=True, cache=True, parallel=True)
def _simulate_lif_inference(
    input_spikes,            # (n_input, n_steps) float64
    w_ie,                    # (n_exc, n_input)
    n_steps,
    v_rest_e, v_thresh_e, tau_m_e, refrac_e,
    tau_ge, tau_gi,
    inh_strength, dt,
    persistent_theta,        # (n_exc,) — 训练后的自适应阈值, 推理时只读
    theta_offset,
):
    """
    纯推理LIF仿真 (无STDP, 无theta修改, 膜电位等每样本重置)。
    使用训练学到的theta作为各神经元的长期适应参数。
    """
    n_exc = w_ie.shape[0]

    # 每样本重置快速变量
    v_e = np.full(n_exc, v_rest_e, dtype=np.float64)
    ge_e = np.zeros(n_exc, dtype=np.float64)
    gi_e = np.zeros(n_exc, dtype=np.float64)
    timer_e = np.full(n_exc, refrac_e + 1.0, dtype=np.float64)

    ge_decay = np.exp(-dt / tau_ge)
    gi_decay = np.exp(-dt / tau_gi)

    counts = np.zeros(n_exc, dtype=np.int32)
    gi_global = 0.0

    for t in range(n_steps):
        ge_e *= ge_decay
        gi_e *= gi_decay
        gi_global *= gi_decay
        timer_e += dt

        inp_sp = np.ascontiguousarray(input_spikes[:, t])
        ge_e += np.dot(w_ie, inp_sp)

        for i in prange(n_exc):
            i_syn_e = ge_e[i] * (-v_e[i])
            i_syn_i = (gi_e[i] + gi_global) * (-100.0 - v_e[i])
            v_e[i] += dt * ((v_rest_e - v_e[i]) + i_syn_e + i_syn_i) / tau_m_e

        thresh = persistent_theta - theta_offset + v_thresh_e
        best_j = -1
        best_v = -1e9
        for i in range(n_exc):
            if v_e[i] > thresh[i] and timer_e[i] >= refrac_e:
                if v_e[i] > best_v:
                    best_v = v_e[i]
                    best_j = i

        if best_j >= 0:
            v_e[best_j] = v_rest_e
            timer_e[best_j] = 0.0
            counts[best_j] += 1
            gi_global += inh_strength

    return counts


# ==================== SNN 网络类 ====================

class SNN:
    """Diehl & Cook (2015) — 在线三重STDP脉冲神经网络 (状态持久)"""

    def __init__(self, n_input=784, n_excitatory=400,
                 dt_ms=1.0, duration_ms=350.0, rest_ms=150.0,
                 max_rate_hz=63.75,
                 v_rest_e=-65.0, v_thresh_e=-52.0, tau_m_e=100.0, refrac_e=5.0,
                 tau_ge=1.0, tau_gi=2.0,
                 nu_pre=0.0001, nu_post=0.01, w_max=1.0,
                 tau_pre=20.0, tau_post1=20.0, tau_post2=40.0,
                 theta_plus=0.05, tc_theta=1e7, theta_offset=20.0,
                 target_weight_sum=78.0,
                 inh_strength=17.0):
        self.n_input = n_input
        self.n_excitatory = n_excitatory
        self.dt = dt_ms
        self.duration_ms = duration_ms
        self.rest_ms = rest_ms
        self.max_rate_hz = max_rate_hz
        self.n_stim_steps = int(duration_ms / dt_ms)
        self.n_rest_steps = int(rest_ms / dt_ms)

        self.v_rest_e = v_rest_e
        self.v_thresh_e = v_thresh_e
        self.tau_m_e = tau_m_e
        self.refrac_e = refrac_e
        self.tau_ge = tau_ge
        self.tau_gi = tau_gi

        self.nu_pre = nu_pre
        self.nu_post = nu_post
        self.w_max = w_max
        self.tau_pre = tau_pre
        self.tau_post1 = tau_post1
        self.tau_post2 = tau_post2

        self.theta_plus = theta_plus
        self.tc_theta = tc_theta
        self.theta_offset = theta_offset
        self.target_weight_sum = target_weight_sum
        self.inh_strength = inh_strength

        # ---- 突触权重 (随机初始化) ----
        rng = np.random.RandomState(42)
        self.w_ie = (rng.rand(n_excitatory, n_input).astype(np.float64) + 0.01) * 0.3
        self.normalize_weights()

        # ---- ⭐ 持久神经元状态 (永不重置, 跨所有样本累积) ----
        self.v_e = np.full(n_excitatory, v_rest_e, dtype=np.float64)
        self.ge_e = np.zeros(n_excitatory, dtype=np.float64)
        self.gi_e = np.zeros(n_excitatory, dtype=np.float64)
        self.theta = np.full(n_excitatory, 20.0, dtype=np.float64)       # ⭐核心
        self.timer_e = np.full(n_excitatory, refrac_e + 1.0, dtype=np.float64)
        self.gi_global = np.zeros(1, dtype=np.float64)                   # 标量用1元素数组

        # ⭐ STDP traces 也跨样本持久
        self.pre_trace = np.zeros(n_input, dtype=np.float64)
        self.post1_trace = np.zeros(n_excitatory, dtype=np.float64)
        self.post2_trace = np.zeros(n_excitatory, dtype=np.float64)

        # 标签相关
        self.assigned_labels = np.full(n_excitatory, -1, dtype=np.int32)
        self.spike_counts_per_class = None
        self.input_intensity = 2.0

    # --- 权重管理 ---

    def normalize_weights(self):
        """权重归一化: 每个神经元的输入权重总和 = target_weight_sum"""
        sums = self.w_ie.sum(axis=1)
        for j in range(self.n_excitatory):
            if sums[j] > 0:
                self.w_ie[j] *= self.target_weight_sum / sums[j]

    # --- 前向仿真 ---

    def forward(self, image, train_mode=False, seed=None):
        """
        前向LIF仿真。

        train_mode=True: 刺激期+静息期, 在线STDP, 状态全部持久
        train_mode=False: 仅刺激期, 无STDP, 快速变量重置, theta只读
        """
        rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        rates = image * self.max_rate_hz * (self.input_intensity / 2.0)
        prob = rates * (self.dt / 1000.0)

        if train_mode:
            input_spikes = (rng.rand(self.n_input, self.n_stim_steps)
                            < prob[:, np.newaxis]).astype(np.float64)
            input_spikes = np.ascontiguousarray(input_spikes)

            return _simulate_lif_stdp(
                input_spikes,
                self.n_stim_steps, self.n_rest_steps,
                self.w_ie,
                self.v_e, self.ge_e, self.gi_e,
                self.theta, self.timer_e,
                self.gi_global,
                self.pre_trace, self.post1_trace, self.post2_trace,
                self.v_rest_e, self.v_thresh_e, self.tau_m_e, self.refrac_e,
                self.tau_ge, self.tau_gi,
                self.theta_plus, self.tc_theta, self.theta_offset,
                self.inh_strength, self.dt,
                self.nu_pre, self.nu_post,
                self.tau_pre, self.tau_post1, self.tau_post2,
                self.w_max,
            )
        else:
            # 推理: 重置快速变量, theta读自训练结果
            self.v_e.fill(self.v_rest_e)
            self.ge_e.fill(0.0)
            self.gi_e.fill(0.0)
            self.timer_e.fill(self.refrac_e + 1.0)
            self.gi_global.fill(0.0)

            n_steps = self.n_stim_steps
            input_spikes = (rng.rand(self.n_input, n_steps)
                            < prob[:, np.newaxis]).astype(np.float64)
            input_spikes = np.ascontiguousarray(input_spikes)

            return _simulate_lif_inference(
                input_spikes, self.w_ie,
                n_steps,
                self.v_rest_e, self.v_thresh_e, self.tau_m_e, self.refrac_e,
                self.tau_ge, self.tau_gi,
                self.inh_strength, self.dt,
                self.theta,      # ⭐ 训练学到的theta
                self.theta_offset,
            )

    def train_on_sample(self, image, seed=None):
        """
        无监督STDP学习 (单样本):
        1. 权重归一化 (每样本前 — 论文标准)
        2. 350ms刺激 + 在线STDP + 150ms静息
        3. 所有状态跨样本持久 (theta累积→自然轮换)
        """
        self.normalize_weights()
        return self.forward(image, train_mode=True, seed=seed)

    # --- 标签分配 ---

    def assign_labels(self, spike_counts, labels):
        """赢家频率分配标签 (消除亮度偏差)"""
        n_classes = 10
        self.spike_counts_per_class = np.zeros((n_classes, self.n_excitatory))
        winners = np.argmax(spike_counts, axis=1)
        for c in range(n_classes):
            mask = labels == c
            for w in winners[mask]:
                self.spike_counts_per_class[c, w] += 1
        self.assigned_labels = np.argmax(self.spike_counts_per_class, axis=0)
        never_won = self.spike_counts_per_class.sum(axis=0) == 0
        self.assigned_labels[never_won] = -1

    # --- 保存/加载 ---

    def save(self, path):
        """保存训练结果: 权重、标签、发放统计、theta、输入强度"""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        spc = (self.spike_counts_per_class
               if self.spike_counts_per_class is not None
               else np.zeros((10, self.n_excitatory)))
        np.savez(path,
                 w_ie=self.w_ie,
                 theta=self.theta,              # ⭐ 保存自适应阈值
                 assigned_labels=self.assigned_labels,
                 spike_counts_per_class=spc,
                 input_intensity=np.array(self.input_intensity))

    @staticmethod
    def load(path, **override_kwargs):
        """从文件加载训练结果, 返回SNN实例。"""
        data = np.load(path, allow_pickle=True)
        kwargs = dict(n_input=784, n_excitatory=data['w_ie'].shape[0])
        kwargs.update(override_kwargs)
        snn = SNN(**kwargs)
        snn.w_ie = data['w_ie']
        snn.assigned_labels = data['assigned_labels']
        snn.spike_counts_per_class = data['spike_counts_per_class']
        if 'theta' in data:
            snn.theta = data['theta']
        if 'input_intensity' in data:
            snn.input_intensity = float(data['input_intensity'])
        return snn

    def predict(self, image, top_k=10, seed=None):
        """Top-K赢家投票预测"""
        counts = self.forward(image, train_mode=False, seed=seed)
        if counts.sum() == 0:
            return -1, counts

        top_winners = np.argsort(counts)[-top_k:]
        votes = np.zeros(10)
        for w in top_winners:
            if self.assigned_labels[w] >= 0:
                votes[self.assigned_labels[w]] += counts[w]
        if votes.sum() > 0:
            return np.argmax(votes), counts
        return -1, counts
