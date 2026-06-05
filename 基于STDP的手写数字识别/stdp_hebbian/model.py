"""
STDP速率近似 — Hebbian LTP + 权重归一化 (竞争学习)
电导型LIF + 泊松编码 + 侧向抑制 + 内在可塑性 + 不应期轮换

理论依据 (Zenke et al., 2015):
  STDP稳态平衡 ⟺ Hebbian LTP + 权重归一化(LTD等效)
  权重向量方向收敛到输入模式聚类中心

学习流程:
  1. LIF仿真 → 发放计数
  2. 赢家 = argmax(发放) (排除不应期神经元)
  3. Hebbian LTP: w[赢家] += lr × image
  4. 权重归一化: w[赢家] *= target / sum(w[赢家])
"""

import os
import numpy as np
from numba import jit, prange


@jit(nopython=True, cache=True)
def _simulate_lif(
    input_spikes,            # (n_input, n_steps) float64
    w_ie,                    # (n_exc, n_input)
    n_steps,
    v_rest_e, v_thresh_e, tau_m_e, refrac_e,
    tau_ge, tau_gi,
    theta_plus, tc_theta, theta_offset,
    inh_strength, dt,
):
    """纯推理LIF仿真 (无权重更新), 返回发放计数"""
    n_exc = w_ie.shape[0]

    v_e = np.full(n_exc, v_rest_e, dtype=np.float64)
    ge_e = np.zeros(n_exc, dtype=np.float64)
    gi_e = np.zeros(n_exc, dtype=np.float64)
    theta = np.full(n_exc, 20.0, dtype=np.float64)
    timer_e = np.full(n_exc, refrac_e + 1.0, dtype=np.float64)

    ge_decay = np.exp(-dt / tau_ge)
    gi_decay = np.exp(-dt / tau_gi)
    theta_decay = np.exp(-dt / tc_theta)

    counts = np.zeros(n_exc, dtype=np.int32)
    gi_global = 0.0

    for t in range(n_steps):
        ge_e *= ge_decay
        gi_e *= gi_decay
        gi_global *= gi_decay
        theta *= theta_decay
        timer_e += dt

        inp_sp = np.ascontiguousarray(input_spikes[:, t])
        ge_e += np.dot(w_ie, inp_sp)

        for i in prange(n_exc):
            i_syn_e = ge_e[i] * (-v_e[i])
            i_syn_i = (gi_e[i] + gi_global) * (-100.0 - v_e[i])
            v_e[i] += dt * ((v_rest_e - v_e[i]) + i_syn_e + i_syn_i) / tau_m_e

        thresh = theta - theta_offset + v_thresh_e
        best_j = -1
        best_v = -1e9
        for i in range(n_exc):
            if v_e[i] > thresh[i] and timer_e[i] >= refrac_e:
                if v_e[i] > best_v:
                    best_v = v_e[i]
                    best_j = i

        if best_j >= 0:
            v_e[best_j] = v_rest_e
            theta[best_j] += theta_plus
            timer_e[best_j] = 0.0
            counts[best_j] += 1
            gi_global += inh_strength

    return counts


class SNN:
    """STDP速率近似 — Hebbian竞争学习脉冲网络"""

    def __init__(self, n_input=784, n_excitatory=2500,
                 dt_ms=1.0, duration_ms=50.0, max_rate_hz=400.0,
                 v_rest_e=-65.0, v_thresh_e=-52.0, tau_m_e=100.0, refrac_e=5.0,
                 tau_ge=1.0, tau_gi=2.0,
                 lr=0.02, w_max=1.0,
                 theta_plus=0.05, tc_theta=1e7, theta_offset=20.0,
                 target_weight_sum=78.0,
                 ref_period_samples=50, inh_strength=17.0):
        self.n_input = n_input
        self.n_excitatory = n_excitatory
        self.dt = dt_ms
        self.duration_ms = duration_ms
        self.max_rate_hz = max_rate_hz
        self.n_steps = int(duration_ms / dt_ms)

        self.v_rest_e = v_rest_e
        self.v_thresh_e = v_thresh_e
        self.tau_m_e = tau_m_e
        self.refrac_e = refrac_e
        self.tau_ge = tau_ge
        self.tau_gi = tau_gi

        self.lr = lr
        self.w_max = w_max

        self.theta_plus = theta_plus
        self.tc_theta = tc_theta
        self.theta_offset = theta_offset
        self.target_weight_sum = target_weight_sum
        self.ref_period_samples = ref_period_samples
        self.inh_strength = inh_strength

        # 不应期计数器
        self.refractory_counter = np.zeros(n_excitatory, dtype=np.int32)

        # 权重随机初始化
        rng = np.random.RandomState(42)
        self.w_ie = (rng.rand(n_excitatory, n_input).astype(np.float64) + 0.01) * 0.3
        self.normalize_weights()

        self.assigned_labels = np.full(n_excitatory, -1, dtype=np.int32)
        self.spike_counts_per_class = None
        self.input_intensity = 2.0

    # ---- 初始化 ----

    def initialize_from_exemplars(self, images, labels, n_per_class=None):
        """基于训练样本初始化权重 (每类均匀采样)"""
        if n_per_class is None:
            n_per_class = self.n_excitatory // 10
        rng = np.random.RandomState(42)
        for c in range(10):
            c_idx = np.where(labels == c)[0]
            chosen = rng.choice(c_idx, size=n_per_class, replace=True)
            start = c * n_per_class
            end = (c + 1) * n_per_class
            self.w_ie[start:end] = images[chosen].astype(np.float64)
        self.w_ie *= 0.3
        self.normalize_weights()
        self.w_ie += rng.normal(0, 0.01, self.w_ie.shape).astype(np.float64)
        np.clip(self.w_ie, 0.0, self.w_max, out=self.w_ie)
        self.normalize_weights()

    # ---- 权重管理 ----

    def normalize_weights(self):
        sums = self.w_ie.sum(axis=1)
        for j in range(self.n_excitatory):
            if sums[j] > 0:
                self.w_ie[j] *= self.target_weight_sum / sums[j]

    # ---- 前向仿真 ----

    def forward(self, image, seed=None):
        rates = image * self.max_rate_hz * (self.input_intensity / 2.0)
        rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        prob = rates * (self.dt / 1000.0)
        input_spikes = (rng.rand(self.n_input, self.n_steps)
                        < prob[:, np.newaxis]).astype(np.float64)
        input_spikes = np.ascontiguousarray(input_spikes)
        return _simulate_lif(
            input_spikes, self.w_ie, self.n_steps,
            self.v_rest_e, self.v_thresh_e, self.tau_m_e, self.refrac_e,
            self.tau_ge, self.tau_gi,
            self.theta_plus, self.tc_theta, self.theta_offset,
            self.inh_strength, self.dt,
        )

    # ---- 训练 ----

    def train_on_sample(self, image, seed=None):
        """Hebbian竞争学习 (单样本)"""
        counts = self.forward(image, seed=seed)

        if counts.sum() == 0:
            return counts

        # 屏蔽不应期神经元
        counts_masked = counts.copy().astype(np.float64)
        counts_masked[self.refractory_counter > 0] = -1.0
        winner = np.argmax(counts_masked)

        # 更新不应期
        self.refractory_counter = np.maximum(0, self.refractory_counter - 1)
        self.refractory_counter[winner] = self.ref_period_samples

        # Hebbian LTP
        self.w_ie[winner] += self.lr * image
        np.clip(self.w_ie[winner], 0.0, self.w_max, out=self.w_ie[winner])

        # 权重归一化 (LTD等效)
        wsum = self.w_ie[winner].sum()
        if wsum > 0:
            self.w_ie[winner] *= self.target_weight_sum / wsum

        return counts

    # ---- 标签分配 ----

    def assign_labels(self, spike_counts, labels):
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

    # ---- 保存/加载 ----

    def save(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        spc = (self.spike_counts_per_class
               if self.spike_counts_per_class is not None
               else np.zeros((10, self.n_excitatory)))
        np.savez(path,
                 w_ie=self.w_ie,
                 assigned_labels=self.assigned_labels,
                 spike_counts_per_class=spc,
                 input_intensity=np.array(self.input_intensity))

    @staticmethod
    def load(path, **override_kwargs):
        data = np.load(path, allow_pickle=True)
        kwargs = dict(n_input=784, n_excitatory=data['w_ie'].shape[0])
        kwargs.update(override_kwargs)
        snn = SNN(**kwargs)
        snn.w_ie = data['w_ie']
        snn.assigned_labels = data['assigned_labels']
        snn.spike_counts_per_class = data['spike_counts_per_class']
        if 'input_intensity' in data:
            snn.input_intensity = float(data['input_intensity'])
        return snn

    def predict(self, image, top_k=10, seed=None):
        counts = self.forward(image, seed=seed)
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
