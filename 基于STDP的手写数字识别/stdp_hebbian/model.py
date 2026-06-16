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
    """
    纯推理LIF仿真 (无权重更新) — 电导型LIF + 侧向抑制 + 自适应阈值 + 不应期。

    在每个时间步依次执行:
      1. 突触电导衰减 (exp(-dt/tau))
      2. 输入脉冲加权求和, 更新兴奋电导
      3. LIF膜电位更新: dv/dt = ((v_rest - v) + ge*(-v) + (gi+gi_global)*(-100 - v)) / tau_m
      4. 赢家通吃: 膜电位超过自适应阈值的神经元中选最高者发放
      5. 发放后: 重置膜电位, 提升阈值 (theta_plus), 全局抑制 (gi_global += inh_strength)

    参数:
        input_spikes: (n_input, n_steps) 输入脉冲矩阵, 值为0或1
        w_ie: (n_exc, n_input) 输入→兴奋权重矩阵
        n_steps: 仿真时间步数
        v_rest_e: 静息电位 (mV)
        v_thresh_e: 基础发放阈值 (mV)
        tau_m_e: 膜时间常数 (ms)
        refrac_e: 不应期时长 (ms)
        tau_ge: 兴奋电导衰减时间常数 (ms)
        tau_gi: 抑制电导衰减时间常数 (ms)
        theta_plus: 每次发放后阈值增量
        tc_theta: 自适应阈值衰减时间常数 (ms)
        theta_offset: 阈值偏置 (mV)
        inh_strength: 全局抑制强度
        dt: 仿真步长 (ms)

    返回:
        counts: (n_exc,) int32 每个神经元的总发放次数
    """
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
        """
        初始化 Hebbian 竞争学习脉冲网络。

        参数:
            n_input: 输入神经元数 (MNIST = 28×28 = 784)
            n_excitatory: 兴奋神经元总数, 每类约 n_excitatory/10 个
            dt_ms: 仿真步长 (ms), 越小精度越高但计算量越大
            duration_ms: 每样本仿真时长 (ms)
            max_rate_hz: 泊松编码最大发放率 (Hz), 像素值=1.0时对应此频率
            v_rest_e: 静息电位 (mV)
            v_thresh_e: 基础发放阈值 (mV), 膜电位超过此值触发脉冲
            tau_m_e: 膜时间常数 (ms), 控制膜电位变化速度
            refrac_e: 不应期时长 (ms), 发放后一段时间内不能再次发放
            tau_ge: 兴奋电导衰减时间常数 (ms), 越小衰减越快
            tau_gi: 抑制电导衰减时间常数 (ms)
            lr: Hebbian学习率, 控制每次权重更新的步长
            w_max: 权重上限, 防止权重无限增长
            theta_plus: 发放后自适应阈值增量, 实现内在可塑性
            tc_theta: 自适应阈值衰减时间常数 (ms), 越大衰减越慢
            theta_offset: 阈值偏置 (mV), 调节神经元兴奋性基线
            target_weight_sum: 每个神经元输入权重和的目标值, 归一化后保持此值
            ref_period_samples: 赢家不应期步数, 防止同一神经元连续获胜
            inh_strength: 全局抑制强度, 发放后抑制所有其他神经元
        """
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
        """
        权重归一化: 将每个神经元的输入权重向量和缩放到 target_weight_sum。

        目的: 模拟 STDP 的 LTD (长时程抑制) 效应 —— Hebbian LTP 增加权重后,
        归一化等效于所有未增强输入的相对权重衰减, 实现竞争学习。
        权重方向不变, 只有模长被缩放。
        """
        sums = self.w_ie.sum(axis=1)
        for j in range(self.n_excitatory):
            if sums[j] > 0:
                self.w_ie[j] *= self.target_weight_sum / sums[j]

    # ---- 前向仿真 ----

    def forward(self, image, seed=None):
        """
        前向推理: 将输入图像编码为泊松脉冲 → LIF仿真 → 返回发放计数。

        流程:
          1. 像素值 × max_rate_hz × (input_intensity/2.0) → 发放率 (Hz)
          2. 发放率 × dt/1000 → 每步发放概率
          3. 伯努利采样生成 (n_input, n_steps) 脉冲矩阵
          4. 送入 _simulate_lif 进行电导型LIF仿真
          5. 返回每个神经元的发放次数

        参数:
            image: (784,) 输入图像, 像素值 ∈ [0, 1]
            seed: 随机种子, 控制泊松编码的随机性

        返回:
            counts: (n_excitatory,) int32 每个兴奋神经元的发放次数
        """
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
        """
        Hebbian竞争学习 (单样本) — 前向推理 → 选赢家 → Hebbian更新 → 权重归一化。

        流程:
          1. forward(): 泊松编码 + LIF仿真, 得到发放计数
          2. 屏蔽处于不应期的神经元 (防止同一神经元连续获胜)
          3. 赢家 = argmax(屏蔽后的发放计数)
          4. Hebbian LTP: w[赢家] += lr × image (权重向输入模式靠拢)
          5. 权重裁剪到 [0, w_max]
          6. 权重归一化: w[赢家] *= target_weight_sum / sum(w[赢家]) (LTD等效)
          7. 更新不应期计数器: 赢家置为 ref_period_samples, 其余递减

        参数:
            image: (784,) 输入图像, 像素值 ∈ [0, 1]
            seed: 随机种子, 控制泊松编码

        返回:
            counts: (n_excitatory,) 本次前向推理的发放计数
        """
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
        """
        标签分配: 基于多次前向推理的发放统计, 为每个兴奋神经元分配数字类别。

        机制:
          对每个输入样本, 取发放最多的神经元作为"赢家", 该赢家获得该样本标签的一票。
          统计完所有样本后, 每个神经元获得各类别的得票数 (spike_counts_per_class[类别, 神经元])。
          最终标签 = argmax(投票数), 即神经元最常响应的数字类别。
          从未获胜的神经元标记为 -1 (未分配)。

        参数:
            spike_counts: (n_samples, n_excitatory) 每个样本对应的发放计数
            labels: (n_samples,) 每个样本的真实标签
        """
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
        """
        保存模型到 .npz 文件。

        保存内容:
          - w_ie: (n_exc, n_input) 输入→兴奋权重矩阵
          - assigned_labels: (n_exc,) 每个神经元的类别标签 (-1=未分配)
          - spike_counts_per_class: (10, n_exc) 各类别投票统计
          - input_intensity: 当前输入强度标量

        参数:
            path: 保存路径 (.npz)
        """
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
        """
        从 .npz 文件加载模型。

        先读取权重文件确定 n_excitatory, 构造 SNN 实例后再恢复所有权重和元数据。
        override_kwargs 可用于覆盖默认超参数 (如 n_input, dt_ms 等)。

        参数:
            path: 模型文件路径 (.npz)
            **override_kwargs: 覆盖 SNN 构造函数的任意参数

        返回:
            snn: 恢复的 SNN 实例
        """
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
        """
        预测输入图像的数字类别 — Top-K 投票机制。

        流程:
          1. forward(): 前向推理, 得到每个神经元的发放次数
          2. 取发放最多的 top_k 个神经元作为"投票委员会"
          3. 按发放次数加权投票: 每个神经元投给其分配的标签, 票数 = 发放次数
          4. 得票最多的类别即为预测结果

        参数:
            image: (784,) 输入图像, 像素值 ∈ [0, 1]
            top_k: 参与投票的神经元数量 (默认10)
            seed: 随机种子

        返回:
            (pred_label, counts): pred_label 为预测的数字 (0-9), -1 表示无发放无法预测;
                                  counts 为 (n_exc,) 发放计数数组
        """
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
