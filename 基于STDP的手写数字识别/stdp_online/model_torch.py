"""
脉冲神经网络 — Diehl & Cook (2015) 在线三重STDP (PyTorch MPS GPU加速版)
电导型LIF + 泊松脉冲编码 + 侧向抑制 + 内在可塑性 + 在线STDP + 权重归一化

与 model.py (Numba CPU) 功能完全一致, 但使用 PyTorch MPS 在 Apple GPU 上运行。
训练速度: ~3-5x CPU版 (400神经元), ~10x+ (1600+神经元)

用法:
  python main.py --gpu    使用GPU加速
  python main.py           默认使用CPU
"""

import os
import numpy as np
import torch


# ==================== 设备管理 ====================

def get_device(gpu=True):
    """获取计算设备"""
    if gpu and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ==================== SNN 网络类 ====================

class SNN:
    """Diehl & Cook (2015) — MPS GPU加速版"""

    def __init__(self, n_input=784, n_excitatory=400,
                 dt_ms=1.0, duration_ms=350.0, rest_ms=150.0,
                 max_rate_hz=63.75,
                 v_rest_e=-65.0, v_thresh_e=-52.0, tau_m_e=100.0, refrac_e=5.0,
                 tau_ge=1.0, tau_gi=2.0,
                 nu_pre=0.0001, nu_post=0.01, w_max=1.0,
                 tau_pre=20.0, tau_post1=20.0, tau_post2=40.0,
                 theta_plus=0.05, tc_theta=1e7, theta_offset=20.0,
                 target_weight_sum=78.0,
                 inh_strength=17.0,
                 gpu=True):
        self.n_input = n_input
        self.n_excitatory = n_excitatory
        self.dt = dt_ms
        self.duration_ms = duration_ms
        self.rest_ms = rest_ms
        self.max_rate_hz = max_rate_hz
        self.n_stim_steps = int(duration_ms / dt_ms)
        self.n_rest_steps = int(rest_ms / dt_ms)
        self.n_total_steps = self.n_stim_steps + self.n_rest_steps

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

        # ---- 设备 ----
        self.device = get_device(gpu)
        self._float_dtype = torch.float32  # MPS对float32最优
        print(f"  设备: {self.device} | 精度: {self._float_dtype}")

        # ---- 衰减因子 (预计算, 存GPU) ----
        self.ge_decay = torch.tensor(np.exp(-dt_ms / tau_ge), dtype=self._float_dtype, device=self.device)
        self.gi_decay = torch.tensor(np.exp(-dt_ms / tau_gi), dtype=self._float_dtype, device=self.device)
        self.theta_decay = torch.tensor(np.exp(-dt_ms / tc_theta), dtype=self._float_dtype, device=self.device)
        self.pre_decay = torch.tensor(np.exp(-dt_ms / tau_pre), dtype=self._float_dtype, device=self.device)
        self.post1_decay = torch.tensor(np.exp(-dt_ms / tau_post1), dtype=self._float_dtype, device=self.device)
        self.post2_decay = torch.tensor(np.exp(-dt_ms / tau_post2), dtype=self._float_dtype, device=self.device)

        # ---- 膜电位常数 (标量, 存GPU) ----
        self._v_rest = torch.tensor(v_rest_e, dtype=self._float_dtype, device=self.device)
        self._v_thresh = torch.tensor(v_thresh_e, dtype=self._float_dtype, device=self.device)
        self._refrac = torch.tensor(refrac_e, dtype=self._float_dtype, device=self.device)
        self._dt_tensor = torch.tensor(dt_ms, dtype=self._float_dtype, device=self.device)
        self._tau_m = torch.tensor(tau_m_e, dtype=self._float_dtype, device=self.device)
        self._v_inh_rev = torch.tensor(-100.0, dtype=self._float_dtype, device=self.device)
        self._theta_plus_t = torch.tensor(theta_plus, dtype=self._float_dtype, device=self.device)
        self._theta_offset_t = torch.tensor(theta_offset, dtype=self._float_dtype, device=self.device)
        self._inh_t = torch.tensor(inh_strength, dtype=self._float_dtype, device=self.device)
        self._nu_pre_t = torch.tensor(nu_pre, dtype=self._float_dtype, device=self.device)
        self._nu_post_t = torch.tensor(nu_post, dtype=self._float_dtype, device=self.device)
        self._one = torch.tensor(1.0, dtype=self._float_dtype, device=self.device)
        self._ninf = torch.tensor(-1e9, dtype=self._float_dtype, device=self.device)
        self._target_sum_t = torch.tensor(target_weight_sum, dtype=self._float_dtype, device=self.device)

        # ---- 突触权重 (GPU) ----
        rng = np.random.RandomState(42)
        w_np = (rng.rand(n_excitatory, n_input).astype(np.float32) + 0.01) * 0.3
        self.w_ie = torch.from_numpy(w_np).to(self.device)
        self.normalize_weights()

        # ---- ⭐ 持久神经元状态 (全部存GPU, 跨样本累积) ----
        self.v_e = torch.full((n_excitatory,), v_rest_e, dtype=self._float_dtype, device=self.device)
        self.ge_e = torch.zeros(n_excitatory, dtype=self._float_dtype, device=self.device)
        self.gi_e = torch.zeros(n_excitatory, dtype=self._float_dtype, device=self.device)
        self.theta = torch.full((n_excitatory,), 20.0, dtype=self._float_dtype, device=self.device)
        self.timer_e = torch.full((n_excitatory,), refrac_e + 1.0, dtype=self._float_dtype, device=self.device)
        self.gi_global = torch.tensor(0.0, dtype=self._float_dtype, device=self.device)

        # STDP traces
        self.pre_trace = torch.zeros(n_input, dtype=self._float_dtype, device=self.device)
        self.post1_trace = torch.zeros(n_excitatory, dtype=self._float_dtype, device=self.device)
        self.post2_trace = torch.zeros(n_excitatory, dtype=self._float_dtype, device=self.device)

        # 标签相关
        self.assigned_labels = np.full(n_excitatory, -1, dtype=np.int32)
        self.spike_counts_per_class = None
        self.input_intensity = 2.0

    # ==================== 权重管理 ====================

    def normalize_weights(self):
        """每神经元权重总和归一化到 target_weight_sum (GPU向量化)"""
        sums = self.w_ie.sum(dim=1)
        mask = sums > 0
        self.w_ie[mask] *= self._target_sum_t / sums[mask].unsqueeze(1)

    # ==================== 前向仿真 ====================

    def forward(self, image, train_mode=False, seed=None):
        """
        前向LIF仿真。image 是 numpy (784,) ∈ [0,1]。
        返回 numpy (n_exc,) int32 发放计数。
        """
        # 泊松输入编码
        rates = image.astype(np.float32) * self.max_rate_hz * (self.input_intensity / 2.0)
        prob = rates * (self.dt / 1000.0)

        if seed is not None:
            torch.manual_seed(seed)

        if train_mode:
            input_spikes = torch.rand(self.n_input, self.n_stim_steps, device=self.device)
            input_spikes = (input_spikes < torch.tensor(prob[:, None], device=self.device)).to(self._float_dtype)
            return self._simulate_stdp(input_spikes)
        else:
            # 推理: 重置快速变量, 保留theta
            self.v_e.fill_(self.v_rest_e)
            self.ge_e.zero_()
            self.gi_e.zero_()
            self.timer_e.fill_(self.refrac_e + 1.0)
            self.gi_global.fill_(0.0)

            n_steps = self.n_stim_steps
            input_spikes = torch.rand(self.n_input, n_steps, device=self.device)
            input_spikes = (input_spikes < torch.tensor(prob[:, None], device=self.device)).to(self._float_dtype)
            return self._simulate_inference(input_spikes)

    def _simulate_stdp(self, input_spikes):
        """
        在线STDP仿真 (GPU向量化)。
        刺激期: input_spikes有效 + STDP
        静息期: 零输入, 仅衰减
        所有状态原地修改, 跨调用持久。
        """
        n_exc = self.n_excitatory
        counts = torch.zeros(n_exc, dtype=torch.int32, device=self.device)

        for t in range(self.n_total_steps):
            is_stim = t < self.n_stim_steps

            # ---- 衰减 ----
            self.ge_e *= self.ge_decay
            self.gi_e *= self.gi_decay
            self.gi_global *= self.gi_decay
            self.theta *= self.theta_decay
            self.timer_e += self.dt

            self.pre_trace *= self.pre_decay
            self.post1_trace *= self.post1_decay
            self.post2_trace *= self.post2_decay

            # ---- 突触输入 + STDP LTD ----
            if is_stim:
                inp_sp = input_spikes[:, t]  # (n_input,)

                # 突触电导: ge += W @ spike
                self.ge_e += self.w_ie @ inp_sp

                # LTD: 每个输入脉冲触发, 全向量化
                # w[:, j] -= nu_pre * post1[:]  for all j where inp_sp[j] > 0
                self.w_ie -= self._nu_pre_t * self.post1_trace.unsqueeze(1) * inp_sp.unsqueeze(0)

                # pre_trace: 发放的输入设为1
                self.pre_trace = torch.where(inp_sp > 0.5, self._one, self.pre_trace)

            # ---- 膜电位更新 (全向量化) ----
            i_syn_e = self.ge_e * (-self.v_e)
            i_syn_i = (self.gi_e + self.gi_global) * (self._v_inh_rev - self.v_e)
            self.v_e += self._dt_tensor * (self._v_rest - self.v_e + i_syn_e + i_syn_i) / self._tau_m

            # ---- WTA: 超过阈值且不在不应期的神经元中选最大v ----
            thresh = self.theta - self._theta_offset_t + self._v_thresh
            eligible = (self.v_e > thresh) & (self.timer_e >= self._refrac)
            best_j = -1
            if eligible.any():
                v_masked = torch.where(eligible, self.v_e, self._ninf)
                best_j = torch.argmax(v_masked).item()

            if best_j >= 0:
                # 发放重置
                self.v_e[best_j] = self.v_rest_e
                self.theta[best_j] += self._theta_plus_t
                self.timer_e[best_j] = 0.0
                self.gi_global += self._inh_t

                if is_stim:
                    counts[best_j] += 1

                    # STDP LTP
                    post2_before = self.post2_trace[best_j].clone()
                    self.w_ie[best_j] += self._nu_post_t * self.pre_trace * post2_before

                # post traces
                self.post1_trace[best_j] = 1.0
                self.post2_trace[best_j] = 1.0

        # ---- 权重裁剪 ----
        self.w_ie.clamp_(0.0, self.w_max)

        return counts.cpu().numpy().astype(np.int32)

    def _simulate_inference(self, input_spikes):
        """纯推理仿真 (无STDP, theta只读)"""
        n_exc = self.n_excitatory
        n_steps = input_spikes.shape[1]
        counts = torch.zeros(n_exc, dtype=torch.int32, device=self.device)

        v_e = self.v_e.clone()
        ge_e = self.ge_e.clone()
        gi_e = self.gi_e.clone()
        timer_e = self.timer_e.clone()
        gi_global = torch.tensor(0.0, dtype=self._float_dtype, device=self.device)

        for t in range(n_steps):
            ge_e *= self.ge_decay
            gi_e *= self.gi_decay
            gi_global *= self.gi_decay
            timer_e += self.dt

            inp_sp = input_spikes[:, t]
            ge_e += self.w_ie @ inp_sp

            i_syn_e = ge_e * (-v_e)
            i_syn_i = (gi_e + gi_global) * (self._v_inh_rev - v_e)
            v_e += self._dt_tensor * (self._v_rest - v_e + i_syn_e + i_syn_i) / self._tau_m

            thresh = self.theta - self._theta_offset_t + self._v_thresh
            eligible = (v_e > thresh) & (timer_e >= self._refrac)
            if eligible.any():
                v_masked = torch.where(eligible, v_e, self._ninf)
                best_j = torch.argmax(v_masked).item()
                v_e[best_j] = self.v_rest_e
                timer_e[best_j] = 0.0
                counts[best_j] += 1
                gi_global += self._inh_t

        return counts.cpu().numpy().astype(np.int32)

    # ==================== 训练API ====================

    def train_on_sample(self, image, seed=None):
        """无监督STDP学习 (单样本)"""
        self.normalize_weights()
        return self.forward(image, train_mode=True, seed=seed)

    # ==================== 标签分配 ====================

    def assign_labels(self, spike_counts, labels):
        """赢家频率分配标签"""
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

    # ==================== 保存/加载 ====================

    def save(self, path):
        """保存训练结果"""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        spc = (self.spike_counts_per_class
               if self.spike_counts_per_class is not None
               else np.zeros((10, self.n_excitatory)))
        np.savez(path,
                 w_ie=self.w_ie.cpu().numpy(),
                 theta=self.theta.cpu().numpy(),
                 assigned_labels=self.assigned_labels,
                 spike_counts_per_class=spc,
                 input_intensity=np.array(self.input_intensity))

    @staticmethod
    def load(path, gpu=True, **override_kwargs):
        """从文件加载模型"""
        data = np.load(path, allow_pickle=True)
        kwargs = dict(n_input=784, n_excitatory=data['w_ie'].shape[0], gpu=gpu)
        kwargs.update(override_kwargs)
        snn = SNN(**kwargs)
        snn.w_ie = torch.from_numpy(data['w_ie']).to(snn.device)
        snn.assigned_labels = data['assigned_labels']
        snn.spike_counts_per_class = data['spike_counts_per_class']
        if 'theta' in data:
            snn.theta = torch.from_numpy(data['theta']).to(snn.device)
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
