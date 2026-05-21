"""竞争STDP脉冲模式学习实验的参数配置。

基于: Masquelier T, Guyonneau R, Thorpe SJ (2009).
Competitive STDP-based Spike Pattern Learning. Neural Computation.
"""

import numpy as np


class Params:
    """所有仿真参数的容器。"""

    def __init__(self, **kwargs):
        # ── 随机状态 ──────────────────────────────────────────────
        self.randomState = kwargs.get("randomState", 0)

        # ── STDP ──────────────────────────────────────────────────
        self.stdp_t_pos = kwargs.get("stdp_t_pos", 16.8e-3)   # LTP时间常数(s)
        self.stdp_t_neg = kwargs.get("stdp_t_neg", 33.7e-3)   # LTD时间常数(s)
        self.stdp_a_pos = kwargs.get("stdp_a_pos", 2 ** -5)   # LTP学习率
        self.stdp_a_neg = kwargs.get("stdp_a_neg", -0.85 * (2 ** -5))  # LTD学习率
        self.stdp_cut = kwargs.get("stdp_cut", 7)             # 截断（tau倍数）
        self.minWeight = kwargs.get("minWeight", 0.0)

        # ── EPSP / PSS / IPSP 核 ─────────────────────────────────
        self.tm = kwargs.get("tm", 10e-3)          # 膜时间常数(s)
        self.ts = kwargs.get("ts", 2.5e-3)         # 突触时间常数(s)
        self.epspCut = kwargs.get("epspCut", 7)    # 截断（tm倍数）
        self.tmpResolution = kwargs.get("tmpResolution", 1e-3)  # 时间分辨率(s)
        self.refractoryPeriod = kwargs.get("refractoryPeriod", 5e-3)  # 不应期(s)
        self.usePssKernel = kwargs.get("usePssKernel", True)

        # PSS系数
        self.pss_coeff_exp = kwargs.get("pss_coeff_exp", 2.0)
        self.pss_coeff_dexp = kwargs.get("pss_coeff_dexp", -3.0)

        # ── 侧向抑制 ──────────────────────────────────────────────
        self.inhibStrength = kwargs.get("inhibStrength", 0.25)
        self.ipspKernelType = kwargs.get("ipspKernelType", "epsp")  # 与EPSP相同

        # ── 脉冲序列 ─────────────────────────────────────────────
        self.nPattern = kwargs.get("nPattern", 1)
        self.nAfferent = kwargs.get("nAfferent", 2000)
        self.nCopyPasteAfferent = kwargs.get(
            "nCopyPasteAfferent", 1000
        )  # 每个模式参与的输入神经元数
        self.dt = kwargs.get("dt", 1e-3)                     # 时间步长(s)
        self.maxFiringRate = kwargs.get("maxFiringRate", 90) # 最大发放率(Hz)
        self.spontaneousActivity = kwargs.get("spontaneousActivity", 10)  # 自发活动(Hz)
        self.copyPasteDuration = kwargs.get("copyPasteDuration", 50e-3)   # 模式长度(s)
        self.jitter = kwargs.get("jitter", 1e-3)             # 高斯抖动标准差(s)
        self.spikeDeletion = kwargs.get("spikeDeletion", 0.0)
        self.maxTimeWithoutSpike = kwargs.get("maxTimeWithoutSpike", 50e-3)
        self.patternFreq = kwargs.get("patternFreq", 1.0 / 3.0)
        self.oscillations = kwargs.get("oscillations", False)

        # ── 输出神经元 ────────────────────────────────────────────
        self.nNeuron = kwargs.get("nNeuron", 3)
        self.threshold = kwargs.get(
            "threshold", 0.55 * self.nCopyPasteAfferent
        )
        self.nuThr = kwargs.get("nuThr", np.inf)
        self.nRun = kwargs.get("nRun", 1)

        # ── 固定发放模式（调试用）────────────────────────────────
        self.fixedFiringMode = kwargs.get("fixedFiringMode", False)
        self.fixedFiringLatency = kwargs.get("fixedFiringLatency", 10e-3)
        self.fixedFiringPeriod = kwargs.get("fixedFiringPeriod", 150e-3)

        # ── 仿真控制 ──────────────────────────────────────────────
        self.beSmart = kwargs.get("beSmart", True)
        self.dump = kwargs.get("dump", False)

        # ── 推导：仿真总时长 ─────────────────────────────────────
        self.T = kwargs.get(
            "T",
            self.nPattern * (500.0 / self.patternFreq) * self.copyPasteDuration,
        )

        # ── 模式位置 ─────────────────────────────────────────────
        self._generate_pattern_positions()

    def _generate_pattern_positions(self):
        """生成随机的模式插入位置（与MATLAB param.m一致）。"""
        rng = np.random.RandomState(self.randomState)
        self.posCopyPaste = {p: [] for p in range(self.nPattern)}

        if self.patternFreq > 0:
            skip = False
            n_windows = int(round(self.T / self.copyPasteDuration))
            for p_idx in range(n_windows):
                if skip:
                    skip = False
                else:
                    if rng.rand() < 1.0 / (1.0 / self.patternFreq - 1):
                        pat = int(rng.rand() * self.nPattern)
                        self.posCopyPaste[pat].append(p_idx)
                        skip = True

        # 转换为列表以便兼容
        self._posCopyPasteLists = {
            p: np.array(v, dtype=int)
            for p, v in self.posCopyPaste.items()
        }

    @property
    def total_simulation_time(self):
        return self.T * self.nRun

    def derive_kernel_params(self):
        """计算推导的核参数。"""
        n_kernel = int(self.epspCut * self.tm / self.tmpResolution) + 1
        return n_kernel
