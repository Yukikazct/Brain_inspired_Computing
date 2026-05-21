"""主仿真引擎：带STDP和侧向抑制的SRM/LIF神经元。

基于 Numba 加速实现，参照：
Masquelier T, Guyonneau R, Thorpe SJ (2009).
Competitive STDP-based Spike Pattern Learning. Neural Computation.
"""

import numpy as np
from numba import njit
from kernels import make_epsp_kernel, make_pss_kernel, make_ipsp_kernel


@njit
def _compute_pot_numba(epsp_amp, epsp_time, epsp_aff, n_epsp, N_EPSP,
                       ipsp_time, n_ipsp, N_IPSP,
                       weight, n_afferent,
                       epsp_kernel, n_epsp_kernel,
                       ipsp_kernel, n_ipsp_kernel,
                       pss_kernel, n_pss_kernel, use_pss,
                       firing_time, n_firing, N_FIRING,
                       threshold, inhib_strength, refractory_period,
                       tmp_resolution, tr_pot,
                       current_time, n_period):
    """Numba加速的膜电位计算。

    返回 (下次发放时间, 最大电位, 当前电位)。
    """
    period = np.arange(n_period) * tmp_resolution + current_time
    potential = np.zeros(n_period, dtype=np.float64)
    nE = n_epsp

    # 求和所有EPSP
    for e_idx in range(nE - 1, -1, -1):
        c = e_idx % N_EPSP
        w = epsp_amp[c]
        if w == 0.0:
            continue
        shift_f = (period[0] - epsp_time[c]) / tmp_resolution
        shift = int(shift_f + 0.5)
        if shift < 0:
            continue
        if shift >= n_epsp_kernel:
            break
        n_contrib = min(n_period, n_epsp_kernel - shift)
        for i in range(n_contrib):
            potential[i] += w * epsp_kernel[shift + i]

    # 求和所有IPSP
    if inhib_strength > 0:
        nI = n_ipsp
        for e_idx in range(nI - 1, -1, -1):
            c = e_idx % N_IPSP
            shift_f = (period[0] - ipsp_time[c]) / tmp_resolution
            shift = int(shift_f + 0.5)
            if shift < 0:
                continue
            if shift >= n_ipsp_kernel:
                break
            n_contrib = min(n_period, n_ipsp_kernel - shift)
            for i in range(n_contrib):
                potential[i] -= (inhib_strength * threshold *
                                 ipsp_kernel[shift + i])

    # 添加PSS（发放后电位）
    if use_pss and n_firing > 0:
        last_fire = firing_time[(n_firing - 1) % N_FIRING]
        shift_f = (period[0] - last_fire) / tmp_resolution
        shift = int(shift_f + 0.5)
        if 0 <= shift < n_pss_kernel:
            n_contrib = min(n_period, n_pss_kernel - shift)
            for i in range(n_contrib):
                potential[i] += threshold * pss_kernel[shift + i]

    current_pot = potential[0]
    max_pot = np.max(potential)

    next_firing = np.inf

    if max_pot >= threshold + tr_pot:
        # 检查不应期
        can_fire = True
        if n_firing > 0:
            last_fire = firing_time[(n_firing - 1) % N_FIRING]
            if current_time - last_fire < refractory_period:
                can_fire = False

        if can_fire:
            for i in range(n_period):
                if potential[i] >= threshold + tr_pot:
                    if i == 0:
                        next_firing = period[i]
                    else:
                        dp = potential[i] - potential[i - 1]
                        dt_period = period[i] - period[i - 1]
                        if abs(dp) > 1e-30:
                            overshoot = potential[i] - (threshold + tr_pot)
                            next_firing = period[i] - dt_period * overshoot / dp
                        else:
                            next_firing = period[i]
                    break

    return next_firing, max_pot, current_pot


class Neuron:
    """轻量级神经元数据容器（逻辑均在Numba函数中）。"""
    __slots__ = ('weight', 'epspAmplitude', 'epspTime', 'epspAfferent',
                 'nEpsp', 'N_EPSP', 'ipspTime', 'nIpsp', 'N_IPSP',
                 'nextFiring', 'firingTime', 'nFiring', 'N_FIRING',
                 'alreadyDepressed', 'maxPotential', 'currentPotential', 'trPot')

    def __init__(self, nAfferent, N_epsp, rng):
        self.weight = (1.0 - rng.rand(nAfferent) ** 1.0).astype(np.float64)
        self.weight = np.clip(self.weight, 0.0, 1.0)

        self.epspAmplitude = np.zeros(N_epsp, dtype=np.float64)
        self.epspTime = np.zeros(N_epsp, dtype=np.float64)
        self.epspAfferent = np.zeros(N_epsp, dtype=np.int32)
        self.nEpsp = 0
        self.N_EPSP = N_epsp

        self.ipspTime = np.zeros(10000, dtype=np.float64)
        self.nIpsp = 0
        self.N_IPSP = 10000

        self.nextFiring = np.inf
        self.firingTime = np.zeros(200000, dtype=np.float64)
        self.nFiring = 0
        self.N_FIRING = 200000

        self.alreadyDepressed = np.zeros(nAfferent, dtype=bool)

        self.maxPotential = 0.0
        self.currentPotential = 0.0
        self.trPot = 0.0


class Simulation:
    """多个SRM/LIF神经元的带STDP和侧向抑制事件驱动仿真。"""

    def __init__(self, params):
        self.params = params
        self.neurons = []
        self.epsp_kernel = None
        self.pss_kernel = None
        self.ipsp_kernel = None
        self.nEpspKernel = 0
        self.nPssKernel = 0
        self.nIpspKernel = 0
        self.epspMaxTime = 0.0
        self.nPeriod = 0

    def initialize(self, spikeList, rng_seed_offset=1):
        """初始化神经元和核。"""
        params = self.params

        # 构建核
        self.epsp_kernel, self.epspMaxTime = make_epsp_kernel(params)
        self.nEpspKernel = len(self.epsp_kernel)
        self.nPeriod = int(np.ceil(self.epspMaxTime / params.tmpResolution)) + 1

        if params.usePssKernel:
            self.pss_kernel = make_pss_kernel(params)
            self.nPssKernel = len(self.pss_kernel)
        else:
            self.pss_kernel = np.zeros(1)
            self.nPssKernel = 1

        self.ipsp_kernel = make_ipsp_kernel(params)
        self.nIpspKernel = len(self.ipsp_kernel)

        # EPSP缓冲区大小：根据脉冲率估计
        total_spikes = len(spikeList)
        if len(spikeList) > 0:
            total_time = spikeList[-1]
        else:
            total_time = 1.0
        avg_rate = total_spikes / total_time
        N_epsp = int(round(3.5 * params.epspCut * params.tm * avg_rate))
        N_epsp = max(N_epsp, 10000)

        # 创建神经元
        rng = np.random.RandomState(params.randomState + rng_seed_offset)
        self.neurons = [
            Neuron(params.nAfferent, N_epsp, rng)
            for _ in range(params.nNeuron)
        ]

    def run(self, spikeList, afferentList):
        """在脉冲序列上运行仿真（Numba加速内循环）。"""
        params = self.params
        neurons = self.neurons
        nNeuron = len(neurons)
        nSpikes = len(spikeList)
        nAfferent = params.nAfferent

        # 提取核数据
        epsp_kernel = self.epsp_kernel
        ipsp_kernel = self.ipsp_kernel
        pss_kernel = self.pss_kernel
        nEpspK = self.nEpspKernel
        nIpspK = self.nIpspKernel
        nPssK = self.nPssKernel
        nPeriod = self.nPeriod
        tmpRes = params.tmpResolution
        threshold = params.threshold
        inhibStrength = params.inhibStrength
        refPeriod = params.refractoryPeriod
        usePss = params.usePssKernel
        beSmart = params.beSmart

        # STDP参数
        stdp_t_pos = params.stdp_t_pos
        stdp_t_neg = params.stdp_t_neg
        stdp_a_pos = params.stdp_a_pos
        stdp_a_neg = params.stdp_a_neg
        stdp_cut = params.stdp_cut
        minWeight = params.minWeight

        nextFiring = np.inf
        nextOneToFire = -1

        report_interval = max(1, nSpikes // 100)

        for s_idx in range(nSpikes):
            if s_idx % report_interval == 0:
                pct = 100 * s_idx / nSpikes
                print(f"  进度: {pct:.0f}%", end="\r")

            spikeTime = spikeList[s_idx]
            afferent = afferentList[s_idx]

            # ── 处理在此输入脉冲之前发生的发放 ──
            while nextFiring <= spikeTime:
                t_fire = nextFiring
                winner = nextOneToFire
                winner_n = neurons[winner]

                # LTP
                self._ltp_numba(winner_n, t_fire,
                                stdp_t_pos, stdp_a_pos, stdp_cut, minWeight)

                # 记录发放
                winner_n.nFiring += 1
                winner_n.firingTime[(winner_n.nFiring - 1) % winner_n.N_FIRING] = t_fire

                # 清空缓冲区
                winner_n.nEpsp = 0
                winner_n.maxPotential = 0.0
                winner_n.nIpsp = 0
                winner_n.nextFiring = np.inf
                winner_n.alreadyDepressed[:] = False

                # 抑制其他神经元
                if inhibStrength > 0:
                    for i in range(nNeuron):
                        if i != winner:
                            n = neurons[i]
                            n.nIpsp += 1
                            cursor = n.nIpsp - 1
                            if cursor < n.N_IPSP:
                                n.ipspTime[cursor] = t_fire

                nextFiring = np.inf

            # ── 处理输入脉冲 ──
            for i in range(nNeuron):
                nrn = neurons[i]

                # 添加EPSP
                nrn.nEpsp += 1
                cursor = (nrn.nEpsp - 1) % nrn.N_EPSP
                nrn.epspAmplitude[cursor] = nrn.weight[afferent]
                nrn.epspTime[cursor] = spikeTime
                nrn.epspAfferent[cursor] = afferent

                # LTD
                if nrn.nFiring > 0 and not nrn.alreadyDepressed[afferent]:
                    lastFire = nrn.firingTime[(nrn.nFiring - 1) % nrn.N_FIRING]
                    dt_ltd = spikeTime - lastFire
                    if dt_ltd <= stdp_cut * stdp_t_neg:
                        dw = stdp_a_neg * np.exp(-dt_ltd / stdp_t_neg)
                        nrn.weight[afferent] = max(minWeight, nrn.weight[afferent] + dw)
                        nrn.alreadyDepressed[afferent] = True

                # 判断是否需要计算电位
                need_compute = True
                if beSmart:
                    if nrn.maxPotential + nrn.weight[afferent] < threshold + nrn.trPot:
                        nrn.maxPotential += nrn.epspAmplitude[cursor]
                        need_compute = False

                if need_compute:
                    nf, maxPot, curPot = _compute_pot_numba(
                        nrn.epspAmplitude, nrn.epspTime, nrn.epspAfferent,
                        nrn.nEpsp, nrn.N_EPSP,
                        nrn.ipspTime, nrn.nIpsp, nrn.N_IPSP,
                        nrn.weight, nAfferent,
                        epsp_kernel, nEpspK,
                        ipsp_kernel, nIpspK,
                        pss_kernel, nPssK, usePss,
                        nrn.firingTime, nrn.nFiring, nrn.N_FIRING,
                        threshold, inhibStrength, refPeriod,
                        tmpRes, nrn.trPot,
                        spikeTime, nPeriod,
                    )
                    nrn.nextFiring = nf
                    nrn.maxPotential = maxPot
                    nrn.currentPotential = curPot

            # 更新所有神经元中的下一次发放
            for i in range(nNeuron):
                nf = neurons[i].nextFiring
                if not np.isinf(nf) and nf < nextFiring:
                    nextFiring = nf
                    nextOneToFire = i

        print("  进度: 100%")

    def _ltp_numba(self, neuron, firingTime, stdp_t_pos, stdp_a_pos, stdp_cut, minWeight):
        """LTP：增强权值。内联实现以提高速度。"""
        weight = neuron.weight
        epspTime = neuron.epspTime
        epspAff = neuron.epspAfferent
        nE = neuron.nEpsp
        N_EPSP = neuron.N_EPSP
        nAfferent = len(weight)
        alreadyPot = np.zeros(nAfferent, dtype=bool)

        for e_idx in range(nE, 0, -1):
            c = (e_idx - 1) % N_EPSP
            dt = firingTime - epspTime[c]
            if dt > stdp_cut * stdp_t_pos:
                break
            aff = epspAff[c]
            if not alreadyPot[aff]:
                dw = stdp_a_pos * np.exp(-dt / stdp_t_pos)
                weight[aff] = min(1.0 - minWeight, weight[aff] + dw)
                alreadyPot[aff] = True
