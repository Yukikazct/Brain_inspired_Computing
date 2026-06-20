"""STDP仿真参数配置 — Diehl & Cook (2015)。

使用: from parameters import Params; p = Params()
"""

from pathlib import Path

class Params:
    """Diehl & Cook (2015) STDP 仿真参数容器。"""

    def __init__(self, **kwargs):
        # ── 网络结构 ──
        self.n_input    = kwargs.get("n_input", 784)   # MNIST像素数
        self.n_exc      = kwargs.get("n_exc", 400)     # 兴奋神经元数
        self.n_inh      = self.n_exc                   # 抑制神经元数

        # ── 仿真时间 ──
        self.dt         = kwargs.get("dt", 0.5)        # 时间步长 (ms)
        self.duration   = kwargs.get("duration", 350.0) # 刺激时长 (ms)
        self.rest       = kwargs.get("rest", 150.0)    # 静息时长 (ms)
        self.intensity  = kwargs.get("intensity", 2.0)  # 输入强度

        # ── 兴奋神经元LIF参数 ──
        self.v_rest_e   = -65.0; self.v_thresh_e = -52.0; self.v_reset_e = -65.0
        self.refrac_e   = 5.0;   self.tau_m_e    = 100.0
        self.tau_ge     = 1.0;   self.tau_gi     = 2.0

        # ── 抑制神经元LIF参数 ──
        self.v_rest_i   = -60.0; self.v_thresh_i = -40.0; self.v_reset_i = -45.0
        self.refrac_i   = 2.0;   self.tau_m_i    = 10.0

        # ── 在线三重trace STDP ──
        self.nu_pre     = kwargs.get("nu_pre", 0.0001)   # LTD学习率
        self.nu_post    = kwargs.get("nu_post", 0.01)     # LTP学习率
        self.w_max      = kwargs.get("w_max", 1.0)
        self.tau_pre    = kwargs.get("tau_pre", 20.0)
        self.tau_post1  = kwargs.get("tau_post1", 20.0)
        self.tau_post2  = kwargs.get("tau_post2", 40.0)

        # ── 内在可塑性 ──
        self.theta_plus   = kwargs.get("theta_plus", 0.05)
        self.tc_theta     = kwargs.get("tc_theta", 1e7)
        self.theta_offset = kwargs.get("theta_offset", 20.0)

        # ── E→I→E侧向抑制 ──
        self.w_exc_inh  = kwargs.get("w_exc_inh", 10.4)
        self.w_inh_exc  = kwargs.get("w_inh_exc", 17.0)
        self.target_sum = kwargs.get("target_sum", 78.0)

        # ── 训练控制 ──
        self.n_train    = kwargs.get("n_train", 10000)
        self.n_observe  = kwargs.get("n_observe", 5000)
        self.n_test     = kwargs.get("n_test", 10000)
        self.seed       = kwargs.get("seed", 42)

        # ── 数据路径 ──
        self.mnist_path = kwargs.get("mnist_path", Path('../data'))
        self.data_path  = kwargs.get("data_path", Path('../data/stdp_full'))
