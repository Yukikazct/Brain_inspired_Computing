"""Parameters for the Competitive STDP Spike Pattern Learning experiment.

Based on: Masquelier T, Guyonneau R, Thorpe SJ (2009).
Competitive STDP-based Spike Pattern Learning. Neural Computation.
"""

import numpy as np


class Params:
    """Container for all simulation parameters."""

    def __init__(self, **kwargs):
        # ── Random state ──────────────────────────────────────────────
        self.randomState = kwargs.get("randomState", 0)

        # ── STDP ──────────────────────────────────────────────────────
        self.stdp_t_pos = kwargs.get("stdp_t_pos", 16.8e-3)   # LTP tau (s)
        self.stdp_t_neg = kwargs.get("stdp_t_neg", 33.7e-3)   # LTD tau (s)
        self.stdp_a_pos = kwargs.get("stdp_a_pos", 2 ** -5)   # LTP learning rate
        self.stdp_a_neg = kwargs.get("stdp_a_neg", -0.85 * (2 ** -5))  # LTD learning rate
        self.stdp_cut = kwargs.get("stdp_cut", 7)             # cut in units of tau
        self.minWeight = kwargs.get("minWeight", 0.0)

        # ── EPSP / PSS / IPSP kernels ─────────────────────────────────
        self.tm = kwargs.get("tm", 10e-3)          # membrane time constant (s)
        self.ts = kwargs.get("ts", 2.5e-3)         # synapse time constant (s)
        self.epspCut = kwargs.get("epspCut", 7)    # cut in units of tm
        self.tmpResolution = kwargs.get("tmpResolution", 1e-3)  # temporal resolution (s)
        self.refractoryPeriod = kwargs.get("refractoryPeriod", 5e-3)  # (s)
        self.usePssKernel = kwargs.get("usePssKernel", True)

        # PSS coefficients
        self.pss_coeff_exp = kwargs.get("pss_coeff_exp", 2.0)
        self.pss_coeff_dexp = kwargs.get("pss_coeff_dexp", -3.0)

        # ── Inhibition ────────────────────────────────────────────────
        self.inhibStrength = kwargs.get("inhibStrength", 0.25)
        self.ipspKernelType = kwargs.get("ipspKernelType", "epsp")  # same as EPSP

        # ── Spike train ───────────────────────────────────────────────
        self.nPattern = kwargs.get("nPattern", 1)
        self.nAfferent = kwargs.get("nAfferent", 2000)
        self.nCopyPasteAfferent = kwargs.get(
            "nCopyPasteAfferent", 1000
        )  # afferents per pattern
        self.dt = kwargs.get("dt", 1e-3)                     # time step (s)
        self.maxFiringRate = kwargs.get("maxFiringRate", 90) # Hz
        self.spontaneousActivity = kwargs.get("spontaneousActivity", 10)  # Hz
        self.copyPasteDuration = kwargs.get("copyPasteDuration", 50e-3)   # pattern length (s)
        self.jitter = kwargs.get("jitter", 1e-3)             # Gaussian jitter std (s)
        self.spikeDeletion = kwargs.get("spikeDeletion", 0.0)
        self.maxTimeWithoutSpike = kwargs.get("maxTimeWithoutSpike", 50e-3)
        self.patternFreq = kwargs.get("patternFreq", 1.0 / 3.0)
        self.oscillations = kwargs.get("oscillations", False)

        # ── Neuron ────────────────────────────────────────────────────
        self.nNeuron = kwargs.get("nNeuron", 3)
        self.threshold = kwargs.get(
            "threshold", 0.55 * self.nCopyPasteAfferent
        )
        self.nuThr = kwargs.get("nuThr", np.inf)
        self.nRun = kwargs.get("nRun", 1)

        # ── Fixed firing mode (for diagnostics) ───────────────────────
        self.fixedFiringMode = kwargs.get("fixedFiringMode", False)
        self.fixedFiringLatency = kwargs.get("fixedFiringLatency", 10e-3)
        self.fixedFiringPeriod = kwargs.get("fixedFiringPeriod", 150e-3)

        # ── Simulation control ────────────────────────────────────────
        self.beSmart = kwargs.get("beSmart", True)
        self.dump = kwargs.get("dump", False)

        # ── Derived: total simulation time ────────────────────────────
        # T = nPattern * (500/patternFreq) * copyPasteDuration
        self.T = kwargs.get(
            "T",
            self.nPattern * (500.0 / self.patternFreq) * self.copyPasteDuration,
        )

        # ── Pattern positions ─────────────────────────────────────────
        self._generate_pattern_positions()

    def _generate_pattern_positions(self):
        """Generate random pattern insertion positions (as in param.m)."""
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

        # Convert to flat lists for compatibility
        self._posCopyPasteLists = {
            p: np.array(v, dtype=int)
            for p, v in self.posCopyPaste.items()
        }

    @property
    def total_simulation_time(self):
        return self.T * self.nRun

    def derive_kernel_params(self):
        """Compute derived kernel parameters."""
        n_kernel = int(self.epspCut * self.tm / self.tmpResolution) + 1
        return n_kernel
