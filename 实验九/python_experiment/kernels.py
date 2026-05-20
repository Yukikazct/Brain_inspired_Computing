"""Kernel functions: EPSP, PSS, IPSP for SRM/LIF neuron model."""

import numpy as np


def psp_kernel(t_array, tau_s, tau_m):
    """Double-exponential PSP kernel (Gerstner 2002).

    K(t) = (exp(-t/tau_m) - exp(-t/tau_s)) * norm
    normalized so max = 1.
    """
    t = np.asarray(t_array, dtype=np.float64)
    # Avoid division by zero when tau_s == tau_m
    if abs(tau_s - tau_m) < 1e-15:
        kernel = (t / tau_s) * np.exp(1 - t / tau_s)
    else:
        kernel = np.exp(-t / tau_m) - np.exp(-t / tau_s)
    kernel[t < 0] = 0.0
    # Normalize
    max_val = np.max(kernel)
    if max_val > 0:
        kernel = kernel / max_val
    return kernel


def make_epsp_kernel(params):
    """Build normalized EPSP kernel."""
    n_kernel = int(params.epspCut * params.tm / params.tmpResolution) + 1
    t_kernel = np.arange(n_kernel) * params.tmpResolution
    kernel = psp_kernel(t_kernel, params.ts, params.tm)
    # Normalize to max = 1
    max_val = np.max(kernel)
    if max_val > 0:
        kernel = kernel / max_val
    epspMaxTime = np.argmax(kernel) * params.tmpResolution
    return kernel, epspMaxTime


def make_pss_kernel(params):
    """Build post-synaptic spike (AHP) kernel.

    PSS = pss_coeff_exp * exp(-t/tm) + pss_coeff_dexp * psp_kernel(t, ts, tm)
    """
    n_kernel = int(params.epspCut * params.tm / params.tmpResolution) + 1
    t_kernel = np.arange(n_kernel) * params.tmpResolution
    dexp = psp_kernel(t_kernel, params.ts, params.tm)
    exp_term = np.exp(-t_kernel / params.tm)
    return params.pss_coeff_exp * exp_term + params.pss_coeff_dexp * dexp


def make_ipsp_kernel(params):
    """Build IPSP kernel (same as EPSP by default)."""
    n_kernel = int(params.epspCut * params.tm / params.tmpResolution) + 1
    t_kernel = np.arange(n_kernel) * params.tmpResolution
    if params.ipspKernelType == "epsp":
        kernel = psp_kernel(t_kernel, params.ts, params.tm)
        max_val = np.max(kernel)
        if max_val > 0:
            kernel = kernel / max_val
        return kernel
    else:
        # Custom IPSP kernel could go here
        return psp_kernel(t_kernel, params.ts, params.tm)
