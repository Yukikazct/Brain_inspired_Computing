"""核函数：SRM/LIF神经元模型的EPSP、PSS、IPSP核。"""

import numpy as np


def psp_kernel(t_array, tau_s, tau_m):
    """双指数PSP核（Gerstner 2002）。

    K(t) = (exp(-t/tau_m) - exp(-t/tau_s)) * 归一化系数
    归一化使最大值为1。
    """
    t = np.asarray(t_array, dtype=np.float64)
    if abs(tau_s - tau_m) < 1e-15:
        kernel = (t / tau_s) * np.exp(1 - t / tau_s)
    else:
        kernel = np.exp(-t / tau_m) - np.exp(-t / tau_s)
    kernel[t < 0] = 0.0
    max_val = np.max(kernel)
    if max_val > 0:
        kernel = kernel / max_val
    return kernel


def make_epsp_kernel(params):
    """构建归一化的EPSP核。"""
    n_kernel = int(params.epspCut * params.tm / params.tmpResolution) + 1
    t_kernel = np.arange(n_kernel) * params.tmpResolution
    kernel = psp_kernel(t_kernel, params.ts, params.tm)
    max_val = np.max(kernel)
    if max_val > 0:
        kernel = kernel / max_val
    epspMaxTime = np.argmax(kernel) * params.tmpResolution
    return kernel, epspMaxTime


def make_pss_kernel(params):
    """构建发放后电位（AHP）核。

    PSS = pss_coeff_exp * exp(-t/tm) + pss_coeff_dexp * psp_kernel(t, ts, tm)
    """
    n_kernel = int(params.epspCut * params.tm / params.tmpResolution) + 1
    t_kernel = np.arange(n_kernel) * params.tmpResolution
    dexp = psp_kernel(t_kernel, params.ts, params.tm)
    exp_term = np.exp(-t_kernel / params.tm)
    return params.pss_coeff_exp * exp_term + params.pss_coeff_dexp * dexp


def make_ipsp_kernel(params):
    """构建IPSP核（默认与EPSP相同）。"""
    n_kernel = int(params.epspCut * params.tm / params.tmpResolution) + 1
    t_kernel = np.arange(n_kernel) * params.tmpResolution
    if params.ipspKernelType == "epsp":
        kernel = psp_kernel(t_kernel, params.ts, params.tm)
        max_val = np.max(kernel)
        if max_val > 0:
            kernel = kernel / max_val
        return kernel
    else:
        return psp_kernel(t_kernel, params.ts, params.tm)
