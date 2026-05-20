"""Shared utilities for all experiments."""

import os
import numpy as np
from parameters import Params

# Output directory for figures
FIGURE_DIR = os.path.join(os.path.dirname(__file__), "figures")


def ensure_figure_dir():
    os.makedirs(FIGURE_DIR, exist_ok=True)


def get_params_single_pattern(inhib_strength=0.0, random_state=0):
    """Get parameters for single-pattern experiments (Parts 1-3)."""
    return Params(
        randomState=random_state,
        nPattern=1,
        nNeuron=3,
        nAfferent=2000,
        nCopyPasteAfferent=1000,
        nRun=3,  # Repeat spike train 3 times for learning
        T=75.0,  # single pattern simulation time per run
        inhibStrength=inhib_strength,
        stdp_a_pos=2 ** -5,   # 0.03125
        stdp_a_neg=-0.85 * (2 ** -5),
        threshold=550.0,
        patternFreq=1.0 / 3.0,
        copyPasteDuration=50e-3,
        maxFiringRate=90,
        spontaneousActivity=10,
        jitter=1e-3,
    )


def get_params_multi_pattern(inhib_strength=0.25, random_state=0):
    """Get parameters for multi-pattern experiment (Part 4)."""
    return Params(
        randomState=random_state,
        nPattern=3,
        nNeuron=9,
        nAfferent=2000,
        nCopyPasteAfferent=1000,
        nRun=3,
        inhibStrength=inhib_strength,
        stdp_a_pos=2 ** -5,
        stdp_a_neg=-0.85 * (2 ** -5),
        threshold=550.0,
        patternFreq=1.0 / 3.0,
        copyPasteDuration=50e-3,
        maxFiringRate=90,
        spontaneousActivity=10,
        jitter=1e-3,
    )
