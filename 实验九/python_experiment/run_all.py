#!/usr/bin/env python3
"""Run all experiments for the Competitive STDP Spike Pattern Learning lab.

Experiment structure:
  1. Single pattern, 3 independent output neurons (no lateral inhibition)
  2. Single pattern, 3 competitive output neurons (with lateral inhibition)
  3. Lateral inhibition strength analysis
  4. Multiple patterns, multiple output neurons

Based on: Masquelier T, Guyonneau R, Thorpe SJ (2009).
Competitive STDP-based Spike Pattern Learning. Neural Computation.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from experiment1_independent import run_experiment1
from experiment2_competitive import run_experiment2
from experiment3_inhib_strength import run_experiment3
from experiment4_multipattern import run_experiment4


def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Competitive STDP-Based Spike Pattern Learning              ║")
    print("║  Python reimplementation for Experiment 8-2                 ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # Parse command-line arguments
    experiments_to_run = sys.argv[1:] if len(sys.argv) > 1 else ["all"]

    run_all = "all" in experiments_to_run

    # ── Experiment 1 ──────────────────────────────────────────────
    if run_all or "1" in experiments_to_run:
        try:
            run_experiment1()
        except Exception as e:
            print(f"Experiment 1 failed: {e}")
            import traceback
            traceback.print_exc()

    # ── Experiment 2 ──────────────────────────────────────────────
    if run_all or "2" in experiments_to_run:
        try:
            run_experiment2()
        except Exception as e:
            print(f"Experiment 2 failed: {e}")
            import traceback
            traceback.print_exc()

    # ── Experiment 3 ──────────────────────────────────────────────
    if run_all or "3" in experiments_to_run:
        try:
            run_experiment3()
        except Exception as e:
            print(f"Experiment 3 failed: {e}")
            import traceback
            traceback.print_exc()

    # ── Experiment 4 ──────────────────────────────────────────────
    if run_all or "4" in experiments_to_run:
        try:
            run_experiment4()
        except Exception as e:
            print(f"Experiment 4 failed: {e}")
            import traceback
            traceback.print_exc()

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
