#!/usr/bin/env python3
"""运行竞争STDP脉冲模式学习实验的全部四个实验。

实验结构：
  1. 单模式、3个独立输出神经元（无侧向抑制）
  2. 单模式、3个竞争输出神经元（有侧向抑制）
  3. 侧向抑制强度分析
  4. 多模式、多输出神经元

基于: Masquelier T, Guyonneau R, Thorpe SJ (2009).
Competitive STDP-based Spike Pattern Learning. Neural Computation.
"""

import sys
import os

# 将当前目录添加到路径
sys.path.insert(0, os.path.dirname(__file__))

from experiment1_independent import run_experiment1
from experiment2_competitive import run_experiment2
from experiment3_inhib_strength import run_experiment3
from experiment4_multipattern import run_experiment4


def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  竞争STDP脉冲模式学习                                      ║")
    print("║  Python复现 - 实验8-2                                      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # 解析命令行参数
    experiments_to_run = sys.argv[1:] if len(sys.argv) > 1 else ["all"]

    run_all = "all" in experiments_to_run

    # ── 实验1：独立神经元 ──────────────────────────────────
    if run_all or "1" in experiments_to_run:
        try:
            run_experiment1()
        except Exception as e:
            print(f"实验1 失败: {e}")
            import traceback
            traceback.print_exc()

    # ── 实验2：竞争神经元 ──────────────────────────────────
    if run_all or "2" in experiments_to_run:
        try:
            run_experiment2()
        except Exception as e:
            print(f"实验2 失败: {e}")
            import traceback
            traceback.print_exc()

    # ── 实验3：抑制强度分析 ────────────────────────────────
    if run_all or "3" in experiments_to_run:
        try:
            run_experiment3()
        except Exception as e:
            print(f"实验3 失败: {e}")
            import traceback
            traceback.print_exc()

    # ── 实验4：多模式 ──────────────────────────────────────
    if run_all or "4" in experiments_to_run:
        try:
            run_experiment4()
        except Exception as e:
            print(f"实验4 失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n全部实验完成。")


if __name__ == "__main__":
    main()
