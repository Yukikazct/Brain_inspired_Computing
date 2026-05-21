# 实验9：多个STDP神经元的竞争学习与多模式检测

基于 Masquelier et al. (2009) *Competitive STDP-Based Spike Pattern Learning* 的 Python 复现。

## 环境

```bash
pip install numpy matplotlib numba
```

## 快速运行

```bash
# 运行全部四个实验
python3 run_all.py

# 或单独运行
python3 experiment1_independent.py   # 实验1: 无侧向抑制
python3 experiment2_competitive.py   # 实验2: 有侧向抑制
python3 experiment3_inhib_strength.py # 实验3: 抑制强度分析 (耗时2-3h)
python3 experiment4_multipattern.py  # 实验4: 多模式多神经元
```

## 输出

所有图表输出至 `figures/`：

| 文件 | 内容 |
|------|------|
| `exp1_independent.png` | 无抑制三神经元潜伏期图 |
| `exp2_competitive.png` | 有抑制三神经元潜伏期图（堆叠现象） |
| `exp3_inhib_analysis.png` | 抑制强度 vs 潜伏期差异 |
| `exp4_multipattern.png` | 3模式×9神经元潜伏期矩阵 |

## 代码结构

```
├── parameters.py          # 全局参数配置
├── kernels.py             # EPSP / PSS / IPSP 核函数
├── spike_train.py         # 脉冲序列生成（copy-paste模式嵌入）
├── simulation.py          # SRM/LIF神经元 + STDP + 侧向抑制 (Numba加速)
├── analysis.py            # 潜伏期 / 命中率 / 误报率 统计
├── plotting.py            # 图表绘制
├── utils.py               # 共享参数预设
├── experiment1_independent.py
├── experiment2_competitive.py
├── experiment3_inhib_strength.py
├── experiment4_multipattern.py
└── run_all.py             # 主入口
```

## 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 输入神经元 | 2000 | 泊松背景发放 |
| 模式长度 | 50ms | copy-paste 嵌入 |
| 模式参与比例 | 50% | 每个模式涉及半数输入 |
| 膜时间常数 τₘ | 10ms | SRM/LIF |
| 突触时间常数 τₛ | 2.5ms | 双指数EPSP核 |
| 发放阈值 | 550 | |
| LTP τ₊ | 16.8ms | STDP |
| LTD τ₋ | 33.7ms | STDP |
| LTP a₊ | 0.03125 | 学习率 |
| LTD a₋ | -0.85×a₊ | 学习率 |
| 侧向抑制 β | 0.25 | (实验3扫描0.05-0.50) |

## 参考

Masquelier T, Guyonneau R, Thorpe SJ (2009). Competitive STDP-based Spike Pattern Learning. *Neural Computation*.
