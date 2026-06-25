# 类脑计算 — 课程实验与项目

> **学期课程仓库**：认知科学与类脑计算（Neuromorphic Computing / Brain-Inspired Computing）  
> 涵盖从基础神经元模型到脉冲神经网络（SNN）监督/无监督学习的完整实验体系，以及一个基于 STDP 的手写数字识别综合项目。

---

##  仓库结构总览

```
类脑计算/
├── 课件/                              # 课程讲义（13个PDF）
├── 实验一/                            # 环境配置（Anaconda/PyTorch）
├── 实验二/                            # McCulloch-Pitts 神经元模型
├── 实验三/                            # Hodgkin-Huxley & LIF 模型
├── 实验四/                            # 神经编码（Poisson / TTFS）
├── 实验五/                            # 神经可塑性与MNIST编码应用
├── 实验六/                            # STDP / 感知机 / 分类与回归
├── 实验七/                            # Hebb & STDP 突触可塑性
├── 实验八/                            # 单神经元STDP重复模式检测
├── 实验8_2/                           # 多神经元STDP竞争学习
├── 实验九/                            # 监督SNN学习（Tempotron / ReSuMe）
├── 基于STDP的手写数字识别/             # 🏆 综合项目（双方法对比）
├── data/                              # 共享数据目录
└── .gitignore
```

---

## 🧪 实验列表

### 实验一：环境配置
- **内容**：Anaconda 安装与管理、PyTorch 环境验证
- **文件**：[test.py](实验一/test.py) — PyTorch 安装验证脚本

### 实验二：McCulloch-Pitts 神经元
- **内容**：MP 神经元模型、逻辑门实现（AND/OR/NOT）、XOR 问题、感知机学习规则
- **文件**：
  - [ex2_1.py](实验二/ex2_1.py) — 加权求和 + 阈值判断
  - [ex2_2.py](实验二/ex2_2.py) — 逻辑门实现
  - [ex2_3.py](实验二/ex2_3.py) — XOR 问题分析
  - [ex2_4.py](实验二/ex2_4.py) — Rosenblatt 感知机

### 实验三：HH & LIF 模型
- **内容**：Hodgkin-Huxley 模型仿真、f-I 曲线、LIF 模型对比
- **文件**：
  - [exp3_1.py](实验三/exp3_1.py) — HH 模型完整仿真
  - [exp3_2.py](实验三/exp3_2.py) — HH 模型 f-I 曲线
  - [exp3_3.py](实验三/exp3_3.py) — LIF 模型实现
  - [exp3_4.py](实验三/exp3_4.py) — LIF 模型 f-I 曲线

### 实验四：神经编码
- **内容**：Poisson 脉冲序列生成、频率编码 vs 时间编码、TTFS 首次脉冲时间编码
- **文件**：
  - [exp4_1.py](实验四/exp4_1.py) — Poisson 脉冲生成
  - [exp4_2.py](实验四/exp4_2.py) — 频率/时间编码对比
  - [exp4_3.py](实验四/exp4_3.py) — TTFS 编码
  - [exp4_4.py](实验四/exp4_4.py) — 编码方案综合分析

### 实验五：神经可塑性与编码应用
- **内容**：STDP 学习窗口、基于 STDP 的无监督特征学习、MNIST 分类
- **文件**：
  - [exp5_1.py](实验五/exp5_1.py) — STDP 窗口可视化
  - [exp5_2.py](实验五/exp5_2.py) — STDP 无监督特征学习
  - [exp5_3.py](实验五/exp5_3.py) — 基于学习特征的分类

### 实验六：STDP / 感知机 / 分类与回归
- **内容**：交互式 STDP 窗口、感知机 MNIST 分类、多类逻辑回归、房价预测回归
- **文件**：
  - [exp6_1.py](实验六/exp6_1.py) — 交互式 STDP 窗口（带滑块）
  - [exp6_2.py](实验六/exp6_2.py) — 感知机 MNIST 分类
  - [exp6_3.py](实验六/exp6_3.py) — 多类逻辑回归
  - [exp6_4.py](实验六/exp6_4.py) — NN 房价预测

### 实验七：Hebb & STDP 突触可塑性
- **内容**：Hebbian 学习规则仿真、STDP 学习规则分析与对比
- **文件**：
  - [exp7_1.py](实验七/exp7_1.py) — Hebb 规则
  - [exp7_2.py](实验七/exp7_2.py) — STDP 规则

### 实验八：单神经元 STDP 模式检测
- **内容**：SRM + STDP 单神经元检测重复脉冲模式（复现 Masquelier & Thorpe 2007 *PLOS ONE*）
- **子目录**：
  - [exp8_1.py](实验八/exp8_1.py) — 单文件实验脚本
  - [最终版本/](实验八/最终版本/) — 🏅 清理后的最终版本（模块化、配置驱动）
  - [STDP-Finds-Start-of-Patterns/](实验八/STDP-Finds-Start-of-Patterns/) — 论文复现代码
- 📖 详见 [实验八/最终版本/README.md](实验八/最终版本/README.md)

### 实验 8-2：多神经元 STDP 竞争学习
- **内容**：多个 STDP 神经元 + 侧抑制实现竞争学习，检测不同脉冲模式
- **实验**：
  1. 独立神经元（无侧抑制）
  2. 竞争学习（含侧抑制）
  3. 抑制强度扫参
  4. 多模式多神经元
- **技术栈**：Numba 加速 SRM/LIF 仿真
- 📖 详见 [实验8_2/python_experiment/README.md](实验8_2/python_experiment/README.md)

### 实验九：监督 SNN 学习
- **内容**：脉冲神经网络的监督学习方法
- **文件**：
  - [exp9_1.py](实验九/exp9_1.py) — Tempotron 学习算法
  - [exp9_2.py](实验九/exp9_2.py) — ReSuMe 远程监督方法

---

## 🏆 综合项目：基于 STDP 的手写数字识别

> 📖 详细文档：[基于STDP的手写数字识别/README.md](基于STDP的手写数字识别/README.md)

**两种方法对比实现：**

| 方法 | 目录 | 框架 | 特点 |
|------|------|------|------|
| **Online STDP** | [`stdp_online/`](基于STDP的手写数字识别/stdp_online/) | Brian2 | 电导型 LIF 神经元 + 三重迹 STDP + E-I 回路 |
| **Hebbian 近似** | [`stdp_hebbian/`](基于STDP的手写数字识别/stdp_hebbian/) | Numba | 频率编码 + 竞争学习 + 胜者更新规则 |

**核心特性：**
- 🔬 双方法独立实现，可直接对比
- 📊 完整的推理评估 + 感受野可视化 + 混淆矩阵
- 💾 预训练权重可用，支持快速复现

---

## 🛠 环境与依赖

- **Python** ≥ 3.8
- **核心依赖**：
  - `numpy` — 数值计算
  - `matplotlib` — 可视化
  - `torch` — 深度学习框架（实验一~六）
  - `brian2` — 脉冲神经网络仿真（Online STDP 方法）
  - `numba` — JIT 加速（Hebbian 近似方法）
  - `tqdm` / `progressbar2` — 进度条

各子项目的精确依赖见对应 `requirements.txt`：
- [stdp_online/requirements.txt](基于STDP的手写数字识别/stdp_online/requirements.txt)
- [stdp_hebbian/requirements.txt](基于STDP的手写数字识别/stdp_hebbian/requirements.txt)

---

## 🚀 快速开始

```bash
# 克隆仓库
git clone <repo-url>
cd 类脑计算

# 安装核心依赖
pip install numpy matplotlib torch brian2 numba tqdm

# 运行实验（以实验三 HH模型为例）
python 实验三/exp3_1.py

# 运行综合项目 — Online STDP 方法
cd 基于STDP的手写数字识别/stdp_online
pip install -r requirements.txt
python train_stdp.py

# 运行综合项目 — Hebbian 近似方法
cd ../stdp_hebbian
pip install -r requirements.txt
python main.py
```

---

##  学习路径建议

```
实验一（环境）→ 实验二（MP神经元）→ 实验三（HH/LIF）
    → 实验四（神经编码）→ 实验五（可塑性）→ 实验七（Hebb/STDP）
    → 实验八（模式检测）→ 实验8_2（竞争学习）→ 实验九（监督SNN）
    → 🏆 综合项目（手写数字识别）
```

实验六是独立的应用模块（感知机/分类/回归），可在实验三之后随时进行。

---

## ️ 注意事项

- 本仓库的模型权重（`.npz`、`.npy`）、数据集（`.gz`）、图片（`.png`）及缓存文件已通过 `.gitignore` 排除，不会出现在版本控制中。
- 实验八和实验 8-2 的部分仿真运行时间较长（数小时），请合理安排。
- Brian2 仿真首次运行会生成 Cython 缓存，需要等待编译完成。

---

##  License

本仓库为课程学习用途，实验代码和项目代码仅供学习参考。
