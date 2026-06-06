# 基于STDP的手写数字识别

MNIST手写数字识别 | 脉冲神经网络 | 无监督STDP学习

## 最终结果

| 方法 | 神经元 | 训练量 | 准确率 | 训练耗时 |
|------|------|------|------|------|
| **在线三元STDP** | 1600 | 180K | **92.59%** | ~11h |
| **STDP速率近似 (Hebbian)** | 3000 | 240K | **91.90%** | 17min |

## 项目结构

```
├── stdp_online/          # 在线三元STDP (Diehl & Cook 2015 完整复现)
│   ├── test_brian.py     # 快速出结果  
│   ├── run_brian.py      # Brian2 Cython 训练脚本
│   ├── model.py          # Numba版在线STDP (备选)
│   ├── README.md
│   └── *.png             # 感受野 + 神经元分配图
│
├── stdp_hebbian/         # STDP速率近似 (对比实验)
│   ├── main.py           # 训练+测试
│   ├── test.py           # 30秒快速测试 + 可视化
│   ├── model.py          # Hebbian竞争学习
│   ├── README.md
│   └── *.png             # 感受野 + 神经元分配图
│
├── stdp-mnist/           # 论文原始Brian代码 (fork)
│   └── Diehl_Cook_2015_brian2.py
│
└── data/                 # MNIST数据集 (各版本共享)
```

---

## 方法一：在线三元STDP (stdp_online)

### 原理

严格复现 Diehl & Cook (2015) 的无监督STDP学习框架：

**神经元模型**：电导型LIF（Leaky Integrate-and-Fire）
- 膜电位动态：`τ_m · dv/dt = (v_rest - v) + I_syn_E + I_syn_I`
- 电导型突触：`I_syn = ge · (-v) + gi · (-100 - v)`（突触电流依赖当前膜电位）
- 不应期：5ms
- 膜时间常数：100ms（增强发放率估计的稳定性）

**STDP学习规则**：在线三重trace机制

```
pre trace:   τ=20ms, 输入脉冲到达 → pre=1
post1 trace: τ=20ms, 神经元发放 → post1=1
post2 trace: τ=40ms, 神经元发放 → post2=1

LTD (输入脉冲触发):    Δw[j,i] = -nu_pre × post1[i]       nu_pre=0.0001
LTP (神经元发放触发):  Δw[j,i] = +nu_post × pre[j] × post2[i]  nu_post=0.01
```

权重更新在仿真过程中**在线进行**，每个spike实时触发，严格依赖脉冲时序。

**网络架构**：
- 输入层：784个泊松神经元（MNIST像素→泊松脉冲序列，350ms，63.75Hz）
- 兴奋层：1600个LIF神经元
- 抑制层：1600个抑制性神经元（E→I→E回路，实现涌现式WTA竞争）
- 内在可塑性：自适应发放阈值θ，防止单神经元主导

**学习流程**：
1. 权重归一化（每样本前，目标总和78.0）
2. 350ms刺激期：泊松输入 + 在线STDP
3. 150ms静息期：零输入，trace衰减
4. 输入强度自适应：发放<5则增量重试
5. 训练后标签分配：赢家频率法（无监督，训练时不使用标签）

**推理方式**：Top-10赢家加权投票

### 所用软件包

| 包 | 作用 |
|------|------|
| **Brian2** | 脉冲神经网络仿真引擎，Cython代码生成加速 |
| **NumPy** | 矩阵运算、数据加载 |
| **Cython** | 将Brian2模型编译为C扩展，10-100x加速 |
| **Matplotlib** | 生成感受野和神经元分配可视化图 |

### 训练配置

- 数据集：MNIST（训练60,000 / 测试10,000）
- 训练方式：无监督（训练时不使用标签）
- 标签分配：事后用赢家频率分配，30000样本
- 随机种子：42

---

## 方法二：STDP速率近似 / Hebbian竞争学习 (stdp_hebbian)

### 原理

基于STDP稳态平衡理论（Zenke et al., 2015）：**当LTP和LTD达到稳态时，权重向量方向收敛到输入模式聚类中心**。这等价于 Hebbian LTP + 权重归一化。

**学习规则**：
1. **LIF仿真**（50ms，电导型LIF + 侧向抑制）
2. **赢家选择**：发放最多的神经元（排除不应期内的）
3. **Hebbian LTP**：`w[赢家] += lr × image`
4. **权重归一化**（LTD等效）：`w[赢家] *= 78 / sum(w)`
5. **不应期轮换**：赢家进入50样本不应期，强制所有神经元参与

```
与STDP的对应关系：
  STDP的LTP ⟺ Hebbian LTP (赢家权重向输入靠拢)
  STDP的LTD ⟺ 权重归一化 (全局约束, 等效稳态平衡)
```

**关键不同**：
- 权重更新在仿真**结束后**一次性完成
- 不依赖精确spike timing，仅依赖发放率
- 额外不应期机制防止神经元垄断

### 所用软件包

| 包 | 作用 |
|------|------|
| **NumPy** | 矩阵运算、MNIST数据解析 |
| **Numba** | JIT编译加速LIF仿真（多核并行） |
| **Matplotlib** | 生成感受野和神经元分配可视化图 |
| **tqdm** | 训练进度显示 |

### 训练配置

- 数据集：MNIST（训练60,000 / 测试10,000）
- 训练方式：无监督（训练时不使用标签）
- 标签分配：事后用赢家频率分配，30000样本
- 初始化：基于训练样本的权重初始化（每类均匀采样）
- 随机种子：42

---

## 训练集与测试集分割

| 用途 | 样本数 | 说明 |
|------|------|------|
| **训练集** | 60,000 | MNIST官方训练集，无监督STDP/Hebbian学习 |
| **测试集** | 10,000 | MNIST官方测试集，**仅用于最终评估** |
| **标签分配集** | 30,000 | 训练集前30K样本，仅用于事后标签分配（不参与权重更新） |

严格遵循：训练/测试分离，测试集不参与训练或调参。

---

## 生成图表说明

### receptive_fields — 感受野
每个神经元的784维权重reshape为28×28。亮区=敏感像素。
- **清晰数字** → 神经元高度特化
- **模糊模板** → 多类别混合激活

### neuron_assignment — 神经元分配
- **左图**：各类别分配的神经元数量（反映竞争学习资源分配）
- **右图**：各类别最强神经元的平均发放数（反映特化程度）

---

## 快速演示

```bash
# STDP 
cd stdp_online && python test_brian.py

# Hebbian 
cd stdp_hebbian && python test.py
```

---

## 总结

本项目实现了两种基于STDP的MNIST手写数字识别方法，均达到90%以上准确率。

**在线三元STDP** 严格复现Diehl & Cook (2015)，使用Brian2仿真器配合Cython C代码生成，1600个LIF神经元通过E→I→E抑制回路实现涌现式WTA竞争，在线三重trace STDP规则在350ms泊松脉冲窗口内实时更新权重。最终92.59%准确率超过论文报告的1600神经元基准（91.9%）。

**STDP速率近似（Hebbian）** 基于Zenke等人(2015)的STDP稳态平衡理论，将LTP/LTD平衡等价为Hebbian LTP + 权重归一化。使用Numba JIT编译加速LIF仿真，3000神经元通过不应期轮换实现均衡竞争。训练仅17分钟即达到91.90%准确率。

两种方法共同验证了：1）STDP无监督学习可在SNN中实现高性能模式识别；2）STDP稳态平衡可被速率近似有效捕获，为实际应用提供更快的训练方案。

---

## 参考

- Diehl, P. U., & Cook, M. (2015). Unsupervised learning of digit recognition using spike-timing-dependent plasticity. *Frontiers in Computational Neuroscience*, 9, 99.
- Zenke, F., Agnes, E. J., & Gerstner, W. (2015). Diverse synaptic plasticity mechanisms orchestrated to form and retrieve memories in spiking neural networks. *Nature Communications*, 6, 6922.
- 原始代码: https://github.com/peter-u-diehl/stdp-mnist
