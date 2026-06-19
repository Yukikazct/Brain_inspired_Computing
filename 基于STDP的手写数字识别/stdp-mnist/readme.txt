Diehl & Cook (2015) 原始代码
=================================
论文: Unsupervised learning of digit recognition using spike-timing-dependent plasticity
DOI:  https://doi.org/10.3389/fncom.2015.00099
代码: https://github.com/peter-u-diehl/stdp-mnist

本目录包含两个版本:
  - Diehl&Cook_spiking_MNIST.py    原始Brian1代码 (需要Python2 + Brian1)
  - Diehl_Cook_2015_brian2.py      Brian2移植版 (Python3, 2024年官方移植)
  - Diehl&Cook_MNIST_evaluation.py 评估脚本

======================================================================
Brian2移植版运行方法
======================================================================

1. 安装依赖:
   python3 -m venv venv
   source venv/bin/activate
   pip install 'numpy<2' brian2 progressbar2

2. 准备MNIST数据:
   将 train-images-idx3-ubyte, train-labels-idx1-ubyte,
   t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte 放到 ../data/ 目录

3. 修改 Diehl_Cook_2015_brian2.py 顶部的参数:
   - MODE = 'train'    # 训练模式
   - N_TRAIN = 50000   # 训练样本数 (论文用 60000*3=180000)
   - MNIST_PATH = Path('../data')

4. 运行三阶段:
   # 阶段1: 训练 (耗时较长, N_TRAIN=50000约需14小时)
   python3 Diehl_Cook_2015_brian2.py     # MODE='train'

   # 阶段2: 标签分配
   python3 Diehl_Cook_2015_brian2.py     # MODE='observe'

   # 阶段3: 测试
   python3 Diehl_Cook_2015_brian2.py     # MODE='test'

5. 预期结果:
   - N_TRAIN=50000+ → 准确率约89-95%
   - 预训练权重 (weights/XeAe.npy) → 91.56%

6. 快速验证 (本项目的500样本测试):
   venv/bin/python3 run_pipeline.py   → 准确率约11%(500样本, 训练不足)

======================================================================
与本项目其他模型的关系
======================================================================
本目录是论文的原始参考实现, 运行需要Brian2仿真器.
项目的主要提交是两个自实现模型:
  - rate_model/     速率竞争学习 (STDP近似), 90.4%, 22秒
  - stdp_spike/     电流型LIF+WTA+STDP, ~67%, ~5分钟
