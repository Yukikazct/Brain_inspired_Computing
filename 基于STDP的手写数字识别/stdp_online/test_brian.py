"""
Brian2 STDP 测试 — 实时推理 + 混淆矩阵验证

分三部分:
  1. 实时推理 — 加载预训练STDP权重, 对N个样本逐张推理并统计准确率
  2. 完整准确率 — 从预存的confusion.npy读取混淆矩阵, 输出10K测试集总体准确率
  3. 可视化 — 生成感受野图 (receptive_fields_stdp.png) 和 神经元分配图
"""
import sys, os, numpy as np, time

# ---- 路径配置 ----
# DATA_DIR: 预训练STDP权重和元数据所在目录 (由train_stdp.py生成)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'stdp-mnist', 'data', 'stdp_full')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))  # 输出图片保存目录

# ---- 实时推理样本数 ----
N_LIVE = 100
# ---- Brian2 配置: 编译优化 ----
# 启用C++代码生成和O3优化, 大幅加速仿真
os.environ['CFLAGS'] = '-O3'
from brian2 import prefs
prefs.codegen.cpp.extra_compile_args_gcc = ['-O3', '-ffast-math']
prefs.codegen.target = 'cython'

# ---- 导入Diehl&Cook STDP模型 ----
# 需要切换到stdp-mnist目录, 因为模型内部使用相对路径
sys.path.insert(0, os.path.join(OUT_DIR, '..', 'stdp-mnist'))
os.chdir(os.path.join(OUT_DIR, '..', 'stdp-mnist'))
import Diehl_Cook_2015_brian2 as dc

# 配置数据路径
dc.MNIST_PATH = dc.Path('../data')
dc.DATA_PATH = dc.Path(DATA_DIR)

# ---- 加载预训练结果 ----
# assign.npy: 每个兴奋神经元的类别标签 (由observe阶段投票产生)
assign = np.load(os.path.join(DATA_DIR, 'assign.npy'))
# groups[c] = 标签为c的所有神经元索引列表
groups = [np.where(assign == i)[0] for i in range(10)]

# 加载测试集并构建Brian2网络 (推理模式)
X_test, Y_test = dc.read_mnist(False)
dc.N_NEURONS = len(assign)
net = dc.build_network(False)

# ---- 1. 实时推理: 逐样本展示, 按类别神经元平均发放数投票 ----
print(f"实时推理 {N_LIVE} 样本...")
t0 = time.time()
correct = 0
for i in range(N_LIVE):
    # show_sample: 输入一张MNIST图像, 运行仿真, 返回每个兴奋神经元的发放次数
    exc = dc.show_sample(net, X_test[i], dc.INTENSITY)
    # 预测: 各类别神经元平均发放数最大的即为预测类别
    guess = np.argmax([exc[grp].mean() for grp in groups])
    if guess == Y_test[i]:
        correct += 1
live_acc = correct / N_LIVE * 100
print(f"实时: {correct}/{N_LIVE} = {live_acc:.1f}% (耗时 {time.time()-t0:.1f}s)")

# ---- 2. 完整准确率: 从预存混淆矩阵直接读取 (10K测试集, 秒出结果) ----
# confusion.npy 由 train_stdp.py 的 dc.test() 阶段生成
conf = np.load(os.path.join(DATA_DIR, 'confusion.npy'))
# 对角元素之和 / 总和 = 整体准确率
full_acc = np.trace(conf) / np.sum(conf) * 100
print(f"\n★ STDP准确率: {full_acc:.2f}% (10K测试集)")
for i in range(10):
    row = conf[i]
    a = row[i] / row.sum() * 100 if row.sum() > 0 else 0
    print(f"  {i}: {a:.1f}%")

# ---- 3. 可视化: 感受野 + 神经元标签分配 ----
# 复用 stdp_hebbian 的可视化工具 (会自动生成横向平铺的灰度图)
sys.path.insert(0, os.path.join(OUT_DIR, '..', 'stdp_hebbian'))
from utils import visualize_weights, visualize_label_responses

# 3a. 感受野: 将每类前2个神经元的输入权重 reshape 为 28×28 灰度图
w_ie = np.load(os.path.join(DATA_DIR, 'weights.npy')).reshape(len(assign), -1)
idxs = []
for c in range(10):
    c_neurons = np.where(assign == c)[0]
    if len(c_neurons) >= 2:
        idxs.extend(c_neurons[:2])
if idxs:
    visualize_weights(w_ie, idxs, save_path=os.path.join(OUT_DIR, 'receptive_fields_stdp.png'))
    print(f"  → receptive_fields_stdp.png")

# 3b. 神经元分配: 左图=各类别神经元数量分布, 右图=各类别最强神经元响应
expanded = np.zeros((10, len(assign)))
for c in range(10):
    expanded[c, assign == c] = 1
visualize_label_responses(expanded, assign, save_path=os.path.join(OUT_DIR, 'neuron_assignment_stdp.png'))
print(f"  → neuron_assignment_stdp.png")
