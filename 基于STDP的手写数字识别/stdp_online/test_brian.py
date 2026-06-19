"""Brian2 STDP 测试 — 实时推理 + 混淆矩阵"""
import sys, os, numpy as np, time
os.environ['CFLAGS'] = '-O3'
from brian2 import prefs
prefs.codegen.cpp.extra_compile_args_gcc = ['-O3', '-ffast-math']
prefs.codegen.target = 'cython'

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, '..', 'stdp-mnist', 'data', 'stdp_full')
sys.path.insert(0, os.path.join(BASE, '..', 'stdp-mnist'))
os.chdir(os.path.join(BASE, '..', 'stdp-mnist'))
import Diehl_Cook_2015_brian2 as dc

dc.MNIST_PATH = dc.Path('../data')
dc.DATA_PATH = dc.Path(DATA_DIR)

# ---- 1. 实时推理 (N个样本, 30秒内) ----
N = int(sys.argv[1]) if len(sys.argv) > 1 else 80
assign = np.load(os.path.join(DATA_DIR, 'assign.npy'))
groups = [np.where(assign == i)[0] for i in range(10)]
X_test, Y_test = dc.read_mnist(False)
dc.N_NEURONS = len(assign)
net = dc.build_network(False)

print(f"实时推理 {N} 样本...")
t0 = time.time()
correct = 0
per_class = np.zeros((10, 2), dtype=int)
for i in range(N):
    exc = dc.show_sample(net, X_test[i], dc.INTENSITY)
    guess = np.argmax([exc[grp].mean() for grp in groups])
    if guess == Y_test[i]: correct += 1; per_class[Y_test[i], 0] += 1
    per_class[Y_test[i], 1] += 1

live_acc = correct / N * 100
print(f"实时: {correct}/{N} = {live_acc:.1f}% (耗时 {time.time()-t0:.1f}s)")
for c in range(10):
    a = per_class[c,0]/per_class[c,1]*100 if per_class[c,1] > 0 else 0
    if per_class[c,1] > 0: print(f"  {c}: {a:.0f}%", end="")
print()

# ---- 2. 完整10K准确率 (预存混淆矩阵) ----
conf = np.load(os.path.join(DATA_DIR, 'confusion.npy'))
full_acc = np.trace(conf) / np.sum(conf) * 100
print(f"\n★ STDP准确率: {full_acc:.2f}% (10K测试集)")
for i in range(10):
    a = conf[i,i] / conf[i].sum() * 100 if conf[i].sum() > 0 else 0
    print(f"  {i}: {a:.1f}%")

# ---- 3. 可视化 ----
sys.path.insert(0, os.path.join(BASE, '..', 'stdp_hebbian'))
from utils import visualize_weights, visualize_label_responses
w_ie = np.load(os.path.join(DATA_DIR, 'weights.npy')).reshape(len(assign), -1)
idxs = []
for c in range(10):
    c_neurons = np.where(assign == c)[0]
    if len(c_neurons) >= 2: idxs.extend(c_neurons[:2])
if idxs: visualize_weights(w_ie, idxs, save_path=os.path.join(BASE, 'receptive_fields_stdp.png'))
expanded = np.zeros((10, len(assign)))
for c in range(10): expanded[c, assign == c] = 1
visualize_label_responses(expanded, assign, save_path=os.path.join(BASE, 'neuron_assignment_stdp.png'))
print("图片已更新")
