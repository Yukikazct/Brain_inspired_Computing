"""调试预训练权重的标签分配"""
import os, sys; os.chdir(os.path.dirname(os.path.abspath(__file__)))
from brian2 import *
from pathlib import Path; from struct import unpack
from collections import defaultdict
from random import randrange, seed as rseed
import numpy as np; rseed(42)

prefs.codegen.target = 'numpy'
MNIST_PATH = Path('../data'); DATA_PATH = Path('data')

images = open(MNIST_PATH/'train-images-idx3-ubyte','rb'); images.read(4)
n_images = unpack('>I',images.read(4))[0]; images.read(8)
labels = open(MNIST_PATH/'train-labels-idx1-ubyte','rb'); labels.read(8)
X = np.frombuffer(images.read(),dtype=np.uint8).reshape(n_images,-1)/8.0
Y = np.frombuffer(labels.read(),dtype=np.uint8)

ng_exc = NeuronGroup(400, Equations('''
dv/dt = (v_rest - v + i_exc + i_inh) / tau_mem  : volt (unless refractory)
i_exc = ge * -v : volt
i_inh = gi * (v_inh_base - v) : volt
dge/dt = -ge/(1 * ms) : 1
dgi/dt = -gi/(2 * ms) : 1
dtimer/dt = 1 : second
theta : volt
''', tau_mem=100*ms, v_rest=-65*mV, v_inh_base=-100*mV),
    threshold='v > (theta - 72 * mV) and (timer > 50 * ms)', refractory=5*ms,
    reset='v = -65*mV; timer = 0*ms', method='euler')
ng_exc.v = -65*mV; ng_exc.theta = np.load(DATA_PATH/'theta.npy') * volt

ng_inh = NeuronGroup(400, Equations('''
dv/dt = (v_rest - v + i_exc + i_inh) / tau_mem : volt (unless refractory)
i_exc = ge * -v : volt
i_inh = gi * (v_inh_base - v) : volt
dge/dt = -ge/(1 * ms) : 1
dgi/dt = -gi/(2 * ms) : 1
dtimer/dt = 1 : second
''', tau_mem=10*ms, v_rest=-60*mV, v_inh_base=-85*mV),
    threshold='v > -40*mV', refractory=2*ms, reset='v = -45*mV', method='euler')
ng_inh.v = -60*mV

Synapses(ng_exc, ng_inh, on_pre='ge_post += 10.4').connect(j='i')
Synapses(ng_inh, ng_exc, on_pre='gi_post += 17.0').connect("i != j")
pg_inp = PoissonGroup(784, 0*Hz)
syns = Synapses(pg_inp, ng_exc, model='w:1', on_pre='ge_post += w')
syns.connect(True)
syns.w = np.load(DATA_PATH/'weights.npy')
exc_mon = SpikeMonitor(ng_exc)
net = Network([pg_inp, ng_exc, ng_inh, syns, exc_mon])
net.run(0*ms)

# 对每个类别取10个样本，看发放模式
print("各类别样本的平均发放（前10神经元）：")
for c in range(10):
    c_idx = np.where(Y == c)[0][:10]
    responses = []
    for ix in c_idx:
        prev = exc_mon.count[:]
        pg_inp.rates = X[ix] * 2 * Hz; net.run(350*ms)
        pat = exc_mon.count[:] - prev
        pg_inp.rates = 0*Hz; net.run(150*ms)
        responses.append(pat)
    mean_resp = np.mean(responses, axis=0)
    print(f"Class {c}: total_spikes={mean_resp.sum():.0f}, top5={np.sort(mean_resp)[-5:][::-1]}")
PYEOF
