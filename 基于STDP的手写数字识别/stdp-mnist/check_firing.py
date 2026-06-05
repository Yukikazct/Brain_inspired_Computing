"""检查预训练权重的发放活动"""
import os, sys; os.chdir(os.path.dirname(os.path.abspath(__file__)))
from brian2 import *
from pathlib import Path
from struct import unpack
import numpy as np

prefs.codegen.target = 'numpy'
MNIST_PATH = Path('../data'); DATA_PATH = Path('data')

images = open(MNIST_PATH/'train-images-idx3-ubyte','rb'); images.read(4)
n_images = unpack('>I',images.read(4))[0]; images.read(8)
labels = open(MNIST_PATH/'train-labels-idx1-ubyte','rb'); labels.read(8)
x = np.frombuffer(images.read(),dtype=np.uint8).reshape(n_images,-1)/8.0

ng_exc = NeuronGroup(400, Equations('''
dv/dt = (v_rest - v + i_exc + i_inh) / tau_mem  : volt (unless refractory)
i_exc = ge * -v                         : volt
i_inh = gi * (v_inh_base - v)           : volt
dge/dt = -ge/(1 * ms)                   : 1
dgi/dt = -gi/(2 * ms)                   : 1
dtimer/dt = 1                           : second
theta                                   : volt
''', tau_mem=100*ms, v_rest=-65*mV, v_inh_base=-100*mV),
    threshold='v > (theta - 72 * mV) and (timer > 50 * ms)', refractory=5*ms,
    reset='v = -65*mV; timer = 0*ms', method='euler')
ng_exc.v = -65*mV; ng_exc.theta = np.load(DATA_PATH/'theta.npy') * volt

ng_inh = NeuronGroup(400, Equations('''
dv/dt = (v_rest - v + i_exc + i_inh) / tau_mem  : volt (unless refractory)
i_exc = ge * -v                         : volt
i_inh = gi * (v_inh_base - v)           : volt
dge/dt = -ge/(1 * ms)                   : 1
dgi/dt = -gi/(2 * ms)                   : 1
dtimer/dt = 1                           : second
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

print("=== 发放活动检查 ===")
# 不同强度
for intensity in [1, 2, 4, 8]:
    prev = exc_mon.count[:]
    pg_inp.rates = x[0] * intensity * Hz; net.run(350*ms)
    pat = exc_mon.count[:] - prev
    pg_inp.rates = 0*Hz; net.run(150*ms)
    print(f'intensity={intensity}: spikes={pat.sum()}, active={(pat>0).sum()}/400, max={pat.max()}')

# 不同样本 (intensity=2)
print()
for ix in [0, 1, 100, 1000, 5000]:
    prev = exc_mon.count[:]
    pg_inp.rates = x[ix] * 2 * Hz; net.run(350*ms)
    pat = exc_mon.count[:] - prev
    pg_inp.rates = 0*Hz; net.run(150*ms)
    print(f'sample {ix}: spikes={pat.sum()}, max={pat.max()}')
