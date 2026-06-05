"""运行 Diehl & Cook (2015) Brian2 复现的完整管线"""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 读取原始脚本并修改关键参数
with open('Diehl_Cook_2015_brian2.py', 'r') as f:
    code = f.read()

code = code.replace("MNIST_PATH = Path('../mnist')", "MNIST_PATH = Path('../data')")
code = code.replace("N_TRAIN = 25_000", "N_TRAIN = 500")
code = code.replace("N_OBSERVE = 2_000", "N_OBSERVE = 500")
code = code.replace("N_TEST = 1_000", "N_TEST = 500")
code = code.replace("N_SAVE_POINTS = 100", "N_SAVE_POINTS = 10")

# 阶段1: 训练
print("="*50)
print("阶段1: 训练 (500样本)")
code_train = code.replace("MODE = 'test'", "MODE = 'train'")
exec(code_train)

# 阶段2: 观察
print("\n" + "="*50)
print("阶段2: 观察 (500样本)")
code_obs = code.replace("MODE = 'test'", "MODE = 'observe'")
exec(code_obs)

# 阶段3: 测试
print("\n" + "="*50)
print("阶段3: 测试 (500样本)")
code_test = code.replace("MODE = 'test'", "MODE = 'test'")
exec(code_test)
