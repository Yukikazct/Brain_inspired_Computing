"""使用预训练权重测试准确率"""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

# 修改原始脚本的参数
import Diehl_Cook_2015_brian2 as script
script.MNIST_PATH = script.Path('../data')
script.N_OBSERVE = 500
script.N_TEST = 500

# 跳过训练，直接用预训练权重
print("="*50)
print("使用预训练权重 — Observe")
script.observe()
print("="*50)
print("使用预训练权重 — Test")
script.test()
