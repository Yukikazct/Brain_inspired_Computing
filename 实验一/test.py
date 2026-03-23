import torch
import importlib.metadata

print("------------------------------------------")
print("PyTorch 版本:", torch.__version__)

try:
    sj_vers;ion = importlib.metadata.version('spikingjelly')
    print("SpikingJelly 版本:", sj_version)
except importlib.metadata.PackageNotFoundError:
    print("SpikingJelly 未安装或无法找到版本信息")
print("------------------------------------------")

# 测试硬件加速是否可用
if torch.cuda.is_available():
    print("硬件加速状态: 你的 Windows NVIDIA GPU 加速已开启！")
elif torch.backends.mps.is_available():
    print("硬件加速状态: 你的 Mac M系列芯片 MPS 加速已开启！")
else:
    print("硬件加速状态: 当前使用的是 CPU 模式运行。")
print("------------------------------------------")
