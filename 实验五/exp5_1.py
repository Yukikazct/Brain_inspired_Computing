import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

# 配置中文字体，避免绘图时标题或坐标轴出现乱码
plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "Deja Sans",
]
plt.rcParams["axes.unicode_minus"] = False


# 从 MNIST 中取出第一张标签为 7 的图像
def load_mnist_seven(root="./data"):
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    for image, label in dataset:
        if int(label) == 7:
            return image.squeeze(0).numpy(), int(label)

    raise RuntimeError("未找到标签为 7 的 MNIST 样本")


def uniform_encode_image(image, T):
    H, W = image.shape
    spikes = np.zeros((H, W, T), dtype=np.float32)
    # 计算每个像素的脉冲数
    spike_num = np.round(image * T).astype(np.int32)
    # 为每个像素均匀分配脉冲
    for i in range(H):
        for j in range(W):
            num = spike_num[i, j]
            if num <= 0:
                continue
            # 均匀分布脉冲时刻
            times = np.linspace(0, T-1, num, dtype=np.int32)
            spikes[i, j, times] = 1.0
    return spikes

def main():
    # 本实验要求修改的参数
    T = 100
    # 先加载原图，再生成脉冲序列并统计时间窗内的脉冲次数
    image, label = load_mnist_seven()
    spikes = uniform_encode_image(image, T=T)
    spike_counts = spikes.sum(axis=2)

    # 左边显示原图，右边显示均匀编码后的脉冲计数图
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title(f"原始 MNIST 图像（标签={label}）")
    axes[0].axis("off")

    axes[1].imshow(spike_counts, cmap="gray")
    axes[1].set_title("均匀编码后的脉冲计数图")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()