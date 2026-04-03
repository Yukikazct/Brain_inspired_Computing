import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def build_demo_image():
    # 10x10 灰度图 小房子
    image = np.array([
        [0.10, 0.10, 0.10, 0.60, 0.85, 0.85, 0.60, 0.10, 0.10, 0.10],
        [0.10, 0.10, 0.60, 0.85, 0.95, 0.95, 0.85, 0.60, 0.10, 0.10],
        [0.10, 0.60, 0.85, 0.95, 1.00, 1.00, 0.95, 0.85, 0.60, 0.10],
        [0.60, 0.85, 0.95, 1.00, 1.00, 1.00, 1.00, 0.95, 0.85, 0.60],
        [0.10, 0.10, 0.20, 0.80, 0.80, 0.80, 0.80, 0.20, 0.10, 0.10],
        [0.10, 0.10, 0.20, 0.80, 0.30, 0.30, 0.80, 0.20, 0.10, 0.10],
        [0.10, 0.10, 0.20, 0.80, 0.30, 0.30, 0.80, 0.20, 0.10, 0.10],
        [0.10, 0.10, 0.20, 0.80, 0.80, 0.80, 0.80, 0.20, 0.10, 0.10],
        [0.10, 0.10, 0.20, 0.80, 0.80, 0.80, 0.80, 0.20, 0.10, 0.10],
        [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
    ])
    return image


def encode_image(image, T):
    encoded_visual = np.zeros_like(image, dtype=float)
    h, w = image.shape

    # TODO4
    #TTFS编码
    for i in range(h):
        for j in range(w):
            val = image[i, j]
            if val <= 0:
                spike_time = T-1
            else:
                spike_time = np.round((1 - val) * (T - 1))
                spike_time = np.clip(spike_time, 0, T - 1)
            encoded_visual[i, j] = spike_time
    # 归一化到[0,1]用于灰度可视化
    encoded_visual = (encoded_visual - encoded_visual.min()) / (encoded_visual.max() - encoded_visual.min())
    visual_title = f"TTFS编码-首次脉冲时间图(T={T})"

    return encoded_visual, visual_title


def setup_pixel_grid(ax, size):
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8, alpha=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)


# 实验参数
T = 3

# 实验执行
image = build_demo_image()
encoded_visual, visual_title = encode_image(image, T=T)

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
im0 = axes[0].imshow(image, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
axes[0].set_title("原始 10x10 灰度图")
setup_pixel_grid(axes[0], 10)

im1 = axes[1].imshow(encoded_visual, cmap="gray", interpolation="nearest")
axes[1].set_title(visual_title)
setup_pixel_grid(axes[1], 10)

# 颜色条
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()