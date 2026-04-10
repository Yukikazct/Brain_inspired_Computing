import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from torchvision import datasets, transforms

# 配置中文字体，避免图像标题或标签出现乱码
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

# 全局仿真参数
T_WINDOW = 100
MAX_SPIKES = 10
TTFS_SPAN = 80.0


# 固定加载一张真实的 MNIST 数字 7
def load_mnist_seven(root="./data"):
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    for image, label in dataset:
        if int(label) == 7:
            return image.squeeze(0).numpy(), int(label)

    raise RuntimeError("未找到标签为 7 的 MNIST 样本")


# 高斯时间抖动：对每个脉冲时刻加入零均值高斯扰动
def add_gaussian_timing_noise(time_value, noise_std, rng):
    return time_value + rng.normal(0.0, noise_std)


# 椒盐型脉冲噪声：随机删除已有脉冲，或者随机注入一个伪脉冲
def add_salt_pepper_spike_noise(spike_times, noise_level, rng):
    # TODO 3：
    prob = noise_level / 100.0
    new_spikes = []
    # Pepper：随机删除脉冲
    for t in spike_times:
        if rng.random() > prob:
            new_spikes.append(t)
    # Salt：随机注入伪脉冲
    if rng.random() < prob:
        fake_t = rng.integers(0, T_WINDOW)
        new_spikes.append(fake_t)
    # 去重并排序
    new_spikes = sorted(list(set(new_spikes)))
    return new_spikes


# 根据频率编码构造理想脉冲时刻，再施加“椒盐型脉冲噪声 + 高斯时间抖动”
def process_rate_coding(image, noise_level):
    H, W = image.shape
    decoded = np.zeros_like(image)
    rng = np.random.default_rng(seed=42)
    noise_std = noise_level / 10.0

    for i in range(H):
        for j in range(W):
            # 频率编码生成脉冲时刻
            xi = image[i, j]
            num_spikes = int(np.round(xi * T_WINDOW))
            if num_spikes <= 0:
                times = []
            else:
                times = np.linspace(0, T_WINDOW-1, num_spikes).tolist()
            # 加椒盐噪声
            times = add_salt_pepper_spike_noise(times, noise_level, rng)
            # 加高斯时间抖动
            times = [add_gaussian_timing_noise(t, noise_std, rng) for t in times]
            # 重构：脉冲数/时间窗
            decoded[i, j] = len(times) / T_WINDOW
    # 归一化到0-1
    decoded = np.clip(decoded, 0.0, 1.0)
    return decoded


# 根据首次脉冲发放时间编码构造理想首次脉冲，再施加“椒盐型脉冲噪声 + 高斯时间抖动”
def process_ttfs_coding(image, noise_level):
    # TODO 5：
    H, W = image.shape
    decoded = np.zeros_like(image)
    rng = np.random.default_rng(seed=42)
    noise_std = noise_level / 10.0

    for i in range(H):
        for j in range(W):
            xi = image[i, j]
            # TTFS编码：灰度值越大，脉冲时间越早
            ideal_t = TTFS_SPAN * (1 - xi)
            times = [ideal_t]
            # 加椒盐噪声
            times = add_salt_pepper_spike_noise(times, noise_level, rng)
            # 加高斯抖动
            times = [add_gaussian_timing_noise(t, noise_std, rng) for t in times]
            # 取最早脉冲
            if len(times) > 0:
                first_t = min(times)
                first_t = np.clip(first_t, 0, TTFS_SPAN)
                decoded[i, j] = 1 - (first_t / TTFS_SPAN)
            else:
                decoded[i, j] = 0.0
    decoded = np.clip(decoded, 0.0, 1.0)
    return decoded


# 比较无噪声重构图与有噪声重构图的平均绝对差
def mad(a, b):
    return float(np.mean(np.abs(a - b)))


# 以像素风格显示 28x28 灰度图
def draw_image(ax, image, title):
    ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    image, label = load_mnist_seven()

    # 先计算无噪声条件下的基线重构图
    clean_rate = process_rate_coding(image, 0.0)
    clean_ttfs = process_ttfs_coding(image, 0.0)

    # 预先计算误差曲线，底部折线图会直接复用这些结果
    noise_levels = np.arange(0.0, 31.0, 1.0)
    rate_curve = []
    ttfs_curve = []

    for n in noise_levels:
        rate_curve.append(mad(clean_rate, process_rate_coding(image, n)))
        ttfs_curve.append(mad(clean_ttfs, process_ttfs_coding(image, n)))

    # 上排三图，底部一张误差曲线
    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.9])
    plt.subplots_adjust(bottom=0.14, hspace=0.35, wspace=0.18)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax_curve = fig.add_subplot(gs[1, :])

    draw_image(ax0, image, f"原始输入\nMNIST 数字 {label}")
    rate_im = ax1.imshow(clean_rate, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax1.set_title("频率编码重构")
    ax1.set_xticks([])
    ax1.set_yticks([])

    ttfs_im = ax2.imshow(clean_ttfs, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax2.set_title("首次脉冲时间编码重构")
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax_curve.plot(noise_levels, rate_curve, label="频率编码", color="tab:blue")
    ax_curve.plot(noise_levels, ttfs_curve, label="首次脉冲时间编码", color="tab:red")
    ax_curve.set_xlabel("噪声强度")
    ax_curve.set_ylabel("MAD")
    ax_curve.set_title("统一噪声链下的重构误差变化")
    ax_curve.grid(alpha=0.3)
    ax_curve.legend()
    vline = ax_curve.axvline(0.0, color="gray", linestyle=":")
    marker_r, = ax_curve.plot([0], [rate_curve[0]], marker="o", color="tab:blue")
    marker_t, = ax_curve.plot([0], [ttfs_curve[0]], marker="o", color="tab:red")

    # 添加统一噪声强度滑块：椒盐先作用于脉冲，再在保留下来的脉冲时刻上施加高斯抖动
    slider_ax = fig.add_axes([0.18, 0.05, 0.64, 0.03])
    noise_slider = Slider(
        ax=slider_ax,
        label="当前噪声强度",
        valmin=0.0,
        valmax=30.0,
        valinit=0.0,
        valstep=1.0,
        color="#e74c3c",
    )

    # 每次滑块变化时，重新计算两种编码在当前噪声强度下的重构结果
    def update(_):
        noise_level = float(noise_slider.val)
        idx = int(noise_level)

        rate_decoded = process_rate_coding(image, noise_level)
        ttfs_decoded = process_ttfs_coding(image, noise_level)

        rate_im.set_data(rate_decoded)
        ttfs_im.set_data(ttfs_decoded)
        vline.set_xdata([noise_level, noise_level])
        marker_r.set_data([noise_level], [rate_curve[idx]])
        marker_t.set_data([noise_level], [ttfs_curve[idx]])

        fig.canvas.draw_idle()

    noise_slider.on_changed(update)
    plt.show()

if __name__ == "__main__":
    main()