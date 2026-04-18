import gzip
import os
import struct
import urllib.request

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
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

CLASS_INFO = {
    1: "trouser",
    8: "bag",
    9: "ankle boot",
}

FASHION_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]
FASHION_BACKUP_BASE = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion"


# 当默认下载源不可用时，自动从备用地址下载 Fashion-MNIST 原始文件
def ensure_fashion_backup_files(root):
    raw_dir = os.path.join(root, "FashionMNIST", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    for filename in FASHION_FILES:
        file_path = os.path.join(raw_dir, filename)
        # ====================== 修复1：os.exists → os.path.exists ======================
        if not os.path.exists(file_path):
            url = f"{FASHION_BACKUP_BASE}/{filename}"
            print(f"正在下载 {filename} ...")
            urllib.request.urlretrieve(url, file_path)

    return raw_dir


# 读取 Fashion-MNIST 的图像 idx.gz 文件
def read_idx_images_gz(path):
    with gzip.open(path, "rb") as f:
        magic, num_images, num_rows, num_cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path} 不是有效的图像 idx 文件")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    images = data.reshape(num_images, num_rows, num_cols).astype(np.float64) / 255.0
    return images


# 读取 Fashion-MNIST 的标签 idx.gz 文件
def read_idx_labels_gz(path):
    with gzip.open(path, "rb") as f:
        magic, num_items = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path} 不是有效的标签 idx 文件")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.reshape(num_items)


# 在备用下载方式下，直接把 Fashion-MNIST 解析成 numpy 数组
def load_fashion_arrays_from_backup(root, train=True):
    raw_dir = ensure_fashion_backup_files(root)
    if train:
        image_file = "train-images-idx3-ubyte.gz"
        label_file = "train-labels-idx1-ubyte.gz"
    else:
        image_file = "t10k-images-idx3-ubyte.gz"
        label_file = "t10k-labels-idx1-ubyte.gz"

    images = read_idx_images_gz(os.path.join(raw_dir, image_file))
    labels = read_idx_labels_gz(os.path.join(raw_dir, label_file))
    return images, labels


# 从 Fashion-MNIST 中抽取平衡的三类子集，并把原标签映射成 0/1/2
def collect_balanced_subset_from_arrays(images, targets, selected_labels, samples_per_class, seed):
    rng = np.random.default_rng(seed)
    label_map = {old_label: new_label for new_label, old_label in enumerate(selected_labels)}

    all_indices = []
    for old_label in selected_labels:
        class_indices = np.where(targets == old_label)[0]
        chosen = rng.choice(class_indices, size=samples_per_class, replace=False)
        all_indices.extend(chosen.tolist())

    rng.shuffle(all_indices)
    all_indices = np.array(all_indices, dtype=int)

    old_targets = targets[all_indices]
    subset_images = images[all_indices].astype(np.float64)
    labels = np.array([label_map[int(old_label)] for old_label in old_targets], dtype=int)
    features = subset_images.reshape(len(subset_images), -1)
    return features, labels, subset_images


# 读取真实图像数据，并分别构造训练集与验证集
def load_fashion_subset(
        root="./data",
        train_samples_per_class=180,
        val_samples_per_class=60,
        selected_labels=(1, 8, 9),
        seed=7,
):
    transform = transforms.ToTensor()

    try:
        train_dataset = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
        train_images_all = train_dataset.data.numpy().astype(np.float64) / 255.0
        train_targets_all = train_dataset.targets.numpy()
        val_images_all = test_dataset.data.numpy().astype(np.float64) / 255.0
        val_targets_all = test_dataset.targets.numpy()
    except RuntimeError:
        train_images_all, train_targets_all = load_fashion_arrays_from_backup(root=root, train=True)
        val_images_all, val_targets_all = load_fashion_arrays_from_backup(root=root, train=False)

    train_X, train_y, train_images = collect_balanced_subset_from_arrays(
        train_images_all,
        train_targets_all,
        selected_labels=selected_labels,
        samples_per_class=train_samples_per_class,
        seed=seed,
    )
    val_X, val_y, val_images = collect_balanced_subset_from_arrays(
        val_images_all,
        val_targets_all,
        selected_labels=selected_labels,
        samples_per_class=val_samples_per_class,
        seed=seed + 1,
    )
    label_names = [CLASS_INFO[label] for label in selected_labels]
    return train_X, train_y, train_images, val_X, val_y, val_images, label_names


# 生成小批量索引，方便你自己组织训练循环
def iterate_minibatches(num_samples, batch_size, seed):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        yield indices[start:end]


# 初始化单隐藏层 MLP 的参数
def init_params(input_dim, hidden_size, output_dim, seed=7):

    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, np.sqrt(2 / input_dim), (input_dim, hidden_size))
    b1 = np.zeros(hidden_size)
    W2 = rng.normal(0, np.sqrt(2 / hidden_size), (hidden_size, output_dim))
    b2 = np.zeros(output_dim)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


# 前向传播：输入展平图像 -> 隐藏层线性变换 -> ReLU -> 输出层 logits -> softmax 概率
def forward_pass(X, y, params):
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]

    # 隐藏层：线性+ReLU
    Z1 = X @ W1 + b1
    H = np.maximum(Z1, 0)

    # 输出层：线性+Softmax
    O = H @ W2 + b2
    O_exp = np.exp(O - np.max(O, axis=1, keepdims=True))  # 防止指数溢出
    Y_hat = O_exp / np.sum(O_exp, axis=1, keepdims=True)

    # 交叉熵损失
    batch_size = X.shape[0]
    loss = -np.mean(np.log(Y_hat[range(batch_size), y] + 1e-8))  # 加微小值防止log(0)

    # 缓存中间变量，用于反向传播
    cache = {"X": X, "Z1": Z1, "H": H, "O": O, "Y_hat": Y_hat, "y": y}
    return loss, cache


# 反向传播：根据 softmax + 交叉熵的梯度更新两层参数
def backward_and_update(cache, params, lr):
    # TODO 3：补全反向传播与参数更新
    X, Z1, H, Y_hat, y = cache["X"], cache["Z1"], cache["H"], cache["Y_hat"], cache["y"]
    batch_size = X.shape[0]

    # 输出层梯度
    dO = Y_hat.copy()
    dO[range(batch_size), y] -= 1
    dO /= batch_size

    # 第二层参数梯度
    dW2 = H.T @ dO
    db2 = np.sum(dO, axis=0)

    # 隐藏层梯度（ReLU反向传播）
    dH = dO @ params["W2"].T
    dZ1 = dH * (Z1 > 0)

    # 第一层参数梯度
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)

    # 梯度下降更新参数
    params["W1"] -= lr * dW1
    params["b1"] -= lr * db1
    params["W2"] -= lr * dW2
    params["b2"] -= lr * db2


# 在完整数据集上评估当前模型的损失和精度
def evaluate_dataset(X, y, params):
    # TODO 4：补全模型评估函数
    loss, cache = forward_pass(X, y, params)
    pred = np.argmax(cache["Y_hat"], axis=1)
    acc = np.mean(pred == y)
    return loss, acc, pred, cache["Y_hat"]


# 训练单隐藏层分类模型，并记录每个 epoch 的训练 / 验证指标
def train_classifier(train_X, train_y, val_X, val_y, hidden_size, lr, num_epochs, batch_size, seed=7):
    # TODO 5：补全完整训练流程
    input_dim = train_X.shape[1]
    output_dim = 3
    params = init_params(input_dim, hidden_size, output_dim, seed)

    # 记录训练历史
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }

    for epoch in range(num_epochs):
        # 小批量训练
        for indices in iterate_minibatches(len(train_X), batch_size, seed + epoch):
            X_batch = train_X[indices]
            y_batch = train_y[indices]
            loss, cache = forward_pass(X_batch, y_batch, params)
            backward_and_update(cache, params, lr)

        # 评估训练集和验证集
        train_loss, train_acc, _, _ = evaluate_dataset(train_X, train_y, params)
        val_loss, val_acc, _, _ = evaluate_dataset(val_X, val_y, params)

        # 保存指标
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 打印日志
        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] | 训练损失:{train_loss:.4f} 准确率:{train_acc:.4f} | 验证损失:{val_loss:.4f} 准确率:{val_acc:.4f}")

    return params, history


# 在左侧区域绘制每个类别的样本图
def draw_sample_grid(fig, spec, images, labels, label_names):
    sample_indices = []
    for class_id in range(len(label_names)):
        class_positions = np.where(labels == class_id)[0][:4]
        sample_indices.extend(class_positions.tolist())

    sub = spec.subgridspec(3, 4, wspace=0.15, hspace=0.35)
    for plot_idx, data_idx in enumerate(sample_indices):
        # ====================== 修复2：add_sub → add_subplot ======================
        ax = fig.add_subplot(sub[plot_idx // 4, plot_idx % 4])
        ax.imshow(images[data_idx], cmap="gray")
        ax.set_title(label_names[labels[data_idx]], fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])


# 中间区域：绘制训练 / 验证损失曲线
def draw_loss_panel(ax, history):
    ax.clear()
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], color="#2563eb", linewidth=2.0, label="训练损失")
    ax.plot(epochs, history["val_loss"], color="#c1111f", linewidth=2.0, label="验证损失")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("训练 / 验证损失曲线")
    ax.legend(loc="upper right")


# 右侧：绘制训练 / 验证准确率曲线
def draw_accuracy_panel(ax, history):
    ax.clear()
    epochs = np.arange(1, len(history["train_acc"]) + 1)
    ax.plot(epochs, history["train_acc"], color="#15803d", linewidth=2.0, label="训练精度")
    ax.plot(epochs, history["val_acc"], color="#f59e0b", linewidth=2.0, label="验证精度")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("训练 / 验证准确率曲线")
    ax.legend(loc="lower right")


def main():
    # 默认训练配置
    hidden_size = 1024
    lr = 0.05
    num_epochs = 25
    batch_size = 32

    train_samples_per_class = 180
    val_samples_per_class = 60
    seed = 7

    train_X, train_y, train_images, val_X, val_y, val_images, label_names = load_fashion_subset(
        root="./data",
        train_samples_per_class=train_samples_per_class,
        val_samples_per_class=val_samples_per_class,
        seed=seed,
    )

    params, history = train_classifier(
        train_X=train_X,
        train_y=train_y,
        val_X=val_X,
        val_y=val_y,
        hidden_size=hidden_size,
        lr=lr,
        num_epochs=num_epochs,
        batch_size=batch_size,
        seed=seed,
    )

    train_loss, train_acc, _, _ = evaluate_dataset(train_X, train_y, params)
    val_loss, val_acc, _, _ = evaluate_dataset(val_X, val_y, params)

    print("------ 三分类 softmax + 交叉熵实验 ------")
    print("hidden_size:", hidden_size)
    print("learning rate:", lr)
    print("num_epochs:", num_epochs)
    print("batch_size:", batch_size)
    print("final train loss:", train_loss)
    print("final train acc:", train_acc)
    print("final val loss:", val_loss)
    print("final val acc:", val_acc)

    # 统一展示：样本图、loss 曲线、准确率曲线
    fig = plt.figure(figsize=(15, 5.8))
    fig.patch.set_facecolor("#eef2f7")
    gs = fig.add_gridspec(1, 3, wspace=0.28)
    fig.text(0.06, 0.96, "真实图像三分类：softmax、交叉熵与手写关键反向传播", fontsize=18, weight="bold")

    draw_sample_grid(fig, gs[0], train_images, train_y, label_names)

    ax_loss = fig.add_subplot(gs[1])
    draw_loss_panel(ax_loss, history)

    ax_acc = fig.add_subplot(gs[2])
    draw_accuracy_panel(ax_acc, history)

    plt.show()


if __name__ == "__main__":
    main()