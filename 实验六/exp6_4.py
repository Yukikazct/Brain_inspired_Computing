import os
import urllib.request

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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


DATA_URL = "http://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_train.csv"
FEATURE_NAMES = ["OverallQual", "GrLivArea", "GarageCars", "YearBuilt", "FullBath"]
TARGET_NAME = "SalePrice"


# 下载 D2L 房价训练数据，保证实验可以在本地直接运行
def download_house_price_csv(data_dir="./data"):
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, "kaggle_house_pred_train.csv")

    if not os.path.exists(file_path):
        print(f"正在下载房价数据到 {file_path} ...")
        urllib.request.urlretrieve(DATA_URL, file_path)

    return file_path


# 只保留固定的 5 个数值特征，并完成缺失值填补、标准化和训练 / 验证划分
def prepare_house_price_data(csv_path, train_ratio=0.8, seed=7):
    df = pd.read_csv(csv_path)
    work_df = df[FEATURE_NAMES + [TARGET_NAME]].copy()

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(work_df))
    split = int(len(indices) * train_ratio)
    train_idx, val_idx = indices[:split], indices[split:]

    train_df = work_df.iloc[train_idx].copy()
    val_df = work_df.iloc[val_idx].copy()

    feature_means = train_df[FEATURE_NAMES].mean()
    feature_stds = train_df[FEATURE_NAMES].std().replace(0, 1.0)

    train_features = train_df[FEATURE_NAMES].fillna(feature_means)
    val_features = val_df[FEATURE_NAMES].fillna(feature_means)

    train_features = (train_features - feature_means) / feature_stds
    val_features = (val_features - feature_means) / feature_stds

    train_X = train_features.to_numpy(dtype=np.float64)
    val_X = val_features.to_numpy(dtype=np.float64)
    train_y = np.log1p(train_df[TARGET_NAME].to_numpy(dtype=np.float64)).reshape(-1, 1)
    val_y = np.log1p(val_df[TARGET_NAME].to_numpy(dtype=np.float64)).reshape(-1, 1)

    return train_X, train_y, val_X, val_y


# 生成小批量索引
def iterate_minibatches(num_samples, batch_size, seed):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        yield indices[start:end]


# 初始化单隐藏层回归网络的参数
def init_regression_params(input_dim, hidden_size, seed=7):
    # TODO 1：补全参数初始化（He初始化）
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, np.sqrt(2 / input_dim), (input_dim, hidden_size))
    b1 = np.zeros(hidden_size)
    W2 = rng.normal(0, np.sqrt(2 / hidden_size), (hidden_size, 1))
    b2 = np.zeros(1)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


# 前向传播：5 维输入 -> 隐藏层 ReLU -> 1 维回归输出
def forward_regression(X, y, params):
    # TODO 2：补全前向传播与 MSE 计算
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]

    # 隐藏层：线性变换 + ReLU激活函数
    Z1 = X @ W1 + b1
    H = np.maximum(Z1, 0)

    # 输出层：线性回归（无激活函数）
    y_pred = H @ W2 + b2

    # 计算 MSE 损失
    loss = np.mean((y_pred - y) ** 2)

    # 缓存中间变量，用于反向传播
    cache = {"X": X, "Z1": Z1, "H": H, "y_pred": y_pred, "y": y}
    return loss, cache


# 反向传播：根据 MSE 的梯度更新单隐藏层回归模型参数
def backward_regression_and_update(cache, params, lr):
    # TODO 3：补全反向传播与参数更新
    X, Z1, H, y_pred, y = cache["X"], cache["Z1"], cache["H"], cache["y_pred"], cache["y"]
    batch_size = X.shape[0]

    # 输出层梯度 (MSE损失)
    dy_pred = 2 * (y_pred - y) / batch_size

    # 第二层参数梯度
    dW2 = H.T @ dy_pred
    db2 = np.sum(dy_pred, axis=0)

    # 隐藏层梯度 (ReLU反向传播)
    dH = dy_pred @ params["W2"].T
    dZ1 = dH * (Z1 > 0)

    # 第一层参数梯度
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)

    # 梯度下降更新参数
    params["W1"] -= lr * dW1
    params["b1"] -= lr * db1
    params["W2"] -= lr * dW2
    params["b2"] -= lr * db2


# 在完整数据集上评估当前模型的 MSE
def evaluate_regression_dataset(X, y, params):
    # TODO 4：补全模型评估函数
    loss, cache = forward_regression(X, y, params)
    return loss, cache["y_pred"]


# 训练回归模型，并记录训练集与验证集损失变化
def train_regression_model(train_X, train_y, val_X, val_y, hidden_size, lr, num_epochs, batch_size, seed=7):
    # TODO 5：补全完整训练流程
    input_dim = train_X.shape[1]
    params = init_regression_params(input_dim, hidden_size, seed)

    # 记录训练历史
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        # 小批量训练
        for indices in iterate_minibatches(len(train_X), batch_size, seed + epoch):
            X_batch = train_X[indices]
            y_batch = train_y[indices]
            loss, cache = forward_regression(X_batch, y_batch, params)
            backward_regression_and_update(cache, params, lr)

        # 评估模型
        train_loss, _ = evaluate_regression_dataset(train_X, train_y, params)
        val_loss, _ = evaluate_regression_dataset(val_X, val_y, params)

        # 保存损失
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # 每20轮打印一次日志
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] 训练损失: {train_loss:.4f} 验证损失: {val_loss:.4f}")

    return params, history


# 仅用于展示：把 log1p 房价还原回原始房价
def recover_price(log_price):
    return np.expm1(log_price)


# 左图：训练 / 验证损失曲线
def draw_loss_panel(ax, history):
    ax.clear()
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], color="#2563eb", linewidth=2.0, label="训练损失")
    ax.plot(epochs, history["val_loss"], color="#c1121f", linewidth=2.0, label="验证损失")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("训练 / 验证损失曲线")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# 中图：按真实房价排序后的真实值 / 预测值对比曲线
def draw_prediction_compare_panel(ax, y_true, y_pred):
    ax.clear()
    y_true_price = recover_price(y_true.reshape(-1))
    y_pred_price = recover_price(y_pred.reshape(-1))
    order = np.argsort(y_true_price)
    sorted_true = y_true_price[order]
    sorted_pred = y_pred_price[order]
    sample_rank = np.arange(len(sorted_true))

    ax.plot(sample_rank, sorted_true, color="#2563eb", linewidth=2.2, label="真实房价")
    ax.plot(sample_rank, sorted_pred, color="#c1121f", linewidth=2.0, label="预测房价")
    ax.set_xlabel("验证样本（按真实房价排序）")
    ax.set_ylabel("房价")
    ax.set_title("验证集真实房价与预测房价对比")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    # 默认训练配置
    hidden_size = 16
    lr = 0.02
    num_epochs = 120

    # 其余固定参数（一般不需要修改）
    batch_size = 64
    seed = 7

    # 先下载并读取房价数据，再完成固定特征的预处理
    csv_path = download_house_price_csv(data_dir="./data")
    train_X, train_y, val_X, val_y = prepare_house_price_data(csv_path=csv_path, seed=seed)

    # 再调用你自己实现的回归训练函数
    params, history = train_regression_model(
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

    train_loss, _ = evaluate_regression_dataset(train_X, train_y, params)
    val_loss, val_pred = evaluate_regression_dataset(val_X, val_y, params)

    print("------ 房价回归实验 ------")
    print("hidden_size:", hidden_size)
    print("learning rate:", lr)
    print("num_epochs:", num_epochs)
    print("train loss:", train_loss)
    print("val loss:", val_loss)

    # 统一展示：训练曲线、真实/预测对比曲线
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    fig.patch.set_facecolor("#eef2f7")
    draw_loss_panel(axes[0], history)
    draw_prediction_compare_panel(axes[1], val_y, val_pred)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()