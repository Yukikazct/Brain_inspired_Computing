import numpy as np
import matplotlib.pyplot as plt

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


# 生成二维高斯簇数据：class_gap 同时控制两类在 x 和 y 方向上的中心间距
def generate_gaussian_blob_data(num_per_class=60, class_gap=3.2, spread=0.45, seed=7):
    rng = np.random.default_rng(seed)

    # class_gap 同时作用于 x 和 y
    mean_pos = np.array([class_gap / 2.0, class_gap / 2.0], dtype=float)
    mean_neg = np.array([-class_gap / 2.0, -class_gap / 2.0], dtype=float)

    covariance = np.array(
        [
            [spread, 0.12 * spread],
            [0.12 * spread, spread],
        ],
        dtype=float,
    )

    pos_X = rng.multivariate_normal(mean=mean_pos, cov=covariance, size=num_per_class)
    neg_X = rng.multivariate_normal(mean=mean_neg, cov=covariance, size=num_per_class)
    pos_y = np.ones(num_per_class, dtype=int)
    neg_y = -np.ones(num_per_class, dtype=int)

    X = np.vstack([pos_X, neg_X])
    y = np.concatenate([pos_y, neg_y])

    shuffle_idx = rng.permutation(len(X))
    return X[shuffle_idx], y[shuffle_idx]


# 根据当前参数对一批样本做预测
def predict_labels(X, w, b):
    return np.sign(X @ w + b)


# 训练感知机
def fit_perceptron(X, y, lr=1.0, num_epochs=30, shuffle_seed=0):
    w = np.zeros(X.shape[1], dtype=float)
    b = 0.0
    rng = np.random.default_rng(shuffle_seed)

    for _ in range(num_epochs):
        # 将样本顺序打乱
        idx = rng.permutation(len(X))
        for xi, yi in zip(X[idx], y[idx]):
            # 预测当前样本
            y_pred = predict_labels(xi.reshape(1, -1), w, b)[0]
            # 预测错误，更新权重和偏置
            if y_pred != yi:
                w += lr * yi * xi
                b += lr * yi

    return w, b


# 绘制样本散点和最终分类边界
def draw_decision_boundary(ax, X, y, w, b):
    ax.clear()

    pos_mask = y == 1
    neg_mask = y == -1
    ax.scatter(X[pos_mask, 0], X[pos_mask, 1], color="#2563eb", s=36, label="类别 +1")
    ax.scatter(X[neg_mask, 0], X[neg_mask, 1], color="#c1121f", s=36, label="类别 -1")

    x_min, x_max = X[:, 0].min() - 0.8, X[:, 0].max() + 0.8
    y_min, y_max = X[:, 1].min() - 0.8, X[:, 1].max() + 0.8

    if np.abs(w[1]) > 1e-8:
        x_line = np.linspace(x_min, x_max, 200)
        y_line = -(w[0] * x_line + b) / w[1]
        ax.plot(x_line, y_line, color="#333333", linewidth=2.2, label="分类边界")
    else:
        x_line = -b / max(abs(w[0]), 1e-8)
        ax.axvline(x_line, color="#333333", linewidth=2.2, label="分类边界")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("特征 x1")
    ax.set_ylabel("特征 x2")
    ax.set_title("二维高斯簇数据与最终分类边界")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    # 本实验要求修改的参数
    # class_gap：控制两类高斯簇中心之间的距离，越大通常越容易分开
    class_gap = 4
    # shuffle_seed：只控制训练时样本访问顺序，不改变样本点本身
    shuffle_seed = 2022

    # 其余默认参数（不需要修改）
    lr = 1.0
    num_epochs = 10
    num_per_class = 60
    # spread：控制每一类样本云团的松散程度，越大类内重叠通常越明显
    spread = 1
    # data_seed：固定生成同一批样本，便于比较不同 shuffle_seed 的影响
    data_seed = 7

    # 先固定生成一份二维高斯簇数据，再只通过 shuffle_seed 改变训练时的样本访问顺序
    X, y = generate_gaussian_blob_data(
        num_per_class=num_per_class,
        class_gap=class_gap,
        spread=spread,
        seed=data_seed,
    )
    w, b = fit_perceptron(X, y, lr=lr, num_epochs=num_epochs, shuffle_seed=shuffle_seed)
    pred = predict_labels(X, w, b)
    train_acc = float(np.mean(pred == y))

    print("------ 感知机线性分类实验 ------")
    print("learning rate:", lr)
    print("num_epochs:", num_epochs)
    print("class_gap:", class_gap)
    print("shuffle_seed:", shuffle_seed)
    print("spread:", spread)
    print("final w:", w)
    print("final b:", b)
    print("train accuracy:", train_acc)

    # 只展示样本散点与最终分类边界
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))
    draw_decision_boundary(ax, X, y, w, b)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()