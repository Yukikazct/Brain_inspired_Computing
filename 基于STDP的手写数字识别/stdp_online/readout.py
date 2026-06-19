"""MLP读出层 — STDP特征 → 监督分类。"""
import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    """400 → 256 → ReLU → Dropout → 10"""

    def __init__(self, n_in=400, n_hidden=256, n_out=10, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(features, labels, n_epochs=300, lr=0.001, wd=1e-4):
    """训练MLP读出层。"""
    model = MLP(n_in=features.shape[1])
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()

    n = len(features)
    for epoch in range(n_epochs):
        opt.zero_grad()
        logits = model(features)
        loss = loss_fn(logits, labels)
        loss.backward()
        opt.step()
        if (epoch + 1) % 50 == 0:
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean().item()
            print(f"  epoch {epoch+1}: loss={loss.item():.4f} acc={acc:.2%}")

    return model


def evaluate_mlp(model, features, labels):
    """评估MLP准确率。"""
    with torch.no_grad():
        logits = model(features)
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
    return acc
