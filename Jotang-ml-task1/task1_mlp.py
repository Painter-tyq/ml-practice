# 导入全部依赖库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim

# 固定随机种子，保证实验可复现
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# 生成月牙二分类数据
X, y = make_moons(n_samples=1000, noise=0.2, random_state=SEED)

# 数据集划分
# 第一步：分出20%作为【测试集】，剩下80%临时数据
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
# 第二步：临时数据再划分：训练集60%，验证集20%
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=SEED)

print(f"训练集样本数: {len(X_train)}")
print(f"验证集样本数: {len(X_val)}")
print(f"测试集样本数: {len(X_test)}")

# 可视化原始数据
plt.figure(figsize=(6,5))
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap="coolwarm", alpha=0.7, label="train")
plt.scatter(X_val[:,0], X_val[:,1], c=y_val, cmap="coolwarm", marker="s", alpha=0.7, label="val")
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap="coolwarm", marker="^", alpha=0.7, label="test")
plt.switch_backend('Agg')
plt.title("make_moons Dataset")
plt.legend()
plt.savefig("moons_dataset.png")

# 转为PyTorch张量
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1)
X_val = torch.FloatTensor(X_val)
y_val = torch.FloatTensor(y_val).unsqueeze(1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).unsqueeze(1)

"""
数据集职责说明
1. 训练集：用来更新模型权重，学习数据模式
2. 验证集：训练过程观察泛化能力，用来调超参数，**不更新权重**
3. 测试集：训练完全结束后，做最终评估，训练全程不能使用，避免数据泄露
数据泄露：测试数据参与训练或者调参，会造成评估结果虚高，不能代表真实性能
"""

# 搭建MLP模型
class MLP(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),  # 输入：2个坐标特征，映射到隐藏层
            nn.ReLU(),                # 激活函数，给网络加上非线性能力（没有它只能学直线）
            nn.Linear(hidden_dim, 1), # 隐藏层 → 输出1个值
            nn.Sigmoid()              # 把输出压缩到0~1，当作分类概率
        )
    def forward(self, x):  
        # forward：定义数据流过网络的计算流程
        return self.net(x) 
    
# 记录曲线用的容器
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []

model = MLP()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 1000
for epoch in range(epochs):
    # ========== 训练阶段 ==========
    model.train()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 训练集准确率
    train_pred = (y_pred > 0.5).float()
    train_acc = (train_pred == y_train).float().mean()

    # ========== 验证阶段（不更新权重） ==========
    model.eval()
    with torch.no_grad(): # 关闭梯度计算，节省显存
        y_val_pred = model(X_val)
        val_loss = criterion(y_val_pred, y_val)
        val_pred = (y_val_pred > 0.5).float()
        val_acc = (val_pred == y_val).float().mean()

    # 保存数据，后续绘图
    train_loss_list.append(loss.item())
    val_loss_list.append(val_loss.item())
    train_acc_list.append(train_acc.item())
    val_acc_list.append(val_acc.item())

    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{epochs}] | Train Loss:{loss.item():.4f}, Train Acc:{train_acc:.4f} | Val Loss:{val_loss.item():.4f}, Val Acc:{val_acc:.4f}")

# ---------------- 训练结束：保存模型 ----------------
torch.save(model.state_dict(), "mlp_moons.pth")
print("模型已保存到 mlp_moons.pth")

# ---------------- 加载模型示例 ----------------
# new_model = MLP()
# new_model.load_state_dict(torch.load("mlp_moons.pth"))
# new_model.eval()

# ---------------------- 绘制 Loss 和 Accuracy 曲线 ----------------------
plt.figure(figsize=(12, 5))

# 左图：Loss曲线
plt.subplot(1, 2, 1)
plt.plot(train_loss_list, label="Train Loss")
plt.plot(val_loss_list, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

# 右图：Accuracy曲线
plt.subplot(1, 2, 2)
plt.plot(train_acc_list, label="Train Acc")
plt.plot(val_acc_list, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()

plt.tight_layout()
# 保存图片到文件（WSL环境，保存为png，不会弹窗）
plt.savefig("loss_acc_curve.png")
print(" loss & acc 曲线图片已保存：loss_acc_curve.png")


# ---------------------- 2. 绘制二维决策边界 ----------------------
def plot_decision_boundary(model, X, y):
    model.eval()
    # 生成网格采样点
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # 网格转为tensor，送入模型预测
    grid_data = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()
    with torch.no_grad():
        pred_prob = model(grid_data)
    pred_prob = pred_prob.reshape(xx.shape)

    # 绘图
    plt.figure(figsize=(6,5))
    plt.contourf(xx, yy, pred_prob, alpha=0.4, cmap="RdBu")
    plt.scatter(X[:,0], X[:,1], c=y, s=30, cmap="RdBu")
    plt.title("MLP Decision Boundary (make_moons)")
    plt.savefig("decision_boundary.png")
    print(" 决策边界图已保存：decision_boundary.png")


# 合并全部数据，画决策边界
X_all = torch.cat([X_train, X_val, X_test])
y_all = torch.cat([y_train, y_val, y_test])
plot_decision_boundary(model, X_all, y_all)


# ---------------------- 任务4：隐藏层神经元数量对照实验 ----------------------
def train_model(hidden_dim, epochs=epochs, lr=0.01):
    """只改变隐藏层宽度，其余条件与基线训练保持一致。"""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = MLP(hidden_dim=hidden_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []

    for epoch in range(epochs):
        model.train()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_pred = (y_pred > 0.5).float()
        train_acc = (train_pred == y_train).float().mean()

        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = criterion(y_val_pred, y_val)
            val_pred = (y_val_pred > 0.5).float()
            val_acc = (val_pred == y_val).float().mean()

        train_loss_hist.append(loss.item())
        val_loss_hist.append(val_loss.item())
        train_acc_hist.append(train_acc.item())
        val_acc_hist.append(val_acc.item())

    return {
        "hidden_dim": hidden_dim,
        "train_loss": train_loss_hist,
        "val_loss": val_loss_hist,
        "train_acc": train_acc_hist,
        "val_acc": val_acc_hist,
        "final_val_acc": val_acc_hist[-1],
    }


hidden_sizes = [4, 16, 128]
results = []
for h in hidden_sizes:
    print(f"\n===== 开始训练 hidden_dim={h} =====")
    result = train_model(h)
    results.append(result)
    print(f"hidden_dim={h} 最终验证集准确率: {result['final_val_acc']:.4f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for result in results:
    h = result["hidden_dim"]
    plt.plot(result["train_loss"], label=f"Train Loss (h={h})")
    plt.plot(result["val_loss"], linestyle="--", label=f"Val Loss (h={h})")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Hidden Size")
plt.legend()

plt.subplot(1, 2, 2)
for result in results:
    h = result["hidden_dim"]
    plt.plot(result["train_acc"], label=f"Train Acc (h={h})")
    plt.plot(result["val_acc"], linestyle="--", label=f"Val Acc (h={h})")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Hidden Size")
plt.legend()

plt.tight_layout()
plt.savefig("task4_compare_hidden_size.png")
print(" 对照实验图已保存：task4_compare_hidden_size.png")

print("\n===== 任务4 最终验证集准确率汇总 =====")
for result in results:
    print(f"隐藏层 {result['hidden_dim']:3d} 个神经元 | Val Acc = {result['final_val_acc']:.4f}")


# ---------------------- 任务4第二组：学习率对照实验 ----------------------
learning_rates = [0.001, 0.01, 0.1]
lr_results = []
for lr in learning_rates:
    print(f"\n===== 开始训练 hidden_dim=16, lr={lr} =====")
    result = train_model(hidden_dim=16, lr=lr)
    result["lr"] = lr
    lr_results.append(result)
    print(f"lr={lr} 最终验证集准确率: {result['final_val_acc']:.4f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for result in lr_results:
    lr = result["lr"]
    plt.plot(result["train_loss"], label=f"Train Loss (lr={lr})")
    plt.plot(result["val_loss"], linestyle="--", label=f"Val Loss (lr={lr})")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Learning Rate")
plt.legend()

plt.subplot(1, 2, 2)
for result in lr_results:
    lr = result["lr"]
    plt.plot(result["train_acc"], label=f"Train Acc (lr={lr})")
    plt.plot(result["val_acc"], linestyle="--", label=f"Val Acc (lr={lr})")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Learning Rate")
plt.legend()

plt.tight_layout()
plt.savefig("task4_compare_lr.png")
print(" 对照实验图已保存：task4_compare_lr.png")

print("\n===== 任务4第二组 最终验证集准确率汇总 =====")
for result in lr_results:
    print(f"学习率 {result['lr']:<6} | Val Acc = {result['final_val_acc']:.4f}")


# ---------------------- 任务5：测试集混淆矩阵与错分样本分析 ----------------------
# 使用前面训练好的基线模型 model（隐藏层16神经元）
model.eval()
with torch.no_grad():
    y_test_prob = model(X_test)

y_test_pred = (y_test_prob > 0.5).float()
X_test_np = X_test.numpy()
y_true_np = y_test.squeeze().numpy().astype(int)
y_pred_np = y_test_pred.squeeze().numpy().astype(int)
y_prob_np = y_test_prob.squeeze().numpy()

cm = confusion_matrix(y_true_np, y_pred_np)
print("\n===== 任务5 测试集混淆矩阵 =====")
print(cm)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Pred 0", "Pred 1"],
    yticklabels=["True 0", "True 1"],
)
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print(" 混淆矩阵图已保存：confusion_matrix.png")

wrong_mask = y_pred_np != y_true_np
wrong_X = X_test_np[wrong_mask]
wrong_y = y_true_np[wrong_mask]
wrong_prob = y_prob_np[wrong_mask]
print(f"\n===== 任务5 错分样本（共 {len(wrong_X)} 个）=====")
for i, (coord, label, prob) in enumerate(zip(wrong_X, wrong_y, wrong_prob)):
    pred_label = int(prob > 0.5)
    print(
        f"[{i}] 坐标=({coord[0]:.4f}, {coord[1]:.4f}), "
        f"真实标签={label}, 预测标签={pred_label}, 预测概率={prob:.4f}"
    )

plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", alpha=0.5, label="dataset")
plt.scatter(
    wrong_X[:, 0],
    wrong_X[:, 1],
    c="black",
    marker="x",
    s=90,
    linewidths=2,
    label="wrong",
)
plt.title("Misclassified Samples on make_moons")
plt.legend()
plt.savefig("wrong_sample.png")
print(" 错分样本图已保存：wrong_sample.png")
