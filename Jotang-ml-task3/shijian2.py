import numpy as np
import torch
from sklearn.datasets import make_moons

# ---------------------- 从make_moons取出样本 ----------------------
X_all, y_all = make_moons(n_samples=100, noise=0.1, random_state=0)
# 取单个样本
X_single = X_all[0:1, :]
y_single = y_all[0:1, None]
# 取小batch batch_size=2
X_batch = X_all[0:2, :]
y_batch = y_all[0:2, None]

# 网络超参
in_dim = 2
h_dim = 3
out_dim = 1
np.random.seed(0)
W1 = np.random.randn(in_dim, h_dim)   # (2,3)
b1 = np.zeros((1, h_dim))             # (1,3)
W2 = np.random.randn(h_dim, out_dim)  # (3,1)
b2 = np.zeros((1, out_dim))           # (1,1)

# ====================== 1.前向 numpy ======================
X = X_single
y = y_single
Z1 = X @ W1 + b1
A1 = np.maximum(Z1, 0.0)
Z2 = A1 @ W2 + b2
A2 = 1.0 / (1.0 + np.exp(-Z2))
loss_np = - y * np.log(A2) - (1 - y) * np.log(1 - A2)

# ====================== 2. Numpy 手动反向传播 ======================
dZ2 = A2 - y
dW2_np = A1.T @ dZ2
db2_np = dZ2          #  db2_np 在这里定义，不会缺失！
dA1 = dZ2 @ W2.T
relu_mask = (Z1 > 0).astype(float)
dZ1 = dA1 * relu_mask
dW1_np = X.T @ dZ1
db1_np = dZ1

# ====================== 3. PyTorch 前向 + autograd自动求导 ======================
X_t = torch.tensor([[0.5, 0.8]], dtype=torch.float32)
y_t = torch.tensor([[1.0]], dtype=torch.float32)
W1_t = torch.tensor(W1, dtype=torch.float32, requires_grad=True)
b1_t = torch.tensor(b1, dtype=torch.float32, requires_grad=True)
W2_t = torch.tensor(W2, dtype=torch.float32, requires_grad=True)
b2_t = torch.tensor(b2, dtype=torch.float32, requires_grad=True)

Z1_t = X_t @ W1_t + b1_t
A1_t = torch.relu(Z1_t)
Z2_t = A1_t @ W2_t + b2_t
A2_t = torch.sigmoid(Z2_t)
loss_t = - y_t * torch.log(A2_t) - (1 - y_t) * torch.log(1 - A2_t)
loss_t.backward()

# 取出梯度转为numpy
dW2_pt = W2_t.grad.numpy()
db2_pt = b2_t.grad.numpy()
dW1_pt = W1_t.grad.numpy()
db1_pt = b1_t.grad.numpy()

# ====================== 4. 第五问：逐项对比函数 ======================
def compare_grad(name, np_grad, pt_grad):
    err = np.max(np.abs(np_grad - pt_grad))
    print(f"\n==== {name} ====")
    print(f"Numpy shape: {np_grad.shape}")
    print(f"PyTorch shape: {pt_grad.shape}")
    print(f"Numpy value:\n{np_grad}")
    print(f"PyTorch value:\n{pt_grad}")
    print(f"最大绝对误差: {err:.2e}")
    same = np.allclose(np_grad, pt_grad, atol=1e-5)
    print(f"是否一致(atol=1e-5): {same}")

# ====================== 5. 逐项输出对比结果 ======================
compare_grad("dW1", dW1_np, dW1_pt)
compare_grad("db1", db1_np, db1_pt)
compare_grad("dW2", dW2_np, dW2_pt)
compare_grad("db2", db2_np, db2_pt)
