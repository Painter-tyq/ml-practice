import numpy as np
import torch

# ===================== Batch版本 batch_size=2 =====================
# 权重初始化，和单样本保持一致
in_dim = 2
h_dim = 3
out_dim = 1
np.random.seed(0)
W1 = np.random.randn(in_dim, h_dim)   # (2,3)
b1 = np.zeros((1, h_dim))             # (1,3)
W2 = np.random.randn(h_dim, out_dim)  # (3,1)
b2 = np.zeros((1, out_dim))           # (1,1)

from sklearn.datasets import make_moons
X_all, y_all = make_moons(n_samples=100, noise=0.1, random_state=0)

batch_size = 2
# 取前2条样本作为batch输入
X_batch = X_all[0:batch_size, :]
y_batch = y_all[0:batch_size, None]
N = batch_size

# ========= Numpy 前向传播 =========
Z1_batch = X_batch @ W1 + b1
A1_batch = np.maximum(Z1_batch, 0.0)
Z2_batch = A1_batch @ W2 + b2
A2_batch = 1.0 / (1.0 + np.exp(-Z2_batch))
# BCE loss，batch取平均
loss_batch = np.mean(- y_batch * np.log(A2_batch) - (1 - y_batch) * np.log(1 - A2_batch))
print(f"Batch Loss = {loss_batch}")

# ========= Numpy 手动反向传播 =========
dZ2_batch = A2_batch - y_batch
N = X_batch.shape[0]
# batch梯度需要求平均
dW2_np = A1_batch.T @ dZ2_batch / N
db2_np = np.sum(dZ2_batch, axis=0, keepdims=True) / N

dA1_batch = dZ2_batch @ W2.T
relu_mask_batch = (Z1_batch > 0).astype(float)
dZ1_batch = dA1_batch * relu_mask_batch

dW1_np = X_batch.T @ dZ1_batch / N
db1_np = np.sum(dZ1_batch, axis=0, keepdims=True) / N

# ========= PyTorch 前向 + autograd自动求导 =========
X_t = torch.tensor(X_batch, dtype=torch.float32)
y_t = torch.tensor(y_batch, dtype=torch.float32)
W1_t = torch.tensor(W1, dtype=torch.float32, requires_grad=True)
b1_t = torch.tensor(b1, dtype=torch.float32, requires_grad=True)
W2_t = torch.tensor(W2, dtype=torch.float32, requires_grad=True)
b2_t = torch.tensor(b2, dtype=torch.float32, requires_grad=True)

Z1_t = X_t @ W1_t + b1_t
A1_t = torch.relu(Z1_t)
Z2_t = A1_t @ W2_t + b2_t
A2_t = torch.sigmoid(Z2_t)
loss_t = torch.mean(- y_t * torch.log(A2_t) - (1 - y_t) * torch.log(1 - A2_t))
loss_t.backward()

# tensor梯度转numpy
dW1_pt = W1_t.grad.numpy()
db1_pt = b1_t.grad.numpy()
dW2_pt = W2_t.grad.numpy()
db2_pt = b2_t.grad.numpy()

# ========= 梯度逐项对比函数 =========
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


print("\n---------- Batch梯度对比结果 ----------")
compare_grad("dW1", dW1_np, dW1_pt)
compare_grad("db1", db1_np, db1_pt)
compare_grad("dW2", dW2_np, dW2_pt)
compare_grad("db2", db2_np, db2_pt)
