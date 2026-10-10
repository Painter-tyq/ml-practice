import numpy as np

# 赋值
x = np.array(2.0)
y = np.array(3.0)

# 前向计算
a = x * y
b = x ** 2
c = a + b
f = c ** 2

# 反向传播，从后往前求导
df_dc = 2 * c
dc_da = 1
dc_db = 1
da_dx = y
da_dy = x
db_dx = 2 * x

df_dx = df_dc * dc_da * da_dx + df_dc * dc_db * db_dx
df_dy = df_dc * dc_da * da_dy

print(f"numpy手动反向 df/dx = {df_dx}")
print(f"numpy手动反向 df/dy = {df_dy}")

import torch

# 设置requires_grad=True，开启梯度追踪
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# 前向
f = (x*y + x**2)**2

# 反向传播
f.backward()

print(f"pytorch自动微分 df/dx = {x.grad}")
print(f"pytorch自动微分 df/dy = {y.grad}")
