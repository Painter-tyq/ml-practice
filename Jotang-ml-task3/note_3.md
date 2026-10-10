# 实践1 
1.把计算过程拆成若干个简单步骤，并画出计算图；  
graph TD  
    A[输入 x, y, z] --> B[计算 a = x * y]  
    A --> C[计算 b = x²]  
    B --> D[计算 c = a + b]  
    C --> D  
    D --> E[计算 f = c * z]  
    E --> F[输出 f]

# 实践2
### 1.flowchart LR    
X[X] --> Z1[Z₁ = XW₁ + b₁]  
W1[W₁] --> Z1  
b1[b₁] --> Z1  
Z1 --> A1[A₁ = ReLU(Z₁)]  
A1 --> Z2[Z₂ = A₁W₂ + b₂]  
W2[W₂] --> Z2  
b2[b₂] --> Z2  
Z2 --> A2[A₂ = σ(Z₂)]  
A2 --> L[L = BCE(A₂, y)]  
y[y] --> L  
### 2.链式反向求导公式

单样本
$$
\begin{cases}
\mathrm{d}Z_2 = A_2 - y \\[4pt]
\mathrm{d}W_2 = A_1^\top \mathrm{d}Z_2 \\[4pt]
\mathrm{d}b_2 = \mathrm{d}Z_2 \\[4pt]
\mathrm{d}A_1 = \mathrm{d}Z_2\, W_2^\top \\[4pt]
\mathrm{d}Z_1 = \mathrm{d}A_1 \odot \mathbb I(Z_1>0) \\[4pt]
\mathrm{d}W_1 = X^\top \mathrm{d}Z_1 \\[4pt]
\mathrm{d}b_1 = \mathrm{d}Z_1
\end{cases}
$$
Batch 版本求导公式

$$
\begin{cases}
\mathrm{d}Z_2 = A_2 - y \\[4pt]
\mathrm{d}W_2 = \dfrac{1}{N}A_1^\top \mathrm{d}Z_2 \\[4pt]
\mathrm{d}b_2 = \dfrac{1}{N}\sum \mathrm{d}Z_2 \\[4pt]
\mathrm{d}A_1 = \mathrm{d}Z_2\, W_2^\top \\[4pt]
\mathrm{d}Z_1 = \mathrm{d}A_1 \odot \mathbb I(Z_1>0) \\[4pt]
\mathrm{d}W_1 = \dfrac{1}{N}X^\top \mathrm{d}Z_1 \\[4pt]
\mathrm{d}b_1 = \dfrac{1}{N}\sum \mathrm{d}Z_1
\end{cases}
$$

# 趁热打铁
1.前向传播保存每层输入输出的原因
- 反向传播链式求导需要前向产生的中间结果，用来计算各层参数梯度。这些中间张量会占用内存/显存；也可以选择不保存中间值，反向阶段重新前向计算，以时间换空间，降低显存占用。

2.链式法则回传梯度过程
- 梯度从损失L沿计算图反向传递：由L求出\mathrm dZ_2，计算第二层权重偏置梯度并回传得到\mathrm dA_1；经过ReLU梯度掩码得到\mathrm dZ_1；最后算出第一层W_1,b_1的梯度，层层相乘传递导数，得到首层参数梯度。 

3.BCE+Sigmoid组合梯度
- \mathrm dZ_2 = A_2-y。该形式合并了Sigmoid与交叉熵求导，避免指数运算，计算简单且数值稳定性更好，极大简化反向传播实现。

4.Numpy与PyTorch梯度不一致排查思路
- 优先核对loss是否同时取均值、batch梯度是否除以样本数；再检查矩阵维度、转置是否正确；接着排查广播对齐、求导公式正误；极小数值差异一般是float32与float64精度不同导致。 

5.loss.backward()原理
- 自动微分，依托前向构建好的计算图，利用链式法则自动反向计算全部可训练参数梯度，并将梯度存入.grad属性，省去手动推导与编写反向传播代码。  

6.遇到的报错记录
- 矩阵维度报错，大多是矩阵乘法缺少转置；
batch场景梯度忘记除以样本数量，梯度数值偏大；
LaTeX公式缺少$$标记，markdown无法渲染数学表达式；
ReLU反向掩码书写错误，造成梯度计算错误；
