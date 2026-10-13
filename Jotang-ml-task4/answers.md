## 任务一：重绘 Transformer 结构图

阅读论文中的 Figure 1，分别找出 Encoder 和 Decoder。参考原图，用自己的方式重新画一张简化/拓展的 Transformer 结构图。

[Encoder]
  Inputs  
    ↓  
  Input Embedding + 位置编码  
    ↓  
  多头自注意力 (Multi-Head Self-Attention)  
    ↓  
  前馈神经网络 (Feed-Forward Network)  
    ↓ (输出 K, V 给 Decoder 的交叉注意力层)  

[Decoder]  
  Outputs (右移一位)  
    ↓  
  Output Embedding + 位置编码  
    ↓  
  掩码多头自注意力 (Masked Multi-Head Self-Attention)  
    ↓  
  交叉注意力 (Cross-Attention) ← 接收 Encoder 的 K, V  
    ↓  
  前馈神经网络 (Feed-Forward Network)  
    ↓  
  Linear + Softmax  
    ↓  
  Output Probabilities

---

## 任务二：阅读论文第 3 节核心组件解析

### 1. Scaled Dot-Product Attention（缩放点积注意力）

**是什么**：可以将其想象成一个信息检索系统。我们有一个查询（Query, $Q$），去匹配一堆键（Key, $K$），匹配度越高，对应提取出的值（Value, $V$）权重就越大。

**核心公式**：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**核心细节**：
- $Q, K, V$ 的由来：在当前的自注意力层中，$Q, K, V$ 都是由同一个输入矩阵 $X$ 乘以不同的可学习权重矩阵（$W^Q, W^K, W^V$）得到的。
- 点积（$QK^T$）：计算 $Q$ 和每个 $K$ 的相似度。相似度越高，点积越大。
- 缩放（$\frac{1}{\sqrt{d_k}}$）：这是极其关键的一步。假设 $d_k$（Key的维度）很大，点积的结果会变得非常大。如果不缩放，直接放进 Softmax，会导致 Softmax 的梯度变得极小（即“梯度消失”），模型无法学习。除以 $\sqrt{d_k}$ 可以让方差保持在 1 左右，保证训练稳定。
- Softmax：将相似度分数转化为概率分布（和为 1）。
- 乘以 $V$：根据概率权重，对 $V$ 进行加权求和，得到最终的注意力输出。

---

### 2. Multi-Head Attention（多头注意力）

**是什么**：与其只做一次注意力计算，不如把 $Q, K, V$ 投影到多个不同的“子空间”（Subspaces）中，并行地进行多次注意力计算，最后把结果拼接起来。

**公式**：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O
$$

其中，$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

**为什么需要多头？**
- 单头注意力会“平均化”所有信息，可能只能关注到一种关系（比如语法结构）。
- 多头注意力允许模型在不同的表示空间里关注不同的信息（比如：一个头关注主谓宾语法，另一个头关注指代关系，第三个头关注语义相似度）。
- 维度变化：原论文中，$d_{model} = 512$，头数 $h = 8$。这意味着每个头的 $d_k = d_v = 512 / 8 = 64$。最后拼接起来（$64 \times 8 = 512$），再经过一次线性映射 $W^O$ 恢复成 512 维。

---

### 3. Position-wise Feed-Forward Network（逐位置前馈网络）

在每个 Encoder 和 Decoder 层的注意力机制之后，都会接一个全连接前馈网络。

**结构**：包含两个线性变换（Linear）和一个 ReLU 激活函数。

**公式**：

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

**“逐位置”（Position-wise）的含义**：这个网络是独立地应用于序列中的每一个位置（每个词）的。也就是说，对于序列里的词 A 和词 B，它们经过这个 FFN 时，使用的权重 $W_1, W_2$ 是完全共享的，但计算过程互不影响。

**维度变化**：原论文中，输入维度 512，中间层维度扩展到 2048，然后再投影回 512。

**作用**：注意力机制主要是做线性加权求和，而 FFN 引入了非线性变换（ReLU），增强了模型的表达能力。

---

### 4. Positional Encoding（位置编码）

**为什么需要它？**
这是 Transformer 与 RNN/LSTM 最大的区别之一。RNN 是按顺序处理的，天然知道词的先后顺序。而 Transformer 的 Self-Attention 是全局并行计算的，对输入序列的顺序毫无感知（如果把输入序列打乱，Self-Attention 算出来的结果完全一样）。所以，必须手动把位置信息注入到输入中。

**怎么做？**
作者没有用可学习的位置嵌入，而是使用了固定公式的正弦和余弦函数：

$$
PE_{(pos, 2i)} = \sin(pos / 10000^{2i / d_{model}})
$$

$$
PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i / d_{model}})
$$

- $pos$ 是位置，$i$ 是维度索引。
- 偶数维度用 sin，奇数维度用 cos。

**为什么用正弦余弦？**
1. 它能表示绝对位置。
2. 由于三角恒等式，它也能让模型轻松学习到相对位置（例如，$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数）。
3. 即使测试时遇到的句子比训练时更长，这种公式也能计算出对应的位置编码，具有很好的外推性。

**如何结合**：将计算出的位置编码直接相加到 Input Embedding（输入词嵌入）上，然后输入到 Encoder/Decoder。

---

## 任务三：核心问题解答笔记

### Q1. 这篇论文最初处理的是什么任务？它为什么尝试不用 RNN 和 CNN？
**任务**：机器翻译（Machine Translation），具体是 WMT 2014 英德和英法翻译任务。

**为什么不用 RNN 和 CNN？**
- **RNN 的痛点**：RNN 是顺序计算的（必须等第 $t-1$ 个词算完才能算第 $t$ 个词），导致无法并行化，训练极其缓慢。而且随着序列变长，长距离依赖信息会丢失（梯度消失）。
- **CNN 的痛点**：CNN 虽然可以并行，但卷积核感受野有限。要捕捉长距离依赖，必须堆叠很多层，计算成本高。
- **Transformer 的优势**：完全基于 Attention，任何两个词之间的计算路径长度都是 $O(1)$（直接连接），既能最大化并行计算，又能完美捕捉长距离依赖。

### Q2. Self-Attention 中的“Self”是什么意思？它和普通 Attention 有什么区别？
**“Self”的含义**：Self-Attention 的 $Q, K, V$ 全部来自同一个输入序列。序列中的每个词都在和序列中的其他词（包括自己）进行交互。

**和普通 Attention（交叉注意力，Cross-Attention）的区别**：
- **普通 Attention**（如 Seq2Seq 中的 Attention）：$Q$ 来自 Decoder，$K$ 和 $V$ 来自 Encoder。它是“查字典”的过程。
- **Self-Attention**：$Q, K, V$ 同源。它是在“自我审视”，目的是让句子内部的词互相建立联系，捕捉句法和语义结构（比如：“it”会自动关联到前文的“the cat”）。

### Q3. Q、K、V 分别来自哪里？它们为什么要经过不同的线性变换？
**来自哪里**：
- Encoder 的 Self-Attention：$Q, K, V$ 全部来自上一层 Encoder 的输出（底层是词嵌入+位置编码）。
- Decoder 的 Masked Self-Attention：$Q, K, V$ 全部来自 Decoder 的上一层输出。
- Decoder 的 Cross-Attention：$Q$ 来自 Decoder 上一层，$K$ 和 $V$ 来自 Encoder 的输出。

**为什么要经过不同的线性变换（$W^Q, W^K, W^V$）**：
同一个词 $X$，乘以不同的权重矩阵，映射到不同的语义空间：
- $Q$ (Query)：代表“我在寻找什么信息？”
- $K$ (Key)：代表“我能提供什么标签？”
- $V$ (Value)：代表“我实际传递的内容是什么？”
如果不做变换（直接用 $XX^T$），模型只能计算词与自身的相似度，表达能力受限。不同的投影让词在不同空间中扮演不同角色。

### Q4. Attention 的结果为什么要经过 softmax？得到的每一行权重表示什么？
**为什么要 Softmax**：
- $QK^T$ 得到的是原始相似度分数（点积），范围不确定，可能有正有负。
- Softmax 将其转化为概率分布（所有值在 $0 \sim 1$ 之间，且和为 $1$），这样才能作为权重去对 $V$ 进行加权求和。
- 同时，Softmax 会拉大差距，让高分更高，低分更低，使注意力更聚焦。

**每一行权重表示什么**：
假设输入序列有 $3$ 个词。Attention 矩阵是 $3 \times 3$ 的。第 $1$ 行表示第 $1$ 个词（作为 Query）对所有 $3$ 个词（作为 Key）的注意力权重。即：第 $1$ 个词在生成自己的表示时，应该从其他词那里提取多少信息。每一行的和为 $1$。

### Q5. Multi-Head Attention 为什么要分成多个头？多个头只是把同一件事重复多次吗？
**绝对不是重复！**
**为什么要多头**：单头注意力只能学习到一种关系。多头相当于把输入投影到多个不同的“子空间”（Subspaces），并行地进行多次 Attention。头 $1$ 可能关注主谓宾语法，头 $2$ 可能关注代词指代，头 $3$ 可能关注语义相似度。最后拼接起来再通过 $W^O$ 融合。
**为什么不是重复**：每个头有自己独立的 $W^Q_i, W^K_i, W^V_i$ 参数矩阵，它们从随机初始化开始，会学到截然不同的注意力模式。

### Q6. Transformer 没有循环结构，它怎样知道词语的先后顺序？
**答案：位置编码（Positional Encoding）**。
因为 Self-Attention 是全局并行的，如果把句子打乱，算出来的结果一模一样。所以必须手动注入位置信息。论文使用了正弦和余弦函数的固定公式：

$$
PE_{(pos, 2i)} = \sin(pos / 10000^{2i / d_{model}})
$$

$$
PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i / d_{model}})
$$

将计算出的位置编码直接相加到 Input Embedding 上。用 sin/cos 是因为三角恒等式能让模型轻松学到相对位置，且具有外推性。

### Q7. Decoder 中为什么需要 Masked Attention？如果能提前看到后面的答案，会发生什么？
**为什么需要**：在训练时，我们使用“Teacher Forcing”，一次性把正确答案喂给 Decoder。但在预测第 $t$ 个词时，模型绝对不能看到第 $t+1$ 个词及之后的词。
**如果没有 Mask 会发生什么**：信息泄露（Information Leakage）。模型会“作弊”直接抄后面的答案。结果就是：训练时 Loss 极低，但推理时完全无法生成合理的句子，彻底失效。
**Mask 的做法**：在计算 Attention 分数后，把未来位置的分数加上一个极大的负数（如 $-\infty$），经过 Softmax 后，这些位置的权重就变成了 $0$，从而被完全屏蔽。

### Q8. 残差连接、Layer Normalization 和 Feed-Forward Network 分别出现在结构图的哪里？它们大致有什么作用？
**残差连接（Add）和 Layer Normalization（Norm）**：
- **位置**：每一个子层（Self-Attention、Cross-Attention、FFN）的外面，都包裹着 `Add & Norm`。结构是：`输出 = LayerNorm(x + Sublayer(x))`。
- **作用**：
  - 残差连接：防止深层网络梯度消失，让梯度可以顺畅回传。
  - LayerNorm：对每个样本的特征维度进行归一化，稳定训练，加速收敛。

**Feed-Forward Network（FFN）**：
- **位置**：在每个注意力子层之后。
- **结构**：两个线性层 + ReLU 激活：

$$
FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

原论文中输入 $512$ 维，中间扩展到 $2048$ 维，再投影回 $512$。
- **作用**：注意力机制主要是线性加权求和，FFN 引入了非线性变换（ReLU），大大增强了模型的特征提取和表达能力。它是“逐位置”的，即对序列中每个词独立计算，参数共享。

---