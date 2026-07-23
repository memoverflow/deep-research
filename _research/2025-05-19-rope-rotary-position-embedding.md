---
title: "RoPE 旋转位置编码：用旋转让模型理解距离"
date: 2025-05-19
level: 3
series: "LLM 原理深度解析"
series_order: 9
series_total: 53
tags: [rope, positional-encoding, rotation, transformer, geometric-algebra]
summary: "RoPE 的核心直觉：把位置信息编码为向量的旋转角度，让两个 token 之间的注意力分数只取决于它们的相对距离——就像时钟上两根指针的夹角只取决于时间差。"
---

# RoPE 旋转位置编码：用旋转让模型理解距离

> Llama、Qwen、Mistral、GPT-NeoX……2022 年之后几乎所有主流大模型都用了同一种位置编码方法：RoPE。它的核心想法简单到优美——把"第几个词"这件事，变成一次旋转。

## 两根时钟指针的类比

在正式讲 RoPE 之前，我想请你看一眼时钟。

假设分针现在指向 3（15 分），再过 10 分钟它会指向 5（25 分）。不管现在几点——无论是下午 2:15 还是早上 9:15——"过了 10 分钟"这件事都对应指针转过 60° 这个固定角度。

换句话说：**两根指针的夹角只取决于时间差，不取决于绝对时间。**

这正是 RoPE 的核心直觉。在 Transformer 中，我们想让两个 token 之间的注意力分数反映它们的"距离"（相对位置），而不是它们各自在序列中的"绝对坐标"。RoPE 的方案是：给每个 token 的向量"转一个角度"，这个角度与它的位置成正比。当我们计算两个 token 的点积时，结果自然只取决于它们转过的角度之差——也就是相对位置。

## 问题：模型为什么需要知道位置？

Transformer 的 Self-Attention 本质上是一个"集合操作"——它看到一堆 token，但如果不额外提供信息，它分不清「猫吃鱼」和「鱼吃猫」。位置编码就是要告诉模型：这些 token 是有顺序的。

### 早期方案的困境

**方案 1：学习绝对位置向量**（GPT-2 风格）

给每个位置 0, 1, 2, ... 学一个向量，加到 token embedding 上。问题：
- 训练时只见过 0~2047，推理时位置 2048 怎么办？→ **无法外推**
- 位置 100 和位置 200 的关系，与位置 500 和位置 600 的关系，模型必须分别学习 → **没有泛化**

**方案 2：正弦位置编码**（原始 Transformer）

用 sin/cos 函数生成位置向量，再加到 embedding 上。虽然理论上可以外推，但实践中效果有限：
- 加法操作会"污染"语义信息——位置和语义混在一起
- 模型很难从加法中干净地提取出相对位置

**方案 3：相对位置偏置**（T5 RPE）

直接在 attention 矩阵上加一个偏置项，表示"距离为 k 的 token 对加多少分"。有效，但：
- 需要构造完整的 N×N 矩阵，与高效 attention 方法不兼容
- 偏置是加在 logit 上的，不够优雅

### RoPE 的设计目标

RoPE 的作者苏剑林（Jianlin Su）2021 年提出了一个干净的问题：

> 能不能找到一个函数 $f(\mathbf{x}, m)$，把位置 $m$ 编码到向量 $\mathbf{x}$ 中，使得两个编码后的向量做内积时，结果只取决于原始向量和它们的相对位置差？

数学上写：

$$\langle f(\mathbf{q}, m), f(\mathbf{k}, n) \rangle = g(\mathbf{q}, \mathbf{k}, m-n)$$

如果能做到这一点，attention score 就自动包含了相对位置信息，不需要额外加偏置，也不需要构造 N×N 矩阵。

## 核心想法：旋转保持角度差

RoPE 的答案是：**用旋转（乘以一个旋转矩阵）来编码位置。**

为什么旋转能工作？回想点积的几何含义：

$$\mathbf{q} \cdot \mathbf{k} = \|\mathbf{q}\| \|\mathbf{k}\| \cos(\theta_{qk})$$

点积取决于两个向量的长度和它们之间的夹角。旋转不改变向量长度，只改变方向。如果我们把 $\mathbf{q}$ 转 $m$ 度、$\mathbf{k}$ 转 $n$ 度，它们之间的夹角就从 $\theta_{qk}$ 变成了 $\theta_{qk} + (m - n)$。看到了吗？夹角的变化只取决于 $m - n$，也就是相对位置！

这就是 RoPE 的全部核心思想。剩下的只是把这个直觉变成精确的数学。

<svg viewBox="0 0 650 300" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:650px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow-rope1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Left: before rotation -->
  <text x="160" y="25" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">旋转前</text>
  <circle cx="160" cy="160" r="100" fill="none" stroke="#3a3a4a" stroke-width="1" stroke-dasharray="4"/>
  <line x1="160" y1="160" x2="250" y2="120" stroke="#6e8eff" stroke-width="2" marker-end="url(#arrow-rope1)"/>
  <text x="255" y="115" fill="#6e8eff" font-size="12" font-family="system-ui">q</text>
  <line x1="160" y1="160" x2="240" y2="200" stroke="#34d399" stroke-width="2" marker-end="url(#arrow-rope1)"/>
  <text x="245" y="205" fill="#34d399" font-size="12" font-family="system-ui">k</text>
  <!-- angle arc -->
  <path d="M 200 140 A 45 45 0 0 1 200 180" fill="none" stroke="#f59e0b" stroke-width="1.5"/>
  <text x="210" y="165" fill="#f59e0b" font-size="11" font-family="system-ui">θ</text>
  <!-- Right: after rotation -->
  <text x="490" y="25" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">旋转后（位置 m 和 n）</text>
  <circle cx="490" cy="160" r="100" fill="none" stroke="#3a3a4a" stroke-width="1" stroke-dasharray="4"/>
  <line x1="490" y1="160" x2="555" y2="80" stroke="#6e8eff" stroke-width="2" marker-end="url(#arrow-rope1)"/>
  <text x="560" y="78" fill="#6e8eff" font-size="12" font-family="system-ui">f(q,m)</text>
  <line x1="490" y1="160" x2="575" y2="140" stroke="#34d399" stroke-width="2" marker-end="url(#arrow-rope1)"/>
  <text x="580" y="138" fill="#34d399" font-size="12" font-family="system-ui">f(k,n)</text>
  <!-- angle arc -->
  <path d="M 530 110 A 45 45 0 0 1 538 130" fill="none" stroke="#f59e0b" stroke-width="1.5"/>
  <text x="545" y="118" fill="#f59e0b" font-size="11" font-family="system-ui">θ+(m−n)ε</text>
  <!-- Bottom note -->
  <text x="325" y="290" text-anchor="middle" fill="#94a3b8" font-size="11" font-family="system-ui">向量长度不变，夹角变化量 (m−n)ε 只取决于相对位置</text>
</svg>

## 用复数让推导变优雅

这里是理解 RoPE 最精彩的部分。如果你有高中复数的知识，接下来的推导会非常自然。

### 从 2D 开始

先考虑最简单的情况：向量只有 2 维，比如 $\mathbf{q} = (q_1, q_2)$。

我们把它看成一个复数：$q = q_1 + i q_2$。

在复数世界里，"旋转一个角度 $\theta$"就是"乘以 $e^{i\theta}$"（欧拉公式）。所以把位置 $m$ 编码进去，就是：

$$f(q, m) = q \cdot e^{im\theta}$$

翻译成人话：**把 query 向量旋转 $m\theta$ 度**。同理，位置 $n$ 的 key：

$$f(k, n) = k \cdot e^{in\theta}$$

现在计算内积（复数内积是一个乘另一个的共轭）：

$$\langle f(q, m), f(k, n) \rangle = q \cdot e^{im\theta} \cdot \overline{k \cdot e^{in\theta}} = q\bar{k} \cdot e^{i(m-n)\theta}$$

看！结果里的位置信息只以 $(m - n)$ 的形式出现。这正是我们想要的——注意力分数只取决于相对距离。

### 推广到高维

实际模型中，向量维度是 64 或 128。RoPE 的做法是：**把 $d$ 维向量两两配对，形成 $d/2$ 个 2D 平面，每个平面独立旋转，但旋转速度不同。**

具体来说，第 $j$ 个平面（即第 $2j-1$ 和第 $2j$ 个维度）使用的旋转频率为：

$$\theta_j = \frac{1}{10000^{2j/d}}$$

位置 $m$ 的 token，在第 $j$ 个平面上旋转角度 $m \cdot \theta_j$。

写成矩阵形式，就是一个分块对角矩阵，每个 2×2 块是一个旋转矩阵：

$$R_m = \begin{pmatrix} \cos m\theta_1 & -\sin m\theta_1 \\ \sin m\theta_1 & \cos m\theta_1 \\ & & \cos m\theta_2 & -\sin m\theta_2 \\ & & \sin m\theta_2 & \cos m\theta_2 \\ & & & & \ddots \end{pmatrix}$$

最终：$f(\mathbf{q}, m) = R_m \cdot W_q \mathbf{x}_m$

先做线性变换得到 query，再旋转。就这么简单。

## 频率的选择：为什么是 10000 的幂次？

上面我们说每个维度对的旋转频率是 $\theta_j = 10000^{-2j/d}$。这个设计不是随意的，它创造了一个**多尺度的位置感知系统**。

想象一个类比：你家地址由"省-市-区-街道-门牌号"组成。高层级变化慢（省很少变），低层级变化快（门牌号每户都不同）。

RoPE 中的频率也是这样：

- **第 1 对维度**：$\theta_1 = 1$（频率最高），每移动一个位置就转 1 弧度 ≈ 57°。这对维度对**近距离**差异最敏感。
- **最后一对维度**：$\theta_{d/2} = 10000^{-1}$（频率极低），每移动一个位置只转 0.0001 弧度。它要走 ~60000 个位置才转一圈，用于编码**长距离**关系。

这种指数级递减的频率分布，让模型同时具备了"看到邻居"和"看到远方"的能力。

<svg viewBox="0 0 700 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <!-- Title -->
  <text x="350" y="22" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">RoPE 多频率旋转示意（位置 0→8）</text>
  <!-- High freq -->
  <text x="50" y="65" fill="#6e8eff" font-size="11" font-family="system-ui">高频维度</text>
  <text x="50" y="80" fill="#94a3b8" font-size="10" font-family="system-ui">θ₁ ≈ 1 rad/pos</text>
  <circle cx="160" cy="75" r="30" fill="none" stroke="#3a3a4a" stroke-width="0.8"/>
  <line x1="160" y1="75" x2="190" y2="75" stroke="#6e8eff" stroke-width="1.5"/>
  <circle cx="240" cy="75" r="30" fill="none" stroke="#3a3a4a" stroke-width="0.8"/>
  <line x1="240" y1="75" x2="253" y2="49" stroke="#6e8eff" stroke-width="1.5"/>
  <circle cx="320" cy="75" r="30" fill="none" stroke="#3a3a4a" stroke-width="0.8"/>
  <line x1="320" y1="75" x2="308" y2="48" stroke="#6e8eff" stroke-width="1.5"/>
  <circle cx="400" cy="75" r="30" fill="none" stroke="#3a3a4a" stroke-width="0.8"/>
  <line x1="400" y1="75" x2="371" y2="68" stroke="#6e8eff" stroke-width="1.5"/>
  <circle cx="480" cy="75" r="30" fill="none" stroke="#3a3a4a" stroke-width="0.8"/>
  <line x1="480" y1="75" x2="455" y2="90" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="160" y="115" text-anchor="middle" fill="#94a3b8" font-size="9" font-family="system-ui">pos 0</text>
  <text x="240" y="115" text-anchor="middle" fill="#94a3b8" font-size="9" font-family="system-ui">pos 1</text>
  <text x="320" y="115" text-anchor="middle" fill="#94a3b8" font-size="9" font-family="system-ui">pos 2</text>
  <text x="400" y="115" text-anchor="middle" fill="#94a3b8" font-size="9" font-family="system-ui">pos 3</text>
  <text x="480" y="115" text-anchor="middle" fill="#94a3b8" font-size="9" font-family="system-ui">pos 4</text>
  <!-- Low freq -->
  <text x="50" y="165" fill="#34d399" font-size="11" font-family="system-ui">低频维度</text>
  <text x="50" y="180" fill="#94a3b8" font-size="10" font-family="system-ui">θ₃₂ ≈ 0.0001</text>
  <circle cx="160" cy="175" r="30" fill="none" stroke="#3a3a4a" stroke-width="0.8"/>
  <line x1="160" y1="175" x2="190" y2="175" stroke="#34d399" stroke-width="1.5"/>
  <circle cx="240" cy="175" r="30" fill="none" stroke="#3a3a4a" stroke-width="0.8"/>
  <line x1="240" y1="175" x2="270" y2="174" stroke="#34d399" stroke-width="1.5"/>
  <circle cx="320" cy="175" r="30" fill="none" stroke="#3a3a4a" stroke-width="0.8"/>
  <line x1="320" y1="175" x2="350" y2="174" stroke="#34d399" stroke-width="1.5"/>
  <circle cx="400" cy="175" r="30" fill="none" stroke="#3a3a4a" stroke-width="0.8"/>
  <line x1="400" y1="175" x2="430" y2="174" stroke="#34d399" stroke-width="1.5"/>
  <circle cx="480" cy="175" r="30" fill="none" stroke="#3a3a4a" stroke-width="0.8"/>
  <line x1="480" y1="175" x2="510" y2="173" stroke="#34d399" stroke-width="1.5"/>
  <text x="160" y="215" text-anchor="middle" fill="#94a3b8" font-size="9" font-family="system-ui">pos 0</text>
  <text x="240" y="215" text-anchor="middle" fill="#94a3b8" font-size="9" font-family="system-ui">pos 1</text>
  <text x="320" y="215" text-anchor="middle" fill="#94a3b8" font-size="9" font-family="system-ui">pos 2</text>
  <text x="400" y="215" text-anchor="middle" fill="#94a3b8" font-size="9" font-family="system-ui">pos 3</text>
  <text x="480" y="215" text-anchor="middle" fill="#94a3b8" font-size="9" font-family="system-ui">pos 4</text>
  <!-- Legend -->
  <text x="350" y="242" text-anchor="middle" fill="#94a3b8" font-size="10" font-family="system-ui">高频维度：指针每步转大角度（感知近距离） | 低频维度：几乎不动（编码远距离）</text>
</svg>

## 一个关键性质：长程衰减

RoPE 还有一个被广泛讨论的性质：**随着两个 token 距离增大，它们的注意力分数在期望意义下会自然衰减**。

直觉上很好理解：当相对距离增大时，高频维度的旋转角 $(m-n)\theta_j$ 变得越来越大，$\cos$ 和 $\sin$ 值开始剧烈振荡。振荡的点积在平均意义上趋向零——就像你随机转两个时钟指针，它们的平均夹角是 90°（点积为零）。

这意味着 RoPE 自带了一种"距离越远、注意力越弱"的归纳偏置，类似于 ALiBi 的线性衰减，但更灵活——模型可以通过学习利用不同频率的维度来选择性地关注远处或近处。

## 与正弦位置编码的区别

初学者常问："RoPE 也用了 sin 和 cos，和原始 Transformer 的正弦位置编码有什么不同？"

两个关键区别：

| | 正弦位置编码 | RoPE |
|---|---|---|
| **操作方式** | 加法（加到 embedding 上） | 乘法（旋转 query/key） |
| **作用对象** | 每个维度独立 | 两两配对，在 2D 平面旋转 |
| **位置信息** | 混入语义向量，难以分离 | 只作用于 attention 计算，不污染表示 |
| **相对位置** | 需要模型自己学着提取 | 数学上保证点积只含相对位置 |

正弦编码是"告诉模型你在哪"（加上一个取决于位置的向量）；RoPE 是"把你转到那个位置"（施加一个旋转）。旋转不改变向量本身的信息（长度不变），只改变向量之间的关系——这正是 attention 需要的。

## 实际实现：比你想象的简单

虽然理论上是一个巨大的分块对角旋转矩阵，但实现时不需要做矩阵乘法。利用旋转矩阵的结构，可以用逐元素操作：

```python
def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [batch, seq, heads, dim]
    # cos, sin: [seq, dim] -- 预计算的 cos(m*θ_j), sin(m*θ_j)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)
```

核心就两行：原向量乘 cos + 旋转半边乘 sin。计算量几乎可以忽略——没有额外的参数，没有额外的矩阵乘法，只有逐元素操作。

这也是 RoPE 被广泛采用的原因之一：**零额外参数，极低计算开销，效果还更好。**

## 上下文长度扩展：当指针转太多圈

RoPE 有一个实际限制：模型在训练时只见过有限的位置。如果训练时最长序列是 4096 token，那么推理时遇到位置 8000，高频维度的旋转角会远超训练时见过的范围——模型没学过"转了这么多圈"意味着什么。

这催生了一系列扩展方法：

### Position Interpolation (PI)

2023 年 Meta 提出：与其让位置超出范围（外推），不如把新的长序列"压缩"到原来的范围内。比如要处理 8K 序列但只训练过 4K，就把所有位置除以 2：位置 8000 变成 4000，回到训练见过的范围。

代价：原来距离为 1 的两个 token，位置差变成了 0.5——分辨率降低。但实验表明，少量微调（~1000 步）就能适应。

### NTK-aware Scaling

社区发现了一个更聪明的方法：**不要均匀缩放所有频率，而是只拉伸低频维度**。

直觉：高频维度本来就只关注"邻近几个 token"，它们不需要变；低频维度负责长距离，它们需要"转得更慢"才能覆盖更长的序列。

具体操作是把 base 从 10000 增大。Llama 3 直接把 base 设为 500000，低频维度的波长从 ~60K 位置扩展到 ~3M 位置，原生支持 128K 上下文。

### YaRN

YaRN（Yet another RoPE extensioN）进一步精细化：把所有频率维度分成三组——高频（不动）、低频（做 PI）、中间过渡（插值）。结合少量微调，在各种长度上都表现良好。

## 为什么几乎所有模型都选了 RoPE？

回顾 RoPE 的优势列表：

1. **数学优雅**：相对位置信息是从旋转的几何性质自然导出的，不是 hack
2. **零参数**：不需要学习任何位置相关的参数
3. **兼容 KV Cache**：旋转在生成 query/key 时就完成了，缓存的 KV 不需要更新
4. **可扩展**：通过调整 base 频率或做 PI/NTK/YaRN 就能扩展上下文
5. **效果好**：EleutherAI 的实验表明，125M 到 1.4B 参数模型上 RoPE 都优于学习绝对编码和 T5 RPE
6. **兼容高效 Attention**：因为编码在 Q/K 向量内部，不需要构造额外的 N×N 矩阵，与 FlashAttention 等方法完全兼容

这些优势的组合，让 RoPE 在 2022-2025 年间成为了事实上的行业标准。GPT-NeoX、Llama 1/2/3、Qwen、Mistral、Gemma——几乎所有开源和闭源大模型都采用了它。

## 局限性与前沿挑战

RoPE 不是完美的。2025-2026 年的研究揭示了一些有趣的问题：

**频率带利用不均**：ICLR 2026 的一篇论文发现，训练好的模型实际上只"有效利用"了一部分频率维度。低于某个频率带的维度几乎没被用到——替换成无位置编码（NoPE）对性能影响很小。

**Token Aliasing**：当序列非常长时，高频维度已经转了太多圈，不同位置的旋转后向量几乎无法区分——就像时钟转了 100 圈后你分不清它到底转了多少圈。

**外推-内插权衡**：增大 base θ 有助于内插（处理更长序列），但会降低外推能力。没有免费午餐。

这些发现正在推动下一代位置编码的研究，但 RoPE 作为"够好"的默认选择，短期内不太可能被取代。

## 总结：一句话回顾

RoPE 的故事很简单：

> **把位置变成旋转角度，让点积自动感知相对距离。**

这个想法之所以强大，是因为它完美契合了 Transformer 注意力机制的数学结构——点积天然对旋转敏感。不需要额外参数，不需要修改架构，只需要在计算 attention 前把 Q 和 K "转一下"。

下次你看到一个大模型的技术报告里写着 "RoPE with base θ = 500000"，你就知道了：这意味着模型的每一对维度都在以不同速度旋转，高频维度负责区分邻近的词，低频维度负责编码段落级的远距离关系。所有这些旋转协同工作，让一个本来"没有时间概念"的 Transformer 获得了精准的位置感知能力。

---

**参考来源：**
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021), arXiv:2104.09864
- EleutherAI, "Rotary Embeddings: A Relative Revolution" (2021)
- Chen et al., "Extending Context Window of Large Language Models via Positional Interpolation" (2023), arXiv:2306.15595
- Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models" (2023), arXiv:2309.00071
- Oka et al., "Frequency Bands in RoPE" (2026), ICLR 2026
