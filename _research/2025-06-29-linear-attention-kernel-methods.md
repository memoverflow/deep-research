---
title: "Linear Attention 的核方法视角：如何用一个数学技巧把 O(N²) 变成 O(N)"
date: 2025-06-29
level: 3
series: "LLM 原理深度解析"
series_order: 19
series_total: 39
tags: [linear-attention, kernel-methods, efficient-transformer, attention]
summary: "从核方法的视角理解 Linear Attention：softmax 其实是一个核函数，而打破二次复杂度的秘密藏在矩阵乘法结合律里。"
---

# Linear Attention 的核方法视角：如何用一个数学技巧把 O(N²) 变成 O(N)

> 标准 Attention 的计算量随序列长度平方增长。但如果你换一种方式看待 softmax——把它当作一个核函数——一条通往线性复杂度的路就会自然浮现。

## 故事从一个乘法顺序说起

假设你是一个老师，班上有 1000 个学生，每个学生需要根据全班同学的表现来调整自己的学习计划。如果每个学生都要逐一对比其他 999 个同学，工作量是 1000 × 1000 = 一百万次比较。

这正是标准 Transformer Attention 面对的处境：每个 token 都要和序列中所有其他 token 计算相似度，产生一个 N×N 的注意力矩阵。当 N 从 512 增长到 100K，计算量从 26 万暴涨到 100 亿——这就是著名的"二次复杂度瓶颈"。

但如果我告诉你，有一种方法可以避免构建这个 N×N 的矩阵，直接算出最终结果呢？秘密就藏在一个你中学就学过的数学性质里：**矩阵乘法的结合律**。

而连接"结合律"和"注意力"的桥梁，就是核方法（Kernel Methods）——一套在机器学习领域已经存在了几十年的数学工具。

## 第一章：Attention 其实是一种古老的统计方法

### 问题回顾：标准 Attention 在做什么

让我们重新审视标准的 Scaled Dot-Product Attention：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V$$

翻译成人话：对于每个查询位置 i，我们计算它和所有键位置 j 的相似度（点积），通过 softmax 归一化成权重，然后用这些权重对值做加权平均。

换个角度看——这不就是**加权投票**吗？每个位置 i 的输出是所有位置的值的加权平均，权重由"和我有多相似"决定。

### 核心洞察：这是 1964 年的 Nadaraya-Watson 回归

统计学家在 1964 年就发明了一个几乎一模一样的东西——**Nadaraya-Watson 核回归**：

$$\hat{f}(x) = \frac{\sum_{j=1}^n K(x, x_j) \cdot y_j}{\sum_{j=1}^n K(x, x_j)}$$

这里 K(x, x_j) 是一个**核函数**，衡量 x 和 x_j 的相似程度。预测值就是所有样本的加权平均，权重正比于和查询点的相似度。

对比 Attention 的逐元素写法：

$$\text{output}_i = \frac{\sum_{j=1}^n \exp(q_i \cdot k_j / \sqrt{d}) \cdot v_j}{\sum_{j=1}^n \exp(q_i \cdot k_j / \sqrt{d})}$$

你发现了吗？**Softmax attention 就是用 $\exp(q \cdot k / \sqrt{d})$ 作为核函数的 Nadaraya-Watson 回归！**

这不是巧合或比喻——数学上它们完全等价。2026 年 arxiv 上甚至有论文严格证明 Multi-Head Attention 就是一组 Nadaraya-Watson 估计器的集成。

### 那么，这个视角有什么用？

在统计学中，核方法有一个著名的技巧：**核函数可以被分解为特征映射的内积**。也就是说，存在某个函数 φ，使得：

$$K(x, y) = \langle \phi(x), \phi(y) \rangle$$

如果我们能对 softmax 核做这样的分解，那么 attention 的计算方式就可以被彻底重写——而重写之后，二次复杂度就会消失。

## 第二章：结合律的魔法——Linear Attention 的核心思想

### 为什么 softmax 会导致二次复杂度

让我们用矩阵符号看标准 attention。分子的计算是：

$$\text{Output} = \text{softmax}(QK^T) \cdot V$$

问题出在 $QK^T$ 这一步——它产生一个 N×N 的矩阵。如果 Q 是 N×d，K 是 N×d，那么 $QK^T$ 是 N×N。当 N=100000，d=128 时，这个中间矩阵有 100 亿个元素。

但更根本的问题是 **softmax 的逐行归一化**——它要求先算出完整的 N×N 分数矩阵，才能做 softmax，然后才能乘 V。计算顺序被锁死为：

$$(QK^T) \rightarrow \text{softmax} \rightarrow \times V$$

你无法改变这个顺序，因为 softmax 是非线性的，它纠缠了 Q 和 K 的关系。

### 核心想法：去掉 softmax，改变乘法顺序

假设我们不用 softmax，而是用一个可以被分解为特征映射内积的核函数：

$$\text{sim}(q_i, k_j) = \phi(q_i)^T \phi(k_j)$$

其中 $\phi: \mathbb{R}^d \rightarrow \mathbb{R}^m$ 是某个特征映射函数。那么注意力变成：

$$\text{output}_i = \frac{\sum_j \phi(q_i)^T \phi(k_j) \cdot v_j}{\sum_j \phi(q_i)^T \phi(k_j)}$$

现在关键来了——分子可以重写为：

$$\sum_j \phi(q_i)^T \phi(k_j) \cdot v_j = \phi(q_i)^T \sum_j \phi(k_j) v_j^T$$

**我们把 $\phi(q_i)$ 提到求和号外面了！**

这意味着我们可以先计算 $\sum_j \phi(k_j) v_j^T$（这是一个 m×d 的矩阵，与 N 无关），然后每个查询只需要和这个固定大小的矩阵做一次乘法。

用矩阵符号：

- 标准 attention 的计算顺序：$(Q' \cdot K'^T) \cdot V$，中间产生 N×N 矩阵
- Linear attention 的计算顺序：$Q' \cdot (K'^T \cdot V)$，中间只产生 m×d 矩阵

其中 $Q' = \phi(Q)$，$K' = \phi(K)$。

**这就是结合律的魔法：(A·B)·C = A·(B·C)，但计算代价完全不同！**

<svg viewBox="0 0 700 320" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Title -->
  <text x="350" y="25" text-anchor="middle" fill="#ededf0" font-size="14" font-weight="bold" font-family="system-ui">标准 Attention vs Linear Attention：乘法顺序的差异</text>
  
  <!-- Standard Attention (top) -->
  <text x="80" y="60" text-anchor="middle" fill="#ff6e6e" font-size="12" font-family="system-ui">标准 Attention: (QK^T)·V</text>
  <rect x="20" y="75" width="50" height="80" rx="4" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="45" y="120" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">Q</text>
  <text x="45" y="135" text-anchor="middle" fill="#6e8eff" font-size="9" font-family="system-ui">N×d</text>
  
  <text x="85" y="118" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui">×</text>
  
  <rect x="100" y="90" width="80" height="50" rx="4" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="140" y="118" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">K^T</text>
  <text x="140" y="133" text-anchor="middle" fill="#6e8eff" font-size="9" font-family="system-ui">d×N</text>
  
  <text x="195" y="118" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui">=</text>
  
  <rect x="210" y="75" width="80" height="80" rx="4" fill="#3a1e1e" stroke="#ff6e6e" stroke-width="2"/>
  <text x="250" y="115" text-anchor="middle" fill="#ff6e6e" font-size="11" font-family="system-ui">N×N</text>
  <text x="250" y="132" text-anchor="middle" fill="#ff6e6e" font-size="9" font-family="system-ui">💥 巨大!</text>
  
  <text x="305" y="118" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui">×</text>
  
  <rect x="320" y="75" width="50" height="80" rx="4" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="345" y="120" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">V</text>
  <text x="345" y="135" text-anchor="middle" fill="#6e8eff" font-size="9" font-family="system-ui">N×d</text>
  
  <text x="500" y="115" text-anchor="start" fill="#ff6e6e" font-size="11" font-family="system-ui">复杂度: O(N²d)</text>
  
  <!-- Linear Attention (bottom) -->
  <text x="80" y="200" text-anchor="middle" fill="#6eff6e" font-size="12" font-family="system-ui">Linear Attention: Q·(K^T·V)</text>
  
  <rect x="100" y="220" width="80" height="50" rx="4" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="140" y="248" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">K^T</text>
  <text x="140" y="263" text-anchor="middle" fill="#6e8eff" font-size="9" font-family="system-ui">d×N</text>
  
  <text x="195" y="248" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui">×</text>
  
  <rect x="210" y="215" width="50" height="80" rx="4" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="235" y="258" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">V</text>
  <text x="235" y="273" text-anchor="middle" fill="#6e8eff" font-size="9" font-family="system-ui">N×d</text>
  
  <text x="275" y="248" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui">=</text>
  
  <rect x="290" y="230" width="50" height="50" rx="4" fill="#1e3a1e" stroke="#6eff6e" stroke-width="2"/>
  <text x="315" y="258" text-anchor="middle" fill="#6eff6e" font-size="11" font-family="system-ui">d×d</text>
  <text x="315" y="273" text-anchor="middle" fill="#6eff6e" font-size="9" font-family="system-ui">✓ 小!</text>
  
  <text x="355" y="258" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui">← Q×</text>
  
  <rect x="20" y="215" width="50" height="80" rx="4" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="45" y="258" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">Q</text>
  <text x="45" y="273" text-anchor="middle" fill="#6e8eff" font-size="9" font-family="system-ui">N×d</text>
  
  <text x="500" y="255" text-anchor="start" fill="#6eff6e" font-size="11" font-family="system-ui">复杂度: O(Nd²)</text>
  
  <!-- Comparison -->
  <text x="500" y="300" text-anchor="start" fill="#ededf0" font-size="10" font-family="system-ui">当 N≫d 时，节省巨大</text>
  <text x="500" y="315" text-anchor="start" fill="#ededf0" font-size="10" font-family="system-ui">N=100K, d=128 → 780x 加速</text>
</svg>

### 复杂度对比

- **标准 Attention**：先算 QK^T (N×N)，再乘 V → O(N²d)
- **Linear Attention**：先算 K^T·V (d×d)，再让 Q 去查 → O(Nd²)

当序列长度 N 远大于特征维度 d 时（比如 N=100000，d=128），这是从 O(N²) 到 O(N) 的质变！

## 第三章：核函数的选择——从 SVM 到 Attention

### 回顾：传统机器学习中的核技巧

在 SVM 中，核技巧的方向是：**避免显式计算高维特征映射 φ(x)**，因为 φ 可能映射到无限维空间。我们只需要知道内积 $K(x,y) = \langle\phi(x), \phi(y)\rangle$ 就够了。

但 Linear Attention 做的恰恰相反——它是**反向核技巧**：

- SVM 核技巧：已知 φ（高维甚至无限维），用 K(x,y) 避免显式计算 φ
- Linear Attention：选择一个**低维的** φ，显式计算 φ(q) 和 φ(k)，从而利用结合律

这个"反向"是理解 Linear Attention 的关键。我们不是在逃避高维映射，而是故意选择一个足够简单的特征映射，使得乘法顺序可以交换。

### Softmax 核的特征映射是什么？

理论上，softmax 核 $K(q,k) = \exp(q^Tk)$ 的精确特征映射是**无限维**的（通过 Taylor 展开）：

$$\exp(q^T k) = \sum_{n=0}^{\infty} \frac{(q^T k)^n}{n!}$$

每一项 $(q^Tk)^n$ 对应越来越高阶的特征交互。这就是为什么 softmax attention 如此强大——它隐式地考虑了所有阶的特征组合。

但这也意味着你无法用有限维的 φ 精确分解 softmax。你只能**近似**。

### 方案一：最简单的 φ —— 恒等映射（elu+1）

Katharopoulos et al. (2020) 的开创性论文"Transformers are RNNs"提出了最朴素的方案：

$$\phi(x) = \text{elu}(x) + 1$$

其中 elu(x) = x if x≥0, else exp(x)-1。加 1 保证非负。

这相当于说：核函数就是 $K(q,k) = (\text{elu}(q)+1)^T (\text{elu}(k)+1)$。它和 softmax 核差距巨大，但好处是特征维度 m=d（没有膨胀），计算极其便宜。

### 方案二：随机特征——Performer 的 FAVOR+

Choromanski et al. (2021) 的 Performer 用了更精巧的方式——随机 Fourier 特征来近似 softmax 核：

$$\phi(x) = \frac{1}{\sqrt{m}} \exp\left(-\frac{\|x\|^2}{2}\right) \left[\exp(w_1^T x), \exp(w_2^T x), ..., \exp(w_m^T x)\right]$$

其中 $w_1, ..., w_m$ 是从标准正态分布中采样的随机向量，并经过正交化处理。

直觉：想象你在一个高维空间中随机撒了 m 个"探测器"。每个探测器测量输入向量在某个随机方向上的"能量"。通过足够多的探测器，你可以近似重建出原始核函数。

这是经典的 Bochner 定理的应用——任何平移不变的正定核都可以写成随机特征的期望。

**正交随机特征（ORF）**的改进：让 $w_1,...,w_m$ 彼此正交，可以减小近似方差，这就是 FAVOR+ 中的"O"。

### 方案三：Taylor 展开——一阶即线性

还有一种理解方式：对 $\exp(q^Tk)$ 做 Taylor 展开并只保留前两项：

$$\exp(q^Tk) \approx 1 + q^Tk$$

这是最粗糙的近似，但它解释了为什么简单的线性 attention（连 φ 都不用，直接 Q·K^T → QK^TV）在某些任务上也能工作。代价是精度大幅下降——你丢掉了所有高阶特征交互。

正如一篇博客精辟总结的：**Linear Attention 和 Softmax Attention 的能力差距，就是一阶 Taylor 近似和完整无穷级数之间的差距。**

## 第四章：当 Linear Attention 变成 RNN

### 一个惊人的发现

Linear Attention 还有一个让人拍案叫绝的性质：**在 causal（自回归）模式下，它等价于一个 RNN。**

在自回归生成中，每个位置 i 只能看到位置 1 到 i。Linear Attention 的输出变成：

$$o_i = \frac{\phi(q_i)^T \sum_{j=1}^i \phi(k_j) v_j^T}{\phi(q_i)^T \sum_{j=1}^i \phi(k_j)}$$

定义 **状态矩阵** $S_i = \sum_{j=1}^i \phi(k_j) v_j^T$ 和 **归一化向量** $z_i = \sum_{j=1}^i \phi(k_j)$，它们有递推关系：

$$S_i = S_{i-1} + \phi(k_i) v_i^T$$
$$z_i = z_{i-1} + \phi(k_i)$$

然后输出只需要：

$$o_i = \frac{\phi(q_i)^T S_i}{\phi(q_i)^T z_i}$$

**这就是一个 RNN！** 状态 $S_i$ 在每步累加新信息，查询时从状态中提取答案。

<svg viewBox="0 0 700 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
    <marker id="arrow3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#34d399"/>
    </marker>
  </defs>
  <!-- Title -->
  <text x="350" y="20" text-anchor="middle" fill="#ededf0" font-size="14" font-weight="bold" font-family="system-ui">Linear Attention 的 RNN 视角：状态累积</text>
  
  <!-- Time steps -->
  <!-- Step 1 -->
  <rect x="50" y="80" width="80" height="50" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="90" y="102" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">S₁ = φ(k₁)v₁ᵀ</text>
  <text x="90" y="120" text-anchor="middle" fill="#6e8eff" font-size="9" font-family="system-ui">状态 (d×d)</text>
  
  <!-- Arrow to step 2 -->
  <line x1="130" y1="105" x2="175" y2="105" stroke="#34d399" stroke-width="1.5" marker-end="url(#arrow3)"/>
  <text x="152" y="98" text-anchor="middle" fill="#34d399" font-size="9" font-family="system-ui">+ φ(k₂)v₂ᵀ</text>
  
  <!-- Step 2 -->
  <rect x="180" y="80" width="80" height="50" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="220" y="102" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">S₂ = S₁ + ...</text>
  <text x="220" y="120" text-anchor="middle" fill="#6e8eff" font-size="9" font-family="system-ui">状态 (d×d)</text>
  
  <!-- Arrow to step 3 -->
  <line x1="260" y1="105" x2="305" y2="105" stroke="#34d399" stroke-width="1.5" marker-end="url(#arrow3)"/>
  <text x="282" y="98" text-anchor="middle" fill="#34d399" font-size="9" font-family="system-ui">+ φ(k₃)v₃ᵀ</text>
  
  <!-- Step 3 -->
  <rect x="310" y="80" width="80" height="50" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="350" y="102" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">S₃ = S₂ + ...</text>
  <text x="350" y="120" text-anchor="middle" fill="#6e8eff" font-size="9" font-family="system-ui">状态 (d×d)</text>
  
  <!-- Arrow to step N -->
  <line x1="390" y1="105" x2="435" y2="105" stroke="#34d399" stroke-width="1.5" stroke-dasharray="4,4"/>
  <text x="413" y="98" text-anchor="middle" fill="#34d399" font-size="9" font-family="system-ui">...</text>
  
  <!-- Step N -->
  <rect x="440" y="80" width="80" height="50" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="480" y="102" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">Sₙ = Sₙ₋₁ + ...</text>
  <text x="480" y="120" text-anchor="middle" fill="#6e8eff" font-size="9" font-family="system-ui">状态 (d×d)</text>
  
  <!-- Query arrows -->
  <line x1="90" y1="130" x2="90" y2="170" stroke="#a78bfa" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <line x1="220" y1="130" x2="220" y2="170" stroke="#a78bfa" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <line x1="350" y1="130" x2="350" y2="170" stroke="#a78bfa" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <line x1="480" y1="130" x2="480" y2="170" stroke="#a78bfa" stroke-width="1.5" marker-end="url(#arrow2)"/>
  
  <!-- Outputs -->
  <rect x="55" y="175" width="70" height="40" rx="8" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="90" y="197" text-anchor="middle" fill="#a78bfa" font-size="10" font-family="system-ui">o₁ = φ(q₁)ᵀS₁</text>
  
  <rect x="185" y="175" width="70" height="40" rx="8" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="220" y="197" text-anchor="middle" fill="#a78bfa" font-size="10" font-family="system-ui">o₂ = φ(q₂)ᵀS₂</text>
  
  <rect x="315" y="175" width="70" height="40" rx="8" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="350" y="197" text-anchor="middle" fill="#a78bfa" font-size="10" font-family="system-ui">o₃ = φ(q₃)ᵀS₃</text>
  
  <rect x="445" y="175" width="70" height="40" rx="8" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="480" y="197" text-anchor="middle" fill="#a78bfa" font-size="10" font-family="system-ui">oₙ = φ(qₙ)ᵀSₙ</text>
  
  <!-- Key insight -->
  <text x="350" y="240" text-anchor="middle" fill="#34d399" font-size="11" font-family="system-ui">推理时：固定大小状态 + O(1) 每步更新 → 无限长序列！</text>
</svg>

### 这意味着什么？

1. **训练时**可以用矩阵并行（先算 K^T·V），复杂度 O(Nd²)
2. **推理时**可以用 RNN 递推（每步只更新状态），复杂度 O(d²) 每步——**常数时间推理！**

这就是 Katharopoulos 论文标题"Transformers are RNNs"的含义：换掉 softmax，Transformer 在数学上就变成了一个 RNN。

### 但这个 RNN 有一个致命弱点

注意状态更新公式：$S_i = S_{i-1} + \phi(k_i) v_i^T$

**它只会累加，永远不会遗忘。**

这意味着：
- 序列开头的无关信息会永远污染状态
- 状态中的信号会被不断堆积的噪声淹没
- 越长的序列，信噪比越低

这就是为什么原始 Linear Attention 在语言模型中表现远不如 softmax attention——语言需要选择性遗忘。

## 第五章：弥合差距——2023-2025 年的进化

### 思路一：学会遗忘——Gated Linear Attention

既然问题是"只累加不遗忘"，解决方案就是加入**衰减门控**：

$$S_i = \gamma_i \odot S_{i-1} + \phi(k_i) v_i^T$$

其中 $\gamma_i$ 是数据依赖的衰减系数（0到1之间），控制旧信息保留多少。

**RetNet (2023)** 用固定的指数衰减 $\gamma^{i-j}$，类似于给记忆加了一个半衰期。
**GLA (2024)** 让衰减率由数据决定——重要的信息衰减慢，无关的快速遗忘。
**HGRN2 (2024)** 将门控机制做了层次化，不同粒度的遗忘速率不同。

### 思路二：学习更好的 φ —— Hedgehog & LUNA

2024 年的 Hedgehog 和 2025 年的 LUNA 走了另一条路：**与其用固定的 φ 近似 softmax，不如直接学习一个最优的 φ。**

Hedgehog 的核心观察是：softmax 产生的注意力权重是**尖锐的**（集中在少数位置）且**单调的**（相似度越高权重越大）。简单的特征映射（如 elu+1）产生的权重过于平滑。所以 Hedgehog 训练一个小型网络作为 φ，目标是模仿 softmax 的尖锐分布。

LUNA (2025) 更进一步证明：不是 linear attention 本身有问题，而是之前所有固定的 φ 都不够好。用可学习的核特征映射，linear attention 可以达到甚至超过 softmax attention 的精度。

### 思路三：接受差异，混合使用

2025 年最实用的方案可能是**混合架构**——在关键层使用完整的 softmax attention（处理需要精确检索的信息），其余层使用 linear attention 或 gated linear RNN（处理一般的信息流动）。

这就像一个公司：少数高管做精细决策（softmax），大量员工按流程办事（linear）——效率和质量兼顾。

## 第六章：直觉总结——为什么核视角重要

让我们把整个故事串起来：

1. **Attention = 核回归**：每个 token 的输出是所有 token 的加权平均，权重由核函数（softmax）决定
2. **核 = 特征内积**：如果核函数可以写成 $\phi(q)^T\phi(k)$，计算就可以重组
3. **结合律 = 线性复杂度**：先算 $\phi(K)^TV$（与 N 无关的固定大小），每个 query 只需一次矩阵向量乘法
4. **Causal + 核分解 = RNN**：状态累积式更新，O(1) 推理
5. **核的质量 = 模型能力**：φ 越接近 softmax 的无限维特征映射，模型越强；差距就是丢掉的高阶交互

**核心 trade-off**：特征维度 m 越大，近似越精确，但 O(Nmd) 的复杂度也越高。整个 Linear Attention 研究都在这条 trade-off 曲线上寻找最优点。

<svg viewBox="0 0 650 280" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:650px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow4" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Title -->
  <text x="325" y="20" text-anchor="middle" fill="#ededf0" font-size="14" font-weight="bold" font-family="system-ui">Linear Attention 的演进路线图</text>
  
  <!-- Timeline -->
  <line x1="50" y1="70" x2="600" y2="70" stroke="#3a3a4a" stroke-width="2" marker-end="url(#arrow4)"/>
  
  <!-- 2020 -->
  <circle cx="100" cy="70" r="5" fill="#6e8eff"/>
  <text x="100" y="55" text-anchor="middle" fill="#6e8eff" font-size="10" font-family="system-ui">2020</text>
  <rect x="55" y="85" width="90" height="55" rx="6" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1"/>
  <text x="100" y="102" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">Linear Transformer</text>
  <text x="100" y="115" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">(Katharopoulos)</text>
  <text x="100" y="130" text-anchor="middle" fill="#6e8eff" font-size="8" font-family="system-ui">φ = elu + 1</text>
  
  <!-- 2021 -->
  <circle cx="200" cy="70" r="5" fill="#6e8eff"/>
  <text x="200" y="55" text-anchor="middle" fill="#6e8eff" font-size="10" font-family="system-ui">2021</text>
  <rect x="155" y="85" width="90" height="55" rx="6" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1"/>
  <text x="200" y="102" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">Performer</text>
  <text x="200" y="115" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">(Choromanski)</text>
  <text x="200" y="130" text-anchor="middle" fill="#a78bfa" font-size="8" font-family="system-ui">FAVOR+ 随机特征</text>
  
  <!-- 2023 -->
  <circle cx="320" cy="70" r="5" fill="#6e8eff"/>
  <text x="320" y="55" text-anchor="middle" fill="#6e8eff" font-size="10" font-family="system-ui">2023</text>
  <rect x="275" y="85" width="90" height="55" rx="6" fill="#1e1e2a" stroke="#34d399" stroke-width="1"/>
  <text x="320" y="102" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">RetNet</text>
  <text x="320" y="115" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">(Microsoft)</text>
  <text x="320" y="130" text-anchor="middle" fill="#34d399" font-size="8" font-family="system-ui">+指数衰减遗忘</text>
  
  <!-- 2024 -->
  <circle cx="430" cy="70" r="5" fill="#6e8eff"/>
  <text x="430" y="55" text-anchor="middle" fill="#6e8eff" font-size="10" font-family="system-ui">2024</text>
  <rect x="385" y="85" width="90" height="55" rx="6" fill="#1e1e2a" stroke="#f59e0b" stroke-width="1"/>
  <text x="430" y="102" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">GLA / Hedgehog</text>
  <text x="430" y="115" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">(数据驱动)</text>
  <text x="430" y="130" text-anchor="middle" fill="#f59e0b" font-size="8" font-family="system-ui">+可学习门控/φ</text>
  
  <!-- 2025 -->
  <circle cx="540" cy="70" r="5" fill="#6e8eff"/>
  <text x="540" y="55" text-anchor="middle" fill="#6e8eff" font-size="10" font-family="system-ui">2025</text>
  <rect x="495" y="85" width="90" height="55" rx="6" fill="#1e1e2a" stroke="#ff6e6e" stroke-width="1"/>
  <text x="540" y="102" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">LUNA / 混合架构</text>
  <text x="540" y="115" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">(学习核+混合)</text>
  <text x="540" y="130" text-anchor="middle" fill="#ff6e6e" font-size="8" font-family="system-ui">≈ softmax 精度</text>
  
  <!-- Bottom summary -->
  <line x1="80" y1="170" x2="560" y2="170" stroke="#3a3a4a" stroke-width="1" stroke-dasharray="4,4"/>
  <text x="100" y="195" text-anchor="start" fill="#ededf0" font-size="10" font-family="system-ui">精度:</text>
  <text x="140" y="195" text-anchor="start" fill="#ff6e6e" font-size="10" font-family="system-ui">差</text>
  <line x1="155" y1="192" x2="500" y2="192" stroke="#34d399" stroke-width="1.5" marker-end="url(#arrow4)"/>
  <text x="510" y="195" text-anchor="start" fill="#34d399" font-size="10" font-family="system-ui">接近 Softmax</text>
  
  <text x="100" y="215" text-anchor="start" fill="#ededf0" font-size="10" font-family="system-ui">核心进展:</text>
  <text x="160" y="215" text-anchor="start" fill="#ededf0" font-size="9" font-family="system-ui">固定简单φ → 随机近似 → 加遗忘机制 → 数据驱动学习φ → 匹配softmax</text>
  
  <text x="100" y="240" text-anchor="start" fill="#ededf0" font-size="10" font-family="system-ui">统一视角:</text>
  <text x="160" y="240" text-anchor="start" fill="#ededf0" font-size="9" font-family="system-ui">核函数的选择决定了"压缩质量"——你愿意用多少信息损失换取多少速度提升</text>
</svg>

## 这意味着什么

Linear Attention 的核方法视角告诉我们一件深刻的事：**Attention 的本质是核回归，而不同的核函数选择定义了不同的"世界观"。**

Softmax 核说："每个位置和其他位置的关系是无穷精细的（无限维特征空间），我要精确计算每一对的相似度。"

Linear Attention 核说："我可以用一个压缩表示来概括所有位置的信息，每个查询去查这个概括就好。"

这就像是全息照片（softmax，保留所有细节）vs 印象派画作（linear，保留本质但丢失细节）的区别。2020-2025 年的研究，本质上是在让那幅印象派画作越画越接近全息照片——通过更好的颜料（特征映射）、更巧妙的技法（门控遗忘）、以及直接向照片学习（可训练 φ）。

对于实际应用，这意味着：如果你的任务需要精确的信息检索（"第 38 段的第 2 句说了什么"），softmax 仍然不可替代。但如果你需要的是对长文本的整体理解和趋势把握，Linear Attention 已经足够好——而且快几个数量级。

未来的方向很可能是混合架构：少量 softmax attention 层负责精确检索，大量 linear attention/gated RNN 层负责高效的信息流动。这在数学上对应的就是：在关键位置使用精确的无限维核，在其他位置使用高效的有限维近似。

---

*本文是"LLM 原理深度解析"系列第 19 篇。这个系列试图把 LLM 的每一个核心组件都讲到"真正理解"的程度。*
