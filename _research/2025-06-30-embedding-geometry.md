---
title: "Embedding 层的几何结构：为什么一张查找表能装下整个世界的意义"
date: 2025-06-30
level: 3
series: "LLM 原理深度解析"
series_order: 20
series_total: 20
tags: [embedding, geometry, high-dimensional, word2vec, transformer, anisotropy, superposition]
summary: "从查找表到高维几何——理解 LLM 最底层的数学结构：为什么 4096 个数字就能编码语言的全部含义"
---

# Embedding 层的几何结构：为什么一张查找表能装下整个世界的意义

> 当你给 LLM 输入一个词，模型做的第一件事是把它变成一串数字。这看似平凡的一步，其实隐藏着令人惊叹的几何结构。

## 故事从一个奇怪的现象开始

2013 年，Google 的研究员 Tomas Mikolov 发现了一个让所有人困惑的事情：如果你把 "king" 这个词对应的向量减去 "man" 的向量，再加上 "woman" 的向量，你会得到一个新向量——它最接近的词是 "queen"。

$$\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}$$

这不是精心设计的结果。没有人告诉模型"国王和王后的关系等同于男人和女人的关系"。模型只是在读大量文本的过程中，自己"发现"了这种结构。

这意味着什么？这意味着**语义关系在向量空间中表现为几何关系**。"性别"不是一个抽象概念，而是空间中一个具体的方向。"王族"是另一个方向。词语的含义被分解成了这些方向的组合。

但这只是故事的开始。今天的 LLM——GPT-4、Claude、Llama——使用的 embedding 层远比 Word2Vec 复杂。它们在 4096 维甚至 12288 维的空间中运作，编码着远超"king-queen"类比的复杂结构。让我们从头理解这个空间的几何学。

## 什么是 Embedding 层？先从最朴素的理解开始

### 问题：计算机不认字

神经网络只能做数学运算——加法、乘法、非线性变换。它不认识"猫"这个字，也不知道"happiness"是什么意思。我们需要一种方式把离散的符号（token）变成连续的数字，让神经网络能处理。

最简单的方案是 one-hot encoding：词汇表有 50000 个 token，就给每个 token 一个 50000 维的向量，只有对应位置是 1，其余全是 0。但这有两个致命问题：

1. **维度灾难**：50000 维的向量太大了，计算成本爆炸
2. **没有关系信息**：在 one-hot 空间里，任何两个不同的词距离都相等——"猫"和"狗"的距离等于"猫"和"经济学"的距离

### 核心想法：用一张查找表做降维

Embedding 层的想法极其简单：维护一张巨大的表格，每一行对应一个 token，每一行有 $d$ 个数字（比如 4096 个）。当输入 token ID 为 $i$ 时，直接查表取出第 $i$ 行。

用数学表示：

$$\mathbf{e}_i = \mathbf{W}_E[i, :]$$

其中 $\mathbf{W}_E \in \mathbb{R}^{V \times d}$ 是 embedding 矩阵，$V$ 是词汇表大小，$d$ 是 embedding 维度。

翻译成人话：这就是一个 $V$ 行 $d$ 列的大表格。输入第 $i$ 个 token，取出第 $i$ 行，得到 $d$ 个数字。没有任何复杂的计算，纯粹的查表操作。

但魔法在于：**这张表的内容不是人写的，是训练出来的。** 通过反向传播，梯度会流过 embedding 层，逐渐调整表格里的每一个数字，直到这些向量的几何排列恰好编码了语言的结构。

<svg viewBox="0 0 680 320" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:680px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Token IDs -->
  <rect x="20" y="40" width="120" height="240" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="80" y="25" text-anchor="middle" fill="#6e8eff" font-size="13" font-family="system-ui" font-weight="bold">Token IDs</text>
  <text x="80" y="75" text-anchor="middle" fill="#ededf0" font-size="12" font-family="monospace">0: "the"</text>
  <text x="80" y="105" text-anchor="middle" fill="#ededf0" font-size="12" font-family="monospace">1: "cat"</text>
  <text x="80" y="135" text-anchor="middle" fill="#ededf0" font-size="12" font-family="monospace">2: "sat"</text>
  <text x="80" y="165" text-anchor="middle" fill="#ededf0" font-size="12" font-family="monospace">3: "on"</text>
  <text x="80" y="195" text-anchor="middle" fill="#ededf0" font-size="12" font-family="monospace">4: "king"</text>
  <text x="80" y="225" text-anchor="middle" fill="#ededf0" font-size="12" font-family="monospace">5: "queen"</text>
  <text x="80" y="260" text-anchor="middle" fill="#888" font-size="11" font-family="system-ui">... (V=50K)</text>
  <!-- Arrow -->
  <line x1="145" y1="160" x2="195" y2="160" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>
  <text x="170" y="150" text-anchor="middle" fill="#888" font-size="10" font-family="system-ui">查表</text>
  <!-- Embedding Matrix -->
  <rect x="200" y="40" width="260" height="240" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="330" y="25" text-anchor="middle" fill="#6e8eff" font-size="13" font-family="system-ui" font-weight="bold">Embedding 矩阵 W_E (V × d)</text>
  <text x="330" y="70" text-anchor="middle" fill="#ededf0" font-size="11" font-family="monospace">[0.12, -0.45, 0.78, ..., 0.33]</text>
  <text x="330" y="100" text-anchor="middle" fill="#34d399" font-size="11" font-family="monospace">[0.91, 0.22, -0.15, ..., 0.67]</text>
  <text x="330" y="130" text-anchor="middle" fill="#ededf0" font-size="11" font-family="monospace">[0.34, -0.88, 0.52, ..., -0.21]</text>
  <text x="330" y="160" text-anchor="middle" fill="#ededf0" font-size="11" font-family="monospace">[0.56, 0.11, -0.73, ..., 0.44]</text>
  <text x="330" y="190" text-anchor="middle" fill="#ededf0" font-size="11" font-family="monospace">[0.83, -0.37, 0.61, ..., 0.95]</text>
  <text x="330" y="220" text-anchor="middle" fill="#ededf0" font-size="11" font-family="monospace">[0.79, -0.31, 0.58, ..., 0.89]</text>
  <text x="330" y="260" text-anchor="middle" fill="#888" font-size="11" font-family="system-ui">每行 d=4096 个浮点数</text>
  <!-- Highlight row -->
  <rect x="205" y="87" width="250" height="20" rx="4" fill="#34d399" fill-opacity="0.15" stroke="#34d399" stroke-width="1" stroke-dasharray="4"/>
  <!-- Arrow to output -->
  <line x1="465" y1="100" x2="520" y2="100" stroke="#34d399" stroke-width="1.5" marker-end="url(#arrow1)"/>
  <!-- Output vector -->
  <rect x="525" y="70" width="140" height="55" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="595" y="92" text-anchor="middle" fill="#34d399" font-size="12" font-family="system-ui" font-weight="bold">token "cat" 的</text>
  <text x="595" y="112" text-anchor="middle" fill="#34d399" font-size="12" font-family="system-ui" font-weight="bold">embedding 向量</text>
  <!-- Note -->
  <text x="595" y="155" text-anchor="middle" fill="#888" font-size="10" font-family="system-ui">送入 Transformer</text>
  <text x="595" y="170" text-anchor="middle" fill="#888" font-size="10" font-family="system-ui">各层处理</text>
</svg>

## 高维空间的反直觉：为什么 4096 维就"够了"

### 问题：维度数远少于概念数

一个直觉的疑问：LLaMA 用 4096 维的 embedding 来表示 32000 个 token。但语言中的概念远不止 32000 个——词义、语法角色、情感色彩、主题领域、频率信息……这些概念的数量可能有几十万甚至上百万。4096 维怎么可能装得下？

答案藏在高维几何的一个反直觉特性里。

### 直觉：高维空间比你想象的大得多

想象你在二维平面上画向量。你最多只能找到 2 个互相垂直（正交）的方向。在三维空间里，最多 3 个。一般人会推断：$d$ 维空间里最多 $d$ 个正交方向，所以只能编码 $d$ 个独立概念。

但这个推理有个漏洞：**我们不需要严格正交，只需要"接近正交"就够了。**

这就是 Johnson-Lindenstrauss 引理告诉我们的惊人事实：在 $d$ 维空间中，你可以找到**指数级数量**（远超 $d$）的向量，它们之间的夹角都接近 90 度。具体来说，$O(e^{cd})$ 个向量可以做到两两之间余弦相似度接近 0。

当 $d = 4096$ 时，$e^{cd}$ 是一个天文数字。这意味着 4096 维的空间可以容纳远超 4096 个"几乎独立"的方向——足以编码语言中的所有概念维度。

### 技术细节：为什么"几乎正交"就够用

在 $d$ 维空间中随机取两个向量，它们的余弦相似度的期望为 0，标准差约为 $1/\sqrt{d}$。当 $d = 4096$ 时，标准差约 $0.016$——也就是说，随机向量之间的相似度几乎完美地等于 0。

这给了模型巨大的编码空间。模型不需要把每个概念对应一个精确的维度（这叫"分布式表示"），它可以用向量方向的组合来编码任意多的概念。这就引出了一个重要现象——**叠加（superposition）**。

## 叠加：当概念比维度多

### 问题：模型如何表示比维度数更多的特征？

Anthropic 的 mechanistic interpretability 研究发现了一个关键现象：神经网络会把多个不同的特征"叠加"在同一组维度上。就像你可以把多个无线电台的信号叠加在同一段频谱上（只要它们频率不同），模型也可以把多个概念叠加在同一组维度上（只要这些概念不经常同时出现）。

### 直觉：多人共用一张桌子

想象一个共享办公空间只有 100 张桌子，但有 500 个会员。这行得通吗？完全可以——只要这 500 人不同时出现。周一来的人和周五来的人共用同一张桌子。

Embedding 空间也是这样。"量子物理"和"烹饪食谱"这两个概念很少同时出现在同一个上下文中，所以它们可以被编码到相近的维度上，而不会互相干扰。只有当两个概念需要在同一个上下文中同时被区分时，它们才真正需要正交的表示。

### 技术细节：稀疏性使叠加成为可能

形式化地说，如果我们有 $m$ 个特征，但每个输入只激活其中 $k$ 个（$k \ll m$），那么 $d$ 维空间可以在最小干扰的前提下表示 $m \gg d$ 个特征。干扰（interference）的大小正比于特征的稀疏度——特征越稀疏，叠加引入的噪声越小。

这解释了为什么 LLM 的 embedding 层虽然"只有"4096 维，却能编码丰富得多的语义信息。

## 从 Word2Vec 到 Transformer：Embedding 的进化

### Word2Vec：第一个有意义的几何空间

Word2Vec 的突破性发现是：如果你训练一个模型来预测一个词的上下文（Skip-gram）或从上下文预测一个词（CBOW），学到的词向量会自动具有语义结构。

为什么会这样？2014 年，Levy 和 Goldberg 证明了一个优美的数学结果：Word2Vec 的 Skip-gram 模型隐式地在对**逐点互信息（PMI）矩阵**做矩阵分解。

$$\mathbf{w}_i \cdot \mathbf{c}_j \approx \text{PMI}(i, j) - \log k$$

翻译成人话：两个词的向量点积，近似等于这两个词共同出现的频率相对于随机共现的"超额程度"。经常一起出现的词，向量点积大，方向接近；从不一起出现的词，向量点积为负，方向相反。

这就是为什么语义关系会变成几何关系——因为共现统计本身就编码了语义。"猫"和"喵"经常一起出现，所以它们的向量接近。"国王"和"王宫"经常一起出现，所以它们也接近。

### Transformer Embedding：起点而非终点

但 Transformer 中的 embedding 层跟 Word2Vec 有本质不同。Word2Vec 的词向量是静态的——"bank"永远是同一个向量，不管它表示"银行"还是"河岸"。

Transformer 的 embedding 层只是**起跑线**。它提供的是一个初始表示，然后经过几十层 attention 和 FFN 的反复加工，同一个 token 在不同上下文中会演化出完全不同的表示。

形象地说：embedding 层给每个词一张"基因蓝图"，Transformer 的各层则是"成长环境"——最终的表示是先天加后天的结合。

<svg viewBox="0 0 700 240" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Static embedding -->
  <rect x="20" y="50" width="140" height="70" rx="8" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="90" y="78" text-anchor="middle" fill="#a78bfa" font-size="12" font-family="system-ui" font-weight="bold">静态 Embedding</text>
  <text x="90" y="100" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">"bank" → 同一向量</text>
  <!-- Arrow -->
  <line x1="165" y1="85" x2="200" y2="85" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <!-- Layer 1 -->
  <rect x="205" y="50" width="100" height="70" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="255" y="78" text-anchor="middle" fill="#6e8eff" font-size="11" font-family="system-ui">Layer 1</text>
  <text x="255" y="96" text-anchor="middle" fill="#888" font-size="10" font-family="system-ui">+上下文信息</text>
  <!-- Arrow -->
  <line x1="310" y1="85" x2="340" y2="85" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <!-- Layer N -->
  <rect x="345" y="50" width="100" height="70" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="395" y="78" text-anchor="middle" fill="#6e8eff" font-size="11" font-family="system-ui">Layer N</text>
  <text x="395" y="96" text-anchor="middle" fill="#888" font-size="10" font-family="system-ui">深度加工</text>
  <!-- Arrow -->
  <line x1="450" y1="85" x2="485" y2="85" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <!-- Contextual output -->
  <rect x="490" y="30" width="190" height="110" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="585" y="55" text-anchor="middle" fill="#34d399" font-size="12" font-family="system-ui" font-weight="bold">上下文化表示</text>
  <text x="585" y="80" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">"river bank" → 向量 A</text>
  <text x="585" y="100" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">"bank account" → 向量 B</text>
  <text x="585" y="125" text-anchor="middle" fill="#888" font-size="10" font-family="system-ui">A 和 B 方向完全不同！</text>
  <!-- Timeline label -->
  <text x="350" y="180" text-anchor="middle" fill="#888" font-size="12" font-family="system-ui">同一个 token，经过 Transformer 各层后变成完全不同的向量</text>
  <!-- Word2Vec comparison -->
  <rect x="20" y="160" width="140" height="55" rx="8" fill="#1e1e2a" stroke="#888" stroke-width="1" stroke-dasharray="4"/>
  <text x="90" y="182" text-anchor="middle" fill="#888" font-size="11" font-family="system-ui">Word2Vec: 到这里结束</text>
  <text x="90" y="200" text-anchor="middle" fill="#888" font-size="10" font-family="system-ui">一个词永远是同一向量</text>
</svg>

## Weight Tying：入口即出口的对称美学

### 问题：模型的入口和出口用同一张表？

大多数现代 LLM（GPT-2、LLaMA、Gemma 等）使用一个叫 **weight tying**（权重绑定）的技巧：embedding 层（把 token 变成向量）和 unembedding 层（把向量变回 token 概率）共享同一个矩阵。

$$P(\text{next token} = j) = \text{softmax}(\mathbf{h} \cdot \mathbf{W}_E^T)_j$$

翻译成人话：预测下一个词时，模型把最后一层的隐藏状态 $\mathbf{h}$ 和 embedding 矩阵的每一行做点积。点积越大，说明隐藏状态和那个词的 embedding 越"对齐"，那个词就越可能是答案。

### 直觉：地图的入口和出口

想象 embedding 空间是一张城市地图。输入时，每个 token 被放到地图上对应的位置（embedding）。经过 Transformer 各层的变换——相当于在地图上移动、旋转、缩放——最终到达一个新位置。输出时，模型看这个最终位置最靠近哪个 token 的"初始位置"，就预测那个 token。

Weight tying 意味着入口和出口使用同一张地图。这带来两个后果：
1. **省参数**：embedding 矩阵通常很大（50000 × 4096 ≈ 2 亿参数），共享可以减半
2. **几何约束**：输入和输出必须在同一个空间里"兼容"——这给 embedding 的几何结构施加了额外的约束

### 这对几何的影响

2025 年的研究（Traylor et al., "Weight Tying Biases Token Embeddings Towards the Output Space"）发现，weight tying 会让 embedding 更多地受到输出任务的塑造——因为输出层的梯度信号通常比输入层强得多。这意味着 embedding 的几何更多反映的是"哪些词经常互相预测"，而非"哪些词在输入中功能相似"。

## Logit Lens：透过 Embedding 看中间层

### 如果 embedding 是地图，我们能用它看见模型在"想"什么吗？

一个名为 **Logit Lens**（2020, nostalgebraist）的技术利用了 weight tying 的性质：既然最后一层通过与 embedding 做点积来预测 token，我们也可以对中间层的隐藏状态做同样的操作——看看模型在中途"倾向于"预测哪个词。

结果令人着迷：
- **底层**（靠近 embedding）：预测几乎是随机的，或者只捕捉了表面统计
- **中间层**：逐渐出现有意义的预测，模型开始"理解"上下文
- **顶层**：预测锐利而准确

这告诉我们：模型的处理像一个渐进的"解密"过程——从 embedding 空间的初始编码开始，逐层细化，最终回到 embedding 空间给出答案。整个 Transformer 的计算可以理解为在 embedding 空间中的一场"旅行"。

## 各向异性：Embedding 空间的"窄锥"问题

### 问题：空间用得太偏了

理想情况下，embedding 向量应该均匀分布在整个空间中——这样每个方向都能被充分利用来编码信息。但 2019 年，Ethayarajh 发现了一个令人不安的事实：BERT、GPT-2 等模型的 embedding 高度**各向异性（anisotropic）**——所有向量都挤在一个狭窄的锥形区域里。

用日常语言说：想象你有一个球形房间，但所有家具都塞在一个角落里。你浪费了 90% 的可用空间。

### 为什么会这样？

原因与 weight tying 和训练动态有关。Gao et al. (2019) 证明了：当使用 weight tying 时，通过最大似然训练，embedding 会自然地退化到一个窄锥中。这是因为：

1. 高频词（"the", "is", "of"）在训练中出现极其频繁
2. 模型需要把大量梯度信号"推"向这些高频词
3. 结果是所有向量都被拉向高频词占据的方向

### 这有什么后果？

各向异性意味着**余弦相似度变得不可靠**。当所有向量都指向差不多的方向时，任何两个词的余弦相似度都很高——"猫"和"经济学"的相似度可能达到 0.7，仅仅因为它们都在同一个窄锥里。

这也解释了为什么直接用 LLM 的 token embedding 做语义搜索效果不好——你需要专门训练的 sentence embedding 模型（如 BGE、E5），这些模型通过对比学习被迫把向量推向空间的不同区域。

### 修复方案

研究者提出了多种应对策略：
- **All-but-the-Top**（Mu & Viswanath, 2018）：去掉 embedding 矩阵的前几个主成分（最强的各向异性方向），恢复各向同性
- **对比学习**：训练时强制相似样本靠近、不相似样本远离，天然推动各向同性
- **白化（whitening）**：用 PCA 白化使向量分布更均匀

## Embedding 维度的选择：一个工程与理论的平衡

### 为什么是 4096？为什么不是 400 或 40000？

不同模型选择了不同的 embedding 维度：

| 模型 | 参数量 | Embedding 维度 |
|------|--------|--------------|
| GPT-2 Small | 117M | 768 |
| GPT-2 XL | 1.5B | 1600 |
| LLaMA-7B | 7B | 4096 |
| LLaMA-70B | 70B | 8192 |
| GPT-3 | 175B | 12288 |

规律很明显：模型越大，embedding 维度越高。但这不是线性增长——参数量增加 100 倍，维度只增加约 3-4 倍。这背后的逻辑是：

1. **表达力**：更高的维度意味着更多"几乎正交"的方向可用，能编码更细粒度的语义区分
2. **计算量**：embedding 维度决定了整个模型的宽度。每一层的计算量都正比于 $d^2$，维度翻倍意味着计算量翻 4 倍
3. **数据需求**：更大的空间需要更多数据来"填满"。如果数据不够，高维空间中大部分区域就是空的（curse of dimensionality 的反面）

Scaling Laws 的研究表明：最优的分配是让模型深度和宽度协调增长。经验法则是 $d \approx 128 \times L$，其中 $L$ 是层数——但这只是粗略近似。

## 余弦相似度 vs 欧几里得距离：选哪个？

### 问题：怎么度量两个 embedding 的"接近程度"？

当我们说"两个词的 embedding 接近"时，到底用什么度量？有两个主流选择：

**余弦相似度**：只看方向，忽略长度
$$\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{a}||\mathbf{b}|}$$

**欧几里得距离**：看绝对位置
$$d(\mathbf{a}, \mathbf{b}) = |\mathbf{a} - \mathbf{b}|$$

### 直觉：方向 vs 位置

余弦相似度就像在看两个箭头指向哪里——不管箭头长短，只看角度。欧几里得距离则像在看两个点隔多远。

在 embedding 空间中，**方向编码语义，长度编码频率**。高频词的 embedding 通常范数（长度）更大，因为训练时它们接受了更多的梯度更新。如果你关心的是语义相似性（"猫"和"狗"是否含义接近），用余弦相似度更合理——你不希望一个常见词仅因为范数大就跟所有词"接近"。

这就是为什么几乎所有语义搜索系统默认使用余弦相似度。

## 这一切意味着什么：Embedding 是模型的世界观

回顾一下我们了解到的：

1. **Embedding 是一张学出来的查找表**，把离散 token 映射到连续向量空间
2. **高维空间的几何特性**允许用远少于概念数的维度来编码丰富的语义
3. **叠加现象**让模型把不经常共现的特征编码到相同的维度上
4. **Weight tying** 让输入和输出共享同一个空间，但也引入了各向异性偏差
5. **各向异性**是个问题，但可以通过对比学习等方法缓解
6. **Logit Lens** 展示了 Transformer 的计算本质上是在 embedding 空间中的渐进变换

Embedding 层看似是模型最简单的部分——一次查表操作。但它定义了模型思考的"坐标系"：什么概念被认为相似，什么方向编码什么信息，模型的整个认知结构都建筑在这个几何基础之上。

当我们训练一个 LLM 时，我们本质上是在教它构建一个高维空间，让语言的结构在这个空间中变得"可计算"。每一个 embedding 向量都是这个模型对一个 token 的全部"初始印象"——凝缩了频率、语法角色、语义领域、共现模式等等信息，等待被 Transformer 的后续层展开、组合、提纯。

这张 4096 维的地图，就是 LLM 用来理解世界的第一把钥匙。

## 下一篇预告

我们讲了 embedding 空间的几何结构——词语如何被安置在高维空间中。但还有一个问题没有回答：当多个 token 同时输入时，模型怎么知道"第三个位置的猫"和"第七个位置的猫"是不同的？这就是位置编码要解决的问题——而 RoPE 用旋转给出了一个优雅的答案。如果你对此感兴趣，可以回看本系列关于 RoPE 的文章。
