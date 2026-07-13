---
title: "Self-Attention 的几何直觉：Q、K、V 到底在向量空间里做什么"
date: 2026-07-06
level: 3
series: "LLM 原理深度解析"
series_order: 27
series_total: 39
tags: [self-attention, QKV, 几何直觉, transformer, 线性代数]
summary: "Query、Key、Value 三个矩阵不是三个随意的名字——它们把一次"信息检索"拆成了三个独立的几何操作。这篇文章带你看清这三步到底在向量空间里做了什么。"
---

> 每个人第一次学 Self-Attention 时都会背下那个公式：$\text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k})V$。但很少有人真正停下来问：**为什么是三个矩阵，不是一个？为什么要做点积？为什么要除以 $\sqrt{d_k}$？** 这篇文章不讲"是什么"，讲"为什么会长这样"。

## 故事从这里开始

假设你在图书馆找一本关于"深度学习优化器"的书。你脑子里有个大概的问题——"我想找讲 Adam 和 SGD 区别的书"——这是你的**查询**。图书馆里每本书的书脊上贴着分类标签，比如"机器学习/优化理论"——这是这本书的**索引标签**。你拿着自己的问题去和每本书的标签做比对，找到匹配度最高的几本，然后翻开这些书,读里面**真正的内容**。

这个过程里，"你想找什么"和"书被贴上什么标签"和"书里到底写了什么"，是三件完全不同的事。一本书的标签可以写得很精确，但内容可能很啰嗕；你的问题可以问得很模糊，但你真正想要的答案可能很具体。如果图书馆强制要求"标签就是内容的完整摘要"，检索效率会大打折扣——因为一个东西要同时干两件不相关的事：既要方便被找到，又要承载全部信息。

Self-Attention 里的 Query、Key、Value，做的正是把"我想找什么"、"我能被怎样找到"、"我实际提供什么内容"这三件事彻底拆开。这不是工程师随手起的名字，而是一次刻意的角色分离——而这次分离，恰好可以用几何的语言讲得很清楚。

## 问题的起点：一个向量不能同时扮演三个角色

在 Self-Attention 出现之前，最朴素的想法是：给每个词一个向量 $x_i$，想知道词 $i$ 和词 $j$ 有多相关，直接算它们的点积或余弦相似度 $x_i \cdot x_j$ 不就行了？

这个想法有一个隐藏的、非常致命的假设：**"$i$ 和 $j$ 相关"是一个对称的、单一的量**。但语言里的关系几乎从来不是对称的。

考虑这句话："The animal didn't cross the street because **it** was too tired."

"it" 需要去关联"animal"——它在问"谁是那个主语？"这是"it"发出的一个**主动的检索请求**。而"animal"要能够被"it"关联到，靠的是"animal"本身作为一个名词、作为句子主语所携带的**可被检索的特征**。这是两件不同的事——一个是"发问的角度"，一个是"被问到时展现的特征"。如果你用同一个向量 $x_{it}$ 和 $x_{animal}$ 直接点积，你其实假设了"it 提问的方式"和"animal 回答的方式"用的是同一套坐标系，这在数学上没有任何理由成立。

更麻烦的是第三件事：一旦"it"确定要看"animal"，它到底要**拿到什么信息**？是要"animal"的词性？还是它的语义类别？还是它在句子里的位置？这又是第三个独立的问题，和"要不要关注它"完全不是一回事。

于是我们得到了三个独立的需求，对应三个独立的向量：

- **Query（查询）**：我（当前词）现在带着什么问题去看别人？
- **Key（键）**：别人（其他词）用什么样的"特征标签"来回应被检索？
- **Value（值）**：一旦确定要看它，它实际提供什么信息？

<svg viewBox="0 0 700 260" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;background:transparent;">
  <defs>
    <marker id="arrow1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <rect x="20" y="30" width="160" height="60" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="100" y="55" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">词向量 x</text>
  <text x="100" y="75" text-anchor="middle" fill="#8888a0" font-size="11" font-family="system-ui">(单一表示)</text>

  <line x1="180" y1="60" x2="240" y2="60" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>

  <rect x="250" y="10" width="150" height="50" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="325" y="40" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">Query: 我想问什么</text>

  <rect x="250" y="105" width="150" height="50" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="325" y="135" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">Key: 我如何被检索</text>

  <rect x="250" y="200" width="150" height="50" rx="8" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="325" y="230" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">Value: 我实际提供什么</text>

  <line x1="180" y1="60" x2="250" y2="35" stroke="#6e8eff" stroke-width="1.2" marker-end="url(#arrow1)"/>
  <line x1="180" y1="60" x2="250" y2="130" stroke="#34d399" stroke-width="1.2" marker-end="url(#arrow1)"/>
  <line x1="180" y1="60" x2="250" y2="225" stroke="#a78bfa" stroke-width="1.2" marker-end="url(#arrow1)"/>

  <text x="420" y="40" fill="#8888a0" font-size="12" font-family="system-ui">W_Q · x → 检索请求向量</text>
  <text x="420" y="135" fill="#8888a0" font-size="12" font-family="system-ui">W_K · x → 索引标签向量</text>
  <text x="420" y="230" fill="#8888a0" font-size="12" font-family="system-ui">W_V · x → 实际内容向量</text>
</svg>

三个可学习的投影矩阵 $W_Q, W_K, W_V$ 把同一个词向量 $x_i$ 分别打到三个不同的子空间：$q_i = W_Q x_i$，$k_i = W_K x_i$，$v_i = W_V x_i$。这三个矩阵在训练开始时是随机初始化的，训练过程会逼着它们学出"提问的坐标系"、"应答的坐标系"和"内容的坐标系"分别应该长什么样。

这就是为什么答案是"三个矩阵"而不是"一个"：**因为提问、被问到、和回答内容本质上是三个不同的几何操作,把它们塞进同一个向量空间是在浪费表达能力。**

## 核心直觉：注意力权重是什么样的几何操作

好，现在我们有了 $q_i, k_j, v_j$。下一步是算 $q_i \cdot k_j$，这个点积到底在几何上意味着什么？

点积有一个非常干净的几何解释：$q_i \cdot k_j = |q_i||k_j|\cos\theta$，其中 $\theta$ 是两个向量之间的夹角。也就是说，点积同时编码了**两个向量指向的方向有多接近**（$\cos\theta$）和**两个向量本身的"强度"**（模长）。

想象每个 Query 向量在这个 $d_k$ 维空间里"指向"一个方向——这个方向代表"我现在想找哪一类信息"。每个 Key 向量也指向一个方向——代表"我提供的是哪一类信息"。当 $q_i$ 和某个 $k_j$ 指向的方向几乎重合时，点积会很大，意味着"这个词正好提供了我要找的东西"。

这里有一个经常被忽略但很重要的直觉：**注意力不是在比较"内容有多像"，而是在比较"提问的方向和应答的方向有多对齐"**。这也是为什么 Query 和 Key 必须用不同的投影矩阵——如果用同一个矩阵（即 $Q=K$），你其实是在问"这个词和自己（以及和其他词）的自相似度是多少"，这会有一个系统性的偏差：每个词和自己的相似度天生就是最大的（对角线上全是自己跟自己的点积，永远是最大值），模型会陷入"总是最关注自己"的坍缩模式，很难去学习真正有意义的跨词关系。用两个不同的矩阵把"提问的方向"和"应答的方向"错开，才能让模型自由地学出"什么样的问题应该被什么样的信息回答"，而不是被"自己和自己最像"这个数学恒等式绑死。

那 Value 呢？一旦注意力权重（也就是 $q_i \cdot k_j$ 经过 softmax 归一化之后的系数）确定了，输出就是所有 Value 向量的加权平均：$\text{output}_i = \sum_j \alpha_{ij} v_j$。这里 Value 向量活在**另一个**子空间里，跟 Query/Key 所在的相似度计算空间完全独立。这意味着"用什么标准去检索"（QK 空间）和"检索到之后拿到什么"（V 空间）是两套不同的坐标系统，模型可以自由地学习"哪些特征用来匹配、哪些特征用来传递信息"，而不必让这两件事共享同一个表示。

Anthropic 的 "A Mathematical Framework for Transformer Circuits" 这篇文章把这个结构总结成两个独立的"电路"：**QK circuit**（决定关注哪里）和 **OV circuit**（决定关注之后传递什么信息）。它们指出这两个电路可以完全独立地分析——你可以先"冻结"某个注意力头的注意力模式（只看它决定关注谁），再单独看它把信息搬运到哪里去。这种解耦正是三个矩阵设计带来的直接好处：检索逻辑和内容传递逻辑，可以被拆开来理解、拆开来调试。

<svg viewBox="0 0 700 220" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;background:transparent;">
  <defs>
    <marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="350" y="20" text-anchor="middle" fill="#8888a0" font-size="12" font-family="system-ui">QK 空间：决定关注谁（方向匹配）</text>
  <circle cx="130" cy="90" r="6" fill="#6e8eff"/>
  <text x="145" y="80" fill="#ededf0" font-size="11" font-family="system-ui">q_i</text>
  <line x1="130" y1="90" x2="80" y2="60" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>

  <circle cx="200" cy="130" r="5" fill="#34d399"/>
  <text x="210" y="140" fill="#ededf0" font-size="11" font-family="system-ui">k_j (方向接近 q_i)</text>
  <line x1="200" y1="130" x2="180" y2="70" stroke="#34d399" stroke-width="1.2" marker-end="url(#arrow2)" stroke-dasharray="2,2"/>

  <circle cx="80" cy="150" r="5" fill="#94a3b8"/>
  <text x="90" y="165" fill="#8888a0" font-size="11" font-family="system-ui">k_m (方向远离)</text>

  <rect x="20" y="185" width="260" height="1" fill="#3a3a4a"/>
  <text x="150" y="205" text-anchor="middle" fill="#8888a0" font-size="11" font-family="system-ui">点积大 → softmax 权重大</text>

  <line x1="330" y1="110" x2="400" y2="110" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <text x="365" y="100" text-anchor="middle" fill="#8888a0" font-size="11" font-family="system-ui">加权</text>

  <text x="470" y="20" text-anchor="middle" fill="#8888a0" font-size="12" font-family="system-ui">V 空间：决定传递什么（内容）</text>
  <rect x="420" y="55" width="60" height="35" rx="6" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="450" y="77" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">v_j</text>
  <rect x="500" y="55" width="60" height="35" rx="6" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1" opacity="0.4"/>
  <text x="530" y="77" text-anchor="middle" fill="#8888a0" font-size="10" font-family="system-ui">v_m</text>
  <line x1="450" y1="95" x2="450" y2="150" stroke="#a78bfa" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <rect x="410" y="155" width="120" height="40" rx="6" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="470" y="180" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">加权平均输出</text>
</svg>

## 技术细节：为什么要除以 $\sqrt{d_k}$，以及"低秩瓶颈"是怎么回事

（这一段给想深入的人看，跳过也不影响理解上面的核心直觉。）

**先说 $\sqrt{d_k}$ 的事。** 假设 $q$ 和 $k$ 的每个分量都是独立同分布、均值为 0、方差为 1 的随机变量，两者的点积 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 是 $d_k$ 个独立随机项的和。根据方差的加法性质，这个和的方差大约是 $d_k$（每一项的方差是 1，$d_k$ 项加起来方差就是 $d_k$）。也就是说，**维度越高，点积的数值天然就越大、波动也越大**——这跟"这两个向量到底有多相关"没有任何关系，纯粹是维度堆出来的数值膨胀。

翻译回人话：如果你把两个 512 维的随机向量做点积，得到的数字天然就会比两个 8 维随机向量的点积大得多——即使它们"真实的相关程度"完全一样。而 softmax 函数对输入的绝对大小非常敏感：一旦输入的差距被放大到几十甚至上百，softmax 会迅速把注意力权重推向 0 和 1 的两个极端（这就是"饱和"），几乎所有概率质量都集中在一个最大值上，其他位置的梯度会被压缩到几乎消失。模型会陷入"永远只看一个词"的死板模式，而且反向传播时几乎学不到东西。

除以 $\sqrt{d_k}$ 正好把方差从 $d_k$ 拉回到 1，不管维度多高，点积的数值分布都保持在一个 softmax 能正常工作的区间——这不是一个经验性的技巧，而是直接从"独立随机变量方差累加"这个统计学事实推出来的修正。

**再说低秩瓶颈。** 这是一个更微妙的问题：每个注意力头把 $d_{model}$ 维的向量压缩到 $d_k = d_{model}/h$（$h$ 是头数）维再做点积。2020 年的一篇论文指出，注意力权重矩阵 $A = QK^T$ 的秩不可能超过 $d_k$——这是线性代数的基本事实（两个 $n \times d_k$ 矩阵相乘得到的 $n \times n$ 矩阵，秩至多是 $d_k$）。但当序列长度 $n$ 远大于 $d_k$ 时（比如序列有 4096 个 token，而每个头只有 64 维），你想表达的注意力模式可能天生就需要更高的秩才能精确刻画——比如"每个 token 只关注序列中距离恰好等于其位置奇偶性的另一个 token"这类复杂模式，可能需要接近满秩（也就是接近 $n$）的表达能力，而 64 维的头根本装不下。

这篇论文（arXiv 2002.07028）的解法是把头的维度设置为跟序列长度绑定而不是跟头数绑定,用更少但更"胖"的头去换取表达力。这个发现直接呼应了后来 Multi-Head Attention 为什么要精心权衡"头数 vs 每个头的维度"——这不是一个自由调的超参数，而是在跟一个真实存在的秩瓶颈做权衡。

## 多头是怎么从这个几何图景里长出来的

理解了单头 Attention 的几何操作之后,一个自然的问题冒出来："既然一个头已经能做检索了,为什么还要很多个头?"

回到图书馆的类比：如果图书馆只有一种分类标签——比如只按"出版年份"分类——你能找到的书就非常有限。真正好用的图书馆会同时维护好几套独立的索引系统：按主题分、按作者分、按语言分、按难度分。你查书的时候,可能同时用上"我想找中文写的、入门级的、关于优化算法的书"——这是四个维度同时在起作用,而每个维度用的是完全不同的标准。

单个注意力头，只用一套 $(W_Q, W_K, W_V)$，相当于只用一套标准去检索——它能学会"关注前一个词"或者"关注句法上的主语"，但它很难同时学会"关注前一个词"**和**"关注句法主语"**和**"关注语义上相关的名词",因为这些不同种类的相关性可能需要指向完全不同方向的 Query/Key 子空间,硬塞进同一个子空间会互相干扰。

2019 年 Voita 等人的论文对训练好的翻译模型做了系统性分析,发现不同注意力头确实学出了清晰可辨认、可解读的不同角色——有的头专门捕捉句法依存关系(比如主谓关系),有的头专门跟踪相邻位置。用一种基于随机门控和 L0 正则化的方法做头剪枝时,他们发现"专业化"程度最高的头是最后被剪掉的——换句话说,模型确实在依赖这些专业化的头做真正有意义的工作,而不是简单的冗余堆叠。在一个英俄翻译任务上,剪掉 48 个编码器头里的 38 个,BLEU 分数只下降了 0.15——这说明大量的头其实在做重复或者不重要的工作,但那一小部分保留下来的头是不可替代的。

更近的一篇理论工作(arXiv 2509.22840, 2025)从信息论的角度给出了一个更精确的解释：把"多头"看成一个通信问题——你有一个固定的总维度预算 $D_K = h \times d_k$，要用它去编码尽可能多的"token-token 关系"。他们证明,把同样的总维度预算拆成更多、更小的头,能够显著降低"嵌入叠加"(embedding superposition)带来的干扰,从而提升系统能编码的关系数量的上限。翻译回直觉：如果你把所有维度都塞进一个头,不同的语义关系会互相"挤"在同一个高维空间里,产生噪声式的干扰;而拆成多个独立的头,相当于给每种关系分配了一块互不干扰的私有空间,信息容量会显著提升。这跟低秩瓶颈的结论看起来矛盾(一个说头要更大,一个说头要更多更小),但其实是同一个权衡的两个侧面：头数与头维度之间,存在一个真实的容量-干扰权衡,而不是"越大越好"或者"越多越好"这么简单。

## 这意味着什么

回到最开始的问题：为什么是三个矩阵，为什么要做点积，为什么要除以 $\sqrt{d_k}$？现在可以给一个完整的答案：

**Query、Key、Value 的拆分，是把"提问的方向"、"应答的方向"、"实际传递的信息"这三件几何上独立的事情，投影到三个独立的子空间去学习**——这避免了单一向量必须同时扮演三个不兼容角色的死结。**点积衡量的是 Query 方向和 Key 方向的对齐程度**，这个对齐程度经过 softmax 归一化后变成注意力权重，本质上是在做一次"基于方向匹配的加权检索"，跟 Hopfield 网络里"用能量函数做联想记忆检索"是同一套数学(Ramsauer 等人 2020 年的论文正式证明了这个等价性)。**除以 $\sqrt{d_k}$ 是纯粹的统计学修正**，把维度堆积带来的数值膨胀校正回 softmax 能正常工作的区间。**多头设计是在一个真实存在的容量约束下做的权衡**——单头受限于低秩瓶颈,无法同时表达多种独立的相关性模式;拆成多个头,相当于把总的表达预算分配到多个互不干扰的子空间,让每个头专注学习一种关系。

理解了这一层几何图景,你会发现后面很多"看起来是工程 trick"的设计,其实都是在这套几何框架里做的自然延伸——GQA/MQA 在问"Key/Value 的子空间是不是可以在多个头之间共享而不损失太多容量","低秩瓶颈"这篇论文的思路后来也启发了 MLA 用低秩投影压缩 KV Cache。这些设计选择,一旦你理解了 QKV 本身在几何上到底做了什么,就不再是孤立的技巧,而是同一套逻辑在不同约束下的重新排列。

## 参考来源

- Vaswani et al., "Attention Is All You Need" (2017), arXiv:1706.03762
- Ramsauer et al., "Hopfield Networks is All You Need" (2020), arXiv:2008.02217 — 证明 Attention 更新规则与连续态 Hopfield 网络检索规则的数学等价性
- Bhojanapalli et al., "Low-Rank Bottleneck in Multi-head Attention Models" (2020), arXiv:2002.07028
- Voita et al., "Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned" (2019), arXiv:1905.09418
- "A Capacity-Based Rationale for Multi-Head Attention" (2025), arXiv:2509.22840
- Anthropic, "A Mathematical Framework for Transformer Circuits" (2021), transformer-circuits.pub
- D2L.ai, "Queries, Keys, and Values" / "Multi-Head Attention" 章节
