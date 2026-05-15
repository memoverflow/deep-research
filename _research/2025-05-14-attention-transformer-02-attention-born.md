---
title: "从「一张便签纸」到「随时翻阅全文」：Attention 的诞生"
date: 2025-05-14
level: 4
series: "理解 Attention 与 Transformer"
series_order: 2
series_total: 8
tags: [attention, seq2seq, Bahdanau, Luong]
summary: "Attention 机制是怎么被发明的？它到底在做什么？用最直觉的方式理解这个改变 AI 历史的想法。"
---

> 2014 年，一个简单的改进让机器翻译的质量跳跃了一大步。这个改进的核心，是让模型学会"哪里该多看两眼"。

## 困境：信息瓶颈

<svg viewBox="0 0 650 130" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:650px;margin:20px auto;display:block;">
  <defs><marker id="a1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10z" fill="#6e8eff"/></marker></defs>
  <text x="325" y="15" text-anchor="middle" fill="#9494a0" font-size="10" font-family="system-ui">Seq2seq 的信息瓶颈</text>
  <rect x="20" y="35" width="80" height="35" rx="6" fill="#1e1e2a" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="60" y="57" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">词₁词₂...词ₙ</text>
  <line x1="100" y1="52" x2="140" y2="52" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#a1)"/>
  <rect x="145" y="30" width="100" height="45" rx="6" fill="#1e1e2a" stroke="#fb7185" stroke-width="2"/>
  <text x="195" y="50" text-anchor="middle" fill="#fb7185" font-size="9" font-family="system-ui">固定向量</text>
  <text x="195" y="65" text-anchor="middle" fill="#6b6b78" font-size="8" font-family="system-ui">所有信息挤在这</text>
  <line x1="245" y1="52" x2="285" y2="52" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#a1)"/>
  <rect x="290" y="35" width="100" height="35" rx="6" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="340" y="57" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">解码 → 翻译</text>
  <text x="325" y="100" text-anchor="middle" fill="#fb7185" font-size="9" font-family="system-ui">⚠️ 句子越长，瓶颈越严重——信息必然丢失</text>
</svg>




上一篇我们用"便签纸"来比喻 Seq2seq 模型的问题。现在让我们更具体地理解它为什么痛。

Seq2seq 模型有两部分：

**编码器**：从左到右，一个词一个词地读输入句子。每读一个词，就更新自己的内部状态（一串数字）。读完最后一个词后，这个最终状态就是那张"便签纸"——整个句子的全部信息都压缩在里面。

**解码器**：接过这张"便签纸"，开始一个词一个词地生成翻译。每生成一个词，也会更新自己的内部状态。

问题出在哪？

出在那个"最终状态"上。不管你的输入句子是 5 个词还是 50 个词，编码器最后交给解码器的都是一个**固定大小**的向量——比如 256 个数字。这是一个物理上的瓶颈：信息被强行挤过一个窄口。

有人可能会问：增大这个向量的维度不就行了？从 256 变成 4096？

理论上可以，但效果有限。因为问题的本质不是"容量不够大"，而是**压缩方式太粗暴**。把 50 个词的所有信息——词义、语法结构、上下文关系——全部塞进一个向量，再优秀的编码器也会丢失细节。而且不同的翻译步骤需要的信息不同——翻译主语时需要输入中的主语信息，翻译动词时需要动词信息——但一个固定向量做不到"按需提供"。

## 关键洞察：让模型自己决定"看哪里"

Bahdanau 的想法（2014 年）是这样的：

既然一个固定向量装不下所有信息，那就**不要压缩成一个向量**。让编码器保留每个位置的输出（每个词读完后的状态），然后解码器在每一步翻译时，可以去"看"编码器所有位置的状态，自己决定重点关注哪些。

打个比方。之前的模型像是一场考试：考试前只能看一遍教材，然后闭卷答题。Attention 之后，变成了开卷考试——答每道题时都可以翻教材，而且会自动翻到最相关的那一页。

## 具体是怎么做到的

<svg viewBox="0 0 600 180" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:600px;margin:20px auto;display:block;">
  <defs><marker id="a2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10z" fill="#6e8eff"/></marker></defs>
  <text x="300" y="15" text-anchor="middle" fill="#9494a0" font-size="10" font-family="system-ui">Attention 计算流程</text>
  <rect x="20" y="30" width="100" height="40" rx="6" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="70" y="50" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">编码器状态</text>
  <text x="70" y="63" text-anchor="middle" fill="#9494a0" font-size="8" font-family="system-ui">h₁, h₂, ... hₙ</text>
  <line x1="120" y1="50" x2="155" y2="50" stroke="#6e8eff" stroke-width="1.2" marker-end="url(#a2)"/>
  <rect x="160" y="30" width="90" height="40" rx="6" fill="#1e1e2a" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="205" y="50" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">打分匹配</text>
  <text x="205" y="63" text-anchor="middle" fill="#9494a0" font-size="8" font-family="system-ui">eᵢⱼ = score(s,h)</text>
  <line x1="250" y1="50" x2="285" y2="50" stroke="#6e8eff" stroke-width="1.2" marker-end="url(#a2)"/>
  <rect x="290" y="30" width="80" height="40" rx="6" fill="#1e1e2a" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="330" y="50" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">Softmax</text>
  <text x="330" y="63" text-anchor="middle" fill="#9494a0" font-size="8" font-family="system-ui">α₁...αₙ (权重)</text>
  <line x1="370" y1="50" x2="405" y2="50" stroke="#6e8eff" stroke-width="1.2" marker-end="url(#a2)"/>
  <rect x="410" y="30" width="100" height="40" rx="6" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="460" y="50" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">加权求和</text>
  <text x="460" y="63" text-anchor="middle" fill="#9494a0" font-size="8" font-family="system-ui">c = Σαⱼhⱼ</text>
  <!-- Bottom annotation -->
  <rect x="100" y="100" width="400" height="50" rx="8" fill="#1a1a24" stroke="#3a3a4a" stroke-width="0.5"/>
  <text x="300" y="120" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">每一步解码都重新打分 → 不同时刻关注不同位置</text>
  <text x="300" y="138" text-anchor="middle" fill="#34d399" font-size="9" font-family="system-ui">翻译"猫"时看 cat 位置，翻译"桌子"时看 table 位置</text>
</svg>




让我们走一遍流程（用直觉而非数学）：

**第 1 步**：编码器正常工作——读完整个句子，但保存每个位置的中间状态。如果输入有 10 个词，就保存 10 个状态向量。

**第 2 步**：解码器开始翻译。在生成每个词之前：
- 解码器有一个"当前状态"，代表"我现在翻译到哪了，接下来要翻什么"
- 把这个状态和编码器保存的每个位置状态做"匹配"——算一个分数，代表"这个位置对我现在有多重要"
- 对所有分数做归一化（softmax），变成权重（加起来等于 1）
- 用这些权重对编码器各位置的状态做加权平均，得到一个"上下文向量"

**第 3 步**：解码器用这个上下文向量（加上自己的状态）来预测下一个词。

**第 4 步**：移动到下一个翻译位置，重复第 2-3 步。

关键在第 2 步的那个"匹配"——它让模型能够在不同时刻关注不同位置。翻译句首时可能关注输入的句首，翻译某个形容词时可能关注输入中对应的形容词。

## "打分"怎么做

上面说到"把解码器状态和编码器状态做匹配算分数"。具体怎么算？

最初的做法（Bahdanau 版）用了一个小型神经网络来打分：把两个状态向量拼在一起，过一个带参数的变换，输出一个数字。这叫 **additive attention**——因为它把两个向量"加"起来（拼接后做线性变换）再处理。

一年后，Luong 提出了一个更简单的做法：直接算两个向量的**内积**（点积）。内积衡量的是两个向量的"方向相似性"——两个向量方向越接近，内积越大。这叫 **multiplicative attention** 或 dot-product attention。

为什么内积能衡量相关性？想象一下：如果编码器把"猫"编码成某个方向，解码器在需要翻译"cat"时，它的状态也会指向相似的方向——于是内积就大，权重就高，模型就"看向"了正确的位置。

这两种打分方法在效果上差别不大，但内积版本计算更快（GPU 擅长做矩阵乘法），所以后来成为了主流。

## 一个意外的礼物：可视化

Attention 带来了一个研究者们没预料到的好处。

之前的神经网络翻译模型是纯黑箱——你给输入，它给输出，中间发生了什么完全不知道。但有了 attention weights，你可以画出一张热力图：横轴是输入词，纵轴是输出词，每个格子的颜色代表"翻译这个输出词时对这个输入词的关注程度"。

这张图通常会呈现出近似对角线的模式（因为很多语言对的词序大致对应），但在语序不同的地方会出现明显的偏移。比如英语的形容词在名词前面，法语可能在后面——attention 图就会在这些地方出现交叉。

这不只是好看。它让研究者第一次能"调试"模型——翻译出错时，可以看看模型是不是关注错了位置。

## 从"看别人"到"看自己"

Attention 最初是用在编码器和解码器之间的——解码器"看"编码器。但很快有人问了一个大胆的问题：

**如果让一个序列中的每个词去关注同一个序列中的所有其他词呢？**

比如"那只猫很胖，因为它吃太多了"。"它"指的是什么？人类一看就知道是"猫"。但如果让机器理解这种指代关系，传统的做法需要 RNN 把信息一步步传递过去——"它"和"猫"之间隔了好几个词，信号会衰减。

但如果用 attention，"它"可以直接和句子中所有词算关注度——和"猫"的关注度最高，模型立刻知道它们有关系。不需要一步步传递，一步到位。

这就是 **self-attention**（自注意力）——2017 年 Transformer 的核心成分。但我们先到这里停一下。

## 这一步有多重要？

回顾一下 attention 带来了什么：

1. **打破了信息瓶颈**：不再把整个句子压进一个固定向量
2. **动态聚焦**：每一步只取需要的信息，按需访问
3. **可解释性**：可以可视化模型的"注意力分布"
4. **直接连接**：任何两个位置之间只需一步 attention 就能交互

前三点在 2014-2015 年就已经改变了 NLP。但第四点——它的全部威力要到 2017 年，当 self-attention 成为架构的唯一核心时，才会完全释放出来。

下一篇，我们会看到这个想法被推到极致后发生了什么——一篇标题狂妄的论文，如何催生了今天所有大模型的基础架构。
