---
title: "Muon 优化器：当我们把梯度矩阵「拉直」成一个旋转"
date: 2026-07-14
level: 3
series: "LLM 原理深度解析"
series_order: 37
series_total: 37
tags: [优化器, Muon, AdamW, Newton-Schulz, 正交化, Kimi K2, 训练]
summary: "为什么 Kimi K2 这样的万亿参数模型会抛弃统治了十年的 Adam，换成一个连大多数工程师都没听过的优化器？Muon 的答案藏在一个简单的问题里：你的梯度矩阵,是不是被少数几个方向绑架了？"
---

> Kimi K2、Moonlight,这些 2025 年之后横空出世的大模型,都悄悄换了一个引擎。不是换架构,不是换数据,是换了**优化器**——训练时用来把梯度变成参数更新的那套算法。而这个新玩家 Muon,声称能用一半的计算量达到 AdamW 的效果。这篇文章讲清楚它到底做了什么,以及为什么这么简单的一个想法,能撬动一个统治了十年的默认选项。

## 一个「偏心」的更新

先讲个反直觉的观察。

训练神经网络时,每一层的权重通常是一个矩阵——比如一个 4096×4096 的矩阵,负责把上一层的输出线性变换到下一层。每一步训练,反向传播会给这个矩阵算出一个梯度,也是一个同样形状的矩阵。Adam(以及它的变体 AdamW)拿到这个梯度矩阵之后,做的事情其实很朴素:把矩阵拉平成一长条向量,对每一个坐标独立地维护一阶矩和二阶矩估计,然后逐坐标地做归一化更新。

这套逐坐标的处理方式,在过去十年里几乎是全场景最优选择。但 2024 年底,有人做了一件很简单的事:观察 Adam(或者更朴素的带动量 SGD)在训练 Transformer 时,产生的权重更新矩阵到底长什么样。

结果发现,这些更新矩阵普遍有一个共同的病:**条件数极高**——用矩阵的语言说,就是它的奇异值分布极不均匀。少数几个方向的奇异值巨大,剩下几百上千个方向的奇异值小到几乎可以忽略。

翻译成人话:每次更新,这个矩阵其实只在"一两个主要方向"上做出了有意义的调整,剩下的绝大多数方向几乎没有动。这就好比一个团队开会,理论上所有 100 个成员都该表达意见,但实际上话筒永远在同一两个人手里传,其余 98 个人一句话都插不上。

这就是这篇文章的主角——Muon 优化器——想要解决的问题:**如果梯度更新长期偏心,那些被压制的"少数方向"会不会正好是学习中重要但被忽略的信号?**

<svg viewBox="0 0 620 220" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:620px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="310" y="24" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">同一个更新矩阵的两种奇异值分布</text>

  <!-- AdamW style: skewed -->
  <rect x="30" y="50" width="260" height="140" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="160" y="72" text-anchor="middle" fill="#ededf0" font-size="12">未处理的更新（高条件数）</text>
  <line x1="55" y1="170" x2="55" y2="90" stroke="#6e8eff" stroke-width="1" marker-end="url(#arrow1)"/>
  <line x1="55" y1="170" x2="270" y2="170" stroke="#6e8eff" stroke-width="1" marker-end="url(#arrow1)"/>
  <rect x="65" y="95" width="14" height="75" fill="#6e8eff"/>
  <rect x="90" y="160" width="14" height="10" fill="#6e8eff"/>
  <rect x="115" y="165" width="14" height="5" fill="#6e8eff"/>
  <rect x="140" y="167" width="14" height="3" fill="#6e8eff"/>
  <rect x="165" y="168" width="14" height="2" fill="#6e8eff"/>
  <rect x="190" y="168" width="14" height="2" fill="#6e8eff"/>
  <rect x="215" y="169" width="14" height="1" fill="#6e8eff"/>
  <text x="160" y="200" text-anchor="middle" fill="#ededf0" font-size="11">一个方向独占，其余接近于零</text>

  <!-- Muon style: even -->
  <rect x="330" y="50" width="260" height="140" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="460" y="72" text-anchor="middle" fill="#ededf0" font-size="12">Muon 正交化后（奇异值≈1）</text>
  <line x1="355" y1="170" x2="355" y2="90" stroke="#6e8eff" stroke-width="1" marker-end="url(#arrow1)"/>
  <line x1="355" y1="170" x2="570" y2="170" stroke="#6e8eff" stroke-width="1" marker-end="url(#arrow1)"/>
  <rect x="365" y="100" width="14" height="70" fill="#6e8eff"/>
  <rect x="390" y="100" width="14" height="70" fill="#6e8eff"/>
  <rect x="415" y="100" width="14" height="70" fill="#6e8eff"/>
  <rect x="440" y="100" width="14" height="70" fill="#6e8eff"/>
  <rect x="465" y="100" width="14" height="70" fill="#6e8eff"/>
  <rect x="490" y="100" width="14" height="70" fill="#6e8eff"/>
  <rect x="515" y="100" width="14" height="70" fill="#6e8eff"/>
  <text x="460" y="200" text-anchor="middle" fill="#ededf0" font-size="11">所有方向被拉到同一高度</text>
</svg>

## 核心直觉:把矩阵"捋平"

那怎么解决"偏心"问题?思路其实很直白——既然问题是奇异值分布不均,那就人为地把它们**拉到同一个高度**。

想象你有一个梯度更新矩阵 $G$,可以用奇异值分解写成 $G = U S V^\top$,其中 $S$ 是那些偏心的奇异值组成的对角矩阵。如果我们能把 $S$ 里所有的奇异值都换成 1(同时保留 $U$、$V$ 里编码的"方向信息"),就得到了一个新的矩阵 $UV^\top$。这个操作叫**正交化**(orthogonalization),数学上等价于:在所有满足"行正交或列正交"的矩阵里,找出离原矩阵 $G$ 最近的那一个。

这就是 Muon 的核心操作。它的全名叫 **MomentUm Orthogonalized by Newton-Schulz**——先用普通的带动量 SGD 算出更新方向,再把这个更新矩阵"捋平",让所有方向拿到同等的话语权,再拿去更新权重。

用开会的比喻讲完:Adam 是"谁声音大就听谁的",Muon 是"开会前先把每个人的音量调成一样,大家轮流发言"。

那具体怎么"捋平"?直接算 SVD 太慢(对于几千维的矩阵,SVD 的计算代价高得离谱,会拖慢整个训练)。这里 Muon 用了一个几十年前就存在的老技巧:**Newton-Schulz 迭代**。

### 技术细节:Newton-Schulz 迭代怎么用矩阵乘法逼近正交化

Newton-Schulz 迭代的巧妙之处在于:它只用矩阵乘法(GPU 极其擅长的操作),反复做几次,就能把一个矩阵的奇异值逼近推到 1,而完全不需要显式算出 SVD。

具体迭代公式是这样的(令 $X_0 = G / \|G\|_F$,先归一化保证初始奇异值落在 $[0,1]$ 区间):

$$X_{k+1} = a X_k + b (X_k X_k^\top) X_k + c (X_k X_k^\top)^2 X_k$$

这个公式看起来复杂,但翻译回人话就是:每一步都用当前矩阵自己的"平方"信息去修正自己,反复迭代几次,奇异值就会被推向 1,而对应的方向(即 $U$、$V$)保持不变。

为什么这招管用?关键在于:如果把 $G$ 写成 $USV^\top$,那么 $X_{k+1}$ 展开后可以写成 $U \varphi(S) V^\top$,其中 $\varphi(x) = ax + bx^3 + cx^5$ 是一个五次多项式,作用在奇异值上——而 $U$、$V$ 完全没有被改变。换句话说,这个迭代**只对奇异值动手,对方向毫发无损**。只要选好系数 $(a,b,c)$ 让 $\varphi$ 反复作用后把 $[0,1]$ 里所有的值都推向 1,迭代个 5 步,就能得到一个几乎正交的矩阵——而且全程只有矩阵乘法,在 bfloat16 精度下也很稳定。

Keller Jordan(Muon 的提出者)最终选定的系数是 $(a,b,c) = (3.4445, -4.7750, 2.0315)$,PyTorch 实现大致是这样:

```python
def newtonschulz5(G, steps=5, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X
```

代价有多大?算一下就知道:对于一个 $n \times m$ 的矩阵,每步 Newton-Schulz 迭代的 FLOP 开销是 $2(2nm^2+m^3)$,五步下来,相对于一次完整的前向反向传播,额外开销只有 $Tm/B$(其中 $T$ 是迭代步数,$B$ 是每步训练用的 token 数)。对 Llama 405B 这种规模的训练,这个比例算下来只有 0.5%——几乎是白送的。

## 为什么"捋平"梯度是个好主意?一个几何视角

前面说的是"怎么做",现在说说"为什么这么做是对的"。

有一个更根本的问题:Adam 为每个参数用的更新量,本质上是在优化"每个坐标独立看待"这件事——数学上,这对应欧几里得范数(每个数字各管各的)。但权重矩阵不是一堆散装数字,它是一个**线性算子**,它的作用是把输入向量映射到输出向量。衡量一个线性算子"改变有多大"的自然量,不是逐坐标的欧几里得范数,而是**谱范数**(spectral norm)——也就是这个矩阵能把任意方向的向量放大多少倍的最大值。

如果你换个视角,把"给权重矩阵找一个最优更新方向"这件事,重新表述成"在谱范数的意义下做最速下降(steepest descent)",答案恰好就是:把梯度矩阵替换成它最近的正交矩阵。这套理论最初是 Jeremy Bernstein 和 Laker Newhouse 在分析 Shampoo 这个更早的优化器时发现的——他们指出如果去掉 Shampoo 里那个昂贵的"预条件累积"步骤,它退化出来的更新恰好就是正交化的梯度。Muon 相当于把这个洞察做成了一个便宜到几乎不增加成本的实用算法。

再换个更朴素的说法:欧几里得范数关心"每个螺丝改变了多少",谱范数关心"这台机器整体上能造成多大的输出偏移"。对权重矩阵这种本质是"变换算子"的对象来说,谱范数显然是更贴切的度量标尺——而按照这个标尺做最速下降,自然就导出了正交化更新。

## 战绩:speedrun 记录与 Kimi K2

理论讲完了,战绩如何?

Muon 最早的舞台是 NanoGPT speedrunning——一个社区竞赛,比谁能用最少的算力把 GPT-2 规模的模型训练到某个固定的验证损失。用 AdamW 达到目标要 13.3 个 8×H100-小时,换成 Muon,只要 10 个小时。CIFAR-10 上,达到 94% 准确率的速度记录,从 3.3 秒 A100-秒压缩到 2.6 秒。这些都还是小模型上的胜利,证明这个想法可行,但没有证明它能扛得住工业级规模。

真正的分水岭是 Moonshot AI(月之暗面)2025 年初发的论文《Muon is Scalable for LLM Training》。他们训练了一个 3B/16B 参数的 MoE 模型 Moonlight,用 5.7 万亿 token,把 Muon 的 scaling law 曲线画出来,结果是:**在算力最优训练的设定下,Muon 相对 AdamW 有大约 2 倍的计算效率**。要让这套东西在大规模上"开箱即用",作者发现必须补上两个关键补丁:

1. **加入权重衰减**——纯正交化的更新如果不配合权重衰减,大模型训练容易失稳。
2. **调整每个参数的更新幅度(Update RMS 对齐)**——让 Muon 产生的更新量级,和大家已经调好参的 Adam 更新量级对齐,这样迁移超参数时不用重新调一遍。

真正把这套优化器推上产业级舞台的是 2025 年中发布的 **Kimi K2**——一个 1 万亿参数、320 亿激活参数的 MoE 模型。K2 用的优化器叫 **MuonClip**,是 Muon 加了一个关键补丁:**QK-Clip**。

问题出在哪?当把 Muon 用到千亿级参数模型时,团队观察到一个新的失稳现象,叫 **MaxLogit 爆炸**——注意力机制里 $QK^\top$ 算出来的最大绝对值,会随着训练线性甚至超线性增长,长期得不到控制。用 Cauchy-Schwarz 不等式拆开看:

$$|q_i \cdot k_j| \le \|q_i\|\|k_j\| \le \|x_i\|\|x_j\|\|W_q\|\|W_k\|$$

由于输入 $x$ 通常经过 RMSNorm 不会爆炸,这个不等式说明:如果 MaxLogit 在爆炸,根源就是 $Q$、$K$ 投影矩阵的谱范数在失控增长。而 Muon 因为擅长把每个方向都拉满(不像 Adam 那样天然抑制某些方向的增长),恰恰更容易放大这个问题。QK-Clip 的解法很直接:直接监控每个注意力头的 MaxLogit,一旦超过阈值,就对该头对应的 $Q$、$K$ 权重矩阵做缩放校正——用作者自己的话讲,这是一种"训练用抗生素":不优雅,但立竿见影,而且不会伤害模型最终性能。

<svg viewBox="0 0 600 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:600px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <rect x="15" y="70" width="130" height="55" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="80" y="102" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">SGD-momentum</text>
  <line x1="145" y1="97" x2="200" y2="97" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>

  <rect x="205" y="70" width="150" height="55" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="280" y="95" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">Newton-Schulz</text>
  <text x="280" y="112" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">正交化 (5 步)</text>
  <line x1="355" y1="97" x2="410" y2="97" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>

  <rect x="415" y="70" width="150" height="55" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="490" y="95" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">权重衰减 +</text>
  <text x="490" y="112" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">RMS 对齐 (+QK-Clip)</text>

  <text x="300" y="30" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">Muon → Moonlight → MuonClip (Kimi K2) 的演化路径</text>
  <text x="300" y="160" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">2024.12 Muon 提出</text>
  <text x="300" y="178" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">2025.02 加权重衰减+RMS对齐 (Moonlight)  →  2025.07 加 QK-Clip (Kimi K2)</text>
</svg>

## 一个诚实的但书:速度不是免费的

如果读到这里觉得"Muon 全面碾压 Adam",还需要补一个更细腻的视角。

2026 年初有一篇分析指出,Muon 之所以快,恰恰是因为它**放弃了 Adam(以及朴素 SGD)天生具备的一种"简洁性偏好"(simplicity bias)**。普通梯度下降有一个隐藏的好习惯:它会先学会数据里最主导的模式,再逐渐学习更细微的信号——收敛速度和奇异值成正比,大奇异值先被学到。这相当于一种隐式的课程学习:模型先吃"营养最丰富的信号",再慢慢啃"细枝末节"。

Muon 恰恰打破了这个节奏——它强制所有方向以同样的速度收敛。这在数据不均衡、需要照顾"长尾"信号时是好事;但在需要模型自己发现"共享结构"而不是死记硬背的场景里(比如让模型学会一个跨领域通用的映射规则,而不是给每个领域单独记一套答案),实验显示 Muon 更容易走向"记忆"而非"泛化"——它会用更高秩、更"记忆化"的方式解决问题,而普通 SGD 反而找到了那个更简洁、泛化更好的低秩解。类似地,在有虚假关联(spurious correlation)的数据上,Muon 因为不偏向"先学主要特征",反而更容易把噪声特征和真实特征一起学进去,没有给你留一个"早停一下就能避开噪声"的安全窗口。

这提醒我们一件事:优化器不是纯粹的"越快越好"的工程问题,它的收敛路径本身,会悄悄决定模型最终学到的是什么样的解。Muon 在大规模预训练——数据量巨大、目标是尽可能压榨每一份算力——的场景里表现优异,是因为这个场景恰好最在意"效率",而较少担心"记忆而非泛化"的风险(海量数据本身就会稀释掉记忆倾向的坏处)。但如果你在做小样本微调,或者关心模型的分布外泛化能力,Muon 是否依然是最佳选择,值得多一分谨慎。

## 这意味着什么

回到开头的问题:为什么 Kimi K2 这样的模型要换优化器?因为它们发现了同一件事——梯度矩阵的"偏心"不是无关紧要的细节,而是一个可以被结构化地纠正、并且纠正之后能省下真金白银算力的机会。Muon 没有发明新的数学,Newton-Schulz 迭代早在 1970 年代就存在,正交化的思想在数值线性代数里也不新鲜。它真正的贡献,是把"权重矩阵应该按照谱范数意义下的最速下降来更新"这个几何洞察,和"用便宜的矩阵乘法迭代逼近正交化"这个工程技巧,严丝合缝地拼在了一起,拼出了一个在生产环境里真的能打的优化器。

而 Kimi K2 的 MuonClip 又提醒我们:任何在小规模上验证过的优化技巧,放大到万亿参数时几乎总会撞上新的意外(这次是 MaxLogit 爆炸),而解决这些意外往往不需要多优雅的理论,一个监控 + 裁剪的"抗生素式"补丁,可能就足够撑住整个训练过程。

优化器的故事从来不是"哪个数学上更漂亮",而是"哪个能在给定的算力预算下,把损失曲线往下压得更快、更稳"。Muon 目前给出的答案是:先看看你的梯度矩阵有没有偏心,如果有,花 0.5% 的额外算力把它捋平,可能就是你训练加速最便宜的一张免费船票。
