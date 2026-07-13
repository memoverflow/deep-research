---
title: "多头注意力为什么有效：一个大脑同时开会，还是八个小组分头调查？"
date: 2026-07-07
level: 3
series: "LLM 原理深度解析"
series_order: 28
series_total: 39
tags: [multi-head-attention, transformer, 子空间分解, 低秩瓶颈, attention]
summary: "把注意力拆成 8 个头,不是为了并行加速这么简单——这背后有一套关于'干扰'、'容量'和'秩'的数学道理,也有一堆关于'其实很多头都是冗余的'的尴尬真相。"
---

> 上一篇讲了 Self-Attention 的几何直觉：Q、K、V 三个矩阵怎么把"检索"拆成三个独立的操作。但如果你翻开任何一个 Transformer 的源码，你会发现没人真的只用一次这个操作——原始论文里用了 8 个头，后来的大模型有 32 个、64 个甚至上百个头。这篇文章要回答的问题很朴素：**为什么要拆成多个头？拆一个头跟拆八个头，数学上到底有什么不一样？**

## 故事从这里开始

假设你要给一家公司做背景调查，决定要不要投资它。你可以找一个全能型侦探，让他一个人跑遍所有信息源——查财报、查专利、查创始人履历、查供应链、查竞品动态，最后凭自己的判断写一份报告。

这个方案有个隐患：一个人的注意力和风格是单一的。如果这个侦探特别擅长财务分析，他可能会不自觉地把所有事情都往"数字对不对"上靠，而对创始人是不是靠谱这种事视而不见。更麻烦的是，当他把"财务风险""团队风险""市场风险"这些完全不同性质的判断，都塞进同一份笔记本、用同一套评分标准去打分时，这些不同类型的信号会开始互相干扰——你很难说清楚最后那个"综合评分 7.2 分"到底是因为财务好、团队一般，还是财务一般、团队特别好。

另一个方案是：分别派四个专门的调查员——一个专盯财务、一个专盯团队背景、一个专盯专利和技术护城河、一个专盯市场和竞品。他们各自用自己的方法、自己的评分体系独立工作，最后再把四份独立报告汇总起来综合判断。这样，"财务不好"这个结论不会因为"团队特别强"而被稀释掉，两件事各自说得清清楚楚。

Transformer 的多头注意力，选的正是第二种方案。而这篇文章要讲的，就是为什么"分头独立判断再汇总"在数学上确实比"一个人打一个笼统的分数"更有道理——以及，这个道理里藏着哪些容易被忽略的坑。

## 第一部分：一个头，为什么会"力不从心"

### 问题是什么

回顾一下 Self-Attention 的计算：对每一对 token，用 Query 和 Key 做点积，得到一个原始相关性分数，再对这一整行分数做 softmax，归一化成权重，最后用这些权重去混合 Value。

现在想一个具体场景。句子里的某个代词 "it" 需要同时判断两件不太相关的事：**它的先行词是谁**（句法层面的指代关系），以及 **这句话整体在讨论什么话题**（语义层面的主题关联）。如果只有一个 Query 向量、一套点积规则，"it" 就必须用同一个坐标系去衡量"谁是我的先行词"和"这句话在聊什么"——这两种判断标准八字不合：判断先行词，你可能更关心词性、句法位置这种"结构性"线索；判断话题，你可能更关心词义、上下文的语义聚类这种"内容性"线索。

用同一个向量、同一次点积去同时完成这两件事，就跟前面那个全能型侦探一样——不是做不了，而是这两种判断会在同一个坐标系里互相拉扯，谁都做不到最好。

### 直觉：把"总分"拆成几张独立的记分卡

一个头的 attention，本质上是给每一对 token 打一个"总相关性分数"，然后一次性归一化。如果你想让"句法相关性"和"语义相关性"各自被公平地衡量，最直接的办法就是：**给每一种关系判断单独开一张记分卡，各自打分，各自归一化，最后把结果汇总起来看。**

这正是多头注意力做的事。每个头有自己独立的 $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}$ 投影矩阵，把原始的词向量投影到一个属于这个头自己的、低维的子空间里，在这个子空间里独立完成"打分—归一化—加权求和"的整套流程。头之间互不干扰，因为它们根本活在不同的坐标系里。

<svg viewBox="0 0 720 300" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:720px;margin:24px auto;display:block;background:transparent;">
  <defs>
    <marker id="arrowmh1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <rect x="20" y="120" width="130" height="55" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="85" y="152" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">词向量 x</text>
  <text x="85" y="170" text-anchor="middle" fill="#8888a0" font-size="10" font-family="system-ui">(d_model 维)</text>

  <line x1="150" y1="147" x2="200" y2="60" stroke="#6e8eff" stroke-width="1.3" marker-end="url(#arrowmh1)"/>
  <line x1="150" y1="147" x2="200" y2="147" stroke="#6e8eff" stroke-width="1.3" marker-end="url(#arrowmh1)"/>
  <line x1="150" y1="147" x2="200" y2="235" stroke="#6e8eff" stroke-width="1.3" marker-end="url(#arrowmh1)"/>

  <rect x="210" y="25" width="150" height="65" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="285" y="52" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">头 1: 句法子空间</text>
  <text x="285" y="70" text-anchor="middle" fill="#8888a0" font-size="10" font-family="system-ui">独立 Q/K/V, d_k 维</text>

  <rect x="210" y="115" width="150" height="65" rx="8" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="285" y="142" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">头 2: 语义子空间</text>
  <text x="285" y="160" text-anchor="middle" fill="#8888a0" font-size="10" font-family="system-ui">独立 Q/K/V, d_k 维</text>

  <rect x="210" y="205" width="150" height="65" rx="8" fill="#1e1e2a" stroke="#f59e0b" stroke-width="1.5"/>
  <text x="285" y="232" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">头 3: 位置子空间</text>
  <text x="285" y="250" text-anchor="middle" fill="#8888a0" font-size="10" font-family="system-ui">独立 Q/K/V, d_k 维</text>

  <line x1="360" y1="57" x2="470" y2="147" stroke="#6e8eff" stroke-width="1.3" marker-end="url(#arrowmh1)"/>
  <line x1="360" y1="147" x2="470" y2="147" stroke="#6e8eff" stroke-width="1.3" marker-end="url(#arrowmh1)"/>
  <line x1="360" y1="237" x2="470" y2="147" stroke="#6e8eff" stroke-width="1.3" marker-end="url(#arrowmh1)"/>

  <rect x="480" y="118" width="200" height="60" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="580" y="143" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">拼接 + 输出投影 W_O</text>
  <text x="580" y="162" text-anchor="middle" fill="#8888a0" font-size="10" font-family="system-ui">Concat(head_1...head_h) W_O</text>
</svg>

关键的工程约束是：为了让计算量和单头 attention 大致相同，每个头分到的维度 $d_k$ 通常是 $d_{model}/h$——如果原来一个头占用全部 512 维，现在 8 个头，每个头只分到 64 维。这不是免费的午餐，这是一次**用维度换视角数量**的交易，交易划不划算，正是下一部分要讲的坑。

## 第二部分：多分几个视角，会不会反而"看不清"了

### 问题是什么

原始论文《Attention Is All You Need》里有一句常被引用的话："Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this." 翻译过来：单头注意力做的是一次加权平均，这个"平均"这个动作本身会把不同类型的信号糊在一起，抑制了模型同时关注多种关系的能力。

但这句直觉性的解释掩盖了一个更硬核的数学问题：**当你把 512 维拆成 8 份 64 维之后，每一份的"表达力"是不是打了折？**

答案是：会。2020 年一篇叫《Low-Rank Bottleneck in Multi-head Attention Models》的论文把这个问题说透了。每个头算出来的 attention 权重矩阵——也就是 softmax(QK^T) 那个方阵，大小是"序列长度 × 序列长度"——它的秩（rank）不可能超过这个头的维度 $d_k$。想象一下：如果序列长度是 1000，而每个头只分到 64 维，这个 1000×1000 的注意力矩阵最多只能有 64 阶的"自由度"。有些复杂的注意力模式，本质上需要更高的秩才能表达（比如同时精确追踪好几种互相独立的长距离依赖关系），而 64 维的头天生就做不到——这就是所谓的"低秩瓶颈"。

### 直觉：分辨率被砍掉了，但视角数量补上来了

打个比方：把一台高分辨率相机换成八台低分辨率相机，从八个不同角度同时拍。单台相机拍出来的每张照片细节都变粗糙了（分辨率下降对应低秩瓶颈——每个头能表达的关系类型变简单了），但你现在有八个不同的观察角度，能看到单一高分辨率相机因为视角固定而永远看不到的东西（多个子空间对应多种关系类型）。

这笔交易值不值得，取决于任务本身需要"更细腻地看一件事"，还是"同时看多件不同的事"。原始论文的实验结果其实已经暗示了这一点：Table 3 显示，把头数从 1 一路加到 8，翻译质量（BLEU 分数）持续提升；但如果继续加到 16、32 个头，质量反而开始下降——单头差 0.9 BLEU，头数过多同样会掉分。这条曲线不是单调的，说明"头数越多越好"是错的，真正发生的是**头数与每头分辨率之间的权衡**，存在一个最优区间。

<svg viewBox="0 0 640 220" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;background:transparent;">
  <defs>
    <marker id="arrowmh2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#8888a0"/>
    </marker>
  </defs>
  <line x1="50" y1="180" x2="600" y2="180" stroke="#3a3a4a" stroke-width="1.5" marker-end="url(#arrowmh2)"/>
  <line x1="50" y1="180" x2="50" y2="20" stroke="#3a3a4a" stroke-width="1.5" marker-end="url(#arrowmh2)"/>
  <text x="600" y="200" text-anchor="end" fill="#8888a0" font-size="12" font-family="system-ui">头数 h →</text>
  <text x="30" y="20" text-anchor="middle" fill="#8888a0" font-size="12" font-family="system-ui">质量</text>

  <path d="M 60 150 Q 150 40 280 40 T 500 90 T 580 140" stroke="#6e8eff" stroke-width="2.5" fill="none"/>

  <circle cx="60" cy="150" r="4" fill="#f59e0b"/>
  <text x="60" y="200" text-anchor="middle" fill="#8888a0" font-size="11" font-family="system-ui">h=1</text>

  <circle cx="280" cy="40" r="5" fill="#34d399"/>
  <text x="280" y="200" text-anchor="middle" fill="#34d399" font-size="11" font-family="system-ui">h=8 (最优区)</text>

  <circle cx="580" cy="140" r="4" fill="#f59e0b"/>
  <text x="580" y="200" text-anchor="middle" fill="#8888a0" font-size="11" font-family="system-ui">h=32+</text>

  <text x="320" y="60" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">视角多样性收益 > 单头分辨率损失</text>
</svg>

### 技术细节（选读）

低秩瓶颈论文的解决方案很直白：既然瓶颈来自"$d_k$ 被头数摊薄"，那就不要让 $d_k$ 依赖头数，而是让 $d_k$ 独立设定为一个足够大的值（甚至接近序列长度），头数和每头维度分开决定，不再是"总维度硬性平分"。这样每个头的 attention 矩阵秩上限更高，实验里用更小的总 embedding 维度就能训练出更好的模型——说明"必须用总维度除以头数"这条约定，其实是一种工程上的简化，不是数学上的必然。

## 第三部分：分头之后，头真的各司其职吗

### 问题是什么

前两部分讲的是"理论上应该分工"，但一个尖锐的问题是：模型真的会自发学出"分工"吗，还是训练出来的八个头其实差不多，只是随机初始化不同，本质上做的是重复的工作？

### 直觉：既有明确分工，也有大量冗余

两条研究路线分别给出了看似矛盾、实际互补的答案。

第一条路线是"看头到底在关注什么"。斯坦福的 Kevin Clark 等人在《What Does BERT Look At?》里，把 BERT 每个头的注意力权重可视化，发现头之间的行为差异非常明显——有的头几乎只关注句子里的分隔符标记，有的头专门盯着固定的相对位置（比如永远看前一个词），有的头广泛地扫视整句话，还有的头精确捕捉特定的句法关系（比如动词和它的宾语）。这证实了"多头分工"确实会自发出现，而且分工的类型是可以被人类解读的。

第二条路线是"如果强行拿掉某个头会怎样"。CMU 的《Are Sixteen Heads Really Better than One?》做了一件挺打脸的实验：训练好的模型里，大部分头在测试时可以直接被裁掉，性能几乎不受影响，有些层甚至能裁到只剩一个头。这说明训练出来的头里，真正"不可替代"的只有一小部分——多数头的功能是重叠的、冗余的。

这两个发现放在一起，画面就完整了：多头确实会分化出各自的功能角色（Clark 的可解释性证据），但这种分化不是"每个头都恰好承担一份独一无二的责任"，而更像是"少数几个头扛起了主要工作，剩下的头处于半失业状态，随时可以被优化掉"（Michel 等人的冗余证据）。这跟前面调查员的比喻有个微妙的修正：现实中的团队协作，往往也不是四个调查员各自精确分工、缺一不可，而是某个能力强的调查员承担了大部分实质工作，另外几个人挂名参与、贡献有限——但你在组建团队之初，并不知道谁会是那个主力,所以还是得多招几个人试一试。这正是"训练动态"起作用的地方：Michel 等人推测，多头带来的收益，很大一部分可能来自**训练过程中**提供了更多探索路径（多个头意味着多组独立初始化的参数在同时尝试不同的关注模式），而不仅是**推理时**每个头都必须存在。

### 技术细节（选读）

2025 年一篇《A Capacity-Based Rationale for Multi-Head Attention》给"分头为什么有意义"提供了一个更硬的信息论论证，回避了"是否冗余"的争议，直接从容量角度证明其必要性。作者设计了一个叫 Relational Graph Recognition 的抽象任务：key-query 通道要编码一张关系图（谁和谁相关），给定一部分节点，要能准确恢复每个节点的邻居关系。在固定的总 key 维度预算 $D_K = h \cdot d_k$ 下，论文证明：**即使是最简单的场景**（每个 query 只对应唯一一个目标，不存在"到底该关注谁"的判断难度），把预算拆成多个头依然能提升可编码的关系数量上限。

原因不是"每个头学不同的东西"这种表示层面的解释，而是一个更底层的几何事实：当很多种不同的"关系"信号被塞进同一个向量空间时，会发生"embedding superposition"——不同关系的编码方向互相重叠，彼此干扰，就像很多个电台挤在同一频段互相串音。把总预算切成独立的头，相当于给每种关系类型分配独立的频段，减少了这种干扰，在完全相同的参数总量下能表达更多、更精确的关系。这个理论预测在受控实验里被验证出清晰的"相变点"——超过某个头数阈值，模型才能可靠地恢复所有关系，这个阈值和理论推导出的下界高度吻合。

有意思的是，这个视角还回应了另一篇 2025 年论文（《The Effect of Attention Head Count on Transformer Approximation》）的极端情形分析：如果只用单头，但把 embedding 维度设得足够大（接近序列长度），attention 层理论上可以完全"记住"整个输入序列，但此时真正的模式识别工作会被甩给后面的 FFN 层去做——attention 本身退化成了一个纯粹的检索/复制机制,失去了"提炼出多种独立关系结构"的能力。这从另一个角度印证了：多头存在的意义之一，正是防止 attention 退化为死记硬背。

## 这意味着什么

回到最开始的比喻：多头注意力选择"分头独立调查再汇总"，不是因为这样风格上更优雅，而是因为背后有三层递进的理由，每一层都在纠正前一层可能带来的误解。

第一层是最直觉的：不同类型的关系判断（句法、语义、位置……）如果被塞进同一个坐标系里做点积，会互相干扰，拆开成独立子空间能避免这种干扰——这是原始论文给出的直觉性理由。

第二层是理论上的提醒：拆分不是免费的，头数越多、每个头分到的维度越小，单个头能表达的关系复杂度（用矩阵的秩来衡量）就越有限，存在一个头数与分辨率之间需要权衡的最优区间，而不是头越多越好。

第三层是最颠覆直觉的：即使模型自发学出了可解释的分工（有的头管句法，有的头管位置），大量实证证据显示，训练出来的头里存在明显冗余，多头的真正价值也许更多来自训练过程中提供的探索多样性，而不是推理时每个头都缺一不可。而最新的信息论分析给出了一个不依赖"是否冗余"争议的、更根本的解释——多头能减少不同关系信号在共享向量空间里的相互干扰，从而在相同参数预算下编码更多信息。

理解这三层，你就不会再满足于"多头能看不同子空间"这句被念了无数遍的套话——你知道这句话背后，有秩的上限、有容量的下界、也有工程实践中被反复验证过的冗余现实。

## 下一篇预告

多头注意力解决了"表达多种关系"的问题，但代价是推理时要为每个头单独缓存一份 Key 和 Value——这正是本系列之前讲过的 KV Cache 膨胀问题的根源之一。如果你已经读过 GQA/MQA/MLA 那一篇，会发现这些方案本质上都是在"多头带来的表达力"和"每个头独立缓存带来的推理成本"之间，重新寻找一个新的平衡点。下一篇，我们会转向另一个战场：当 attention 的头数和维度都已经确定，模型还能靠改变**训练时的初始化策略**来获得更稳定、更快速的收敛——这是另一场关于"分配"的战争,只是这次分配的不是关注的维度,而是梯度流动的起点。
