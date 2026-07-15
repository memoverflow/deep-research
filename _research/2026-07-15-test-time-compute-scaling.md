---
title: "测试时计算扩展：当模型学会「多想一会儿」而不是「变得更大」"
date: 2026-07-15
level: 3
series: "LLM 原理深度解析"
series_order: 38
series_total: 43
tags: [test-time-compute, reasoning, o1, deepseek-r1, chain-of-thought, scaling-laws, process-reward-model]
summary: "为什么让模型多想一会儿，有时比造一个更大的模型更划算？测试时计算扩展的原理、边界与代价。"
---

> 一块钱的算力，你会选择造一个更大的脑子，还是给现在这个脑子多一点思考时间？2024 年之后，几乎整个行业都在赌后者。

## 故事从这里开始

2020 年到 2023 年，AI 圈信奉一条朴素的信条：模型越大越聪明。GPT-2 到 GPT-3 到 GPT-4，参数量一路从 15 亿涨到万亿级别，每一次跃升都伴随着能力的跃升。这条路径被称为"预训练时代的规模法则"（Scaling Laws）——多堆参数、多堆数据、多堆算力，性能就会按照可预测的曲线往上走。

但堆参数这件事，越往后越贵，越往后收益越小。训练一个万亿参数模型要烧掉几亿美元的电费和芯片钱，而性能提升却越来越像挤牙膏。更麻烦的是，模型不管拿到什么问题，思考的"力气"都是一样的——问它"1+1 等于几"和问它"证明费马大定理的一个特例"，模型都是"啪"地一下直接吐出答案，中间没有任何停顿去真正"想"。

这就是荒谬所在：人类面对难题会多想一会儿，甚至会打草稿、会验算、会推翻自己重来。但直到 2024 年之前的大模型不会——它们的计算量在推理阶段是固定的，跟问题难度完全无关。一道简单的算术题和一道奥数难题，模型都是一次前向传播（forward pass）解决，用的计算量分毫不差。

2024 年 8 月，一篇叫《Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters》的论文把这件事捅破了：如果让模型在回答问题之前多花点计算——多想、多试、多验证——效果可能比单纯把模型做大更划算。几个月后，OpenAI 的 o1 和后来的 DeepSeek-R1 把这套想法产品化，"推理模型"（reasoning model）这个新品类诞生了。这篇文章要讲的，就是这背后的原理：为什么"多想一会儿"有用？想多久才够？这条路有没有尽头？

<svg viewBox="0 0 640 220" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow0" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="320" y="24" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">两条不同的"变强"路径</text>

  <rect x="30" y="50" width="180" height="60" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="120" y="75" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">预训练扩展</text>
  <text x="120" y="95" text-anchor="middle" fill="#9a9ab0" font-size="11" font-family="system-ui">更多参数+数据+算力</text>

  <line x1="120" y1="110" x2="120" y2="150" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow0)"/>
  <rect x="30" y="155" width="180" height="50" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="120" y="184" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">更大的脑子（一次性）</text>

  <rect x="430" y="50" width="180" height="60" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="520" y="75" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">测试时计算扩展</text>
  <text x="520" y="95" text-anchor="middle" fill="#9a9ab0" font-size="11" font-family="system-ui">推理阶段多想/多试/多验证</text>

  <line x1="520" y1="110" x2="520" y2="150" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow0)"/>
  <rect x="430" y="155" width="180" height="50" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="520" y="184" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">同一个脑子（按需思考）</text>
</svg>

## 第一个概念：为什么"多想"会有用？

### 问题是什么

先想清楚一件反直觉的事：如果模型的所有知识和能力都已经固化在参数里了，那"多想一会儿"到底能挤出什么新东西？参数没变，模型的"智商"应该也没变才对吧？

答案藏在一个理论细节里：标准 Transformer 在没有思维链的情况下，能表达的计算类别是有限的。具体来说，固定深度、有限精度的 Transformer 一次前向传播只能解决"并行可判定"的问题——用理论计算机科学的术语，这类问题属于 $\mathsf{TC}^0$（甚至更弱的 $\mathsf{AC}^0$）复杂度类。这类问题的共同特点是：计算过程可以被拆成很多独立的、同时进行的小步骤，不需要严格的先后依赖。

但很多我们真正在乎的问题不是这样的。比如多位数乘法、图的可达性判断、复杂的逐步推理——这些问题本质上是**串行**的：第 5 步的结果依赖第 4 步,第 4 步依赖第 3 步。而一次前向传播的深度是固定的（假设 24 层），如果问题需要的串行步骤数超过了这个深度，模型就算不出来,不是知识不够,是"结构性"算不出来。

### 直觉：核心想法

这里有一个很生活化的类比：想象你被要求心算 $47 \times 83$，但规则是"只能用一步得出答案，不能在纸上写中间结果"。大部分人做不到——不是不会乘法，而是这道题需要好几步进位、好几次中间结果的记录，一步到位超出了大脑"寄存器"的容量。

但如果允许你打草稿——先算 $47 \times 80 = 3760$，再算 $47 \times 3 = 141$，再把两者加起来——同样的大脑,同样的知识,却能算出同样一次性做不到的题。草稿纸本身没有增加你的智力，它做的事情是：**把一个需要很多步骤的问题，拆成很多个"一步就能算"的小问题，然后把中间结果重新喂给你自己**。

思维链（Chain-of-Thought）对 Transformer 起的正是这个作用。模型每生成一个 token，这个 token 就会被重新拼回输入序列里，参与下一次前向传播——这本质上是给模型开了一条"回路"，让它可以把计算拆成多步，每一步都在原本受限的深度里完成，但步骤数不再受限于层数,而是受限于你允许它生成多少 token。理论上已经证明:给足够长的思维链，固定深度、常数精度的 Transformer 可以模拟任意规模的布尔电路计算——这就把可解问题的范围从 $\mathsf{AC}^0$ 一路扩展到了理论上的 $\mathsf{P}$（多项式时间可解问题）。

### 技术细节（选读）

用复杂度理论的语言精确表述:给定输入长度 $n$，不带思维链的常数深度、常数比特精度 Transformer 只能计算 $\mathsf{AC}^0$ 中的函数。但如果允许生成 $T$ 步思维链,同样架构的 Transformer 可以模拟任意大小为 $T$ 的布尔电路——也就是说，思考的"时长" $T$ 直接兑换成了计算能力的"深度"。

翻译回人话：**思维链不是让模型"更聪明"，而是让模型把一次性的浅层计算，换成了可以按需拉长的深层计算**。这也解释了一个经验现象——思维链对"算术题"、"多步逻辑题"这类天生需要串行计算的任务提升巨大,但对"这段话是什么情绪"这类本质上并行、一步判断的任务提升有限甚至没有提升。

<svg viewBox="0 0 640 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="320" y="24" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">思维链把"一步算不出"拆成"多步能算出"</text>

  <rect x="20" y="50" width="150" height="50" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="95" y="80" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">直接输出答案</text>
  <text x="95" y="115" text-anchor="middle" fill="#9a9ab0" font-size="10" font-family="system-ui">固定深度 · 一次前向</text>
  <text x="95" y="132" text-anchor="middle" fill="#ff8080" font-size="10" font-family="system-ui">受限于 AC⁰</text>

  <line x1="180" y1="75" x2="230" y2="75" stroke="#3a3a4a" stroke-width="1.5" marker-end="url(#arrow1)"/>
  <text x="205" y="65" text-anchor="middle" fill="#9a9ab0" font-size="10">✗ 超出深度</text>

  <rect x="250" y="50" width="90" height="50" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="295" y="80" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">步骤 1</text>
  <line x1="340" y1="75" x2="380" y2="75" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>
  <rect x="390" y="50" width="90" height="50" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="435" y="80" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">步骤 2</text>
  <line x1="480" y1="75" x2="520" y2="75" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>
  <rect x="530" y="50" width="90" height="50" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="575" y="80" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">答案</text>
  <text x="435" y="115" text-anchor="middle" fill="#9a9ab0" font-size="10" font-family="system-ui">每步都在深度限制内，步骤数=T 不受层数限制</text>
  <text x="435" y="132" text-anchor="middle" fill="#34d399" font-size="10" font-family="system-ui">可解范围扩展到 ~P</text>
</svg>

## 第二个概念：想多久才算够？"计算最优"分配策略

### 问题是什么

既然多想有用,那自然的想法是"想得越久越好"。但真实世界里算力和延迟都是钱,不能无限烧。这就带来一个新问题：给定固定的额外计算预算，应该怎么花？

早期最粗暴的做法叫 Best-of-N：让模型对同一个问题独立生成 N 个答案，然后用一个"打分器"（verifier）挑出最好的一个。这个方法简单粗暴地"用数量换质量"——但代价也很粗暴：N 越大越贵，而且不管问题难不难，都花一样多的力气去采样,这跟没有测试时扩展之前"不管难度都一次算完"的问题异曲同工，只是换了个层面。

更根本的问题是：一道简单题，模型第一次就大概率对了,采多少次都是浪费；一道特别难的题，模型可能怎么采样都对不了,再采也是浪费。真正值得砸算力的，是那些"模型有一定几率蒙对，但需要多试几次才能撞上正确答案"的中等难度题。

### 直觉：核心想法

这就像医生给病人开检查——阑尾炎这种一眼就能确诊的病，不需要做十项检查；癌症晚期这种确诊了也没法治的病，做更多检查也是徒劳；真正该多做检查的，是那种"可能是,也可能不是"的中间地带病例,多一项检查确实能提高诊断准确率。

Google DeepMind 那篇论文（Snell 等人，2024）提出的"计算最优"（compute-optimal）策略正是这个思路：先估计一道题对当前模型的难度（可以用模型自己对这道题的置信度分布来估），然后按难度动态分配采样次数或搜索深度——难题多分配、易题少分配，而不是所有题目一刀切用同样的预算。这个策略比"无脑 Best-of-N"效率高出 4 倍以上：达到同样的准确率，用四分之一的计算量就够了。

论文还发现一个更值得警惕的事实：在某些"FLOPs 对等"的比较下，一个小模型配合聪明的测试时计算策略，可以打赢一个大 14 倍的模型。换句话说，同样烧掉这么多算力，花在"让小模型多想"上，可能比花在"训练一个更大的模型"上更划算——这正是这篇论文标题里"can be more effective than scaling model parameters"这句话的分量所在。

### 技术细节（选读）

具体的两种"多花计算"的方式，论文里对应两种机制：

**机制一：搜索（Search），配合过程奖励模型。** 不是让模型一口气写完整条推理链再打分，而是把推理拆成一步一步，每一步都用一个训练好的过程奖励模型（Process Reward Model，PRM）打分，模型可以在中途剪掉看起来走偏的分支，把算力集中投给看起来靠谱的路径——这跟下棋时的搜索树剪枝逻辑一致。PRM 的训练数据来自 OpenAI 那篇《Let's Verify Step by Step》论文公开的 PRM800K 数据集：80 万条人工标注的"这一步对不对"的反馈,证明了逐步反馈（process supervision）比只看最终答案对不对（outcome supervision）的反馈质量高得多——在 MATH 数据集上，用过程监督训练的模型能解出 78% 的题，明显超过只用结果监督的版本。

**机制二：修正（Revision），让模型迭代改写自己的答案。** 训练一个"修正模型"，输入是问题加上模型自己之前给出的若干次尝试，输出是一个改进后的新答案——本质上是让模型对自己的输出做"批评—重写"的循环,而不是每次从零采样一个独立的新答案。

数学上，"计算最优分配"要解的是这样一个优化问题：给定问题 $x$ 和总预算 $N$，找到一个把预算分配给不同策略/难度的方案 $\theta^*(x)$，使得：

$$\theta^*(x) = \arg\max_{\theta} \mathbb{E}\big[\text{正确率} \mid \text{预算分配方案 } \theta, \text{问题 } x\big]$$

翻译回人话就是：这不是"一刀切"地给所有问题分配同样多的计算，而是先估计这道题"多想能不能想出来"，再决定要不要在它身上多花预算。

<svg viewBox="0 0 640 240" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;">
  <text x="320" y="24" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">计算最优分配：难度决定预算</text>

  <rect x="20" y="50" width="180" height="150" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="110" y="75" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">简单题</text>
  <text x="110" y="100" text-anchor="middle" fill="#9a9ab0" font-size="10" font-family="system-ui">第一次几乎必对</text>
  <text x="110" y="170" text-anchor="middle" fill="#34d399" font-size="20" font-family="system-ui" font-weight="bold">少采样</text>

  <rect x="230" y="50" width="180" height="150" rx="8" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="320" y="75" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">中等难度题</text>
  <text x="320" y="100" text-anchor="middle" fill="#9a9ab0" font-size="10" font-family="system-ui">有几率蒙对，多试能撞上</text>
  <text x="320" y="170" text-anchor="middle" fill="#a78bfa" font-size="20" font-family="system-ui" font-weight="bold">多采样/搜索</text>

  <rect x="440" y="50" width="180" height="150" rx="8" fill="#1e1e2a" stroke="#ff8080" stroke-width="1.5"/>
  <text x="530" y="75" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">超难题</text>
  <text x="530" y="100" text-anchor="middle" fill="#9a9ab0" font-size="10" font-family="system-ui">怎么试都对不了</text>
  <text x="530" y="170" text-anchor="middle" fill="#ff8080" font-size="20" font-family="system-ui" font-weight="bold">再多也没用</text>
</svg>

## 第三个概念：o1 和 DeepSeek-R1 把这套理论变成了产品

### 问题是什么

前面讲的"搜索 + PRM"、"多次采样 + 打分器"，本质上都是在**推理阶段**做额外的工程编排——需要外部的验证器、需要多次调用模型、整个系统很复杂。有没有可能，把"多想一会儿"这个能力直接训练进模型本身，让模型自己知道什么时候该多想、该怎么想，而不需要外部脚手架？

### 直觉：核心想法

这就是 OpenAI o1（2024 年 9 月发布）和 DeepSeek-R1（2025 年 1 月发布）做的事：不再靠外部脚手架拼凑"多想"的过程，而是直接用强化学习去训练模型，让它自己学会生成又长又有效的思维链——包括自我反思、自我验证、甚至中途推翻重来。

一个很关键但容易被忽略的事实是：DeepSeek-R1 的前身 R1-Zero 完全没有用人工标注的推理示范来做监督微调，纯粹靠强化学习——只给模型一个"这道题最终答案对不对"的信号，模型自己在训练过程中，逐渐"涌现"出了长篇推理、自我反思、验证中间步骤这些行为模式。这有点像把一个学生扔进一堆只批"对/错"、不给任何解题过程提示的题海里，学生自己慢慢摸索出了"打草稿"、"检查一遍"这些策略——没有人明确教他,但强化学习的奖励信号自然地把这些行为筛选出来了，因为长思维链、自我验证平均而言能提高最终答案对的概率。

这跟第一节讲的"思维链拓展计算能力上限"是同一件事的两个层面：第一节讲的是"多想为什么理论上有用"，这里讲的是"怎么让模型自己学会想得又长又好"，而不是靠人工写好的思维链示范去教它。

### 技术细节（选读）

DeepSeek-R1 的训练分为几个阶段（简化版）：

1. **R1-Zero**：直接在基础模型上跑强化学习，奖励函数只包含"最终答案是否正确"（可以自动验证的任务，比如数学、代码）和"格式是否规范"两项，完全不需要人工标注推理过程。
2. 观察到 R1-Zero 存在可读性差、语言混杂等问题后，**R1** 引入了一小部分高质量的冷启动数据做监督微调，再叠加强化学习,进一步提升推理质量和可读性。
3. 训练出的"涌现推理模式"（自我反思、验证、动态调整策略）还可以被"蒸馏"（distill）到更小的模型上——用大模型生成的长推理轨迹作为训练数据，让小模型也学会类似的思考方式，即使小模型自己跑强化学习跑不出这些能力。

这也是为什么"推理模型"在产品层面会多出一种新的计费维度——"推理 token"（reasoning tokens）。模型在给出最终可见答案之前，会先生成一大段用户看不到的内部思考过程，这段思考本身也要计入 token 消耗和延迟。一道题可能只需要输出 100 个字的最终答案，但内部可能悄悄"想"了 2000 个 token——这 2000 个 token 就是测试时计算扩展在账单上的真实体现。

<svg viewBox="0 0 640 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="320" y="22" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui" font-weight="bold">推理模型的一次请求：看不见的"思考"占了大头</text>

  <rect x="30" y="50" width="120" height="50" rx="8" fill="#1e1e2a" stroke="#94a3b8" stroke-width="1.5"/>
  <text x="90" y="80" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">用户提问</text>
  <text x="90" y="115" text-anchor="middle" fill="#9a9ab0" font-size="10" font-family="system-ui">~500 tokens</text>

  <line x1="150" y1="75" x2="200" y2="75" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>

  <rect x="210" y="50" width="220" height="50" rx="8" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="320" y="80" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">内部推理链（用户不可见）</text>
  <text x="320" y="115" text-anchor="middle" fill="#a78bfa" font-size="10" font-family="system-ui">可能高达 2000+ tokens</text>

  <line x1="430" y1="75" x2="480" y2="75" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>

  <rect x="490" y="50" width="120" height="50" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="550" y="80" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">最终答案</text>
  <text x="550" y="115" text-anchor="middle" fill="#9a9ab0" font-size="10" font-family="system-ui">~100 tokens</text>
</svg>

## 这条路的边界在哪里？

任何"越多越好"的叙事都值得警惕，测试时计算扩展也一样有明确的天花板。

**第一个边界是收益递减，甚至反转。** 实证研究显示，准确率相对测试时计算量的曲线一开始很陡，但很快就会趋于平缓，某些任务上甚至会出现"越想越错"的反转——这被称为逆向扩展（inverse scaling）。原因不难理解：模型的思维链一旦跑偏,继续想只会在错误的方向上越走越远，而不是自我纠正回来。

**第二个边界是"过度思考"（overthinking）和"思考不足"（underthinking）并存的怪现象。** 有研究发现，长推理模型面对简单问题时会不必要地生成冗长的思维链——相当于医生给感冒病人做全身核磁；而面对真正复杂的问题时，反而会在不同的推理思路之间来回切换，还没把一条路径想透就跳到另一条，导致哪条都没想清楚。也就是说,模型目前还不太擅长"准确判断这道题值得想多久",这也是"计算最优分配"这个理论问题在工程上远未解决的地方。

**第三个边界是复合误差在长任务链上的累积。** 如果任务需要很多步骤连续正确（比如一个需要几十步操作的智能体任务），即便单步的准确率提升看起来很小，多步复合下来失败概率也会指数级增长——反过来说，单步准确率的微小提升，在长链条任务上也可能带来完成率的巨大跃升。这意味着测试时计算扩展在"多步智能体"场景里的价值评估，比在单轮问答场景里复杂得多，一刀切的"想得越久越好"的直觉在这里并不总是成立。

**第四个边界是纯粹的经济学问题。** 测试时计算扩展把原来"一次性"的训练成本，变成了"每一次请求都要重复付出"的边际成本。训练一个更大的模型是一次性投入，之后每次推理成本相对固定；而让模型多想，是每次调用都要多花钱、多花时间——这笔账在大规模部署时会指数级放大,尤其是当推理 token 的定价往往比输出 token 更贵的时候。

## 这意味着什么

测试时计算扩展给整个 AI 行业提供了第二条增长曲线：当预训练的规模法则开始显现收益递减的迹象时，"让模型在推理阶段多想一会儿"打开了一条新的、独立的性能提升路径。它的理论根基并不神秘——思维链本质上是把 Transformer 从"一次前向传播的浅层并行计算"，扩展成了"任意步数的深层串行计算"，这在复杂度理论上是可以严格证明的能力跃升，不是玄学。

但这条路也不是免费的午餐。它把成本从"训练一次"挪到了"推理每一次"，把工程问题从"怎么堆更多参数"换成了"怎么判断一道题该想多久"。o1 和 DeepSeek-R1 证明了这套思路可以规模化落地，但"计算最优分配"、"过度思考 vs 思考不足"这些问题，恰恰说明我们离真正让模型"自知之明"地分配思考力气,还有很长的路要走。

下一次当你看到某个模型答题前"转圈思考"了好几秒钟，你可以想到：那几秒钟不是延迟故障，而是这台机器正在花你的钱,把一个理论上受限于层数的计算问题,拆解成很多个它自己能算得动的小问题。
