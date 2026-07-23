---
title: "多token预测：如果训练时就逼模型「看得更远」会怎样？"
date: 2026-07-16
level: 3
series: "LLM 原理深度解析"
series_order: 40
series_total: 53
tags: [multi-token-prediction, DeepSeek-V3, 预训练目标, 归纳头, 投机解码]
summary: "从 Meta 的一篇实验论文到 DeepSeek-V3 的生产级设计：让模型在训练时一次预测好几个未来的 token，为什么反而能让它更聪明、跑得更快？"
---

# 多token预测：如果训练时就逼模型「看得更远」会怎样？

> 几乎所有大模型的训练目标都是"预测下一个词"。但如果你逼模型在训练时一次预测未来 4 个词呢？结果不仅没变差，反而在代码生成、推理能力和推理速度上全面变好——这背后藏着一个关于"教师强制"（teacher forcing）的隐藏缺陷。

## 故事从这里开始

想象你在教一个学生写作文。你的教学方法是：每次只给他看到目前为止写好的所有句子，然后让他猜"下一个字应该是什么"。猜对了，继续往下教；猜错了，你把正确答案告诉他，然后翻到下一页，继续这个游戏。

这个学生练习了几百万遍之后,确实变得很会"接话"——给他任何一句话的开头，他都能顺畅地续写下去。但你有没有想过一个问题：这种训练方式,会不会让他养成一种"短视"的写作习惯？

因为他每次只需要对付眼前这一个字。他不需要真的想清楚"这一段接下来要往哪个方向发展"，只需要根据最近几个字的局部规律,给出一个统计上合理的续写。就好像你问他"中国的首都是"，他不需要理解"首都"这个概念，只需要记住"中国的首都是"后面经常跟着"北京"这个统计规律就够了。

这正是 2020 年代大语言模型训练方式的真实写照。GPT、Llama 这些模型的训练目标全都是**next-token prediction**——给定前面所有的词，预测下一个词。这个目标简单、优雅，也确实训练出了极其强大的模型。但 2024 年 4 月，一群来自 Meta FAIR 的研究者（Fabian Gloeckle 等人）提出了一个大胆的问题：**如果我们从一开始就逼模型同时预测未来好几个词,而不只是下一个,会发生什么?**

这篇论文叫《Better & Faster Large Language Models via Multi-token Prediction》。它的结论后来被 DeepSeek-V3 直接搬进了生产级模型的架构设计里,成为其技术报告中明确列出的核心创新之一。这篇文章要讲的,就是这个看似简单的改动背后,到底藏着什么样的原理。

<svg viewBox="0 0 700 260" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrowA" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="20" y="30" fill="#6e8eff" font-size="13" font-weight="bold" font-family="system-ui">传统：Next-Token Prediction</text>
  <rect x="20" y="45" width="90" height="45" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="65" y="72" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">共享主干</text>
  <line x1="110" y1="67" x2="150" y2="67" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrowA)"/>
  <rect x="155" y="45" width="90" height="45" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="200" y="72" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">预测 t+1</text>
  <text x="200" y="112" text-anchor="middle" fill="#9a9aac" font-size="11" font-family="system-ui">只关心眼前这一步</text>

  <text x="20" y="160" fill="#6e8eff" font-size="13" font-weight="bold" font-family="system-ui">Multi-Token Prediction (n=4)</text>
  <rect x="20" y="175" width="90" height="45" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="65" y="202" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">共享主干</text>
  <line x1="110" y1="197" x2="145" y2="150" stroke="#6e8eff" stroke-width="1.2" marker-end="url(#arrowA)"/>
  <line x1="110" y1="197" x2="145" y2="185" stroke="#6e8eff" stroke-width="1.2" marker-end="url(#arrowA)"/>
  <line x1="110" y1="197" x2="145" y2="220" stroke="#6e8eff" stroke-width="1.2" marker-end="url(#arrowA)"/>
  <line x1="110" y1="197" x2="145" y2="255" stroke="#6e8eff" stroke-width="1.2" marker-end="url(#arrowA)"/>
  <rect x="150" y="128" width="90" height="40" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="195" y="152" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">头1: t+1</text>
  <rect x="150" y="175" width="90" height="40" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="195" y="199" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">头2: t+2</text>
  <rect x="150" y="222" width="90" height="40" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="195" y="246" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">头4: t+4</text>
  <text x="380" y="150" fill="#9a9aac" font-size="11" font-family="system-ui">推理时只用头1；</text>
  <text x="380" y="170" fill="#9a9aac" font-size="11" font-family="system-ui">训练时四个头同时</text>
  <text x="380" y="190" fill="#9a9aac" font-size="11" font-family="system-ui">给主干反馈梯度，</text>
  <text x="380" y="210" fill="#9a9aac" font-size="11" font-family="system-ui">逼它学到更远的结构</text>
</svg>

## 问题的根源：教师强制的「近视症」

### 问题是什么

要理解为什么"只预测下一个词"会有局限，我们得先搞清楚模型训练时到底在做什么。

模型训练用的是一种叫**教师强制（teacher forcing）**的方法：给模型看一段真实文本的前半部分，让它猜下一个词，然后不管猜得对不对，都直接把**真实的**下一个词喂给它，让它接着猜下下个词。这个过程从头到尾都有"标准答案"在手，模型永远不会真的"走错路"。

这在训练阶段效率极高——你可以并行计算一整个句子里每个位置的损失。但它有一个隐藏的副作用:**模型学会的是"给定完美的历史，预测下一步"，而不是"在自己可能犯错的情况下，如何把整段话讲圆"。** 到了推理阶段（也就是你真正跟模型聊天的时候），情况完全不同——没有人会在旁边随时纠正它,它自己生成的每个词都会变成下一步的输入。如果某一步说错了，错误会一直累积下去。

这种训练和推理之间的**脱节**（论文里称为 distributional discrepancy）有另一个更微妙的后果：因为模型只需要对付"下一个词"这一个任务，它有充分的理由走捷径——只学习局部的、字面上的统计规律，而不是真正理解文本背后的结构。

Meta 论文里做了一个很有意思的实验来证明这一点。他们训练了一个字节级（byte-level）的模型——也就是把文本切到最细的粒度，一个字节一个字节地预测。在这种极端情况下，"预测下一个字节"几乎完全没有信息量,因为一个单词的大部分字节其实是可以由前面的字节机械推出来的（比如你打出"transf"，后面几个字节大概率是"ormer"）。这时候如果只做单字节预测，模型基本学不到什么有意义的东西。但如果让模型一次预测未来 8 个字节，效果发生了戏剧性的变化——在 MBPP 编程测试上，预测正确率提升了 67%。这说明当"下一步"这个任务本身缺乏信息量时，把目标拉长到"未来几步"能强迫模型去捕捉更本质的规律。

### 直觉：不是学一个更难的任务，而是学一个更"诚实"的任务

这里有一个关键的类比可以帮你建立直觉。

想象你在教一个孩子下棋。方法 A：每次只让他看当前局面，问他"下一步该走哪"，走完你直接告诉他正确答案，进入下一个局面。方法 B：每次看到当前局面时,不仅让他想"下一步"，还得同时说出"我认为再往后两三步，局势会往什么方向发展"。

方法 B 显然更难，孩子刚开始可能表现更差。但如果他真的能在方法 B 里做得不错，说明他理解的不只是"这一步棋的套路"，而是**局面背后的战略结构**。这种能力会反过来让他在方法 A 的单步预测上也表现得更好——因为一个理解全局的棋手，自然也能选对每一步。

Multi-token prediction 做的正是这件事：**它不改变模型最终要完成的任务（还是自回归地一个词一个词生成），但改变了训练时的监督信号——除了要求模型对下一个词负责，还要求它对未来第 2、3、4 个词也负责。** 这个额外的压力,逼着模型的中间表征（也就是共享主干产生的隐藏状态）必须携带更多"面向未来"的信息，而不能只满足于"刚好够猜对下一个字"。

论文提出了一个特别精妙的解释,叫「**选择点的隐式加权（implicit weighting of choice points）**」。文本里不是每个词的地位都一样。有些词是"风格性"的——换个说法结果都差不多；但有些词是"关键抉择点"——一旦选定，后面很长一段文字的走向就被锁死了。比如写一个函数，你决定用递归还是用循环，这个选择会决定接下来十几行代码的结构。

如果只做单步预测，模型给这两种词分配的训练信号权重是一样的（都只是"猜一个词，对或错"）。但如果你同时预测未来 n 个词，情况就不同了：一个"关键抉择点"后面跟着的所有词，都会因为这个抉择的正确与否而变得容易或困难预测。这意味着——通过多步预测loss的累积效应——**关键抉择点会被隐式赋予更高的训练权重**，因为它的"连带责任"更大。论文给出了具体的数学结果：n-token 预测会给一个"抉择点"分配大约 n(n+1)/2 倍的隐式权重，而对"无关紧要"的转折点只分配 n 倍的权重。换句话说，模型被自动引导去更用心地学那些真正重要的决策，而不是均匀地对待每一个词。

### 技术细节（选读）

标准的语言模型训练目标是最小化下一个词的负对数似然：

$$L_1 = -\sum_t \log P_\theta(x_{t+1} \mid x_{t:1})$$

翻译成人话：给定到目前为止的所有词 $x_{t:1}$，让模型给真实的下一个词 $x_{t+1}$ 打尽量高的概率。

Multi-token prediction 把这个目标推广为同时预测未来 $n$ 个词：

$$L_n = -\sum_t \sum_{i=1}^{n} \log P_\theta(x_{t+i} \mid x_{t:1})$$

翻译成人话：不只让模型对下一个词负责，还要它对未来第 2、3、...、n 个词都给出准确的概率估计——而且这些估计全都只依赖于**同一段**已知历史 $x_{t:1}$，不能偷看真实的中间结果。

架构上怎么实现？论文的做法很巧妙：所有的预测头共享同一个 Transformer 主干（把输入编码成一个隐藏表征 $z_t$），然后接上 $n$ 个**独立的**输出头 $f_{h_1}, ..., f_{h_n}$，第 $i$ 个头专门负责预测未来第 $i$ 个词：

$$P_\theta(x_{t+i} \mid x_{t:1}) = \text{softmax}(f_u(f_{h_i}(f_s(x_{t:1}))))$$

这里 $f_s$ 是共享主干，$f_{h_i}$ 是第 $i$ 个头专属的几层 Transformer，$f_u$ 是共享的输出投影（unembedding）矩阵。**推理时只用第一个头**（也就是标准的 next-token 预测），其他头被抛弃或者用来加速——这点很关键：模型在部署时的实际使用方式完全没变，唯一变化是训练时多花了点计算去"逼"主干学得更好。

有个工程细节值得一提：如果直接把 n 个头的 logits 都在内存里同时展开，会因为词表体积巨大（通常几万到十几万个词）而爆显存。论文的解法是让 n 个头**依次**做前向和反向传播，做完一个头就立刻释放它占用的显存,只在主干上累积梯度。这样峰值显存占用从 $O(nV+d)$ 降到了 $O(V+d)$，几乎不增加训练时间。

## 效果验证:不是纸上谈兵

论文在真实规模的实验里验证了几个核心发现：

**规模效应**：在 300M 到 130 亿参数的范围内做对比实验，小模型上 multi-token prediction 反而略差，但从 30 亿参数开始就明显反超，而且优势随模型变大而扩大。130 亿参数的模型在 HumanEval 上多解出 12% 的问题，MBPP 上多解出 17%。这可能解释了为什么这个想法在更早的文献里（Qi 等人 2020 年就提出过类似思路）一直没有火起来——它需要足够大的模型才能显现优势。

**推理反而更快**：因为训练时已经有了额外的 3 个预测头，推理阶段可以直接拿它们做"自我投机解码"（self-speculative decoding）——不需要额外训练一个小模型来做草稿，主模型自己就能一次"蹦"出好几个候选词，再用真正的验证步骤确认。实验测得推理速度提升了最多 3 倍。这是一个意外之喜：本来只是想改善训练质量，结果顺手解决了推理速度问题。

**归纳头形成得更早、更稳**：论文专门设计了一个小规模合成任务，测试模型学会"归纳"模式的能力（如果之前见过 "A→B"，再次看到 A 时预测下一个是 B）。这正是 Anthropic 的 Olsson 等人 2022 年发现的**归纳头（Induction Head）**机制——一种被认为是 in-context learning 底层电路的注意力模式。实验发现，用 multi-token prediction 训练的模型，形成归纳头的速度明显更快，尤其是在模型规模较小、数据质量一般的情况下。这为"multi-token prediction 促进更深层次的语言理解"提供了一个可解释的证据。

## DeepSeek-V3：把实验室想法搬进生产模型

2024 年底，DeepSeek 发布 V3 技术报告，在架构创新清单里明确写道："We investigate a Multi-Token Prediction (MTP) objective and prove it beneficial to model performance."——这不是巧合式的复现，而是把这个想法工程化改造后大规模部署。

DeepSeek-V3 的 MTP 设计跟 Meta 论文有一个关键区别，值得专门讲清楚。Meta 论文里的 $n$ 个预测头是**并行独立**的——它们都直接接在共享主干上，互不依赖，同时给出对 $t+1, t+2, ..., t+n$ 的预测。而 DeepSeek-V3 采用的是**顺序链式**结构：设置 $D$ 个 MTP 模块（发布的版本里 $D=1$，也就是额外多预测一步），每个模块由一个专属的 Transformer block 和投影矩阵组成，但**第 $k$ 个模块的输入依赖于第 $k-1$ 个模块的输出**，形成一条完整的因果链。

这个设计选择背后的考虑是：与其让每个头孤立地猜"未来第 i 步"，不如让预测过程本身模拟真实的自回归生成——每一步的预测都建立在"假设前面几步预测正确"的基础上，这样训练出来的表征更贴近推理时真实发生的情况。这也呼应了前面讲的"教师强制脱节"问题:如果训练时预测未来几步的方式本身就是链式的、依赖前一步结果的，那模型学到的能力就更接近它在推理时真正需要用到的能力。

<svg viewBox="0 0 700 240" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrowB" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="350" y="25" text-anchor="middle" fill="#6e8eff" font-size="13" font-weight="bold" font-family="system-ui">DeepSeek-V3 的 MTP：顺序因果链</text>

  <rect x="20" y="60" width="130" height="55" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="85" y="88" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">主模型</text>
  <text x="85" y="105" text-anchor="middle" fill="#9a9aac" font-size="10" font-family="system-ui">预测 t+1</text>

  <line x1="150" y1="87" x2="200" y2="87" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrowB)"/>

  <rect x="205" y="60" width="130" height="55" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="270" y="88" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">MTP 模块 1</text>
  <text x="270" y="105" text-anchor="middle" fill="#9a9aac" font-size="10" font-family="system-ui">预测 t+2</text>

  <line x1="335" y1="87" x2="385" y2="87" stroke="#34d399" stroke-width="1.5" marker-end="url(#arrowB)" stroke-dasharray="4,2"/>
  <text x="360" y="75" text-anchor="middle" fill="#9a9aac" font-size="9" font-family="system-ui">D&gt;1 时继续链式</text>

  <rect x="390" y="60" width="130" height="55" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5" opacity="0.5"/>
  <text x="455" y="88" text-anchor="middle" fill="#9a9aac" font-size="12" font-family="system-ui">MTP 模块 2</text>
  <text x="455" y="105" text-anchor="middle" fill="#9a9aac" font-size="10" font-family="system-ui">预测 t+3</text>

  <rect x="20" y="150" width="500" height="50" rx="8" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="270" y="180" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">共享 Embedding 层 + 共享输出头（跨主模型与所有 MTP 模块复用）</text>

  <text x="20" y="225" fill="#9a9aac" font-size="11" font-family="system-ui">发布版本 D=1：只加一个模块，额外预测下下个 token；推理阶段也可用于投机解码加速</text>
</svg>

工程上还有一个精巧之处：这些 MTP 模块和主模型**共享 embedding 层和输出头（unembedding）**。这意味着新增的模块只贡献了很小一部分额外参数（DeepSeek-V3 的主模型 671B 参数，MTP 模块只占约 14B），却能给整个模型的表征质量带来实打实的提升。损失函数上，MTP 的贡献是一个加权项：

$$\mathcal{L}_{\text{MTP}} = \frac{\lambda}{D}\sum_{k=1}^{D} \mathcal{L}_{\text{MTP}}^{k}$$

翻译回人话：把每个 MTP 模块自己的预测损失取平均，再乘上一个权重系数 $\lambda$ 加到总损失里。DeepSeek 在训练过程中还专门调整了 $\lambda$ 的大小——训练前期设得大一点（比如 0.3），让 MTP 信号充分发挥作用；训练后期调小，避免它干扰主任务的收敛。这种务实的工程调参,正是把一个"实验室发现"变成"生产级特性"所必需的打磨。

而且和 Meta 论文一样，DeepSeek-V3 部署时同样利用了 MTP 模块做**推理加速**：额外的预测头训练好之后可以直接拿来做投机解码的"草稿模型"，不需要单独再训练一个小模型来做这件事——这是一个几乎零成本的额外收益，因为训练阶段已经把这个能力"内置"进模型里了。

## 这意味着什么

回过头看，multi-token prediction 讲的其实是一个很朴素的道理：**你给模型设定的训练目标,决定了它会发展出什么样的内部表征。** 如果目标只是"猜对下一个字"，模型有充分的理由走捷径，只捕捉局部统计规律。而如果目标逼着它同时对未来好几步负责，它就不得不在内部构建出更"面向未来"的表征——这种表征恰好也是归纳、推理这些高阶能力所需要的原料。

更妙的是，这个改动几乎是"免费的午餐"：不需要新数据，不需要改变推理时的使用方式，训练时的计算开销也被精心的工程设计压到接近于零，却换来了训练质量和推理速度的双重提升。DeepSeek-V3 把它从一篇实验论文变成了一个生产级模型的标配组件，这也提示我们：**很多改善大模型的空间，不在于堆更多参数或更多数据，而在于重新设计"你到底在教它学什么"这个问题本身。**

这也让我们对"预测下一个词就是智能的全部"这个说法有了更细腻的理解——不是说这个目标错了，而是它的"实现方式"（教师强制、单步监督）本身可能限制了模型能学到多深。当你稍微改变一下监督信号的形状，同样的数据、同样的架构，就能挖出更多潜力。

---

*本文是「LLM 原理深度解析」系列的第 40 篇。系列聚焦大语言模型底层原理的教学讲解，从注意力机制到训练目标设计,逐步揭开现代 LLM 背后的数学与工程直觉。*

## 参考来源

1. Gloeckle, F., Idrissi, B. Y., Rozière, B., Lopez-Paz, D., & Synnaeve, G. (2024). *Better & Faster Large Language Models via Multi-token Prediction*. arXiv:2404.19737.
2. DeepSeek-AI. (2024). *DeepSeek-V3 Technical Report*. arXiv:2412.19437.
3. Olsson, C., et al. (2022). *In-context Learning and Induction Heads*. Transformer Circuits Thread, Anthropic.
4. Qi, W., et al. (2020). *ProphetNet: Predicting Future N-gram for Sequence-to-SequencePre-training*.
5. HackerNoon 系列解读: "Multi-Token Prediction: Higher Sample Efficiency for Large Language Models" (2025).
