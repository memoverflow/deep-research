---
title: "Attention Sink：为什么每个大模型都有一个「情绪垂直区」，把注意力都倒在第一个词上？"
date: 2026-07-14
level: 3
series: "LLM 原理深度解析"
series_order: 36
series_total: 36
tags: [attention, transformer, interpretability, kv-cache, quantization]
summary: "几乎所有 Transformer 语言模型都会把大量注意力砸在序列的第一个 token 上——不管这个 token 是什么、句子在讲什么。这不是 bug，而是模型自己发明出来的一种「安全阀」机制。"
---

> 如果你打开任何一个训练好的 LLM，画出它某一层某个注意力头的注意力矩阵，你几乎肯定会看到一列异常显眼的竖条：几乎所有 token 都把一部分注意力分给了序列里的第一个词。这个词甚至可能只是一个不带任何语义的起始符 `<bos>`。这篇文章讲清楚这件怪事到底是怎么回事，以及它为什么重要到能决定一个模型能不能处理百万级上下文、能不能被量化压缩、甚至能不能被越狱攻击。

## 故事从这里开始

假设你在开一个百人的圆桌会议。每个人发言前，规定要先"回顾"一下在场所有人已经说过的话，并且要给每个人的发言分配一个权重——权重必须严格加起来等于 100%，一分都不能多，一分都不能少。

现在设想一种情况：这轮讨论进行到第 80 个人发言，前面 79 个人讲的内容五花八门，跟他此刻真正想表达的观点关系都不大——没有一个人的话是"强相关"的。可规则摆在这里：这 100% 的权重必须分配出去，不能留白。他要怎么办？

一种办法是把权重均匀撒给这 79 个人，谁都不特别在意。但这样会有一个副作用：均匀混合意味着他自己原本清晰的想法，会被这 79 份"其实不重要"的信息按比例稀释掉，变得模糊。

另一种办法：把绝大部分权重都丢给会议里坐在最前面、第一个发言、且几乎所有人在整场会议里始终能"看到"的那个人——不管他说了什么。这样，剩下极小一部分权重才需要去"认真"分配给真正相关的发言人。用这种方式,他保住了自己想法的清晰度,同时又满足了"权重必须加起来是 100%"这条铁律。

这正是 2023 年 MIT 韩松实验室团队在研究如何让 LLM 处理超长文本流时,意外发现的现象。他们把这个"专门用来倒垃圾权重"的位置,称为 **attention sink（注意力汇）**。而几乎无一例外地,这个位置就是序列里最开始的那个 token。

这件事乍看只是个奇怪的统计规律，但它牵连出一整条因果链：为什么长文本推理会突然崩溃、为什么模型量化压缩这么难、为什么有的攻击者能靠操纵开头几个 token 就影响模型行为。我们从头讲起。

<svg viewBox="0 0 640 220" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow0" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="320" y="24" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">一层注意力头的分配示意</text>
  <!-- tokens row -->
  <rect x="20" y="60" width="60" height="40" rx="8" fill="#1e1e2a" stroke="#ffb86e" stroke-width="2"/>
  <text x="50" y="85" text-anchor="middle" fill="#ffb86e" font-size="12" font-family="system-ui">&lt;bos&gt;</text>
  <rect x="100" y="60" width="60" height="40" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="130" y="85" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">今天</text>
  <rect x="180" y="60" width="60" height="40" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="210" y="85" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">天气</text>
  <rect x="260" y="60" width="60" height="40" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="290" y="85" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">很</text>
  <rect x="340" y="60" width="60" height="40" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="370" y="85" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">好</text>
  <!-- current token -->
  <rect x="420" y="60" width="80" height="40" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="2"/>
  <text x="460" y="85" text-anchor="middle" fill="#6e8eff" font-size="12" font-family="system-ui">当前 token</text>

  <!-- arrows from current token down to each, thickness ~ weight -->
  <path d="M 460 100 Q 300 150 50 100" stroke="#ffb86e" stroke-width="7" fill="none" marker-end="url(#arrow0)"/>
  <path d="M 460 100 Q 400 130 130 100" stroke="#6e8eff" stroke-width="1.5" fill="none" marker-end="url(#arrow0)" opacity="0.5"/>
  <path d="M 460 100 Q 400 130 210 100" stroke="#6e8eff" stroke-width="1.5" fill="none" marker-end="url(#arrow0)" opacity="0.5"/>
  <path d="M 460 100 Q 420 120 290 100" stroke="#6e8eff" stroke-width="1.5" fill="none" marker-end="url(#arrow0)" opacity="0.5"/>
  <path d="M 460 100 Q 430 110 370 100" stroke="#6e8eff" stroke-width="1.5" fill="none" marker-end="url(#arrow0)" opacity="0.5"/>

  <text x="50" y="180" text-anchor="middle" fill="#ffb86e" font-size="12" font-family="system-ui">吸收 60-90%</text>
  <text x="250" y="200" text-anchor="middle" fill="#8a8a9a" font-size="12" font-family="system-ui">其余 token 只分到剩下的一小份注意力权重</text>
</svg>

## 第一部分：这现象是怎么被发现的

### 问题是什么

2023 年，很多团队想让 LLM 支持"无限长"的流式对话——比如一个聊天机器人要一直运行，处理几百万个 token 的历史对话，而不是每次都重新计算全部上下文（那样太贵）。

一个自然的想法是"滑动窗口"：只保留最近 N 个 token 的 KV cache，超出窗口的旧 token 直接丢弃。听起来很合理——旧内容反正也不太相关了。

但当研究者真的这么做时，结果出乎意料：一旦最初的几个 token 被挤出 cache 窗口，模型的困惑度（perplexity，衡量模型预测准确性的指标，数值越低越好）会瞬间爆炸，从十几点几直接飙升到几万点——模型直接"疯"了，输出变得完全不可用。

这很奇怪。直觉上，删掉最早、最"过时"的几个 token 应该是影响最小的操作，但实际却是灾难性的。

### 直觉：核心想法

MIT 团队做了个简单的实验：可视化注意力矩阵，看看模型到底在关注什么。结果发现了本文开头讲的那个现象——不管中间内容讲了什么，几乎每个 token 都会把大量注意力砸在序列最开头的几个 token 上，哪怕那几个 token 语义上毫无意义（比如仅仅是一个起始符）。

回到会议室的类比：那个"坐在最前面、始终可见"的人，就是序列里第一个 token。它有两个特殊属性：

1. **它一直在场**——因为 Transformer 是自回归的（用因果掩码，每个 token 只能看到自己之前的 token），第一个 token 是唯一一个对*所有*后续位置永远可见的位置。
2. **它是安全的"垫底"选项**——把权重丢给它，不会像丢给一个"看起来相关但其实不相关"的中间 token 那样,把错误的语义信息带进当前 token 的表示里。

于是模型在训练中学到了一种"偷懒但聪明"的策略：把 softmax 分配权重时那部分"不知道该给谁"的多余概率质量，稳定地倾倒进第一个 token。这样，其余的注意力权重才能真正、干净地聚焦在语义上相关的内容上。

研究者把这个位置命名为 **attention sink**——就像下水道里的排水口，把系统里"多余"的、必须排出但又没有明确去向的东西，导向一个固定的、安全的出口。

一旦你理解了这一点，前面 perplexity 爆炸的原因就清楚了：删掉这几个 sink token 之后，模型突然没有了"安全垫底选项"，softmax 被迫把原本要倾倒进 sink 的那部分权重强行摊到剩余的、真正携带语义的 token 上——这相当于往每个语义相关的注意力权重里注入了大量噪音，整个注意力分布被彻底破坏。

### 技术细节（选读）

具体来说，第 $\ell$ 层第 $h$ 个注意力头对位置 $i$ 的输出可以写成：

$$
\mathbf{z}_i^{(\ell,h)} = \sum_{j \le i} \alpha_{ij}^{(\ell,h)} \, \mathbf{W}^{(\ell,h)} \mathbf{v}_j^{(\ell)}, \qquad \alpha_{ij}^{(\ell,h)} = \frac{\exp(\text{score}_{ij})}{\sum_{w \le i} \exp(\text{score}_{iw})}
$$

翻译回人话：位置 $i$ 的新表示，是它能看到的所有位置 $j$（$j \le i$，因果掩码）的 value 向量的加权平均，权重 $\alpha_{ij}$ 由 softmax 决定，而 softmax 的分母保证了所有权重加起来必须等于 1。

**这个"必须加起来等于 1"的约束，就是问题的根源。** 无论 query 和所有可见 key 的相关性有多低，softmax 的分母都会强迫把这些低相关性分数"归一化"成一组合法的概率分布——总要凑够 100%。当没有任何 key 真正相关时，模型没有"弃权"这个选项。

MIT 团队给出的解决方案 **StreamingLLM** 很直接：把 KV cache 分成两部分——最初的几个 sink token（固定保留，不随窗口滑动被逐出）+ 滑动窗口内的最近 token。同时给 cache 内的 token 重新编号位置（用"cache 内的相对位置"而不是"原文中的绝对位置"），这样 RoPE / ALiBi 这类相对位置编码依然能正常工作。

更进一步，他们发现如果训练阶段就专门加入一个可学习的、不携带语义的"Sink Token"，效果比单纯依赖模型自然选中的第一个 token 更稳定——在实验中，只保留这一个 sink token（不需要额外保留 3-4 个真实的开头 token）就能把 1024 长度窗口下的 perplexity 稳定在 18.01，跟保留全部原始上下文几乎一样好。

<svg viewBox="0 0 640 260" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="320" y="24" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">滑窗 vs StreamingLLM 的 KV Cache 策略</text>

  <text x="20" y="55" fill="#ff6e6e" font-size="12" font-family="system-ui">❌ 普通滑动窗口</text>
  <rect x="20" y="65" width="600" height="40" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <rect x="20" y="65" width="60" height="40" rx="8" fill="#2a1e1e" stroke="#ff6e6e" stroke-width="1.5" stroke-dasharray="4,2"/>
  <text x="50" y="90" text-anchor="middle" fill="#ff6e6e" font-size="11" font-family="system-ui">被逐出</text>
  <rect x="300" y="65" width="320" height="40" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="460" y="90" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">滑动窗口保留的最近 token</text>
  <text x="140" y="90" text-anchor="middle" fill="#8a8a9a" font-size="11" font-family="system-ui">...中间也被逐出...</text>
  <text x="20" y="120" fill="#ff6e6e" font-size="11" font-family="system-ui">→ sink 消失 → PPL 爆炸</text>

  <text x="20" y="160" fill="#5eff9e" font-size="12" font-family="system-ui">✅ StreamingLLM</text>
  <rect x="20" y="170" width="60" height="40" rx="8" fill="#1e2a20" stroke="#5eff9e" stroke-width="2"/>
  <text x="50" y="195" text-anchor="middle" fill="#5eff9e" font-size="11" font-family="system-ui">固定保留</text>
  <rect x="90" y="170" width="180" height="40" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1" stroke-dasharray="4,2"/>
  <text x="180" y="195" text-anchor="middle" fill="#8a8a9a" font-size="11" font-family="system-ui">中间逐出</text>
  <rect x="280" y="170" width="340" height="40" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="450" y="195" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">滑动窗口保留的最近 token</text>
  <text x="20" y="225" fill="#5eff9e" font-size="11" font-family="system-ui">→ sink 永远在 → 可稳定处理 400 万+ token</text>
</svg>

## 第二部分：为什么模型会"学"出这种策略——理论解释

发现现象是一步，理解"为什么会这样"是另一步。StreamingLLM 提出了直觉解释，但没有严格证明。2025 年，一篇来自牛津大学和 Google DeepMind 的论文《Why do LLMs attend to the first token?》给出了更完整的理论框架。

### 问题是什么

要理解这篇论文的贡献，得先理解一个更基础的问题：**Transformer 叠了很多层之后，会有什么"副作用"？**

每一层的注意力机制,本质上都是在做"信息混合"——让每个 token 的表示,融合进它能看到的其他 token 的信息。如果只叠一两层,这没什么问题。但 LLM 动辄叠上几十层甚至上百层，每一层都在做混合，会发生什么？

图神经网络领域早就研究过一个类似的现象，叫 **over-squashing（过度压缩）**：当信息要在图上传播很多步时，来自远处节点的信息会被压缩得越来越模糊，最终淹没在噪声里，模型分不清哪些信息真正重要。类比到 Transformer：如果每一层都强制每个 token 去"平均"看它所有能看到的邻居，叠加几十层后，所有 token 的表示会逐渐变得彼此相似——这就是所谓的 **rank collapse（秩坍塌）**,一种比"表示坍塌"更强、更精确的退化条件：所有向量挤到了一个低维子空间里，模型区分不同 token 的能力大幅下降。

用一个更接地气的比喻：如果开会时每个人都必须把发言前所有人的话"平均消化"一遍才能发言,开到第 50 轮时,大家的观点已经被磨得几乎一模一样——没有人还记得自己最初想说什么了。

### 直觉：核心想法

作者的核心论点是：**attention sink 是模型进化出来对抗"过度混合"的一种防御机制。**

回到会议室类比：那个把大量权重丢给"坐在最前面的人"的做法，表面上看很浪费——分给一个不相关的人 90% 的注意力。但反过来想：这也意味着他只用剩下 10% 的权重去真正混合其他人的发言。**换句话说，把注意力集中到一个"安全、无害"的锚点上，实际上是在给自己保留"不被过度混合"的空间。**

论文里管这个叫构造出一个 **近似 no-op（近似的空操作）**：当某个注意力头把绝大部分注意力都给了 sink token 时，这个头对当前 token 的表示几乎没有做任何实质性的更新——上一层已经算好的、清晰的表示，被完好地保留了下来，没有被这一层的注意力操作"搅浑"。

这跟残差连接（Residual Connection）的哲学是相通的：残差连接允许网络选择"跳过"某一层、不做任何变换；而 attention sink 则允许**某个注意力头**在**某个特定的层**选择"不参与混合"，把决定权下放到了更细粒度的层面——不是整层跳过，而是某个头选择性地把自己的输出锁定为近似恒等映射。

这也解释了为什么模型越深、上下文越长，sink 现象越强——层数越多，"过度混合"的风险越大，模型就越需要这种"刹车机制"来保护表示不被稀释。作者在真实模型上验证了这一点：LLaMA 3.1 405B 里，竟有高达 **80%** 的注意力头形成了强烈的 sink 模式。

<svg viewBox="0 0 640 240" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="320" y="24" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">深层堆叠中的信息混合 vs Sink 保护</text>

  <text x="130" y="55" text-anchor="middle" fill="#ff9e6e" font-size="12" font-family="system-ui">无 sink：逐层过度混合</text>
  <rect x="30" y="65" width="60" height="30" rx="6" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="60" y="85" text-anchor="middle" fill="#ededf0" font-size="10">L1 表示</text>
  <line x1="90" y1="80" x2="150" y2="80" stroke="#ff9e6e" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <rect x="150" y="65" width="60" height="30" rx="6" fill="#241e1e" stroke="#ff9e6e" stroke-width="1.5"/>
  <text x="180" y="85" text-anchor="middle" fill="#ededf0" font-size="10">趋同</text>
  <line x1="210" y1="80" x2="270" y2="80" stroke="#ff9e6e" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <rect x="230" y="65" width="60" height="30" rx="6" fill="#241e1e" stroke="#ff9e6e" stroke-width="1.5"/>
  <text x="260" y="85" text-anchor="middle" fill="#ededf0" font-size="10">趋同</text>
  <text x="320" y="85" fill="#ff9e6e" font-size="14">→ rank collapse</text>

  <text x="130" y="150" text-anchor="middle" fill="#5eff9e" font-size="12" font-family="system-ui">有 sink：部分头执行 no-op，保留信息</text>
  <rect x="30" y="160" width="60" height="30" rx="6" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="60" y="180" text-anchor="middle" fill="#ededf0" font-size="10">L1 表示</text>
  <line x1="90" y1="175" x2="150" y2="175" stroke="#5eff9e" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <rect x="150" y="160" width="60" height="30" rx="6" fill="#1e2a20" stroke="#5eff9e" stroke-width="1.5"/>
  <text x="180" y="180" text-anchor="middle" fill="#5eff9e" font-size="10">保留</text>
  <line x1="210" y1="175" x2="270" y2="175" stroke="#5eff9e" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <rect x="230" y="160" width="60" height="30" rx="6" fill="#1e2a20" stroke="#5eff9e" stroke-width="1.5"/>
  <text x="260" y="180" text-anchor="middle" fill="#5eff9e" font-size="10">保留</text>
  <text x="320" y="180" fill="#5eff9e" font-size="14">→ 表示区分度保留</text>
  <text x="330" y="215" fill="#8a8a9a" font-size="11" font-family="system-ui">部分 head 把注意力权重丢给 sink token,相当于对当前 token 执行"近似不更新"</text>
</svg>

### 技术细节（选读）

论文进一步做了两个关键验证：

1. **深度和长度预测 sink 强度**：作者用一个精细化的 over-squashing 分析，预测"模型更深、训练上下文更长，sink 应该更强"。他们分别在自己预训练的模型上和 LLaMA 3.1 系列（8B/70B/405B）上验证了这个预测，趋势一致。

2. **`<bos>` 本身不特殊**：作者做了对照实验——无论预训练时 `<bos>` 是否被固定插入到序列第一个位置，sink 现象都会稳定地出现在序列的第一个位置（不一定是语义上的 `<bos>`）。但如果预训练阶段确实把 `<bos>` 固定在第一位，会影响模型具体"构造" sink 的方式（比如更依赖某些特定的低频位置编码分量）。

这个结论很重要：它说明 sink 不是因为模型"认出"了某个特殊 token 才格外关注它，而是**位置本身**——"对所有后续 token 永远可见的那个最靠前的位置"——天然具备成为安全垫底选项的资格,不管坐在那里的是谁。

## 第三部分：机制的另一半——massive activations

理论解释告诉我们"为什么模型想要这么做"，但还有一个问题没解决：**模型具体是怎么在数值层面实现"把注意力锁定到某个 token"这件事的?**

### 问题是什么

Attention 的核心运算是 query 向量和 key 向量做点积，得到一个分数，再经过 softmax。如果模型想让某个 token 无条件地获得高分——不管当前 query 是什么——它需要在数值上"作弊"：让这个 token 的 key 向量在某些维度上有一个极端的、和其余内容无关的固定值,这样只要它和任何 query 做点积,结果都会异常大。

2024 年一篇论文《Massive Activations in Large Language Models》直接在隐藏状态里找到了这种"作弊"的证据。

### 直觉：核心想法

研究者发现，LLM 内部隐藏状态里存在极少数几个激活值，大小是其他激活值的**十万倍**级别——这些异常巨大的数值被称为 **massive activations（巨量激活）**。

最关键的特征是：这些数值几乎不随输入内容变化——不管你输入的是诗歌、代码还是新闻，这几个位置上的数值大小几乎恒定。这说明它们根本没有在编码"这句话具体讲了什么"这种信息，而更像是模型内部的一种**固定偏置量（bias term）**——就像一个常年开着不熄的指示灯，跟当前处理的具体内容无关。

这正好和 attention sink 拼在了一起：因为这些巨量激活值几乎恒定，只要它们出现在某个 token（通常是序列第一个 token）的表示里，这个 token 在做 QK 点积时,天然就会在某些维度上产生异常高的分数——不管当前 query 想找什么，这个 token 都会"抢戏"。而 softmax 归一化之后，这就直接表现为该 token 吸收了不成比例的注意力权重。

用一个类比：如果会议室里那个坐在最前面的人，说话声音天生就比其他人大十万倍（不管他说了什么内容），那么在任何"注意力测量仪"上，他都会自动获得最高的关注度——这不是因为他说的话更重要,纯粹是物理声压的问题。

### 技术细节（选读）

论文指出了完整的因果链：**massive activations 出现在特定层的特定维度 → 导致注意力概率集中到对应 token（attention sink 现象）→ 进一步在 self-attention 输出中形成隐式偏置项，影响下游所有层。**

值得注意的是，这不只是语言模型的现象——论文同样在 Vision Transformer 里观察到了 massive activations，说明这是注意力这种架构本身的通用属性，而不是语言数据独有的产物。后续的研究（比如 Vision Transformer 的"register tokens"工作）进一步确认，在视觉模型里加入几个专门的"寄存器 token"（不对应任何真实图像 patch），可以主动承担这个"垫底"角色，从而让真正的图像 patch token 不必再承担这个副作用——这是把 sink 现象从"意外发现的 bug"变成"主动设计的功能"的一个直接例证。

而更早（2023 年，比 StreamingLLM 还早）来自 Qualcomm AI Research 的论文《Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing》，从量化压缩的角度独立发现了同样的问题：模型训练后会出现极端的激活值 outlier，这些 outlier 对应的注意力头，本质上就是在执行"什么都不做"的操作。他们提出的解法很有意思——直接改softmax，让它允许"裁剪到接近于负数"的输出（Clipped Softmax），或者给每个注意力头加一个显式的门控开关，让模型可以直接说"这个头不更新"，而不需要靠制造数值 outlier 这种迂回的方式去实现同样的效果。

这三条研究路径——StreamingLLM（工程角度）、Barbero et al.（信息论角度）、Massive Activations + Quantizable Transformers（数值机制角度）——分别独立地收敛到了同一个结论：**attention sink 本质上是模型在"必须给出一个和为 1 的概率分布"这条数学铁律下，为了实现"某些位置该被跳过、不该被混合"这一诉求，被迫发明出来的一种数值把戏。**

## 第四部分：这件事到底重要在哪——真实世界的连锁反应

理解了成因，我们再回头看这个现象为什么值得整整一篇文章来讲——因为它牵连了 LLM 工程里四个完全不同的痛点。

**1. 长文本流式推理（已经讲过）**：如果做 KV cache 淘汰策略时不小心把 sink token 挤掉了,模型会直接崩溃。这也是为什么现在几乎所有做长上下文/流式推理优化的系统（H2O、MInference、DuoAttention 等）都会显式地把"保留 sink token"作为一条硬性策略写进代码里，而不是任其自然发生。

**2. 模型量化压缩**：把模型压缩成低比特整数（int8、int4）来省显存、加速推理时，需要给每一层的激活值确定一个数值范围。而 massive activations 和它们导致的极端 outlier，会把这个范围硬生生撑大几个数量级，导致除了这几个"巨量"数值外，其余所有正常数值的量化精度被严重压低。这就是为什么"量化感知保护"（quantization-aware protection）几乎是所有量化框架的标配——必须识别出这些 sink/pivot token，单独用更高精度处理它们，其余 token 才能被大胆地压缩。

**3. 安全与越狱**：研究者发现,操纵输入序列开头几个 token 的内容或位置,可以异常有效地影响模型后续的行为——因为这些位置天然占据了不成比例的"话语权"。这为分析某些提示注入（prompt injection）和越狱攻击的机制提供了一个新的切入角度：攻击者某种程度上是在利用模型自己发明的这个"权力过度集中"的架构漏洞。

**4. 从 bug 到 feature 的转变**：Diffusion Transformer（DiT）架构里出现的"Sink Registers"，以及 Vision Transformer 里的"register tokens"，都是主动把这个现象设计进架构里——专门留出几个不承载真实信息的"垫底"位置，让真正有意义的 token 不再需要靠制造数值异常来自我保护。这标志着 attention sink 正在经历一次身份转变：从"训练意外产生的怪癖"变成"经过验证有效、值得主动设计进架构里的机制"。

<svg viewBox="0 0 640 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="320" y="24" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">Attention Sink 的四个下游影响</text>
  <rect x="20" y="45" width="140" height="60" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="90" y="70" text-anchor="middle" fill="#ededf0" font-size="12">流式推理 /</text>
  <text x="90" y="88" text-anchor="middle" fill="#ededf0" font-size="12">KV Cache</text>

  <rect x="180" y="45" width="140" height="60" rx="8" fill="#1e1e2a" stroke="#ffb86e" stroke-width="1.5"/>
  <text x="250" y="70" text-anchor="middle" fill="#ededf0" font-size="12">量化压缩</text>
  <text x="250" y="88" text-anchor="middle" fill="#ededf0" font-size="12">(GPTQ/AWQ)</text>

  <rect x="340" y="45" width="140" height="60" rx="8" fill="#1e1e2a" stroke="#ff6e9e" stroke-width="1.5"/>
  <text x="410" y="70" text-anchor="middle" fill="#ededf0" font-size="12">安全性 /</text>
  <text x="410" y="88" text-anchor="middle" fill="#ededf0" font-size="12">提示注入</text>

  <rect x="500" y="45" width="120" height="60" rx="8" fill="#1e1e2a" stroke="#5eff9e" stroke-width="1.5"/>
  <text x="560" y="70" text-anchor="middle" fill="#ededf0" font-size="12">主动架构设计</text>
  <text x="560" y="88" text-anchor="middle" fill="#ededf0" font-size="12">(Register Token)</text>

  <line x1="320" y1="130" x2="90" y2="105" stroke="#6e8eff" stroke-width="1.2" marker-end="url(#arrow3)"/>
  <line x1="320" y1="130" x2="250" y2="105" stroke="#6e8eff" stroke-width="1.2" marker-end="url(#arrow3)"/>
  <line x1="320" y1="130" x2="410" y2="105" stroke="#6e8eff" stroke-width="1.2" marker-end="url(#arrow3)"/>
  <line x1="320" y1="130" x2="560" y2="105" stroke="#6e8eff" stroke-width="1.2" marker-end="url(#arrow3)"/>
  <rect x="240" y="140" width="160" height="40" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="2"/>
  <text x="320" y="165" text-anchor="middle" fill="#6e8eff" font-size="12" font-weight="bold">Attention Sink</text>
</svg>

## 这意味着什么

回顾一下我们走过的路：一个看似"应该没有大问题"的架构选择——softmax 强制所有注意力权重加起来等于 1——在深度堆叠、长序列训练的现实条件下，逼出了一个意想不到的"漏洞"：模型没有"弃权"选项，只能把多余的注意力权重倾倒进某个固定、安全的位置。而这个位置几乎总是序列里第一个 token,因为它是唯一一个对所有后续位置永远可见的锚点。

更有意思的是，这个"漏洞"其实是一种优雅的自我保护机制：它让模型能够在必须服从"归一化"这条数学铁律的同时，依然保留住部分注意力头对当前 token 表示"什么都不做"的自由——从而对抗深层网络里信息被过度混合、过度平均化的风险。

沿着"为什么"这条线追下去，我们看到理论（over-mixing / rank collapse）、机制（massive activations）、工程验证（量化、softpick 消融实验）三条完全独立的研究路径,不约而同地指向同一个结论。这在研究里是很难得的——通常意味着我们真正抓住了问题的本质,而不是一个表面的相关性。

也正因为理解了这个机制，工程界才能从"被动防御"（保留 sink token 别删掉）走向"主动设计"（专门设计寄存器 token、修改 softmax 函数本身）。这是一个典型的"从观察现象到理解机制到主动利用"的科研闭环,也提醒我们:LLM 训练过程中涌现出来的很多"奇怪行为",往往不是随机的噪音,而是模型在既定约束下找到的、值得我们认真对待的解决方案。

## 延伸阅读

- Xiao et al. (2023), *Efficient Streaming Language Models with Attention Sinks*, ICLR 2024 — [arxiv.org/abs/2309.17453](https://arxiv.org/abs/2309.17453)
- Barbero et al. (2025), *Why do LLMs attend to the first token?* — [arxiv.org/abs/2504.02732](https://arxiv.org/abs/2504.02732)
- Sun et al. (2024), *Massive Activations in Large Language Models* — [arxiv.org/abs/2402.17762](https://arxiv.org/abs/2402.17762)
- Bondarenko et al. (2023), *Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing*, NeurIPS 2023 — [arxiv.org/abs/2306.12929](https://arxiv.org/abs/2306.12929)
- Softpick (2025), *No Attention Sink, No Massive Activations with Rectified Softmax* — [arxiv.org/abs/2504.20966](https://arxiv.org/abs/2504.20966)
- *Attention Sink in Transformers: A Survey on Utilization, Interpretation, and Mitigation* (2026) — [arxiv.org/abs/2604.10098](https://arxiv.org/abs/2604.10098)
- *When Attention Sink Emerges in Language Models: An Empirical View*, ICLR 2025 — [arxiv.org/abs/2410.10781](https://arxiv.org/abs/2410.10781)
