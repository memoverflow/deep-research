---
title: "AI 吃 AI 会变傻吗？模型坍缩的数学真相与被夸大的恐慌"
date: 2026-07-25
level: 3
series: "LLM 原理深度解析"
series_order: 34
series_total: 53
tags: [model-collapse, synthetic-data, 训练数据, 统计学习理论, scaling-law]
summary: "当模型开始吃自己生成的数据训练自己，会发生什么？'模型坍缩'背后有严谨的数学证明，但现实世界的坍缩风险，可能没有新闻标题说得那么可怕。"
---

> 2024 年，Nature 上一篇论文的标题是"AI models collapse when trained on recursively generated data"（AI 模型在递归训练自身生成数据时会坍缩）。媒体的解读是："互联网正在被 AI 生成内容淹没，未来的 AI 会因为吃了太多 AI 垃圾而变傻。"这个故事听起来很吓人，但它对吗？

## 故事从这里开始

想象一个游戏：你找一百个人玩"传话游戏"，第一个人听到一句话，转述给第二个人，第二个人转述给第三个人……到第一百个人的时候，这句话大概已经面目全非了。每次转述都会丢失一点细节，加入一点误解，经过足够多轮之后，原始信息基本消失。

现在把"人"换成"AI 模型"。ChatGPT 刚出来的时候，互联网上的文字几乎全是人写的。但过了几年，情况变了——今天你随便打开一个博客、一条社交媒体评论、一篇产品说明，很可能是 AI 生成或 AI 辅助写的。而下一代 AI 模型，正是靠"爬取整个互联网"来训练的。

这就带来一个诡异的问题：**如果模型 A 生成了大量文本，这些文本混进了互联网，模型 B 又拿这些混合了 AI 生成内容的互联网数据去训练自己，那么模型 B 会不会像传话游戏里的最后一个人一样，学到的东西越来越走偏？**

2023 年，牛津大学的 Ilia Shumailov 团队发了一篇论文，专门研究这个问题，起了个名字叫"model collapse"（模型坍缩）。他们做了一个很直观的实验：拿一个语言模型在一段维基百科文本上反复"自我蒸馏"——用第 0 代模型生成的文本去训练第 1 代模型，再用第 1 代生成的文本训练第 2 代，一直循环到第 9 代。结果令人不安：

第 0 代的输出还算连贯，讲的是"中世纪教堂建筑的历史"；到第 7 代,输出已经开始跑题,变成了一段关于"接受采访"的莫名其妙的话;到第 9 代,模型输出彻底崩坏成一段语义上毫无意义的重复列表——**"这里生活着世界上数量最多的黑尾长耳大野兔、白尾长耳大野兔、蓝尾长耳大野兔、红尾长耳大野兔、黄尾……"**

这不是段程序 bug，这是这个模型经过 9 代自我训练之后，"忘记"了怎么好好说话,只剩下对某个高频词模式的病态偏爱。

这个现象被 Nature 报道后引发了广泛焦虑：互联网正在被 AI 内容污染，未来的模型会不会集体"变傻"？但就像很多引人注目的科学发现一样，真相比标题复杂得多——而理解这个复杂真相，恰好是一次很好的统计学习理论教学机会。

<svg viewBox="0 0 700 220" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <rect x="10" y="80" width="120" height="55" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="70" y="103" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">真实数据</text>
  <text x="70" y="120" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">(人类文本)</text>

  <line x1="130" y1="107" x2="180" y2="107" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>

  <rect x="185" y="80" width="120" height="55" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="245" y="103" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">模型 Gen 0</text>
  <text x="245" y="120" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">训练</text>

  <line x1="305" y1="107" x2="355" y2="107" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>

  <rect x="360" y="80" width="120" height="55" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="420" y="103" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">生成数据</text>
  <text x="420" y="120" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">(采样)</text>

  <line x1="480" y1="107" x2="530" y2="107" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>

  <rect x="535" y="80" width="120" height="55" rx="8" fill="#1e1e2a" stroke="#ff6e6e" stroke-width="1.5"/>
  <text x="595" y="103" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">模型 Gen 1</text>
  <text x="595" y="120" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">用生成数据训练</text>

  <path d="M 595 135 C 595 190, 245 190, 245 135" stroke="#ff6e6e" stroke-width="1.5" fill="none" marker-end="url(#arrow1)"/>
  <text x="420" y="205" text-anchor="middle" fill="#ff6e6e" font-size="12" font-family="system-ui">递归循环：一代一代重复这个过程</text>
</svg>

## 坍缩的两个数学根源

要理解为什么会坍缩，得先搞清楚一件事：**一个语言模型永远不是它所训练数据的完美复制品。** 每一次"从数据学出一个模型，再从模型采样出新数据"，都会经过两道会走样的关卡。

### 第一道关卡：统计近似误差（主犯）

### 问题是什么

假设有一枚真实的硬币，正面概率恰好是 50%。你抛 10 次，得到 6 次正面、4 次反面——这很正常，抽样本身就有随机波动。如果你现在把"抛 10 次得到 6 正 4 反"当作新的"真实规律"去教下一个人,他学到的硬币就已经不是 50/50 了，而是 60/40。

这就是**统计近似误差**的本质：任何有限次抽样，都无法完美还原背后的概率分布，抽样次数越少，偏差越大。哪怕样本量已经很大，偏差也不会精确归零——论文里做了个实验，用一千万个样本去估计一个标准正态分布（真实均值是 0）的均值，估计出来的结果仍然是 0.00024899，而不是精确的 0。这个误差很小，但绝不为零。

### 核心直觉

现在把这件事套进"模型训练→采样→再训练"的循环里：模型 Gen 0 从真实数据里学到一个近似分布,然后从这个近似分布里采样出一批"合成数据"喂给 Gen 1。Gen 1 学到的分布，天生就带着 Gen 0 那次抽样引入的偏差。Gen 1 再采样喂给 Gen 2，Gen 2 又在 Gen 1 的偏差基础上叠加了自己新的抽样偏差……

这就像那个传话游戏——不是因为某一个人特别笨,而是因为每一次转述都存在不可避免的随机误差，误差在代际之间累积、复合、可能被进一步放大。数学上，这个过程可以被建模成一个**随机游走**：每一代模型的分布参数（比如均值、方差）相对于上一代做一次小的随机跳动，而这些跳动没有"回归原点"的力，只会越走越远，直到方差要么塌缩到 0（模型输出变得高度重复、单一），要么发散到失控。

### 技术细节（选读）

论文构造了一个"用生成数据学习"（Learning with Generational Data）的随机过程模型：第 i 代数据集由分布 p_i 生成，通过函数近似 F_θ: p_i → p_{θ(i+1)}，再从

  p_{i+1} = α_i · p_{θ(i+1)} + β_i · p_i + γ_i · p_0

采样出下一代数据（这个混合公式允许"完全替换"、"部分累积旧数据"、"保留一点原始数据"三种策略的任意组合）。在单维高斯的简化情形下，可以精确推导出方差随代数的演化规律——翻译回人话：**每一代都在原有基础上叠加一次随机扰动，扰动没有被系统性抵消的机制，于是方差会像喝醉了走路一样，随着代数增加越走越偏。**

### 第二道关卡：函数近似误差（配角）

模型本身的表达能力也不是无限的。想象你只有一个"单峰"的高斯分布模型，却要去拟合一个真实是"双峰"（两群人身高分布，例如男女分开算）的数据——无论你给多少数据，单峰模型永远学不出双峰的形状，它会把两个峰强行"揉"成一个偏斜的单峰。这就是函数近似误差：**模型的假设空间（它能表达的所有可能分布的集合）如果不包含真实分布，就永远存在这道无法消除的系统性偏差。**

好消息是，如果没有统计误差这道关卡（比如你有无限数据），函数近似误差只会在第一代出现一次——一旦模型稳定收敛到它能表达的最佳近似，后面代际不会继续恶化。真正让坍缩持续加剧、代代变差的，是统计误差的复合效应。

<svg viewBox="0 0 680 260" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:680px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="340" y="24" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">分布的尾部如何一步步消失</text>

  <!-- Gen0 -->
  <text x="90" y="55" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">Gen 0（原始分布）</text>
  <path d="M 20 130 Q 90 40 160 130" stroke="#6e8eff" stroke-width="2" fill="none"/>
  <line x1="20" y1="130" x2="160" y2="130" stroke="#3a3a4a" stroke-width="1"/>

  <line x1="180" y1="90" x2="220" y2="90" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>

  <!-- Gen3 -->
  <text x="320" y="55" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">Gen 3（尾部变窄）</text>
  <path d="M 260 130 Q 320 55 380 130" stroke="#a78bfa" stroke-width="2" fill="none"/>
  <line x1="260" y1="130" x2="380" y2="130" stroke="#3a3a4a" stroke-width="1"/>

  <line x1="400" y1="90" x2="440" y2="90" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>

  <!-- Gen9 -->
  <text x="560" y="55" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">Gen 9（几乎坍缩为一点）</text>
  <path d="M 530 130 Q 560 70 590 130" stroke="#ff6e6e" stroke-width="3" fill="none"/>
  <line x1="480" y1="130" x2="640" y2="130" stroke="#3a3a4a" stroke-width="1"/>

  <text x="340" y="180" text-anchor="middle" fill="#a0a0b0" font-size="12" font-family="system-ui">每一代都在丢弃概率密度较低的"罕见样本"（分布的尾部）</text>
  <text x="340" y="200" text-anchor="middle" fill="#a0a0b0" font-size="12" font-family="system-ui">经过多代累积后，模型只会输出高频、常见、"安全"的内容</text>
  <text x="340" y="225" text-anchor="middle" fill="#a0a0b0" font-size="12" font-family="system-ui">这就是"jackrabbit 现象"背后的几何图像：分布方差趋于 0</text>
</svg>

## 事情没那么简单：数据"替换"和"累积"是两回事

如果坍缩真的这么不可避免，为什么我们今天用的 GPT、Claude、Gemini 并没有明显变傻？这里有个关键的细节，Shumailov 原始论文里的实验设计，容易被媒体报道忽略。

### 问题是什么

原始实验里，每一代模型的训练数据都是**完全替换**的：Gen 1 只用 Gen 0 生成的数据训练，不掺杂任何真实数据。这在数学上等价于——你把传话游戏里"原始那句话的文字记录"直接烧掉，只留下每一轮口头转述的记忆。当然到最后什么都不剩。

但现实世界的互联网不是这样运作的。今天写在维基百科上的一句话，明天不会因为有人发了篇 AI 生成的博客就被删除。**真实世界的数据是累积的，不是替换的。**

### 核心直觉

2024 年 Stanford 团队的 Gerstgrasser 等人做了一个很关键的后续研究，专门检验"如果不删除真实数据，只是往里面加合成数据"会发生什么。结果是：**只要保留原始真实数据、把合成数据当作补充而非替代，坍缩就可以被数学上证明是有界的（不会无限恶化下去）。**

用一个比喻理解为什么会有本质区别：如果传话游戏里每一轮都允许你回头看一眼原始纸条（哪怕只是偶尔看一眼），信息就不会无限失真——因为总有一个"锚点"在不断校正积累的误差。累积策略正是给了模型这样一个"锚点"：真实数据始终占据训练集的一部分，且随着代数增加，早期引入的噪声在训练集中所占的权重会被后续不断加入的新数据稀释，而不是被放大。

### 技术细节（选读）

在线性回归这个可以精确求解的简化设定下，累积策略下第 n 代模型的测试误差满足：

  E_test(ŵ_n) = (σ²d)/(T-d-1) · Σ_{i=1}^{n} 1/i²

翻译回人话：每一代新增的噪声项被 1/i² 加权，i 越大（代数越晚）权重越小。这个级数 Σ 1/i² 是数学里一个经典的收敛级数，随着 n → ∞ 它收敛到 π²/6 ≈ 1.645，也就是说——**即使无限代地训练下去,测试误差也有一个封顶的上界，不会像"替换"策略下那样随代数线性增长直到彻底崩溃。**

这个证明的意义远大于它看起来的技术细节：它把"model collapse 是否不可避免"从一个纯经验问题，变成了一个和"你的训练协议设计（累积 vs 替换）"直接相关的工程决策问题。

<svg viewBox="0 0 700 280" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <text x="350" y="24" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">替换 vs 累积：误差随代数的演化</text>

  <!-- axes -->
  <line x1="70" y1="230" x2="650" y2="230" stroke="#3a3a4a" stroke-width="1.5"/>
  <line x1="70" y1="230" x2="70" y2="50" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="360" y="255" text-anchor="middle" fill="#a0a0b0" font-size="12" font-family="system-ui">训练代数 n</text>
  <text x="35" y="140" text-anchor="middle" fill="#a0a0b0" font-size="12" font-family="system-ui" transform="rotate(-90 35 140)">测试误差</text>

  <!-- replace curve: grows linearly/unbounded -->
  <path d="M 70 220 L 150 190 L 230 150 L 310 110 L 390 75 L 470 55 L 550 40 L 630 30"
        stroke="#ff6e6e" stroke-width="2.5" fill="none"/>
  <text x="590" y="25" fill="#ff6e6e" font-size="12" font-family="system-ui">替换策略：无界发散</text>

  <!-- accumulate curve: converges -->
  <path d="M 70 220 L 150 195 L 230 180 L 310 172 L 390 168 L 470 166 L 550 165 L 630 165"
        stroke="#6e8eff" stroke-width="2.5" fill="none"/>
  <text x="590" y="180" fill="#6e8eff" font-size="12" font-family="system-ui">累积策略：收敛到 π²/6 上界</text>

  <line x1="70" y1="165" x2="630" y2="165" stroke="#3a3a4a" stroke-width="1" stroke-dasharray="4,4"/>
</svg>

## 那么谁说得更对：Nature 论文还是反驳论文？

这里必须诚实地面对一个复杂局面：2025 年，Rylan Schaeffer 等人发了一篇很尖锐的"立场论文"（position paper），标题直接叫《模型坍缩不是你想的那样》。他们指出一个很扎实的批评：**"model collapse"这个词在不同论文里实际有 8 种不同、有时互相矛盾的定义。** 有些研究说的坍缩是"输出多样性下降"，有些说的是"生成质量下降"，有些说的是"分布支撑集收缩"——这些是不同的现象，却被同一个词笼统地指代。

更重要的是，他们重新审视了那些"证明坍缩不可避免"的研究方法，发现很多研究依赖的假设本身脱离现实：完全替换数据、完全不过滤合成数据、假设无限代数递归下去。而现实中的大模型训练流程，通常会有 RLHF、reward model 过滤、人工审核这些"质检关卡"，不会盲目吞下互联网上所有 AI 生成的文字。

这不是说 model collapse 的数学证明是错的——恰恰相反，前面讲的统计误差随机游走、方差收缩，这些都是被严格证明成立的数学事实,在受控实验里也确实能复现出"jackrabbit 现象"那样的退化。**真正有争议的是：这套数学结论套用到"整个互联网 + 所有大模型公司的训练流程"这个复杂现实系统时，到底有多严重、多紧迫。** 这就像"复利公式"是数学真理,但"你会不会因为复利而破产"取决于你的具体借贷条件——公式本身不会说谎，但套用场景的假设匹配度决定了结论的现实分量。

2025 年之后的一批研究也在往"缓解"方向努力：比如引入"验证器"（verifier）对合成数据先做质量筛选，再喂给模型训练——这确实能延缓坍缩,但也带来新问题：如果验证器本身有偏差，长期迭代后模型会被拉向"验证器的知识中心",而不是真实世界的知识中心。这提示我们，坍缩问题并不会被一个简单的补丁彻底解决,而是变成了一系列需要持续工程决策的权衡：数据要不要保留、要不要过滤、过滤器本身可信吗、真实数据的比例底线是多少。

## 这意味着什么

回到开头那个吓人的新闻标题："AI 模型在吃自己生成的数据后会坍缩。"现在你可以给出一个更精确的回答：

**数学上，是的**——如果你把真实数据完全替换成 AI 生成数据、不加任何过滤、无限代数递归下去，坍缩是可以被严格证明会发生的，其根源是有限样本采样带来的统计误差在代际间累积成一个没有回归力的随机游走，尾部信息（罕见但重要的样本）率先消失，最后方差趋于坍缩。

**但现实中，情况没那么绝望**——因为真实世界的训练协议不是简单的"代际替换"：数据是累积而非替换的，合成数据会经过筛选，人类产出的高质量内容仍持续注入互联网。理论证明也表明，"累积"这个简单的工程决策就能把发散的误差变成有界的误差。这也是为什么今天的模型没有像最初实验那样在几代之内就退化成"jackrabbit 列表"。

这背后其实藏着一个更普遍的教训：**任何"AI 训练 AI"的自我循环系统，都需要一个不被这个循环污染的外部真实性锚点。** 无论是保留人类数据、引入外部验证、还是持续采集真实世界反馈,本质上都是在给这个循环装一个刹车片。这个原则不只适用于语言模型——推荐系统靠用户点击数据训练、再用模型生成的内容影响用户点击，同样面临类似的自我强化循环问题。理解 model collapse 的数学机制，其实是在理解所有"自我参照学习系统"共同的脆弱性。

## 参考来源

1. Shumailov et al., "The Curse of Recursion: Training on Generated Data Makes Models Forget" (2023, arXiv:2305.17493)
2. Shumailov et al., "AI models collapse when trained on recursively generated data" (Nature, 2024)
3. Gerstgrasser et al., "Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data" (2024, arXiv:2404.01413)
4. Dohmatob, Feng, Kempe, "Model Collapse Demystified: The Case of Regression" (NeurIPS 2024, arXiv:2402.07712)
5. Schaeffer et al., "Position: Model Collapse Does Not Mean What You Think" (2025, arXiv:2503.03150)
6. "Escaping Model Collapse via Synthetic Data Verification: Near-Optimal Rates" (2025, arXiv:2510.16657)
