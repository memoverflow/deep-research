---
title: "秩坍缩：如果没有残差连接，Transformer 会把所有词变成同一个词"
date: 2026-07-13
level: 3
series: "LLM 原理深度解析"
series_order: 39
series_total: 39
tags: [rank-collapse, over-smoothing, residual-connection, self-attention, token-uniformity, transformer-depth]
summary: "纯注意力网络会以双指数速度把所有 token 的表示压成同一个向量——这不是训练事故，是数学定理。残差连接和 MLP 才是真正阻止这件事发生的英雄。"
---

> 一个只有注意力、没有残差连接的 Transformer，深度每增加一层，所有 token 的表示就会以恐怖的速度变得越来越像——像到最后，模型读进去"猫追狗"和"狗追猫"，吐出来的中间表示居然是同一个向量。

## 故事从这里开始

2017 年那篇论文的标题喊得响亮：《Attention Is All You Need》。这句话后来几乎成了整个深度学习领域的座右铭——只要有注意力机制，其他的都是细节。

但四年后，2021 年,三位研究者（Yihe Dong、Jean-Baptiste Cordonnier、Andreas Loukas）写了一篇论文，标题几乎是在跟前者对着骂：《Attention is Not All You Need》。他们证明了一件挺吓人的事：如果你把 Transformer 里除了自注意力之外的东西全部拿掉——没有残差连接，没有 MLP，只留纯粹的多头自注意力堆叠起来——那么随着层数增加,模型的输出会以"双指数"的速度坍缩成一个秩为 1 的矩阵。

秩为 1 是什么意思？直白点说：不管你输入的句子里有多少个不同的词，跑过足够多层纯注意力之后,每一个词位置输出的向量都会趋向于完全相同。"我"、"爱"、"你"这三个原本应该携带不同信息的词，最后会变成三个几乎无法区分的向量。模型看什么都是一个样子——这在信息论意义上等于模型瞎了。

更吓人的是"双指数"这个词的分量。线性收敛是"每次减少一半"，指数收敛是"每次减少到平方"，而这里的收敛速度是三次方级别的——论文里给出的具体数字是：正常情况下要把某个量从 1000 降到 1 大概需要十几次线性迭代，而这种坍缩只需要两三层。也就是说，如果不做任何补救，一个 10 层的纯注意力网络,可能在还没翻过第三层的时候,所有 token 的表示就已经混成一锅粥了。

那为什么现实中的 GPT、BERT、LLaMA 这些跑了几十层甚至上百层的模型没有变成"信息一锅粥"？答案就是这篇文章要讲的核心悬念——真正撑住这栋楼的，不是注意力，而是那两个看起来毫不起眼的组件：残差连接（skip connection）和 MLP 层。这篇论文把"注意力是万能的"这个神话,换成了一个更精确、也更谦逊的真相。

<svg viewBox="0 0 640 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrowA" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="320" y="24" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">纯注意力网络：token 表示随深度坍缩</text>
  <rect x="20" y="50" width="80" height="110" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="60" y="75" text-anchor="middle" fill="#ededf0" font-size="11">猫</text>
  <text x="60" y="95" text-anchor="middle" fill="#6e8eff" font-size="11">追</text>
  <text x="60" y="115" text-anchor="middle" fill="#ededf0" font-size="11">狗</text>
  <text x="60" y="135" text-anchor="middle" fill="#ededf0" font-size="10">(不同向量)</text>
  <text x="60" y="180" text-anchor="middle" fill="#8a8a99" font-size="10">Layer 0</text>

  <line x1="105" y1="105" x2="165" y2="105" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrowA)"/>

  <rect x="170" y="60" width="80" height="90" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="210" y="90" text-anchor="middle" fill="#ededf0" font-size="10">渐渐趋同</text>
  <text x="210" y="110" text-anchor="middle" fill="#ededf0" font-size="10">...</text>
  <text x="210" y="180" text-anchor="middle" fill="#8a8a99" font-size="10">Layer 2</text>

  <line x1="255" y1="105" x2="315" y2="105" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrowA)"/>

  <rect x="320" y="70" width="80" height="70" rx="8" fill="#1e1e2a" stroke="#e05a5a" stroke-width="1.5"/>
  <text x="360" y="100" text-anchor="middle" fill="#e05a5a" font-size="10">几乎相同</text>
  <text x="360" y="180" text-anchor="middle" fill="#8a8a99" font-size="10">Layer 4</text>

  <line x1="405" y1="105" x2="465" y2="105" stroke="#e05a5a" stroke-width="1.5" marker-end="url(#arrowA)"/>

  <rect x="470" y="85" width="90" height="40" rx="8" fill="#2a1a1a" stroke="#e05a5a" stroke-width="2"/>
  <text x="515" y="110" text-anchor="middle" fill="#e05a5a" font-size="11" font-weight="bold">秩 = 1</text>
  <text x="515" y="180" text-anchor="middle" fill="#8a8a99" font-size="10">Layer 6+</text>
</svg>

## 秩坍缩到底是什么：从"一群人达成共识"讲起

### 问题是什么

先抛开公式，想象一个场景：一个会议室里有 20 个人，每个人对某个议题都有自己独特的看法。会议规则是——每一轮，每个人都要"听取"房间里所有人的意见，然后把自己的看法更新成"大家意见的某种加权平均"。

第一轮之后，极端的看法会被拉向中间。第二轮之后，剩下的分歧继续缩小。你多开几轮会，会发生什么？直觉上，如果每次更新纯粹是"加权平均"，那么最终所有人的看法会趋向于收敛成同一个值——房间里 20 个人变成了 20 个说着同一句话的复读机。

这正是自注意力机制的数学结构：注意力矩阵（softmax 之后的那个概率分布矩阵）每一行都是非负、和为 1 的权重——这在数学上叫"行随机矩阵"（row-stochastic matrix）。用这样一个矩阵去乘以 token 的表示矩阵，本质上就是在做"加权平均"操作。而"反复做加权平均"这件事,在数学上早就有名字——它会把所有向量拉向同一个点。这跟马尔可夫链最终收敛到平稳分布是同一类现象,也跟图神经网络里那个让人头疼的"过平滑"（over-smoothing）问题是同一件事的不同马甲。

### 直觉：为什么"多头"、"多层"都救不了这个问题

你可能会想：那我们不是有很多"头"（multi-head）,每个头看的角度都不一样吗？多头不是应该能保留多样性吗？

论文里给出的分析结果有点反直觉：多头和多层反而让坍缩变得更快，不是更慢。原因在于一个叫"路径分解"（path decomposition）的巧妙数学工具——这是这篇论文最核心的贡献,值得专门讲一讲。

想象整个多层多头的自注意力网络，是一个"关卡迷宫"：每一层有 H 个头（H 个"门"），一个 token 的信息要从第 1 层走到第 L 层，等价于在每一层选一个门穿过去，一共有 $H^L$ 种不同的穿门方式，每一种方式叫一条"路径"（path）。论文证明了一个精确的等式：整个网络的输出,恰好等于所有这些路径各自贡献的结果加总起来。

这个分解很漂亮，因为它把一个复杂的多层网络问题，拆成了研究"单条路径会怎样"的简单问题——而单条路径,其实就是一串"行随机矩阵"依次相乘的结果。数学上有个很干净的性质：行随机矩阵乘行随机矩阵,还是行随机矩阵；而这一串矩阵反复相乘,恰好就是不断做加权平均,平均次数越多，坍缩越彻底。

关键的反直觉之处在于：秩坍缩的速度不是线性叠加的，而是"三次方"级别加速的。论文给出的解释是——注意力矩阵本身的形成,又依赖于输入的秩。当输入已经开始变得"扁"（低秩）时，由它算出来的注意力权重会变得更极端、更集中（因为不同 token 之间的差异变小了，softmax 更容易给出近乎一样的分布）,而更集中的权重反过来又会让下一层的输出更扁。这是一个自我加强的恶性循环：坍缩得越厉害，接下来坍缩得就越快。层数增加，路径数量确实是指数增长的（$H^L$ 条），但每一条路径本身的坍缩速度是"双指数"的——路径数量的增长完全跑不过坍缩速度的增长。这就是为什么"多头"、"多层"不但没能阻止坍缩，反而把整个系统推向了坍缩得更快的方向。

### 技术细节：路径分解与收敛速率（选读）

单头自注意力层的输出可以写成：

$$
\text{SA}(\bm{X}) = \bm{P}\bm{X}\bm{W}_V
$$

其中 $\bm{P}$ 是 softmax 得到的行随机注意力矩阵。多头、多层堆叠展开后，可以严格证明输出等价于：

$$
\text{SAN}(\bm{X}) = \sum_{\text{path} \in [H]^L} \bm{P}_{\text{path}} \, \bm{X} \, \bm{W}_{\text{path}}
$$

翻译回人话：整个网络的输出，是把所有可能的"跨层选头方式"（一共 $H^L$ 种）各自的贡献加起来。每一种选头方式对应一个"路径矩阵" $\bm{P}_{\text{path}}$，它是一串行随机矩阵连乘的结果——而连乘行随机矩阵,天然就会把矩阵推向秩更低的方向。

定义"残差" $\text{res}(\bm{X}) = \bm{X} - \mathbf{1}\bm{x}^\top$（也就是 $\bm{X}$ 减去它每一列的均值，衡量各 token 表示之间"到底有多不一样"），论文的核心定理给出：

$$
\|\text{res}(\text{SAN}(\bm{X}))\|_{1,\infty} \leq \left(\frac{4\gamma\beta H}{\sqrt{d_{qk}}}\right)^{\frac{3^L - 1}{2}} \|\text{res}(\bm{X})\|_{1,\infty}^{3^L}
$$

这个公式看起来吓人，但核心信息很简单：右边残差的指数是 $3^L$——层数每增加 1，指数就乘 3。这就是"双指数"收敛的来源：不是每层减少固定比例（那是线性/指数收敛），而是每层把"还剩多少差异"这个量本身拿去做三次方，差异消失的速度呈爆炸式加快。

## 那为什么真实的 Transformer 没有坍缩：残差连接才是真英雄

### 问题是什么

如果秩坍缩这么猛烈，那为什么 GPT-4、LLaMA 这些跑几十层的真实模型输出仍然是有意义的、每个 token 依然携带独特信息？答案必须在 Transformer 相对于"纯注意力网络"多出来的三个组件里找：残差连接、MLP、LayerNorm。论文对这三者逐一做了拆解，结论出乎很多人的意料。

### 直觉：残差连接给了每个 token 一条"永不被稀释"的备用通道

回到路径分解的框架里加入残差连接后会发生什么？残差连接相当于在每一层都新增了一个"什么都不做，直接跳过这一层"的选项。数学上，这意味着路径的集合不再只有"必须穿过全部 L 层的门"这一种可能，还包括"跳过第 3 层、第 7 层，只穿过其余层"等等所有组合方式。

这带来一个关键的结构变化：现在存在一条特殊的路径——完全跳过所有层，长度为 0 的路径。这条路径的贡献恰好等于原始输入本身，一点没被"平均"污染。论文证明了一个简洁却有力的结论：只要这条"全跳过"路径存在，网络的输出残差就永远不可能收敛到 0——哪怕网络无限深。换句话说，残差连接给每个 token 留了一条"直达电梯"，不管经过多少层加权平均的"稀释"，原始信息总有一部分能原封不动地传到最后。

这也解释了论文里一个很有意思的推论：带残差连接的深层 Transformer，行为上更像是很多"浅层网络的集成（ensemble）"，而不是一个真正意义上的"深"网络。因为残差连接极大丰富了短路径的数量（$\binom{L}{l} \cdot H^l$ 条长度为 $l$ 的路径），而短路径受到的"稀释"远比长路径轻——所以网络的输出更多是被这些短路径主导的。这个洞察跟 ResNet 领域早年"深度残差网络其实类似浅层网络集成"的发现是同一件事的不同侧面——只不过 Dong 等人第一次把它和秩坍缩联系了起来。

### 直觉：MLP 帮忙，但帮得没那么彻底

MLP 层带来的效果不太一样。因为 MLP 是对每个 token 独立进行的非线性变换（不涉及跨 token 的"平均"），它不会像残差连接那样阻止坍缩，但它能拖慢坍缩的速度——具体拖慢多少取决于 MLP 的"利普希茨常数"（Lipschitz constant，粗略理解成"这个变换能把输入的差异放大多少倍"）。MLP 越"有力气"（放大能力越强），坍缩就越慢。

但这里有个微妙的权衡：一个 Lipschitz 常数很大的 MLP 固然能更好地对抗秩坍缩，但同时也意味着这个网络对输入的微小扰动更敏感、更不稳定，训练时梯度方差也更大、更难调。这是一个典型的"鱼与熊掌"局面——你想要更强的抗坍缩能力，就要付出更大的训练不稳定性作为代价。

### 直觉：LayerNorm——一个大家都以为有用、其实没用的组件

这是论文里最反直觉的结论。很多人的直觉是 LayerNorm 通过重新缩放和居中,应该能帮着"打散"坍缩的表示——毕竟它把每个 token 的向量都重新拉回了单位球附近。但论文用严格证明给出了相反的答案：**LayerNorm 对秩坍缩没有任何缓解作用**。

原因其实很简单：LayerNorm 本质上是对每一行（每个 token）做独立的、逐列的缩放和平移，这在矩阵运算上等价于右乘一个对角矩阵。而右乘操作永远不会增加一个矩阵的秩——一个本来就是秩 1 的矩阵，右乘任何对角矩阵之后依然是秩 1。所以 LayerNorm 可以让坍缩后的向量"看起来"分布得更均匀（因为重新做了归一化），但它没有能力真正阻止坍缩这件事本身发生。

不过后续 2024 年的研究（Wu 等人《On the Role of Attention Masks and LayerNorm in Transformers》）在更细致的分析框架下，指出如果结合因果注意力掩码（causal mask，也就是解码器模型里那种"只能看过去、不能看未来"的限制）一起考虑，LayerNorm 的作用会变得更微妙——它能在某些情形下帮助避免坍缩到单点，但机制跟人们原本设想的"重新拉开距离"完全不是一回事。这提醒我们：直觉常常会在数学证明面前碰壁,而这恰恰是研究这类理论问题的价值所在。

<svg viewBox="0 0 640 260" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrowB" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="320" y="24" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">三个组件对秩坍缩的作用</text>

  <rect x="30" y="50" width="170" height="70" rx="8" fill="#1a2a1a" stroke="#4ade80" stroke-width="1.5"/>
  <text x="115" y="75" text-anchor="middle" fill="#4ade80" font-size="12" font-weight="bold">残差连接</text>
  <text x="115" y="95" text-anchor="middle" fill="#ededf0" font-size="10">彻底阻止坍缩</text>
  <text x="115" y="110" text-anchor="middle" fill="#8a8a99" font-size="9">(留一条跳过全部层的路径)</text>

  <rect x="235" y="50" width="170" height="70" rx="8" fill="#2a2a1a" stroke="#facc15" stroke-width="1.5"/>
  <text x="320" y="75" text-anchor="middle" fill="#facc15" font-size="12" font-weight="bold">MLP</text>
  <text x="320" y="95" text-anchor="middle" fill="#ededf0" font-size="10">拖慢坍缩速度</text>
  <text x="320" y="110" text-anchor="middle" fill="#8a8a99" font-size="9">(取决于 Lipschitz 常数)</text>

  <rect x="440" y="50" width="170" height="70" rx="8" fill="#2a1a1a" stroke="#e05a5a" stroke-width="1.5"/>
  <text x="525" y="75" text-anchor="middle" fill="#e05a5a" font-size="12" font-weight="bold">LayerNorm</text>
  <text x="525" y="95" text-anchor="middle" fill="#ededf0" font-size="10">不影响坍缩</text>
  <text x="525" y="110" text-anchor="middle" fill="#8a8a99" font-size="9">(右乘对角矩阵不改变秩)</text>

  <rect x="130" y="160" width="380" height="70" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="320" y="185" text-anchor="middle" fill="#ededf0" font-size="11">真实 Transformer = 纯注意力 + 残差连接 + MLP</text>
  <text x="320" y="205" text-anchor="middle" fill="#8a8a99" font-size="10">残差连接是"防坍缩"的主力，MLP 是辅助刹车</text>
  <line x1="115" y1="120" x2="280" y2="160" stroke="#4ade80" stroke-width="1.2" marker-end="url(#arrowB)"/>
  <line x1="320" y1="120" x2="320" y2="160" stroke="#facc15" stroke-width="1.2" marker-end="url(#arrowB)"/>
</svg>

## 因果掩码：解码器模型的额外一层保护

值得单独一提的是，2024 年 MIT 的 Karagodin 和 Polyanskiy 团队专门研究了带因果掩码（causal attention mask）的场景——也就是今天所有 GPT 类自回归模型实际使用的注意力形式（每个 token 只能看到它之前的 token，看不到之后的）。他们的分析显示，因果掩码下的坍缩行为跟双向注意力（BERT 那种）不完全一样：token 会趋向于"聚类"（clustering）而不是完全塌缩到单点——在某些参数条件下，序列里的 token 会分裂成两个或几个簇，簇内高度相似，簇间仍保留区分度。这某种程度上比"全部坍缩成一个点"温和一点，但本质上还是同一种"信息被过度平均掉"的现象，只是掩码结构给了它一点点喘息空间。

这也侧面解释了为什么解码器模型（GPT 系列）相对更"耐深"一些——因果掩码天然限制了每个 token 能接触到的"平均对象"数量（越靠前的 token 能看到的 token 越少），这本身就减慢了信息被拉向统一的速度。

## 这件事背后更大的图景：秩坍缩、各向异性和注意力汇聚

秩坍缩不是一个孤立的怪现象，它跟 LLM 领域另外几个广为人知的"病理"现象共享同一套数学根源。

第一个是"表示退化问题"（representation degeneration problem），也叫各向异性（anisotropy）：研究者发现训练好的语言模型的词嵌入普遍挤在一个很窄的锥形区域里，导致很多本该无关的词在余弦相似度上意外地"高度相似"。这跟秩坍缩讲的其实是同一枚硬币的两面——都是"表示空间被压扁了，丧失了区分能力"，只是一个发生在嵌入层，一个发生在注意力堆叠的深层网络里。

第二个更值得玩味的联系是"注意力汇聚"（attention sink）现象——几乎所有大模型都会把大量注意力权重"倒"给序列的第一个 token，即便这个 token 语义上毫无意义。2026 年初一篇来自 Yann LeCun 所在团队的论文（《The Spike, the Sparse and the Sink》）给出了一个精巧的解释：注意力汇聚和它伴随的"巨量激活"（massive activation，某些通道的数值异常大）,实际上是模型自己进化出来的一种"防秩坍缩机制"。让大量注意力权重集中砸向一个固定的、跟内容无关的汇聚 token，相当于人为地在注意力矩阵里插入了一个近似恒定的分量，这恰好起到了"抑制 token 之间过度混合"的效果——用一种略显笨拙但有效的方式，模型自己学会了对抗秩坍缩这个数学诅咒。换句话说，那个让人费解、看起来像是训练缺陷的"注意力汇聚"，可能根本不是缺陷，而是模型在跟秩坍缩这个物理定律做斗争时,自发演化出的补丁。

最后要提一句工程上的呼应：微软研究院 2022 年提出的 DeepNorm（用于 DeepNet 论文，成功训练了 1000 层的 Transformer）,本质上就是在"精细调节残差连接的强度"，通过给残差分支乘上一个恰当的常数,让极深的网络在保持训练稳定的同时,不至于让注意力层的"平均化效应"占据主导。这可以看作是秩坍缩理论在工程实践里的正面回应——如果残差连接是防坍缩的核心武器，那么把这件武器的"火力"调节得恰到好处，就是让 Transformer 敢于变得更深的关键。

## 这意味着什么

回过头看，秩坍缩这个理论结果给了我们一个精确得多的答案，来回应"Transformer 为什么有效"这个问题：**不是因为注意力本身有多神奇，而是因为注意力、残差连接和 MLP 三者形成了一种精妙的动态平衡**。注意力提供了"看全局、做信息聚合"的能力，但它天生带有把一切拉向雷同的强烈冲动；残差连接和 MLP 则像两个刹车，一个是绝对刹车（保证信息永远有一条不被稀释的路径），一个是相对刹车（拖慢混合的速度）。去掉任何一个刹车，纵使注意力机制设计得再精巧，模型都会在深度增加的过程中，以肉眼可见的速度失去表达能力。

这也给了我们一个新的视角去理解很多看似互不相关的现象：为什么极深的 Transformer 需要特殊的归一化技巧（DeepNorm）？为什么词嵌入会呈现各向异性？为什么每个大模型都会莫名其妙地把注意力"扔"给第一个 token？这些看似孤立的怪癖，追根溯源，都跟同一个数学定律脱不开关系——一堆行随机矩阵的连乘，天生就想把一切变得一样。

对于任何正在设计新架构、或者试图理解现有模型为什么这样运作的人来说，秩坍缩理论提供了一条清晰的检验标准：任何声称能替代传统注意力+残差架构的新方案，都必须回答一个问题——你的架构里，谁在负责防止表示坍缩？如果答案含糊不清，那可能只是运气好，还没跑到深到让坍缩现身的那一层。

---

*参考文献：Dong, Cordonnier & Loukas (2021), "Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth", ICML 2021 (Oral)；Wu et al. (2024), "On the Role of Attention Masks and LayerNorm in Transformers", NeurIPS 2024；Karagodin & Polyanskiy (2024), "Clustering in Causal Attention Masking"；Wang et al. (2022), "DeepNet: Scaling Transformers to 1,000 Layers"；相关团队 (2026), "The Spike, the Sparse and the Sink: Anatomy of Massive Activations and Attention Sinks"。*
