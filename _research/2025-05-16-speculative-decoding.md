---
title: "投机解码：让大模型「猜」着跑的数学魔术"
date: 2025-05-16
level: 3
series: "LLM 原理深度解析"
series_order: 2
series_total: 3
tags: [speculative-decoding, inference, rejection-sampling, LLM]
summary: "Speculative Decoding 用一个小模型先猜、大模型再验的方式，在数学上保证输出分布完全不变的前提下，把推理速度提升 2-3 倍。"
---

# 投机解码：让大模型「猜」着跑的数学魔术

> 如果有一种方法，能让 70B 参数的大模型跑得快 3 倍，而且生成的文字和原来一模一样——不是"差不多一样"，是数学意义上的完全相同——你会不会觉得这是在变魔术？

## 故事从一个荒谬的瓶颈开始

想象你在一个高档餐厅里，主厨（大模型）做菜极其精湛，但有一个怪癖：他每做完一道菜，必须亲眼看到客人吃完这一口，才肯开始做下一口。哪怕桌上有 500 道菜要做，他也坚持一口一口来。

这就是大语言模型推理时的真实困境。GPT-4、Claude、Llama 这些模型生成文字的方式叫做**自回归解码**（autoregressive decoding）：每生成一个 token，都要把整个模型从头跑一遍。一个 70B 参数的模型，生成 100 个 token 就要做 100 次前向传播——即使这 100 次计算中，大部分信息都是重复的。

更讽刺的是，现代 GPU 的算力其实远远没有被用满。自回归解码是**内存带宽瓶颈**（memory-bandwidth bound），不是计算瓶颈。GPU 大部分时间在等数据从显存搬到计算单元，而不是在做乘法。这就好比主厨手速飞快，但每做一口菜都要等服务员从仓库跑一趟拿原料。

问题的核心在于：大模型验证一批 token 的成本和生成一个 token 几乎一样。验证 5 个 token 时，GPU 可以并行处理这 5 个位置，充分利用那些闲置的计算单元。但生成时却不行——因为第 2 个 token 依赖第 1 个的结果，第 3 个依赖第 2 个，天然串行。

那如果我们换个思路呢？如果我们能"猜"出接下来的几个 token，然后让大模型一次性验证这些猜测——猜对的就用，猜错的就改——那不就能把串行变并行了吗？

这就是 Speculative Decoding（投机解码）的核心思想。

## 「小弟先猜，大哥验收」——核心想法

2023 年，两组研究者几乎同时想到了同一个点子。Google 的 Leviathan 等人发表了"Fast Inference from Transformers via Speculative Decoding"，DeepMind 的 Chen 等人发表了"Accelerating Large Language Model Decoding with Speculative Sampling"。两篇论文的核心思想惊人地一致：

**用一个小而快的模型（draft model）先"猜"出 k 个 token，然后让大模型（target model）一次性验证这些猜测。**

类比一下：你是一个忙碌的总编辑（大模型），手下有一个写作能力还行的实习生（小模型）。以前你每个字都亲自写，写一个字要思考 1 分钟。现在改成新流程：

1. **实习生先写** 5 个字（很快，每个字只要 3 秒）
2. **你一次性审阅**这 5 个字（审阅 5 个字和写 1 个字花的时间差不多）
3. 前 3 个字你觉得可以，直接用；第 4 个字不行，你改掉它
4. 第 4 个字之后的都作废，因为它们是基于错误的第 4 个字写的
5. 回到步骤 1，实习生从你改的位置继续写

一次循环就产出了 4 个字（3 个接受的 + 1 个你改的），但只花了你 1 次"审阅时间"。如果原来你 1 分钟写 1 个字，现在大约 1 分 15 秒出 4 个字——**快了 3 倍多**。

但这里有一个关键问题，也是这个方法最精妙之处：

> **验收标准是什么？怎么保证最终产出的文字，和你亲自一个字一个字写的结果，统计分布完全一样？**

答案不是"看起来差不多就行"，不是"质量损失可以接受"，而是**数学意义上的完全等价**。输出的 token 分布和直接用大模型采样的分布是同一个分布——不是近似，是精确相等。

这个魔术般的保证来自一个经典的统计学工具：**拒绝采样**（rejection sampling）。

## 拒绝采样：一个优雅的统计学把戏

在理解投机解码的具体算法之前，我们需要先认识拒绝采样这个工具。这是 1950 年代冯·诺依曼发明的方法，核心思想出人意料地简单：

**如果你想从一个复杂分布 p 中采样，但只能从一个简单分布 q 中采样，那可以这样做：从 q 采一个样本，然后以某个精心设计的概率接受或拒绝它。被接受的样本就服从分布 p。**

打个比方：你想买到某个城市中均匀分布的房子（目标分布 p），但你的房产中介只给你推荐他有提成的房子（提案分布 q）。你的策略是：看一下这套房子的"质量分数"（p/q 的比值），质量越高就越容易接受。如果中介拼命推一套其实不怎么样的房子（q(x) 很大但 p(x) 很小），你大概率拒绝它；如果他偶然推了一套真正的好房子（p(x) 相对 q(x) 很大），你一定接受。

经过这个筛选后，你最终买到的房子的分布就是你真正想要的均匀分布。

在投机解码中：
- **目标分布 p(x)** = 大模型对下一个 token 的概率分布
- **提案分布 q(x)** = 小模型对下一个 token 的概率分布
- **我们的目标** = 从 q 中采样，但最终得到的结果服从 p

## 验收标准的精确数学

现在我们来看投机解码最核心的公式。当小模型提出一个 token x 时，我们以如下概率接受它：

$$\alpha(x) = \min\left(1, \; \frac{p(x)}{q(x)}\right)$$

翻译成人话：**接受概率 = 大模型喜欢这个 token 的程度 ÷ 小模型喜欢这个 token 的程度（最多为 1）。**

这个公式有两层含义：

**情况一：大模型比小模型更喜欢这个 token（p(x) ≥ q(x)）**

比值 ≥ 1，截断为 1，**一定接受**。直觉：小模型本来就不太愿意生成这个 token（q(x) 小），结果它还是生成了，而大模型更喜欢这个 token——这相当于"歪打正着"，没有理由拒绝。

**情况二：小模型比大模型更喜欢这个 token（q(x) > p(x)）**

比值 < 1，接受概率 = p(x)/q(x)。直觉：小模型"过度推荐"了这个 token，我们需要按比例拒绝一些，把它的频率降下来，降到大模型认为合理的水平。

<svg viewBox="0 0 700 320" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- 坐标轴 -->
  <line x1="60" y1="260" x2="660" y2="260" stroke="#6e8eff" stroke-width="1.5"/>
  <line x1="60" y1="260" x2="60" y2="30" stroke="#6e8eff" stroke-width="1.5"/>
  <!-- Y轴标签 -->
  <text x="30" y="150" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui" transform="rotate(-90, 30, 150)">概率</text>
  <!-- X轴标签 -->
  <text x="360" y="295" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">Token (词表)</text>
  <!-- 标题 -->
  <text x="360" y="20" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">接受/拒绝区域示意</text>
  <!-- Token A: p > q (总是接受) -->
  <rect x="100" y="80" width="60" height="180" rx="4" fill="#34d399" fill-opacity="0.3" stroke="#34d399" stroke-width="1.5"/>
  <rect x="100" y="150" width="60" height="110" rx="4" fill="#6e8eff" fill-opacity="0.3" stroke="#6e8eff" stroke-width="1.5" stroke-dasharray="4"/>
  <text x="130" y="275" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">Token A</text>
  <!-- Token B: q > p (部分拒绝) -->
  <rect x="220" y="120" width="60" height="140" rx="4" fill="#6e8eff" fill-opacity="0.3" stroke="#6e8eff" stroke-width="1.5"/>
  <rect x="220" y="160" width="60" height="100" rx="4" fill="#34d399" fill-opacity="0.3" stroke="#34d399" stroke-width="1.5" stroke-dasharray="4"/>
  <text x="250" y="275" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">Token B</text>
  <!-- Token C: p ≈ q -->
  <rect x="340" y="180" width="60" height="80" rx="4" fill="#34d399" fill-opacity="0.3" stroke="#34d399" stroke-width="1.5"/>
  <rect x="340" y="180" width="60" height="80" rx="4" fill="#6e8eff" fill-opacity="0.15" stroke="#6e8eff" stroke-width="1.5" stroke-dasharray="4"/>
  <text x="370" y="275" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">Token C</text>
  <!-- Token D -->
  <rect x="460" y="210" width="60" height="50" rx="4" fill="#34d399" fill-opacity="0.3" stroke="#34d399" stroke-width="1.5"/>
  <rect x="460" y="200" width="60" height="60" rx="4" fill="#6e8eff" fill-opacity="0.3" stroke="#6e8eff" stroke-width="1.5" stroke-dasharray="4"/>
  <text x="490" y="275" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">Token D</text>
  <!-- 图例 -->
  <rect x="500" y="40" width="15" height="15" rx="3" fill="#34d399" fill-opacity="0.5" stroke="#34d399" stroke-width="1"/>
  <text x="520" y="52" fill="#ededf0" font-size="11" font-family="system-ui">p(x) 目标模型</text>
  <rect x="500" y="65" width="15" height="15" rx="3" fill="#6e8eff" fill-opacity="0.5" stroke="#6e8eff" stroke-width="1"/>
  <text x="520" y="77" fill="#ededf0" font-size="11" font-family="system-ui">q(x) 草稿模型</text>
  <!-- 注解 -->
  <text x="130" y="65" text-anchor="middle" fill="#34d399" font-size="10" font-family="system-ui">p &gt; q: 必接受</text>
  <text x="250" y="107" text-anchor="middle" fill="#f87171" font-size="10" font-family="system-ui">q &gt; p: 部分拒绝</text>
</svg>

### 拒绝之后怎么办？——残差分布

如果我们拒绝了小模型的提议，不能简单地让小模型再猜一次（那会引入偏差）。我们需要从一个特殊的**残差分布**（residual distribution）中采样：

$$p_{\text{resid}}(x) = \frac{\max(0, \; p(x) - q(x))}{Z}$$

其中 $Z = \sum_x \max(0, p(x) - q(x))$ 是归一化常数。

残差分布的直觉是：它只包含那些"大模型想要但小模型忽略的"token。如果把两个分布想象成两个柱状图叠在一起，残差分布就是大模型的柱子"高出"小模型柱子的那些部分。

**为什么这样能保证正确性？** 让我们追踪一个 token x 被最终产出的总概率：

- **路径 1（接受）**：小模型提出 x 的概率 q(x) × 接受概率 min(1, p(x)/q(x)) = min(q(x), p(x))
- **路径 2（拒绝后重采样）**：总拒绝概率 × 从残差分布采到 x 的概率 = max(0, p(x) - q(x))

两条路径加起来：min(q(x), p(x)) + max(0, p(x) - q(x)) = **p(x)**

这是一个恒等式！不管 p 和 q 怎么分配，两条路径加起来永远精确等于目标分布。这就是投机解码"零损失"保证的数学根基。

## 多 Token 验证：左到右，遇错即停

实际使用中，小模型一次猜 k 个 token（通常 k = 3~8）。大模型通过一次前向传播同时得到这 k 个位置的条件概率分布，然后从左到右逐个验证：

1. 验证第 1 个 token：接受？继续。拒绝？从残差分布重采样，**后面全部作废**。
2. 验证第 2 个 token（条件于第 1 个已接受）：接受？继续。拒绝？重采样，后面作废。
3. ...依此类推
4. 如果全部 k 个都接受了，还能从大模型的输出中**白嫖一个额外 token**（因为大模型的前向传播已经计算了第 k+1 个位置的分布）。

为什么必须左到右？因为语言模型的概率是条件概率。"the cat sat on"的概率取决于前面每一个词。如果第 2 个词"cat"被拒绝了，改成了"dog"，那后面基于"cat"算出来的概率就全错了。

**一次迭代的保底收益：至少 1 个 token。** 即使所有猜测都被拒绝，我们也会从残差分布得到 1 个正确的 token。最好情况：k+1 个 token（全部接受 + 奖励 token）。

## 加速比公式：什么时候值得投机？

设 α 为平均接受概率（衡量小模型和大模型有多像），k 为猜测长度，c 为成本比（大模型一次前传时间 ÷ 小模型一次前传时间）。

**每次迭代期望产出的 token 数：**

$$E[N] = \frac{1 - \alpha^{k+1}}{1 - \alpha}$$

这是一个截断几何级数。当 α → 1 时，E[N] → k+1（几乎全部接受）；当 α → 0 时，E[N] → 1（几乎全部拒绝，只靠残差分布续命）。

**加速比公式：**

$$S = \frac{E[N]}{1 + k/c} = \frac{(1 - \alpha^{k+1}) / (1 - \alpha)}{1 + k/c}$$

分子是收益（期望产出 token 数），分母是成本（一次迭代 = 大模型 1 次 + 小模型 k 次，折算为大模型次数就是 1 + k/c）。

**实际数字：**
- α = 0.8, k = 5, c = 10：E[N] ≈ 3.7, 成本 = 1.5, 加速比 ≈ **2.5×**
- α = 0.9, k = 5, c = 10：E[N] ≈ 4.7, 成本 = 1.5, 加速比 ≈ **3.1×**
- α = 0.6, k = 5, c = 10：E[N] ≈ 2.5, 成本 = 1.5, 加速比 ≈ **1.7×**

当小模型太慢（c 小）或者猜得太差（α 小），投机解码可能反而更慢（S < 1）。这就是为什么选择合适的 draft model 至关重要。

<svg viewBox="0 0 700 350" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- 标题 -->
  <text x="350" y="25" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">投机解码完整流程（k=4 示例）</text>
  <!-- Step 1: Draft Model -->
  <rect x="30" y="55" width="130" height="50" rx="8" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="95" y="75" text-anchor="middle" fill="#a78bfa" font-size="10" font-family="system-ui">Draft Model (7B)</text>
  <text x="95" y="92" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">快速猜 4 个 token</text>
  <!-- Arrow -->
  <line x1="160" y1="80" x2="195" y2="80" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <!-- Draft tokens -->
  <rect x="200" y="55" width="200" height="50" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="220" y="78" fill="#34d399" font-size="12" font-family="monospace">t₁</text>
  <text x="260" y="78" fill="#34d399" font-size="12" font-family="monospace">t₂</text>
  <text x="300" y="78" fill="#34d399" font-size="12" font-family="monospace">t₃</text>
  <text x="340" y="78" fill="#f87171" font-size="12" font-family="monospace">t₄</text>
  <text x="370" y="78" fill="#94a3b8" font-size="10" font-family="system-ui">草稿</text>
  <!-- Arrow down -->
  <line x1="300" y1="105" x2="300" y2="135" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <!-- Step 2: Target Model -->
  <rect x="200" y="140" width="200" height="50" rx="8" fill="#1e1e2a" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="300" y="160" text-anchor="middle" fill="#22d3ee" font-size="10" font-family="system-ui">Target Model (70B)</text>
  <text x="300" y="177" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">并行验证全部 4 个位置</text>
  <!-- Arrow down -->
  <line x1="300" y1="190" x2="300" y2="220" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <!-- Step 3: Accept/Reject -->
  <rect x="130" y="225" width="340" height="55" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="300" y="245" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">逐个验证 (α = min(1, p/q))</text>
  <text x="180" y="268" fill="#34d399" font-size="12" font-family="monospace">t₁ ✓</text>
  <text x="240" y="268" fill="#34d399" font-size="12" font-family="monospace">t₂ ✓</text>
  <text x="300" y="268" fill="#34d399" font-size="12" font-family="monospace">t₃ ✓</text>
  <text x="360" y="268" fill="#f87171" font-size="12" font-family="monospace">t₄ ✗</text>
  <text x="420" y="268" fill="#fbbf24" font-size="12" font-family="monospace">→ t₄'</text>
  <!-- Result -->
  <line x1="300" y1="280" x2="300" y2="310" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <rect x="180" y="312" width="240" height="30" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="300" y="332" text-anchor="middle" fill="#34d399" font-size="12" font-family="system-ui">产出 4 个 token（3 接受 + 1 重采样）</text>
  <!-- 右侧注释 -->
  <text x="530" y="78" fill="#94a3b8" font-size="10" font-family="system-ui">耗时：≈ 4×T_draft</text>
  <text x="530" y="168" fill="#94a3b8" font-size="10" font-family="system-ui">耗时：1×T_target</text>
  <text x="530" y="253" fill="#94a3b8" font-size="10" font-family="system-ui">数学保证 ≡ 原始分布</text>
</svg>

## 最优猜测长度 k：贪心还是保守？

k 太小（比如 1），投机解码退化为普通解码，没有加速效果。k 太大，猜的 token 越多，后面被浪费的概率越高（因为一旦中间某个被拒绝，后面全废了），而且小模型的开销也随 k 线性增长。

最优 k 取决于 α 和 c 的平衡：

- **α 高**（小模型和大模型很像）→ 可以猜长一点，k = 5~8
- **α 低**（差异大）→ 猜短一点，k = 2~3
- **c 大**（大模型比小模型慢很多）→ 猜长一点更划算
- **c 小**（两模型速度差不多）→ 猜太长反而亏

实践中，最常见的配置是 k = 4~6。一些系统（如 SpecDec++）会动态调整 k，根据上一轮的接受率自适应地增加或减少猜测长度。

## 不需要额外模型的投机：Medusa 和 EAGLE

原始的投机解码需要一个独立的 draft model，这带来了工程负担：要额外加载一个模型、管理两套权重、保证两者使用相同的 tokenizer。于是研究者们开始探索"自投机"（self-speculative）的方案。

### Medusa：给大模型装上多个预测头

Medusa（2024, ICML）的想法非常直观：既然大模型在生成当前 token 时已经算出了丰富的隐藏状态表示，为什么不在这个表示上多接几个轻量的"预测头"，直接预测后面第 2、第 3、...个 token？

具体来说，Medusa 在大模型的最后一层之上添加了 k 个额外的 MLP 头。第 i 个头负责预测当前位置之后第 i 个 token。这些头很小（只有一两层 MLP），训练也很快（只需要微调这些头，冻结主模型）。

更巧妙的是，Medusa 使用**树状注意力**（tree attention）来组织候选 token。每个头可能给出 top-2 或 top-3 的预测，组合起来形成一棵候选树，一次验证整棵树的所有路径。树中被接受的最长路径就是最终输出。

Medusa 在 Vicuna-7B 上实现了 2.2~3.6× 的加速。

### EAGLE：在特征层做自回归

EAGLE（2024, ICML）的观察更深入：为什么 draft 不一定要在 token 层面做自回归？token 的分布是经过 softmax 的、高熵的、难以预测的。但如果我们看模型的**倒数第二层特征**（second-to-top-layer features），情况就不一样了——特征层的自回归比 token 层容易得多。

EAGLE 的做法：在大模型的倒数第二层输出之上，接一个轻量的自回归头，让它在特征空间预测下一步的特征向量，再通过原模型的 LM head 转化为 token 分布。这个特征层草稿模型非常小（单层 Transformer），但因为特征空间比 token 空间"平滑"得多，预测精度很高。

EAGLE 结合树状投机验证，在多个模型上实现了 2.5~3.8× 的加速。最新的 EAGLE-3 进一步使用多层特征融合（低层+中层+高层），报告了高达 6.5× 的加速。

## 投机解码的适用场景

投机解码并非万能。它在以下场景效果最好：

**效果好的场景：**
- 翻译、摘要等"输入驱动"任务（输出高度依赖输入，可预测性强，α 高）
- 代码补全（代码有强规律性，很多 token 是确定性的）
- 长文本生成（延迟敏感，投机解码的收益累积效应明显）
- Batch size = 1 的场景（内存带宽瓶颈最严重）

**效果差的场景：**
- 创意写作（输出分布高熵，小模型很难猜对）
- 很短的回复（投机的 overhead 还没回本就结束了）
- 大 batch 推理（此时瓶颈从带宽转向计算，投机的优势减弱）

## 这意味着什么

投机解码的深远意义在于它证明了一个反直觉的命题：**推理速度和输出质量不一定要做权衡。**

传统的加速方法——量化、蒸馏、剪枝——都在"快一点但差一点"的权衡曲线上滑动。投机解码跳出了这条曲线：它不改变模型、不降低精度、不近似任何东西，只是更聪明地安排计算顺序。

从信息论的角度看，投机解码利用了一个事实：大模型的输出中有大量"低信息量"的 token（比如"the"、"of"、标点符号），这些 token 是高度可预测的。小模型花极少的算力就能猜对它们。真正需要大模型"动脑子"的只有那些高信息量的关键词。投机解码本质上是**让算力分配跟随信息量分布**——简单的 token 用小模型，困难的 token 才麻烦大模型。

这也是为什么 2024-2025 年，几乎所有主流推理框架（vLLM、TensorRT-LLM、SGLang、HuggingFace TGI）都将投机解码作为内置特性。在 NVIDIA H200 GPU 上，生产部署实测加速 2~3.6×，这已经是工程界的共识，不再是论文里的数字游戏。

## 延伸思考

投机解码打开了一个更大的问题：**如果我们能"猜"token，是否意味着语言生成中的大部分计算是冗余的？** Mixture of Depths（条件计算）、early exit、自适应计算等研究方向，都在从不同角度回答这个问题。也许未来的模型不需要对每个 token 都投入同等的计算量——就像人类写作时，写"Hello"不需要思考，但写一个关键论点可能需要深思熟虑。

投机解码是这个"自适应计算"愿景中最成功的一个具体实例。它告诉我们：**最好的加速不是让硬件更快，而是让计算更聪明。**

---

**参考文献：**
- Leviathan et al. "Fast Inference from Transformers via Speculative Decoding" (ICML 2023)
- Chen et al. "Accelerating Large Language Model Decoding with Speculative Sampling" (2023)
- Cai et al. "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads" (ICML 2024)
- Li et al. "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty" (ICML 2024)
- Sun et al. "A Theoretical Perspective for Speculative Decoding Algorithm" (NeurIPS 2024)
