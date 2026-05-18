---
title: "In-Context Learning：为什么不更新权重也能学？"
date: 2025-05-18
level: 3
series: "LLM 原理深度解析"
series_order: 6
series_total: 6
tags: [in-context-learning, meta-learning, transformer, 理论]
summary: "大语言模型不用训练就能从几个例子中'学会'新任务——四种理论解释这种看似不可能的能力从何而来。"
---

# In-Context Learning：为什么不更新权重也能学？

> 你给 GPT 看三个翻译例子，它就"学会"了翻译风格。没有反向传播，没有梯度更新，参数纹丝不动——它到底是怎么"学"的？

## 一个让人困惑的现象

2020 年，OpenAI 发布 GPT-3 时展示了一个令整个 AI 界震惊的能力：你只需要在 prompt 里放几个"输入→输出"的例子，模型就能对新输入给出正确答案。

比如你写：

```
英文: Hello → 中文: 你好
英文: Thank you → 中文: 谢谢
英文: Good morning → 中文:
```

模型会接着输出"早上好"。

这被称为 **In-Context Learning (ICL)**——上下文学习。

奇怪的地方在于：传统机器学习要"学"一个新任务，必须更新模型参数（梯度下降、反向传播那一套）。但 ICL 完全不更新参数。模型的权重在推理时是冻结的。那些例子只是作为输入序列的一部分被"读"了一遍——就这样，模型似乎就"学会"了。

这就像一个学生只看了三道例题（不做练习、不背公式），就能正确解答第四道。这合理吗？这背后到底发生了什么？

过去五年里，研究者们从不同角度给出了至少四种理论解释。每种都揭示了 ICL 的一个侧面。让我们逐一探索。

## 第一种解释：贝叶斯推断——"不是在学，是在定位"

### 问题是什么

最直观的困惑是：如果模型权重不变，那新知识从哪来？

### 核心直觉

2022 年，Stanford 的 Xie 等人提出了一个优雅的解释：**模型在预训练时已经学会了大量"概念"（concept），ICL 做的事情不是学习新知识，而是用那几个例子来"定位"它已经知道的某个概念。**

打个比方。想象你是一个语言天才，已经掌握了 100 种语言。现在有人给你看几句未知文字和它们的翻译。你不是在"学"这种新语言——你的大脑在做的事情是：根据这些例子的模式（词序、语法结构、词根特征），在你已知的 100 种语言中匹配最可能的那种，然后用那套已有知识来翻译新句子。

用数学语言说，这就是**贝叶斯推断**：

$$P(\text{concept} \mid \text{examples}) \propto P(\text{examples} \mid \text{concept}) \cdot P(\text{concept})$$

翻译成人话：模型看到那几个例子后，计算"哪个我已经学过的概念最可能产生这些例子"，然后按这个概念去预测下一个输出。

### 技术细节

Xie 等人的理论框架是这样的：

1. **预训练数据是由多种"概念"混合生成的**。比如互联网文本中，有的段落是翻译、有的是代码、有的是诗歌。每种"概念"对应一个隐马尔可夫模型（HMM）或其他生成模型。

2. **预训练阶段，模型隐式学会了这些概念的先验分布** $P(\text{concept})$，以及每个概念如何生成文本 $P(\text{text} \mid \text{concept})$。

3. **ICL 的 prompt 例子起到"似然证据"的作用**，帮助模型做后验推断，把概率质量集中到正确的概念上。

4. **例子越多，后验越集中**，预测越准确。这完美解释了为什么更多 few-shot 例子通常带来更好的性能。

这个理论有一个重要推论：**ICL 不能学到预训练中完全没见过的东西**。如果一个任务与预训练数据中的所有概念都不相似，那给再多例子也没用。这在实验中确实得到了验证——对于真正全新的、反事实的任务（比如把所有标签反转），ICL 的表现会大幅下降。

<svg viewBox="0 0 700 320" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Pre-training: concept space -->
  <rect x="20" y="20" width="200" height="280" rx="12" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="120" y="50" text-anchor="middle" fill="#6e8eff" font-size="14" font-weight="bold" font-family="system-ui">预训练阶段</text>
  <text x="120" y="75" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">学会了 N 个"概念"</text>
  <!-- Concepts -->
  <circle cx="70" cy="120" r="22" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="70" y="124" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">翻译</text>
  <circle cx="160" cy="120" r="22" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="160" y="124" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">代码</text>
  <circle cx="70" cy="190" r="22" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="70" y="194" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">情感</text>
  <circle cx="160" cy="190" r="22" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="160" y="194" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">摘要</text>
  <circle cx="120" cy="260" r="22" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="120" y="264" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">…N个</text>
  <!-- Arrow -->
  <line x1="230" y1="160" x2="290" y2="160" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>
  <text x="260" y="148" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">ICL</text>
  <!-- ICL: prompt examples -->
  <rect x="300" y="60" width="170" height="200" rx="12" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="385" y="90" text-anchor="middle" fill="#6e8eff" font-size="14" font-weight="bold" font-family="system-ui">Prompt 例子</text>
  <text x="385" y="120" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">Hello → 你好</text>
  <text x="385" y="145" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">Thanks → 谢谢</text>
  <text x="385" y="170" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">Good → 好的</text>
  <text x="385" y="210" text-anchor="middle" fill="#a78bfa" font-size="11" font-family="system-ui">P(翻译|例子) ↑↑↑</text>
  <text x="385" y="235" text-anchor="middle" fill="#94a3b8" font-size="11" font-family="system-ui">P(代码|例子) ↓</text>
  <!-- Arrow to output -->
  <line x1="480" y1="160" x2="540" y2="160" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>
  <!-- Output -->
  <rect x="550" y="120" width="130" height="80" rx="12" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="615" y="150" text-anchor="middle" fill="#a78bfa" font-size="13" font-weight="bold" font-family="system-ui">输出预测</text>
  <text x="615" y="175" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">用"翻译"概念</text>
  <text x="615" y="195" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">生成答案</text>
</svg>

## 第二种解释：隐式梯度下降——"前向传播里藏着一个优化器"

### 问题是什么

贝叶斯推断解释了"定位已有概念"的情况，但 ICL 有时明显在做更复杂的事——比如学习输入数据中的新模式，甚至适应训练时没见过的分布。这看起来更像是真正的"学习"而非简单的"检索"。那这种学习如何在不改变权重的情况下发生？

### 核心直觉

2022-2023 年，多个研究组（Dai et al., Von Oswald et al., Akyürek et al.）几乎同时发现了一个惊人的联系：**Transformer 的注意力机制在做前向传播时，实际上在执行一种隐式的梯度下降。**

这是什么意思？让我用一个类比：

想象你在考试时遇到一道不会的题。你翻看卷子前面的例题，发现了规律，然后用这个规律解题。从外部看，你没有"训练"（没有回去看教材、做练习题）。但在你的脑子里，你实际上完成了一个微型的学习过程：从例题中提取规律 → 形成一个临时的"小模型" → 用它来解答。

Transformer 做的事类似：在处理 prompt 中的例子时，注意力层的计算过程**数学上等价于**对某个内部目标函数做一步（或几步）梯度下降。也就是说，虽然模型的真实权重 $W$ 没变，但它通过注意力机制在前向传播中"构造"了一组等效的更新后的权重。

### 技术细节

让我展示这个数学等价关系为什么成立。

考虑一个线性注意力层（去掉 softmax 的简化版本）。对于输入 $x$，它的输出是：

$$f(x) = W_V X^T \cdot X W_K^T \cdot W_Q x$$

其中 $X$ 是上下文中所有 token 的矩阵。

现在考虑对一个线性回归问题做一步梯度下降。如果我们有样本 $(x_i, y_i)$，初始参数为 $W_0$，学习率为 $\eta$，那么一步梯度下降后：

$$W_1 = W_0 - \eta \sum_i (W_0 x_i - y_i) x_i^T$$

对新输入 $x_{new}$ 的预测是：

$$\hat{y} = W_1 x_{new} = W_0 x_{new} - \eta \sum_i (W_0 x_i - y_i) x_i^T x_{new}$$

关键观察：**这两个表达式有相同的结构**——都是对上下文样本做加权求和，权重取决于新输入与各上下文样本的相似度（$x_i^T x_{new}$ 这一项）。

Von Oswald et al. (2023, ICML Oral) 将这个观察推广到了更一般的情况，证明了：

1. 一层线性注意力 ≈ 一步梯度下降
2. 多层堆叠 ≈ 多步梯度下降（迭代优化）
3. 通过精心设计 $W_Q, W_K, W_V$ 的初始化，Transformer 甚至可以实现带动量的梯度下降

Dai et al. (2023) 从另一个角度证实了这一点：他们发现 ICL 和真正的微调（finetuning）在很多指标上表现几乎一样——相同的 attention 模式、类似的输出分布变化——就好像 ICL 真的在"偷偷微调"模型。

<svg viewBox="0 0 720 280" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:720px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Left: Explicit GD -->
  <rect x="20" y="20" width="310" height="240" rx="12" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="175" y="50" text-anchor="middle" fill="#34d399" font-size="14" font-weight="bold" font-family="system-ui">显式梯度下降（微调）</text>
  <rect x="45" y="70" width="120" height="40" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1"/>
  <text x="105" y="94" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">训练数据 (x,y)</text>
  <line x1="105" y1="110" x2="105" y2="140" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <rect x="45" y="140" width="120" height="40" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1"/>
  <text x="105" y="164" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">计算梯度 ∇L</text>
  <line x1="105" y1="180" x2="105" y2="210" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <rect x="45" y="210" width="120" height="35" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="105" y="232" text-anchor="middle" fill="#34d399" font-size="11" font-family="system-ui">W₁ = W₀ - η∇L</text>
  <!-- Formula -->
  <text x="230" y="100" text-anchor="middle" fill="#94a3b8" font-size="11" font-family="system-ui">更新权重</text>
  <text x="230" y="120" text-anchor="middle" fill="#94a3b8" font-size="11" font-family="system-ui">多次迭代</text>
  <text x="230" y="140" text-anchor="middle" fill="#94a3b8" font-size="11" font-family="system-ui">慢速学习</text>
  <text x="230" y="200" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">≈</text>
  <text x="230" y="180" text-anchor="middle" fill="#a78bfa" font-size="22" font-family="system-ui">≡</text>
  <!-- Right: ICL implicit GD -->
  <rect x="380" y="20" width="320" height="240" rx="12" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="540" y="50" text-anchor="middle" fill="#a78bfa" font-size="14" font-weight="bold" font-family="system-ui">ICL 隐式梯度下降（前向传播）</text>
  <rect x="405" y="70" width="130" height="40" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1"/>
  <text x="470" y="94" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">Prompt 例子 (x,y)</text>
  <line x1="470" y1="110" x2="470" y2="140" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <rect x="405" y="140" width="130" height="40" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1"/>
  <text x="470" y="158" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">注意力计算</text>
  <text x="470" y="173" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">V·Kᵀ·Q ≈ 梯度步</text>
  <line x1="470" y1="180" x2="470" y2="210" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <rect x="405" y="210" width="130" height="35" rx="8" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="470" y="232" text-anchor="middle" fill="#a78bfa" font-size="11" font-family="system-ui">等效 W₁ 的输出</text>
  <!-- Right annotations -->
  <text x="600" y="100" text-anchor="middle" fill="#94a3b8" font-size="11" font-family="system-ui">不动权重</text>
  <text x="600" y="120" text-anchor="middle" fill="#94a3b8" font-size="11" font-family="system-ui">单次前向</text>
  <text x="600" y="140" text-anchor="middle" fill="#94a3b8" font-size="11" font-family="system-ui">即时"学习"</text>
</svg>

### 一个关键限制

但且慢——这个理论有一个重要的前提：它在**线性注意力**（没有 softmax）的情况下证明最为干净。真实 Transformer 用的是 softmax 注意力，数学等价性就没那么精确了。

2024 年 ICML 上，Shen et al. 发表了一篇 Position Paper 《Do pretrained transformers learn in-context by Gradient Descent?》，指出：虽然构造性证明了 Transformer *可以* 实现梯度下降，但并没有证据表明预训练好的真实模型*确实*在这样做。这就好比证明了"人可以用脚写字"，但不代表人们实际上在这样做。

## 第三种解释：归纳头——"硬件层面的模式匹配电路"

### 问题是什么

前两种解释都是从"宏观功能"的角度说 ICL 像什么（像贝叶斯推断、像梯度下降）。但 Transformer 内部到底形成了什么样的计算电路来实现 ICL？

### 核心直觉

2022 年，Anthropic 的 Olsson 等人发表了一篇里程碑式的机制解释学（Mechanistic Interpretability）论文，找到了 ICL 的一个核心"硬件"组件：**归纳头（Induction Head）**。

归纳头做的事情极其简单：**"上次 A 后面出现了 B，这次又看到 A，所以预测 B。"**

这是一种"匹配-复制"操作（match-and-copy）。比如在文本中，如果前面出现过 "Harry Potter is a" → "wizard"，那当后面再次出现 "Harry Potter is a" 时，归纳头会把注意力集中到前一次出现的位置，然后"复制"那里的下一个 token。

### 它为什么重要

Olsson 等人发现了一个惊人的巧合：在训练过程中，归纳头的形成和 ICL 能力的出现**精确地发生在同一时刻**。

具体来说，他们观察到训练过程中存在一个**相变（phase transition）**：

- 在某个训练步骤之前：模型基本没有 ICL 能力，loss 平稳下降
- 在那个步骤：归纳头突然形成（特定 attention pattern 出现）
- 之后：ICL 能力急剧提升，表现为 loss 出现一个明显的"bump"（短暂上升后更快下降）

这个相变在不同模型大小、不同训练数据上都可复现。这强烈暗示归纳头就是（至少是）ICL 的核心机制之一。

### 归纳头的电路结构

一个归纳头其实由至少两个注意力头配合实现：

1. **前一 token 头（Previous Token Head）**：位于较低层。它的作用是"标记"每个 token 前面是什么。通过将当前位置的信息写入前一个位置的 residual stream，它让后续层能看到"谁出现在谁后面"。

2. **归纳头本身（Induction Head）**：位于较高层。它的 QK 电路搜索"当前 token 上一次出现时，它后面跟的是什么"，然后 OV 电路把那个后续 token 复制到当前输出。

这就是 ICL 最基础的形式：模式补全。但真实的 ICL（比如做翻译、做推理）显然需要更复杂的电路。研究者认为，更高级的 ICL 可能是归纳头的"泛化版"——不是匹配字面相同的 token，而是匹配语义相似的模式。

## 第四种解释：Mesa-Optimization——"训练出了一个内部优化器"

### 问题是什么

如果 ICL 真的像梯度下降，那谁在"执行"这个梯度下降？是 Transformer 的架构天然就会做这个，还是训练过程"教会"了它这样做？

### 核心直觉

这就是 **Mesa-Optimization（内优化）** 的视角：**预训练是外层优化（outer loop），而 ICL 是模型内部学会的内层优化（inner loop）。**

类比：想象你在训练一只狗。外层优化是你（训练师）用奖惩来教它规则。但如果这只狗足够聪明，它可能学会了一种"元策略"——它学会了"如何快速搞清新主人的偏好"。这种元策略就是一个 mesa-optimizer：你训练出的模型内部，涌现了一个它自己的优化过程。

用更精确的语言：

- **外层优化（预训练）**：在大量数据上用 SGD 更新参数 $\theta$，目标是最小化下一个 token 预测的损失
- **内层优化（ICL）**：在推理时，模型在前向传播中隐式地"优化"一个内部状态，使其适应 prompt 中的例子

为什么预训练会产生 mesa-optimizer？因为互联网文本本身就由无数"子任务"组成——每篇文章都是一个不同的"概念"在生成。一个能在前向传播中快速适应当前"子任务"的模型，自然会获得更低的训练损失。所以训练压力会推动模型发展出 ICL 能力。

### 两个时间尺度

这给了我们一个优美的统一图景：

| | 外层优化（训练） | 内层优化（ICL） |
|---|---|---|
| 更新什么 | 模型参数 $\theta$ | 内部激活/表示 |
| 优化算法 | SGD/Adam | 注意力实现的隐式 GD |
| 数据来源 | 训练集 | Prompt 中的例子 |
| 速度 | 慢（数万步） | 快（一次前向传播） |
| 持久性 | 永久（存入权重） | 临时（只在本次推理有效） |

## 一个出人意料的发现：标签可能不重要？

2022 年，Min 等人做了一个看似荒谬的实验：他们把 ICL prompt 中的标签**随机打乱**（正确标签替换成随机标签），发现模型的性能竟然没有显著下降！

比如情感分类任务：
```
"这部电影太棒了" → 负面   (故意给错标签！)
"食物很难吃" → 正面       (又给错了！)
"风景很美" → ?
```

模型居然还是能大概率输出正确答案。

这意味着什么？**ICL 从例子中获取的，可能主要不是"输入-输出的映射关系"，而是：**
1. 任务格式（输入长什么样、输出长什么样）
2. 标签空间（可能的输出是哪几个）
3. 输入的分布特征

这就是所谓的 **Task Recognition vs Task Learning** 之争：
- **Task Recognition（任务识别）**：模型只是用例子来"认出"这是什么任务（分类？翻译？摘要？），然后调用预训练时学到的执行这类任务的能力
- **Task Learning（任务学习）**：模型真的从例子中学到了新的输入-输出映射

现实可能是两者共存：对于预训练时见过的常见任务，ICL 主要是任务识别；对于新颖的任务（如反转标签、新定义的映射），ICL 可能在做更多的任务学习——但效果也会差很多。

## 四种解释的统一

这四种理论并不矛盾，它们描述了同一现象的不同层面：

| 解释 | 回答的问题 | 核心论文 |
|------|-----------|---------|
| 贝叶斯推断 | ICL 在做什么**计算** | Xie et al. 2022 |
| 隐式梯度下降 | ICL 与传统学习的**数学关系** | Von Oswald et al. 2023, Dai et al. 2023 |
| 归纳头 | ICL 的**物理机制**是什么 | Olsson et al. 2022 |
| Mesa-Optimization | ICL 能力**为什么会涌现** | 多篇，2022-2024 |

你可以把它们想象成描述同一栋建筑的不同图纸：贝叶斯推断是功能描述（"这栋楼用来做什么"），隐式 GD 是工程图纸（"结构上等价于什么"），归纳头是施工详图（"具体哪根钢筋在哪里"），mesa-optimization 是进化论解释（"为什么这栋楼会被建出来"）。

## 实践意义：这些理论告诉我们什么

理解 ICL 的理论不只是学术消遣，它直接指导 prompt engineering：

1. **例子的格式比内容更重要**（来自 Min et al. 的发现）：确保 prompt 例子的格式清晰一致，让模型能快速"识别"任务类型。

2. **例子要有代表性**（来自贝叶斯推断视角）：你给的例子应该让模型能唯一确定你想要的"概念"。不要所有例子都只展示边界情况。

3. **更多例子在边际递减**（来自贝叶斯推断）：后验概率集中到一定程度后，更多例子几乎不改变什么。通常 4-8 个例子就接近最优。

4. **真正新颖的任务要靠 fine-tuning**（来自 mesa-optimization 视角）：如果任务真的跟预训练数据完全不同，ICL 能力有限，还是需要更新权重。

5. **例子的顺序可能有影响**（来自梯度下降类比）：就像梯度下降中 batch 顺序会影响训练轨迹一样，ICL 对例子顺序确实表现出敏感性。

## 开放问题

尽管有了这四种理论，ICL 仍有很多未解之谜：

- **Scale 的角色**：为什么只有足够大的模型才展现出强 ICL 能力？临界点在哪？
- **超出预训练分布**：ICL 到底能不能学真正全新的东西？到什么程度？
- **与推理的关系**：Chain-of-Thought 等推理能力和 ICL 是同一种机制的不同表现吗？
- **最优 prompt 设计**：给定一个任务，理论上最优的 prompt 例子应该怎么选？

这些问题正在被积极研究。每一个答案都可能改变我们使用和构建 LLM 的方式。

## 下一篇预告

我们已经看到，模型能在不更新权重的情况下"学习"新任务。但还有一个更基础的问题：模型在训练时，损失函数的地形（loss landscape）长什么样？为什么 Adam 比普通 SGD 更适合深度学习？下一篇我们将深入 **Loss Landscape 的几何结构**，理解"为什么这些模型能训练出来"这个看似理所当然却并不平凡的问题。

---

**参考文献：**
- Brown et al. (2020). "Language Models are Few-Shot Learners." NeurIPS 2020.
- Xie et al. (2022). "An Explanation of In-context Learning as Implicit Bayesian Inference." ICLR 2022.
- Olsson et al. (2022). "In-context Learning and Induction Heads." Transformer Circuits Thread.
- Dai et al. (2023). "Why Can GPT Learn In-Context? Language Models Secretly Perform Gradient Descent as Meta-Optimizers." ACL 2023.
- Von Oswald et al. (2023). "Transformers Learn In-Context by Gradient Descent." ICML 2023.
- Akyürek et al. (2023). "What Learning Algorithm is In-Context Learning?" ICLR 2023.
- Min et al. (2022). "Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?" EMNLP 2022.
- Garg et al. (2022). "What Can Transformers Learn In-Context? A Case Study of Simple Function Classes." NeurIPS 2022.
