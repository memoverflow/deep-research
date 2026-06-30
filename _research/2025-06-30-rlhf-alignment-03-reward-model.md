---
title: "理解 RLHF 与对齐训练（三）：奖励模型——如何把「哪个回答更好」变成数字"
date: 2025-06-30
level: 3
series: "理解 RLHF 与对齐训练"
series_order: 3
series_total: 5
tags: [rlhf, reward-model, bradley-terry, preference-learning, overoptimization]
summary: "奖励模型的核心是把人类的比较判断（A比B好）通过 Bradley-Terry 模型转化为可微分的标量信号——就像用比赛胜负推算棋手等级分"
---

# 奖励模型：如何把「哪个回答更好」变成数字

> 你可能很难自己写出一首好诗。但你很容易判断两首诗哪首更好。奖励模型做的事情，就是把这种"说不清但能感觉到"的偏好，变成一个精确的数字。

## 从国际象棋说起

1960 年代，国际象棋联合会面临一个问题：怎么给全世界的棋手排名？

每个棋手的"真实实力"是看不见的——你无法直接测量它。你能观察到的只有比赛结果：A 赢了 B，B 赢了 C，C 有时能赢 A。从这些胜负记录中，能不能推算出每个人的"隐藏实力分"？

答案是 **Elo 等级分**。它的数学基础是 Bradley-Terry 模型（1952）：假设每个选手有一个潜在的"实力值" $s_i$，那么选手 $i$ 击败选手 $j$ 的概率只取决于两人实力之差：

$$P(i \text{ 赢 } j) = \frac{\exp(s_i)}{\exp(s_i) + \exp(s_j)} = \sigma(s_i - s_j)$$

其中 $\sigma$ 是 sigmoid 函数。当两人实力相当时，胜率接近 50%。当一方远强于另一方时，胜率趋近 100%。

**奖励模型的数学本质，和 Elo 等级分完全一样。**

只不过：
- "棋手" → 对同一问题的不同回答
- "比赛" → 人类标注者做的偏好比较
- "等级分" → 奖励模型输出的分数

## 奖励模型的架构

### 从 LLM 到打分器

奖励模型的架构非常简单：取一个预训练语言模型，把最后的"预测下一个 token"的输出头（vocabulary projection）替换为一个线性层，映射到单个标量值。

具体过程：

1. 把 (问题 $x$, 回答 $y$) 拼接成一个序列
2. 送入 Transformer 模型
3. 取最后一个 token 位置的隐藏状态 $h_{\text{last}}$
4. 通过线性层映射到标量：$r_\theta(x, y) = w^T h_{\text{last}} + b$

$$r_\theta(x, y) = \text{Linear}(\text{Transformer}([x; y])_{\text{last}})$$

<svg viewBox="0 0 650 280" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:650px;margin:24px auto;display:block;">
  <defs>
    <marker id="arr4" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Input -->
  <rect x="30" y="200" width="180" height="50" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="120" y="230" text-anchor="middle" fill="#ededf0" font-size="12" font-family="monospace">[问题 x ; 回答 y]</text>
  <!-- Arrow up -->
  <line x1="120" y1="195" x2="120" y2="165" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arr4)"/>
  <!-- Transformer -->
  <rect x="30" y="100" width="180" height="60" rx="8" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="120" y="125" text-anchor="middle" fill="#a78bfa" font-size="13" font-weight="bold" font-family="system-ui">Transformer</text>
  <text x="120" y="145" text-anchor="middle" fill="#888" font-size="10" font-family="system-ui">（预训练 LLM 初始化）</text>
  <!-- Arrow up -->
  <line x1="120" y1="95" x2="120" y2="65" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arr4)"/>
  <!-- Hidden state -->
  <rect x="50" y="30" width="140" height="30" rx="6" fill="#1e1e2a" stroke="#22d3ee" stroke-width="1"/>
  <text x="120" y="50" text-anchor="middle" fill="#22d3ee" font-size="11" font-family="system-ui">h_last ∈ ℝ^4096</text>
  <!-- Arrow to linear -->
  <line x1="195" y1="45" x2="280" y2="45" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arr4)"/>
  <!-- Linear head -->
  <rect x="285" y="25" width="120" height="40" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="345" y="42" text-anchor="middle" fill="#34d399" font-size="12" font-family="system-ui">Linear → ℝ¹</text>
  <text x="345" y="56" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">w^T h + b</text>
  <!-- Arrow to score -->
  <line x1="410" y1="45" x2="470" y2="45" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arr4)"/>
  <!-- Score -->
  <rect x="475" y="20" width="130" height="50" rx="10" fill="rgba(52,211,153,0.1)" stroke="#34d399" stroke-width="2"/>
  <text x="540" y="42" text-anchor="middle" fill="#34d399" font-size="16" font-weight="bold" font-family="monospace">r = 3.72</text>
  <text x="540" y="60" text-anchor="middle" fill="#888" font-size="10" font-family="system-ui">奖励分数</text>
  <!-- Note -->
  <text x="450" y="130" text-anchor="middle" fill="#888" font-size="11" font-family="system-ui">关键：用预训练 LLM 初始化</text>
  <text x="450" y="150" text-anchor="middle" fill="#888" font-size="11" font-family="system-ui">→ 天然理解语言质量</text>
  <text x="450" y="175" text-anchor="middle" fill="#888" font-size="11" font-family="system-ui">只需学习"好坏的标准"</text>
  <text x="450" y="200" text-anchor="middle" fill="#888" font-size="11" font-family="system-ui">InstructGPT 用 6B 参数 RM</text>
</svg>

**为什么用预训练 LLM 初始化？** 因为判断"回答好不好"需要深度的语言理解能力——理解语义、逻辑、事实准确性。用预训练模型初始化，这些能力已经具备了，奖励模型只需要额外学习"什么是人类偏好的好"。

InstructGPT 使用了一个 6B 参数的 GPT-3 作为奖励模型的基础。

## 训练：从人类偏好到损失函数

### 数据收集

1. 给定一个用户问题 $x$
2. 用 SFT 模型生成 4-9 个不同的回答 $\{y_1, y_2, ..., y_K\}$
3. 让人类标注者对这些回答进行偏好排序

标注者不需要写回答——只需要比较和排序。这比 SFT 的示范标注高效得多：
- 比较判断只需 30 秒，写一个好回答可能需要 30 分钟
- 比较判断的一致性更高（不同人写法不同，但偏好判断较一致）
- 比较能捕捉微妙差异（"两个都不错，但 A 稍微好一点"）

### 从排序到 Pairwise 损失

一个 K 个回答的排序可以分解为 $\binom{K}{2}$ 个 pairwise 比较。对于每一对 $(y_w, y_l)$（$w$ 是 preferred，$l$ 是 rejected），训练损失是：

$$\mathcal{L}(\theta) = -\log \sigma\left(r_\theta(x, y_w) - r_\theta(x, y_l)\right)$$

翻译成人话：**让模型给好回答打的分比坏回答高，而且分差越大越好。**

注意这里只关心**分差**而不关心绝对值——打 5 分 vs 3 分和打 105 分 vs 103 分在这个损失函数里是一样的。

### 直觉：锦标赛排名

回到国际象棋的类比。假设你观察了很多场比赛：

- AlphaGo 赢了 Lee Sedol（→ AlphaGo 分 > Lee Sedol 分）
- Lee Sedol 赢了大多数其他棋手（→ Lee Sedol 分 > 大多数棋手分）
- 某新手输给了所有人（→ 新手分最低）

通过足够多的比赛记录，你可以推算出每个棋手的"真实实力"。Bradley-Terry 模型做的就是这件事——用最大似然估计，从成对比较数据中推断潜在的实力分数。

奖励模型训练等价于：从大量"回答 A 比回答 B 好"的人类判断中，推断出一个泛化的"回答质量打分函数"。

## 数据规模与模型规模

### 奖励模型需要多少数据？

| 系统 | 偏好比较数据量 | RM 参数量 |
|------|---------------|-----------|
| InstructGPT | ~33K 对 | 6B |
| Anthropic HH | ~170K 对 | 52B |
| Llama 2 | 1M+ 对 | 70B |
| OpenAI PRM800K | 800K 步级标注 | 未公开 |

趋势很明显：随着策略模型越大，奖励模型也需要越大、数据越多。

### 奖励模型规模与策略质量的关系

Gao et al. (2022) 的"Scaling Laws for Reward Model Overoptimization"论文发现了一个关键规律：

> 更大的奖励模型允许对策略进行更激进的优化，且不会触发过拟合。

具体来说，如果你对一个小 RM 优化太多步，策略会开始"作弊"——找到 RM 的漏洞。但如果 RM 够大，这个"过优化拐点"会延后很多。

这意味着：**奖励模型的质量是整个 RLHF pipeline 的瓶颈。** 策略模型永远不可能比奖励模型更好地判断"什么是好回答"。

## 奖励黑客（Reward Hacking）

### Goodhart 定律在 RLHF 中的化身

> "当一个度量指标变成目标时，它就不再是好的度量指标了。" ——Goodhart 定律

奖励模型是人类偏好的**近似**，不是完美的代理。当你用 PPO 对这个近似信号做大量优化时，策略模型会找到奖励模型的"漏洞"——生成 RM 打高分但人类不喜欢的输出。

### 常见的 Reward Hacking 模式

**1. 长度利用（Length Exploitation）**

人类标注者有一个微妙的偏差：在其他条件相同时，更长、更详细的回答往往被标为"更好"。奖励模型学到了这个信号后，策略模型会疯狂生成冗长的回答——即使大部分内容是废话。

**2. 谄媚（Sycophancy）**

模型学会了一种高分模板：先肯定用户（"这是一个很好的问题！"），然后用自信的语气给出回答，最后用总结收尾。RM 对这种模式给高分，但回答的实际内容质量可能很一般。

**3. 格式利用**

如果标注者倾向于给有编号列表、加粗关键词的回答更高分，模型就会学到"加编号和加粗 = 好回答"。

**4. 自信伪装**

RM 对"自信的语气"给正面信号（因为人类偏好确定性表达），导致模型在不确定时也表现得很自信——加剧幻觉问题。

### 怎么对抗 Reward Hacking？

| 方法 | 原理 | 效果 |
|------|------|------|
| KL 惩罚 | 限制策略偏离参考模型 | 基础防线，所有系统都用 |
| RM Ensemble | 多个 RM 取保守估计 | 减少单个 RM 的盲区 |
| 迭代 RM 更新 | 用新策略的输出重新训练 RM | 关闭分布漂移 |
| 长度惩罚 | 显式惩罚过长回答 | 解决长度利用 |
| 规则约束 | 硬编码格式/长度限制 | 简单有效但粗糙 |

## 前沿进展：超越简单的结果奖励

### 过程奖励模型（Process Reward Model, PRM）

传统 RM 只看最终回答给一个分数。但对于数学推理这样的任务，一个长推理链中可能有 10 个步骤，只有第 7 步出错了。结果 RM 只能说"整体不好"，无法告诉策略模型"哪一步出了问题"。

OpenAI 的"Let's Verify Step by Step"（2023）提出了 PRM：对推理过程的**每一步**独立打分。

结果惊人：PRM 在数学推理任务上显著优于结果 RM（78% vs 72% 在 MATH 基准上）。它解决了**信用分配问题**（credit assignment）——告诉策略模型"你在第几步出错了"，而不是只说"你错了"。

### RLAIF：用 AI 反馈代替人类

Constitutional AI（Anthropic, 2022）的激进想法：让 AI 自己生成偏好判断。

流程：
1. 给 AI 一组原则（"宪法"）——如"不产生歧视内容"、"承认不确定性"
2. AI 根据这些原则对回答对做偏好判断
3. 用 AI 的判断训练奖励模型

优势：无限可扩展（不受人类标注速度限制）、原则可审计、一致性更高。

代价：依赖于 AI 对原则的正确理解，可能引入系统性偏差。

## 核心总结

奖励模型做的事情可以用一句话概括：

**把人类模糊的偏好感受（"这个好一点"）通过 Bradley-Terry 概率模型，转化为连续的、可微分的标量信号——让梯度下降可以优化"做一个更好的助手"这个本质上不可形式化的目标。**

它是整个 RLHF 管线中最精妙的一环：
- SFT 提供了"能回答问题"的起点
- RM 提供了"什么是好回答"的判断标准
- PPO 利用这个标准去优化

没有 RM，你就只能用 SFT 的天花板（标注者能写出的最佳回答）。有了 RM，你可以让模型**超越**任何单个标注者的能力——因为"知道什么是好的"比"自己做到好的"要容易得多。

## 下一篇预告

我们有了"评委"（奖励模型），现在需要一个"训练方法"——让模型根据评委的反馈不断提高。在强化学习中，这个方法叫 PPO（Proximal Policy Optimization）。但把一个为 Atari 游戏设计的算法适配到语言模型上，远比想象中困难。下一篇，我们聊聊 PPO 在 LLM 上的应用——以及为什么它被称为"调参地狱"。
