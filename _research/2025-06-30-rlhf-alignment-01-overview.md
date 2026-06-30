---
title: "理解 RLHF 与对齐训练（一）：从预训练到对齐——为什么 ChatGPT 不只是一个更大的 GPT-3"
date: 2025-06-30
level: 3
series: "理解 RLHF 与对齐训练"
series_order: 1
series_total: 5
tags: [rlhf, alignment, instructgpt, chatgpt, pretraining, sft]
summary: "预训练语言模型的目标函数和人类需求之间存在根本性错位——RLHF 就是桥接这个 gap 的优雅方案"
---

# 从预训练到对齐：为什么 ChatGPT 不只是一个更大的 GPT-3

> GPT-3 有 1750 亿参数，InstructGPT 只有 13 亿。但人类评估者更喜欢 InstructGPT 的回答。这不是因为小模型"更聪明"，而是因为它终于学会了"用人类期望的方式使用它的智能"。

## 一个让所有人困惑的实验结果

2022 年 3 月，OpenAI 发表了 InstructGPT 论文。里面有一个看起来不可能的结论：

**一个 13 亿参数的模型，在人类评估中击败了 1750 亿参数的 GPT-3。**

参数量差了 100 多倍。按照"越大越好"的直觉，这就像一个高中生在专业考试中击败了一位教授。这怎么可能？

答案藏在一个核心洞察里：**模型的能力和模型的行为是两回事。** GPT-3 拥有惊人的语言能力和世界知识，但它不知道怎么"当一个好助手"。它的训练目标——预测下一个 token——和"做一个有用、安全、诚实的 AI 助手"之间，存在着根本性的错位。

InstructGPT 的秘密武器不是更多的参数，而是一套让模型"学会做人"的训练流程：RLHF（Reinforcement Learning from Human Feedback，基于人类反馈的强化学习）。

## 预训练模型到底"错"在哪里？

### 目标函数的错位

让我们先理解预训练的本质。GPT-3 的训练目标极其简单：

> 给定前面的所有文字，预测下一个最可能出现的词。

这个目标让模型学到了语法、常识、推理能力、甚至编程——但它学到的是**互联网文本的统计分布**，而不是"如何做一个好助手"。

打个比方：一个人读了人类有史以来的所有书籍，他会变成一个非常博学的人。但他不一定能当好一个客服代表——因为"博学"和"善于服务"是两种完全不同的能力。预训练给了模型"博学"，但没教它"服务"。

### 三个具体的问题

**1. 毒性（Toxicity）**

2020 年 Allen AI 的研究发现，即使是完全无害的提示词，GPT-2/GPT-3 也可能生成种族歧视、性别歧视或暴力内容。原因很简单：互联网数据里有大量有毒内容，模型忠实地学会了再现这些模式。对模型来说，生成一段仇恨言论和生成一首诗没有本质区别——都只是"统计上合理的文本续写"。

**2. 无用性（Unhelpfulness）**

如果你问 GPT-3"如何做番茄炒蛋"，它可能会给你一篇关于番茄的维基百科摘录，或者续写出一段小说对话。在预训练数据中，这种模式的出现概率确实很高。模型不知道你想要的是"一个直接、实用的菜谱"。

**3. 失控（Misalignment）**

模型可能帮用户编写恶意代码、编造看似可信的虚假信息（幻觉）、或在争议话题上表达极端立场。这些不是 bug——它们是预训练目标的自然后果。

<svg viewBox="0 0 700 280" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arr1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Left: Pretrain objective -->
  <rect x="20" y="30" width="200" height="220" rx="10" fill="#1e1e2a" stroke="#ff6b6b" stroke-width="1.5"/>
  <text x="120" y="60" text-anchor="middle" fill="#ff6b6b" font-size="13" font-weight="bold" font-family="system-ui">预训练目标</text>
  <text x="120" y="90" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">预测下一个 token</text>
  <text x="120" y="120" text-anchor="middle" fill="#888" font-size="11" font-family="system-ui">学会了：语法、知识、推理</text>
  <text x="120" y="150" text-anchor="middle" fill="#888" font-size="11" font-family="system-ui">没学会：什么时候该说什么</text>
  <line x1="120" y1="170" x2="120" y2="170" stroke="#3a3a4a" stroke-width="1"/>
  <text x="120" y="195" text-anchor="middle" fill="#ff6b6b" font-size="11" font-family="system-ui">❌ 有毒内容</text>
  <text x="120" y="215" text-anchor="middle" fill="#ff6b6b" font-size="11" font-family="system-ui">❌ 答非所问</text>
  <text x="120" y="235" text-anchor="middle" fill="#ff6b6b" font-size="11" font-family="system-ui">❌ 编造事实</text>
  <!-- Arrow -->
  <line x1="230" y1="140" x2="280" y2="140" stroke="#6e8eff" stroke-width="2" marker-end="url(#arr1)"/>
  <text x="255" y="125" text-anchor="middle" fill="#6e8eff" font-size="11" font-family="system-ui">RLHF</text>
  <!-- Middle: Gap -->
  <rect x="285" y="70" width="120" height="140" rx="8" fill="transparent" stroke="#6e8eff" stroke-width="1" stroke-dasharray="5,5"/>
  <text x="345" y="130" text-anchor="middle" fill="#6e8eff" font-size="12" font-family="system-ui">桥接 Gap</text>
  <text x="345" y="150" text-anchor="middle" fill="#888" font-size="10" font-family="system-ui">人类偏好 → 训练信号</text>
  <!-- Arrow -->
  <line x1="415" y1="140" x2="465" y2="140" stroke="#6e8eff" stroke-width="2" marker-end="url(#arr1)"/>
  <!-- Right: Aligned model -->
  <rect x="470" y="30" width="200" height="220" rx="10" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="570" y="60" text-anchor="middle" fill="#34d399" font-size="13" font-weight="bold" font-family="system-ui">对齐后的模型</text>
  <text x="570" y="90" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">最大化人类偏好</text>
  <text x="570" y="120" text-anchor="middle" fill="#888" font-size="11" font-family="system-ui">保留了：语法、知识、推理</text>
  <text x="570" y="150" text-anchor="middle" fill="#888" font-size="11" font-family="system-ui">新增了：知道该怎么回答</text>
  <text x="570" y="195" text-anchor="middle" fill="#34d399" font-size="11" font-family="system-ui">✓ 有用（Helpful）</text>
  <text x="570" y="215" text-anchor="middle" fill="#34d399" font-size="11" font-family="system-ui">✓ 无害（Harmless）</text>
  <text x="570" y="235" text-anchor="middle" fill="#34d399" font-size="11" font-family="system-ui">✓ 诚实（Honest）</text>
</svg>

## "对齐"到底是什么意思？

Anthropic 在 2021 年提出了一个被广泛接受的框架——**HHH 原则**：

**Helpful（有用）**：尽力帮助用户完成任务，提供准确、相关、完整的信息。

**Harmless（无害）**：不产生有毒内容，不协助危险活动，不操纵或欺骗用户。

**Honest（诚实）**：只说有合理信心的事情，承认不确定性，不编造信息。

这三者听起来很和谐，实际上存在**天然的张力**。一个极端"无害"的模型会变成那种什么都说"对不起我无法帮助您"的废物。一个极端"有用"的模型可能会帮用户做危险的事。RLHF 的核心挑战之一就是在这三个维度之间找到正确的平衡。

## 三阶段对齐管线：全景图

RLHF 并不是一个单一的技术，而是一个三阶段的训练流程。让我们先建立全局图景，后面的文章会逐一深入。

<svg viewBox="0 0 750 320" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:750px;margin:24px auto;display:block;">
  <defs>
    <marker id="arr2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Stage 1 -->
  <rect x="20" y="60" width="200" height="200" rx="10" fill="#1e1e2a" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="120" y="45" text-anchor="middle" fill="#22d3ee" font-size="14" font-weight="bold" font-family="system-ui">Stage 1: SFT</text>
  <text x="120" y="90" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">监督微调</text>
  <text x="120" y="120" text-anchor="middle" fill="#888" font-size="11" font-family="system-ui">输入：(问题, 示范回答)</text>
  <text x="120" y="145" text-anchor="middle" fill="#888" font-size="11" font-family="system-ui">方法：标准有监督学习</text>
  <text x="120" y="175" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">作用：教模型"回答问题"</text>
  <text x="120" y="195" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">而不是"续写文本"</text>
  <text x="120" y="230" text-anchor="middle" fill="#22d3ee" font-size="10" font-family="system-ui">~13K 示范数据</text>
  <text x="120" y="248" text-anchor="middle" fill="#888" font-size="10" font-family="system-ui">局限：受限于标注者写作能力</text>
  <!-- Arrow 1->2 -->
  <line x1="225" y1="160" x2="265" y2="160" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arr2)"/>
  <!-- Stage 2 -->
  <rect x="270" y="60" width="200" height="200" rx="10" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="370" y="45" text-anchor="middle" fill="#a78bfa" font-size="14" font-weight="bold" font-family="system-ui">Stage 2: RM</text>
  <text x="370" y="90" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">奖励模型训练</text>
  <text x="370" y="120" text-anchor="middle" fill="#888" font-size="11" font-family="system-ui">输入：(问题, 回答A, 回答B)</text>
  <text x="370" y="140" text-anchor="middle" fill="#888" font-size="11" font-family="system-ui">标注："A 比 B 好"</text>
  <text x="370" y="175" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">作用：把人类偏好</text>
  <text x="370" y="195" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">编码为可微分的分数</text>
  <text x="370" y="230" text-anchor="middle" fill="#a78bfa" font-size="10" font-family="system-ui">~33K 偏好比较数据</text>
  <text x="370" y="248" text-anchor="middle" fill="#888" font-size="10" font-family="system-ui">关键洞察：比较比示范容易得多</text>
  <!-- Arrow 2->3 -->
  <line x1="475" y1="160" x2="515" y2="160" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arr2)"/>
  <!-- Stage 3 -->
  <rect x="520" y="60" width="200" height="200" rx="10" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="620" y="45" text-anchor="middle" fill="#34d399" font-size="14" font-weight="bold" font-family="system-ui">Stage 3: RL (PPO)</text>
  <text x="620" y="90" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">强化学习优化</text>
  <text x="620" y="120" text-anchor="middle" fill="#888" font-size="11" font-family="system-ui">奖励：RM 打分</text>
  <text x="620" y="140" text-anchor="middle" fill="#888" font-size="11" font-family="system-ui">约束：不能偏离 SFT 太远</text>
  <text x="620" y="175" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">作用：最大化人类偏好</text>
  <text x="620" y="195" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">同时保持语言能力</text>
  <text x="620" y="230" text-anchor="middle" fill="#34d399" font-size="10" font-family="system-ui">KL 散度惩罚防止退化</text>
  <text x="620" y="248" text-anchor="middle" fill="#888" font-size="10" font-family="system-ui">最终目标：对齐的 LLM</text>
</svg>

### Stage 1: 监督微调（SFT）

第一步是"格式化"——让模型学会"被问问题 → 回答问题"的交互模式。就像教一个博学的教授"请用回答学生问题的方式说话"。

InstructGPT 收集了约 13,000 条高质量的 (问题, 示范回答) 数据对。标注者写出他们认为最好的回答，然后用标准的有监督学习微调模型。

**但 SFT 有一个根本局限**：标注者能**写出**的最佳回答，质量是有上限的。写出一个完美的技术解释可能需要 30 分钟，但从两个回答中**选出**更好的一个只需要 30 秒——而且选择判断往往更准确。

这个洞察直接导向了第二阶段。

### Stage 2: 奖励模型训练（RM）

这是 RLHF 最精妙的环节。给定同一个问题的多个回答，让人类标注者做偏好排序（"A 比 B 好"）。然后训练一个独立的模型，学习预测人类的偏好——输入一个 (问题, 回答)，输出一个分数。

**核心洞察**：人类的比较判断比直接示范更 cheap、更 scalable、更能捕捉微妙的偏好差异。你可能很难自己写出一首好诗，但你很容易判断两首诗哪首更好。

奖励模型本质上是把人类**隐式的**（存在于脑中的、无法直接写成代码的）偏好，转化为**显式的**、可微分的信号。

### Stage 3: 强化学习优化（PPO）

有了奖励模型作为"评委"，就可以用强化学习来优化策略了。模型生成回答 → RM 打分 → 模型根据分数调整参数 → 循环往复。

关键约束：加入 KL 散度惩罚，限制新策略不能偏离 SFT 模型太远。没有这个约束，模型会找到奖励模型的"漏洞"——生成一些 RM 给高分但人类觉得奇怪的退化输出。

最终优化目标可以用一行公式概括：

$$\text{maximize} \quad \mathbb{E}[R(x, y)] - \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{SFT}})$$

翻译成人话：**在让奖励模型满意的同时，别忘了自己是谁。**

## InstructGPT 的关键发现

InstructGPT 论文不只是提出了方法，还给出了几个改变行业认知的结论：

**1. 对齐 >> 规模：** 1.3B 参数的 InstructGPT 在人类评估中胜过 175B 的 GPT-3。这颠覆了"参数越多越好"的简单逻辑。

**2. "对齐税"很小：** 对齐训练后，模型在传统 NLP 基准上的性能几乎没有下降。对齐和能力并不严重冲突。

**3. 幻觉自然减少：** RLHF 训练后，模型编造虚假信息的频率显著降低——尽管这不是训练的显式目标。

**4. 数据效率极高：** 仅需约 33,000 条偏好比较数据，就能产生如此巨大的行为改变。

## 从 InstructGPT 到 ChatGPT

2022 年 11 月，OpenAI 发布了 ChatGPT。它本质上就是在更大的模型上应用了 InstructGPT 相同的 RLHF 管线。ChatGPT 的爆火验证了一个深刻的真理：

**不是模型变聪明了，而是模型终于学会了"说人话"。**

从技术角度看，ChatGPT 和 GPT-3 的根本区别不在于参数量或架构创新，而在于训练目标的转变：从"预测互联网上最可能出现的下一个词"到"生成人类觉得最好的回答"。

## 对齐研究的时间线

| 年份 | 里程碑 | 意义 |
|------|--------|------|
| 2017 | Christiano et al. "Deep RL from Human Preferences" | RLHF 奠基工作，Atari 游戏验证可行性 |
| 2020 | "Learning to Summarize from Human Feedback" | 首次在 NLP 任务上规模化应用 RLHF |
| 2022.03 | InstructGPT 论文 | 三阶段管线系统化，1.3B 胜 175B |
| 2022.11 | ChatGPT 发布 | RLHF 的产品化，震动全球 |
| 2022.12 | Constitutional AI | AI 反馈替代人类反馈（RLAIF） |
| 2023.05 | DPO 论文 | 颠覆性简化：无需 RM 和 PPO |
| 2024 | GRPO (DeepSeek) | 去掉 value function，推理任务大放异彩 |
| 2025 | 对齐训练民主化 | 开源模型普遍采用，不再是大厂专利 |

## 这个系列要带你去哪里

这是一个 5 篇的系列。我们会逐步深入 RLHF 的每一个环节：

- **本篇（第 1 篇）**：全局图景——为什么需要对齐，三阶段管线是什么
- **第 2 篇**：SFT——用示范数据教模型说人话的第一步
- **第 3 篇**：奖励模型——如何把"A 比 B 好"变成一个可优化的数字
- **第 4 篇**：PPO——用强化学习微调 LLM 的艺术与苦难
- **第 5 篇**：DPO 与 GRPO——跳过奖励模型的新范式

## 下一篇预告

我们说过，RLHF 的第一步是 SFT——监督微调。这听起来很简单：收集好的回答，训练模型模仿它们。但魔鬼藏在细节里：什么样的数据才算"好"？需要多少条？为什么 SFT 能用如此少的数据产生如此大的行为变化？下一篇，我们深入这些问题。
