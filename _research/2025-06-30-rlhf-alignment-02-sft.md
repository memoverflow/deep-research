---
title: "理解 RLHF 与对齐训练（二）：SFT——用示范教会模型「怎么说话」"
date: 2025-06-30
level: 3
series: "理解 RLHF 与对齐训练"
series_order: 2
series_total: 5
tags: [rlhf, sft, supervised-finetuning, instruction-tuning, alignment]
summary: "SFT 是对齐的第一步：用少量高质量示范数据，把一个'文字续写机器'变成一个'问答助手'"
---

# SFT：用示范教会模型「怎么说话」

> 预训练模型像一个读了所有书的学者。SFT 的作用，是教这个学者"请在别人问你问题时，用回答问题的方式说话"。听起来简单，但这一步改变了一切。

## 一个反直觉的观察

InstructGPT 的 SFT 阶段只用了大约 13,000 条标注数据。

13,000 条。

GPT-3 的预训练数据量是**几千亿** token。而仅仅 13,000 条微调数据，就能让模型的行为发生质的改变——从一个随机续写文本的生成器，变成一个能正经回答问题的助手。

这怎么可能？答案揭示了一个关于大语言模型的深刻真理。

## SFT 到底在做什么？

### 表面理解：教模型模仿好回答

SFT（Supervised Fine-Tuning，监督微调）的操作非常直观：

1. 收集一批 (用户问题, 优质回答) 的配对数据
2. 用标准的有监督学习（最小化交叉熵损失）微调预训练模型
3. 完成

损失函数就是语言模型经典的 next-token prediction，但只计算在回答部分的 loss：

$$\mathcal{L}_{\text{SFT}} = -\sum_{t} \log P_\theta(y_t | x, y_{<t})$$

其中 $x$ 是用户问题，$y$ 是标注者写的回答。

### 深层理解：激活已有的能力

但这只是表象。真正有趣的问题是：**为什么 13,000 条数据就够了？**

答案是：SFT 并没有教会模型新的知识或能力。这些能力在预训练阶段就已经学会了——模型已经知道什么是好的写作、什么是准确的信息、什么是有条理的回答。SFT 做的事情更像是"翻开正确的开关"。

类比：想象你雇了一个精通多国语言的翻译官，但他不知道自己被雇来做翻译。他可能会用这些语言能力写小说、做文字游戏、或者只是随便聊天。你需要做的不是教他语言（他已经会了），而是告诉他"当有人递给你一段英文时，请翻译成中文"。

**SFT 本质上是在做行为格式化（behavioral formatting）**——在模型已有的巨大能力空间中，标定一个特定的输出分布区域："当收到问题时，用这种方式回答。"

<svg viewBox="0 0 650 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:650px;margin:24px auto;display:block;">
  <defs>
    <marker id="arr3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Left circle: pretrain distribution -->
  <ellipse cx="160" cy="130" rx="130" ry="100" fill="none" stroke="#3a3a4a" stroke-width="1.5" stroke-dasharray="4,4"/>
  <text x="160" y="30" text-anchor="middle" fill="#888" font-size="11" font-family="system-ui">预训练模型的输出空间</text>
  <text x="80" y="80" fill="#ff6b6b" font-size="10" font-family="system-ui">小说续写</text>
  <text x="200" y="70" fill="#ff6b6b" font-size="10" font-family="system-ui">新闻报道</text>
  <text x="60" y="180" fill="#ff6b6b" font-size="10" font-family="system-ui">代码</text>
  <text x="220" y="190" fill="#ff6b6b" font-size="10" font-family="system-ui">诗歌</text>
  <!-- SFT target zone -->
  <ellipse cx="160" cy="130" rx="40" ry="30" fill="rgba(34,211,238,0.1)" stroke="#22d3ee" stroke-width="2"/>
  <text x="160" y="125" text-anchor="middle" fill="#22d3ee" font-size="11" font-weight="bold" font-family="system-ui">问答助手</text>
  <text x="160" y="142" text-anchor="middle" fill="#22d3ee" font-size="10" font-family="system-ui">行为模式</text>
  <!-- Arrow -->
  <line x1="310" y1="130" x2="370" y2="130" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arr3)"/>
  <text x="340" y="118" text-anchor="middle" fill="#6e8eff" font-size="10" font-family="system-ui">SFT</text>
  <text x="340" y="148" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">13K 数据</text>
  <!-- Right: focused distribution -->
  <ellipse cx="490" cy="130" rx="130" ry="100" fill="none" stroke="#3a3a4a" stroke-width="1" stroke-dasharray="4,4"/>
  <ellipse cx="490" cy="130" rx="45" ry="35" fill="rgba(52,211,153,0.15)" stroke="#34d399" stroke-width="2"/>
  <text x="490" y="125" text-anchor="middle" fill="#34d399" font-size="11" font-weight="bold" font-family="system-ui">专注于</text>
  <text x="490" y="142" text-anchor="middle" fill="#34d399" font-size="11" font-weight="bold" font-family="system-ui">问答模式</text>
  <text x="490" y="30" text-anchor="middle" fill="#888" font-size="11" font-family="system-ui">SFT 后的输出分布</text>
  <text x="490" y="200" text-anchor="middle" fill="#888" font-size="10" font-family="system-ui">其他能力仍在，但概率降低</text>
</svg>

## SFT 数据的构成

### 数据从哪来？

InstructGPT 的 SFT 数据来自三个来源：

1. **用户提交的真实问题**（通过 OpenAI API 收集）——保证了分布的真实性
2. **标注者自创的问题**——确保覆盖多样化的指令类型
3. **从少量种子任务扩展**——通过变换措辞、增加约束来增加多样性

### 什么算"好回答"？

标注者被要求遵守几个原则：

- **有用**：直接回答问题，不绕弯子
- **真实**：不编造信息，不确定时说"我不确定"
- **无害**：不产生歧视、暴力或危险内容
- **结构化**：分点列出、有条理、易于阅读

一个关键的设计选择：标注者**不需要是领域专家**。他们写的不是最专业的回答，而是一个有教养的、负责任的助手会给出的回答。这个标准虽然不完美，但它足够清晰，让不同标注者之间能保持一致性。

### 数据多样性比数据量更重要

后来的研究（LIMA, 2023）进一步证实了一个激进的观点：**SFT 可能只需要 1000 条精选数据**。LIMA 用仅仅 1000 条高质量示范数据微调 LLaMA-65B，效果接近用几万条数据的 GPT-4。

这强化了我们之前的洞察：SFT 不是在"注入新知识"，而是在"激活正确的行为模式"。只要示范数据覆盖了足够多的**格式类型**（问答、创作、分析、编程……），数量本身并不是关键。

## SFT 为什么有效？——更深的解释

### 从信息论角度

预训练后，模型的输出分布 $P_{\text{pretrain}}(y|x)$ 是一个非常宽泛的分布——给定任何输入，它对各种各样的续写都给出非零概率。

SFT 的本质是做 **分布窄化（distribution narrowing）**：把模型的输出分布从"所有统计上合理的续写"收窄到"符合助手行为规范的回答"。

$$P_{\text{SFT}}(y|x) \approx P_{\text{pretrain}}(y|x, \text{context}=\text{"你是一个有用的助手"})$$

某种意义上，SFT 等价于找到一个隐式的 system prompt，然后把它"烧录"到模型的权重里。

### 从优化景观角度

另一个理解方式：预训练已经把模型的参数带到了一个"平坦的高原"上——在这个区域内有很多不同的"好"配置。SFT 的梯度更新只需要做很小的参数移动，就能把模型推到这个平坦区域内一个特定的位置——"助手行为"对应的位置。

这解释了为什么：
- 需要的数据量很少（不需要翻越大的 loss barrier）
- 学习率要很小（2e-5 级别，避免跳出好的参数区域）
- Epoch 数通常只需 1-3（多了会过拟合到特定表达方式）

## SFT 的局限性

### 天花板问题

SFT 的上限取决于标注者能写出的最佳回答质量。问题是：

- 写出一个好回答**很难**（需要专业知识、写作能力、耐心）
- 写出一个**最优**回答几乎不可能（总有更好的措辞、更全面的覆盖）
- 但**判断**两个回答哪个更好要容易得多

这个不对称性——"评判比创作容易"——正是 RLHF 第二阶段（奖励模型）存在的根本原因。

### 模式模仿的局限

SFT 训练出的模型学到的是**标注者行为的表面模式**。如果标注者倾向于写很长的回答，模型就会学到"回答要长"。如果标注者倾向于使用特定的句式开头（"好的，让我来解释……"），模型也会模仿。

但模型学到的是"形"而非"神"——它模仿了**什么样的回答看起来像好回答**，但没有真正理解**好回答好在哪里**。这种浅层模仿会导致：

- 有时候答案看起来很专业，但内容是错的（幻觉）
- 模型学会了"回答的格式"，但对内容质量的判断力仍然不够
- 面对边界情况（模糊请求、有争议话题），模型缺乏内在的"判断标准"

### SFT 是必要的但不充分的

总结来说，SFT 解决了一个基础问题（让模型知道"该回答问题"），但留下了一个更深层的问题（让模型知道"什么是好回答"）。后者需要更精细的信号——这就是奖励模型的舞台。

## 现代 SFT 的实践要点

如果你今天要对一个开源模型做 SFT，以下是经过大量实践验证的关键参数：

| 参数 | 推荐值 | 原因 |
|------|--------|------|
| 学习率 | 1e-5 ~ 2e-5 | 过大会破坏预训练知识 |
| Epoch | 1-3 | 多了过拟合表面模式 |
| 数据量 | 5K-50K | 质量 >> 数量 |
| 数据多样性 | 覆盖 10+ 任务类型 | 格式多样性是关键 |
| Loss 计算 | 只算回答部分 | 问题部分不需要学习 |
| 上下文长度 | 2048-8192 | 根据目标应用调整 |

## 下一篇预告

SFT 教会了模型"回答问题"，但它无法教会模型"什么是好回答"。下一步，我们需要一种方法，把人类脑中那些模糊的、直觉性的偏好判断（"这个回答感觉更好"）转化为精确的数字信号。这就是奖励模型的工作——而它建立在一个 1952 年的数学模型之上：Bradley-Terry 模型。下一篇，我们深入这个巧妙的数学工具。
