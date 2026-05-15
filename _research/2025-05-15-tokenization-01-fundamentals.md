---
title: "什么是 Token？为什么 AI 不直接读文字"
date: 2025-05-15
level: 3
series: "理解 Tokenization"
series_order: 1
series_total: 6
tags: [tokenization, BPE, vocabulary, LLM]
summary: "ChatGPT 数不清 strawberry 里有几个 r——因为它根本看不到单个字母。它看到的是 'token'：比字母大、比单词小的文本碎片。BPE 算法决定了这些碎片怎么切。"
---

> 你有没有好奇过，为什么 ChatGPT 数不清 "strawberry" 里有几个 r？为什么同样的内容，中文用的 token 比英文多好几倍？答案藏在 tokenization 这个看似简单的步骤里。

## 一个根本性的问题：机器不认字

神经网络是数学机器。它做矩阵乘法、向量加法、非线性变换——但它**不认字**。你不能把字符串 "Hello" 直接塞给它。

所以我们需要一个翻译层：**文字 → 数字 ID → 向量（embedding）**。这个"文字 → 数字 ID"的过程就是 tokenization。

但关键问题是：**怎么切？** 按什么粒度把文字切成一个个 token？

## 三种切法和它们的问题

**按字符切：** `"Hello" → ['H','e','l','l','o'] → [72, 101, 108, 108, 111]`

词表很小（只需 256 个 ASCII 或几千个 Unicode 字符），但序列变得**极长**。一篇 1000 词的文章变成 5000+ token。而 attention 是 O(n²) 的——序列长度翻倍，计算量翻四倍。而且单个字符几乎没有语义信息。

**按单词切：** `"Hello world" → ['Hello', 'world'] → [4521, 8923]`

语义丰富，序列短。但词表爆炸（英语 17 万单词 + 变形 + 俚语 + 专有名词...），而且遇到新词（"ChatGPT"、"defenestration"）就只能标记为 `<UNK>`——信息丢失。

**按子词切（subword）：** `"unhappiness" → ['un','happi','ness'] → [359, 43922, 2516]`

这是现代 LLM 的统一选择：
- 词表可控（32K-200K）
- 常用词保持完整（"the"、"is" 是单个 token）
- 罕见词被拆成有意义的碎片（"un" + "happi" + "ness"）
- 永远不会遇到 OOV（未知词）

但具体怎么决定"哪些碎片应该合并成一个 token"？答案是 **BPE（Byte Pair Encoding）**。

## BPE：交互式动画

<iframe src="/assets/bpe-animation.html" width="100%" height="530" style="border:1px solid #23232e; border-radius:12px; background:#0a0a0f;" loading="lazy"></iframe>

<p style="color:#6b6b78; font-size:0.85em; text-align:center; margin-top:8px;">↑ 点击「下一步」逐步观察 BPE 如何从字符开始，一步步合并成子词 token</p>

## BPE 的核心逻辑

BPE 的算法极其简洁：**不断合并语料中出现频率最高的相邻字符对。**

从一个小例子理解：

假设语料是：`"low"×5, "lower"×2, "newest"×6, "widest"×3`

1. **起点**：把所有词拆成字符 → `[l,o,w], [l,o,w,e,r], [n,e,w,e,s,t], [w,i,d,e,s,t]`
2. **数相邻对**：`(e,s)` 出现 6+3=9 次最多
3. **合并**：`(e,s) → es`，现在 "newest" 变成 `[n,e,w,es,t]`
4. **重复**：再数，`(es,t)` 最多 → 合并为 `est`
5. 继续直到词表达到目标大小（如 100K）

每次合并就给词表加一个新 token。合并顺序就是词表的"构建历史"——推理时按同样顺序应用这些规则。

## 训练 vs 推理

**训练 tokenizer**（一次性，用大语料）：
- 统计全部相邻对频率
- 贪心合并最频繁的
- 保存：词表（token→ID）+ 合并规则列表

**使用 tokenizer**（每次推理都要做）：
- 把新文本拆成基础字符
- 按预存的合并规则顺序，逐步合并
- 相同文本 → 永远得到相同的 token 序列（确定性的）

## 实际的数字

| 模型 | 词表大小 | 算法 |
|------|----------|------|
| GPT-2 | 50,257 | Byte-level BPE |
| GPT-4 (cl100k) | 100,256 | BPE + regex 预切分 |
| GPT-4o (o200k) | ~200,000 | BPE + regex |
| LLaMA 2 | 32,000 | SentencePiece BPE |
| LLaMA 3 | 128,000 | BPE (tiktoken) |

为什么 LLaMA 3 从 32K 跳到 128K？因为 32K 词表对非英语语言太不友好——中文一个词可能要 3-4 个 token，扩大词表后压缩到 1-2 个。

## 一些让人惊讶的事实

**空格是 token 的一部分：** `" hello"`（带前导空格）和 `"hello"` 是**完全不同**的 token，有不同的 ID 和 embedding。正文中的词几乎都带着前导空格。

**大小写产生不同 token：** `"Hello"`, `"hello"`, `"HELLO"` 是三个完全不同的 token。

**ChatGPT 数不清字母：** `"strawberry"` 可能被切成 `["str","aw","berry"]`——模型看到的是 3 个子词 token，根本不知道里面有几个 r。

**中文用户付更多钱：** API 按 token 计费。同样的语义内容，中文可能需要 2-3× 的 token 数（因为词表主要基于英语训练），直接导致费用翻倍。

## 特殊 Token

除了从语料中学到的 token，还有一些手动添加的**特殊 token**：

- `<BOS>` / `<s>` — 序列开始
- `<EOS>` / `</s>` — 序列结束（模型学会在该停时输出它）
- `<|im_start|>` / `<|im_end|>` — ChatML 对话格式标记
- `<|tool_call|>` — 工具调用标记

这些是模型的"控制信号"——告诉它角色切换、对话边界、何时停止生成。

下一篇我们对比三种主流 tokenizer 算法（BPE vs WordPiece vs Unigram），看看同一段话被它们切出来有多不一样。
