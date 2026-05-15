---
title: "Tokenizer 的工程实战：那些让人踩坑的细节"
date: 2025-05-15
level: 3
series: "理解 Tokenization"
series_order: 3
series_total: 3
tags: [tokenization, 多语言, token-healing, 工程实践]
summary: "为什么 LLM 数不清字母？为什么中文用户 API 费用更高？为什么 prompt 末尾加个空格就影响生成质量？这些奇怪现象背后都是 tokenization 的锅。"
---

> Tokenization 看似只是预处理的第一步，但它的设计决策会渗透到模型的每一个角落——从计费到质量到安全性。这篇聊那些实际踩过的坑。

## 为什么 LLM 数不清字母

"strawberry 里有几个 r？"——这个经典问题难倒了很多 LLM。原因很简单：

`"strawberry"` 被 tokenized 为 `["str", "aw", "berry"]`（3 个 token）。模型看到的是这三个子词的 embedding——它**根本不知道每个 token 里有哪些字母**。

这不是"智力"问题，是**感知**问题。就像你看一个像素化的图片——信息在输入阶段就丢了。

<svg viewBox="0 0 600 100" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:600px;margin:20px auto;display:block;">
  <text x="300" y="15" text-anchor="middle" fill="#9494a0" font-size="9" font-family="system-ui">模型看到的 vs 你看到的</text>
  <!-- What you see -->
  <text x="70" y="45" text-anchor="middle" fill="#6b6b78" font-size="8" font-family="system-ui">你看到的:</text>
  <text x="70" y="65" text-anchor="middle" fill="#ededf0" font-size="12" font-family="monospace">s t r a w b e r r y</text>
  <!-- What model sees -->
  <text x="400" y="45" text-anchor="middle" fill="#6b6b78" font-size="8" font-family="system-ui">模型看到的:</text>
  <rect x="310" y="52" width="45" height="24" rx="4" fill="#1e1e2a" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="332" y="68" text-anchor="middle" fill="#22d3ee" font-size="10" font-family="monospace">str</text>
  <rect x="360" y="52" width="35" height="24" rx="4" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="377" y="68" text-anchor="middle" fill="#34d399" font-size="10" font-family="monospace">aw</text>
  <rect x="400" y="52" width="60" height="24" rx="4" fill="#1e1e2a" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="430" y="68" text-anchor="middle" fill="#fbbf24" font-size="10" font-family="monospace">berry</text>
  <text x="400" y="92" text-anchor="middle" fill="#fb7185" font-size="8" font-family="system-ui">← 3个不透明的"原子"，字母信息丢失</text>
</svg>

## 多语言不公平：Token Fertility

**Token fertility**（token 丰度）= 表达同样语义需要多少 token。

同样说"你好"：
- 英语 `"hello"` → 1 token
- 中文 `"你好"` → 可能 2-3 个 token（因为词表主要基于英语训练）
- 缅甸语 → 可能 5-8 个 token

**后果是三重的：**
1. **费用翻倍**：API 按 token 计费，中文用户为同等内容多付 2-3×
2. **Context window 缩水**：4096 token 上下文能装 ~3000 个英文词，但只能装 ~1000 个中文词
3. **质量下降**：碎片化的 token 更难学习语义

这就是为什么 LLaMA 3 把词表从 32K 扩到 128K——大量新增的 token 是中文、日文、韩文的常用词，让这些语言的 token fertility 降低。

## Token Healing：边界效应

这是一个微妙但重要的问题。假设你的 prompt 以 `"https://"` 结尾，让模型续写 URL。

**问题：** tokenizer 贪心地把 `"https://"` 切成了 `["https", "://"]`。但模型在训练时，几乎总是看到 `"://"` 和后面的内容连在一起（如 `"://www"` 是一个 token）。

现在模型被迫从 `"://"` 这个 token 的**边界**开始生成——这个起点在训练中极少见。结果可能是生成 `" www"`（多了个空格）或其他异常。

**研究发现：最常用的 10000 个 token 中，约 70% 是更长 token 的前缀。** 这意味着大部分 prompt 的末尾都可能碰到这个问题。

**Token healing 的解决方案（Guidance 库）：**
1. 往回退一个 token
2. 让模型在生成时被约束必须以这个 token 的内容开头
3. 模型自然地"修复"边界，生成 `"://www"` 这样的完整 token

## 空格是 Token 的一部分

这是最容易让人困惑的事实之一：

```
"hello"  → token ID 15339
" hello" → token ID 24748  ← 完全不同的 token！
```

正文中几乎所有词都带着前导空格——`" the"`, `" is"`, `" cat"` 都是独立的 token，和 `"the"`, `"is"`, `"cat"` 不同。

这解释了为什么：
- 在 prompt 末尾加/不加空格会影响生成质量
- `"Hello World"` 不是 `["Hello", "World"]` 而是 `["Hello", " World"]`（空格属于第二个 token）

## Chat Template：特殊 token 的格式战争

不同模型家族用不同的格式标记对话：

**OpenAI ChatML：**
```
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
```

**LLaMA 3：**
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are helpful.<|eot_id|>
```

这些特殊 token（`<|im_start|>` 等）是手动加入词表的，tokenizer 永远不会把它们拆开。它们是模型的"控制信号"——区分谁在说话、何时停止、何时调用工具。

如果你用错了 chat template（比如把 LLaMA 的格式喂给了 ChatML 模型），模型就会困惑——因为它从未在训练中见过这种 token 序列。

## 实用建议

1. **计费估算**：用 tiktoken 提前数 token。中文内容 × 2-3 估算。
2. **Prompt 设计**：注意末尾空格。固定格式的 prompt 有助于 prompt caching 命中。
3. **多语言**：如果主要服务中/日/韩用户，选用大词表模型（LLaMA 3 128K > LLaMA 2 32K）。
4. **调试**：当模型行为诡异时，先看 tokenization 结果——很多"幻觉"其实是 token 边界问题。
5. **选 tokenizer**：OpenAI 模型用 tiktoken，开源模型用 HuggingFace Tokenizers。

## 系列回顾

三篇文章走完了 tokenization 的完整图景：

1. **基础**：为什么需要 token，BPE 如何工作（附交互动画）
2. **算法对比**：BPE vs WordPiece vs Unigram 的不同哲学
3. **工程实战**：字母数不清、多语言不公平、token healing、空格陷阱

核心认识：**Tokenization 不是"预处理小事"，它定义了模型"看见"文本的方式。** 模型能力的上限，从 tokenizer 设计的那一刻就已经被划定了。
