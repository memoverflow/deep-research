---
title: "BPE vs WordPiece vs Unigram：三种切词算法对决"
date: 2025-05-15
level: 3
series: "理解 Tokenization"
series_order: 2
series_total: 3
tags: [BPE, WordPiece, Unigram, SentencePiece, tiktoken]
summary: "同一段话，GPT-4 和 BERT 切出来的 token 完全不同。BPE 看频率，WordPiece 看统计显著性，Unigram 用概率论。三种哲学，殊途同归。"
---

> BPE 说："出现最多的就合并。" WordPiece 说："统计上最不寻常的组合才值得合并。" Unigram 说："我从大词表开始，把没用的砍掉。" 三种完全不同的哲学，最终都产出 32K-128K 的子词词表。

## 三条路，一个目的

所有 tokenizer 算法的目标都一样：**找到一组子词（subwords），让它们能高效地表示任意文本**。但"高效"怎么定义？这就是分歧所在。

<svg viewBox="0 0 650 160" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:650px;margin:20px auto;display:block;">
  <!-- BPE -->
  <rect x="15" y="30" width="190" height="110" rx="8" fill="#1a1a24" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="110" y="20" text-anchor="middle" fill="#22d3ee" font-size="10" font-family="system-ui" font-weight="bold">BPE (GPT/LLaMA)</text>
  <text x="110" y="52" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">方向: 自底向上 ↑</text>
  <text x="110" y="72" text-anchor="middle" fill="#9494a0" font-size="8" font-family="system-ui">策略: 合并最频繁的对</text>
  <text x="110" y="92" text-anchor="middle" fill="#9494a0" font-size="8" font-family="system-ui">char → subword → word</text>
  <text x="110" y="115" text-anchor="middle" fill="#22d3ee" font-size="8" font-family="system-ui">简单、快速、确定性</text>
  <text x="110" y="132" text-anchor="middle" fill="#6b6b78" font-size="7" font-family="system-ui">GPT-2/3/4, LLaMA, Mistral</text>
  <!-- WordPiece -->
  <rect x="225" y="30" width="190" height="110" rx="8" fill="#1a1a24" stroke="#34d399" stroke-width="1.5"/>
  <text x="320" y="20" text-anchor="middle" fill="#34d399" font-size="10" font-family="system-ui" font-weight="bold">WordPiece (BERT)</text>
  <text x="320" y="52" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">方向: 自底向上 ↑</text>
  <text x="320" y="72" text-anchor="middle" fill="#9494a0" font-size="8" font-family="system-ui">策略: 合并 PMI 最高的对</text>
  <text x="320" y="92" text-anchor="middle" fill="#9494a0" font-size="8" font-family="system-ui">freq(ab)/(freq(a)×freq(b))</text>
  <text x="320" y="115" text-anchor="middle" fill="#34d399" font-size="8" font-family="system-ui">统计显著性 > 原始频率</text>
  <text x="320" y="132" text-anchor="middle" fill="#6b6b78" font-size="7" font-family="system-ui">BERT, DistilBERT, Electra</text>
  <!-- Unigram -->
  <rect x="435" y="30" width="190" height="110" rx="8" fill="#1a1a24" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="530" y="20" text-anchor="middle" fill="#fbbf24" font-size="10" font-family="system-ui" font-weight="bold">Unigram (T5/mBART)</text>
  <text x="530" y="52" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">方向: 自顶向下 ↓</text>
  <text x="530" y="72" text-anchor="middle" fill="#9494a0" font-size="8" font-family="system-ui">策略: 从大词表中剪枝</text>
  <text x="530" y="92" text-anchor="middle" fill="#9494a0" font-size="8" font-family="system-ui">移除对 likelihood 影响最小的</text>
  <text x="530" y="115" text-anchor="middle" fill="#fbbf24" font-size="8" font-family="system-ui">概率模型, 多种切法加权</text>
  <text x="530" y="132" text-anchor="middle" fill="#6b6b78" font-size="7" font-family="system-ui">T5, mBART, XLNet, ALBERT</text>
</svg>

## BPE：谁出现多就合并谁

上一篇详细讲过。核心逻辑：数所有相邻对的频率，合并最多的那个。简单粗暴。

`"th"` 在英语中出现了 10 万次？合并。不管 `t` 和 `h` 各自有多常见。

## WordPiece：统计上"不寻常"的组合才有价值

WordPiece 的打分方式不同：

```
score(a, b) = freq(ab) / (freq(a) × freq(b))
```

这其实就是**互信息（PMI）**——衡量两个符号一起出现的频率，**相对于它们各自独立出现的频率**，高出了多少。

**一个关键区别：**

假设 `"th"` 出现 1000 次，但 `"t"` 出现 5 万次，`"h"` 出现 3 万次：
- BPE 分数：1000（频率高！优先合并）
- WordPiece 分数：1000/(50000×30000) = 极低（`t` 和 `h` 各自太常见了，一起出现不稀奇）

而 `"qi"` 出现 100 次，`"q"` 出现 150 次，`"i"` 出现 200 次：
- BPE 分数：100（低于 "th"）
- WordPiece 分数：100/(150×200) = 很高！（`q` 和 `i` 一起出现的概率**远超随机预期**）

WordPiece 会优先合并 `"qi"`，因为它是一个**统计上有意义的组合**——而 BPE 会优先合并 `"th"`，仅仅因为它频繁。

**BERT 的 `##` 前缀：** WordPiece 用 `##` 标记续接片段。`"playing"` → `["play", "##ing"]`。这区分了独立词 `"in"` 和后缀 `"##in"`——BPE 不做这个区分。

## Unigram：逆向思维——从大到小

Unigram 完全是另一个方向：

1. 从一个**巨大的候选词表**开始（所有可能的子串，10 万+）
2. 用 EM 算法给每个 token 分配概率
3. 计算"如果删掉这个 token，整体 likelihood 下降多少"
4. 删掉影响最小的 10-20%
5. 重复，直到词表缩小到目标大小

**Unigram 的独特之处：** 它是概率性的。对于 `"tokenization"`，它会同时考虑多种切法：
- `["token", "ization"]` — 概率 0.03
- `["token", "iza", "tion"]` — 概率 0.02
- `["to", "ken", "ization"]` — 概率 0.001

训练时加权所有切法，推理时选概率最高的那个（Viterbi 解码）。

## SentencePiece：不是算法，是框架

SentencePiece 经常被和 Unigram 搞混。它其实是一个**库**，可以实现 BPE 或 Unigram。它的创新在于：
- 把输入视为**原始字节流**，不做任何语言相关的预处理
- 用 `▁`（下划线）表示空格，所以 `"Hello World"` → `["▁Hello", "▁World"]`
- 语言无关——中文、阿拉伯语、代码都一样处理

LLaMA 1/2 用 SentencePiece 实现 BPE，T5 用 SentencePiece 实现 Unigram。

## Byte-Level BPE：GPT-2 的发明

GPT-2 的创新：BPE 不在 Unicode 字符上操作，而是在**原始字节（256 个）**上操作。

好处：
- **永远不会 OOV**——任何输入（emoji、任何语言、甚至二进制数据）都能被表示
- 基础词表只有 256 个
- 多字节字符（如中文）一开始是多个 byte token，训练后常用的会合并为单个 token

## tiktoken vs HuggingFace Tokenizers

| | tiktoken (OpenAI) | HuggingFace Tokenizers |
|---|---|---|
| 语言 | Rust + Python | Rust + Python |
| 速度 | 极快 (3-6× faster) | 快 |
| 算法 | 仅 BPE | BPE + WordPiece + Unigram |
| 灵活性 | 只用于 OpenAI 模型 | 任意模型 |
| 特色 | regex 预切分 | 可训练自定义 tokenizer |

tiktoken 更快是因为它高度特化——只做一件事，做到极致。

## 词表大小的权衡

**大词表（128K，如 LLaMA 3）：**
- ✅ 序列更短（同等内容少用 token） → 推理更快、context window 装更多
- ✅ 多语言更友好
- ❌ Embedding 矩阵更大（128K × 4096 = 5.4 亿参数）
- ❌ 罕见 token 训练信号不足

**小词表（32K，如 Mistral）：**
- ✅ Embedding 小，模型更紧凑
- ✅ 每个 token 被充分训练
- ❌ 序列更长
- ❌ 非英语支持差

**趋势：** 词表在变大。因为推理成本（按 token 计费）比训练成本（embedding 参数）更重要——压缩一个 token 节省的推理计算远超 embedding 多占的内存。

下一篇来看 tokenization 在实际工程中的坑：token healing、多语言不公平、以及为什么 LLM 数不清字母。
