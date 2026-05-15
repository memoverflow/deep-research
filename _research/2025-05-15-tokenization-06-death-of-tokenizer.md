---
title: "Tokenizer 之死：当模型学会自己切词"
date: 2025-05-15
level: 3
series: "理解 Tokenization"
series_order: 6
series_total: 6
tags: [tokenization, BLT, MEGABYTE, byte-level, dynamic-patching, future]
summary: "固定词表、贪婪切分、数据分布不匹配——传统 tokenizer 的种种问题催生了一个激进的想法：直接处理原始字节，让模型自己决定怎么'看'文本。Byte Latent Transformer 正在把这个想法变成现实。"
---

> 如果 tokenization 带来了这么多问题——数学失败、安全漏洞、多语言不公平、字符级任务无能——那最彻底的解决方案是什么？**把 tokenizer 扔掉。**

## Tokenizer 的原罪

前面五篇，我们看到了 tokenization 的全貌：它是如何工作的、不同算法的取舍、工程陷阱、安全漏洞、以及多模态扩展。

但一个根本性的问题一直悬而未决：**为什么我们需要 tokenizer？**

答案很功利：因为 Transformer 的计算量与序列长度的平方成正比。如果直接处理字节（每个字符 1 byte），一段 1000 词的文本大约有 5000 个字节——比 token 化后的 ~1300 个 token 长了 4 倍。计算量增加 16 倍。

但 tokenizer 作为一个**固定的、在模型训练之前就确定的**预处理步骤，引入了一系列根本性问题：

1. **训练数据偏差**：tokenizer 在数据集 A 上训练，模型在数据集 B 上训练——两者的分布未必一致
2. **固定粒度**：不管文本多简单或多复杂，都用同样的切分规则
3. **字符盲区**：模型永远无法进行比 token 更细粒度的推理
4. **语言不公平**：英文天然占优，其他语言付出 2-10 倍的 token 成本
5. **域外脆弱**：遇到训练时没见过的领域（代码、数学符号），tokenizer 的切分可能完全不合理

如果有一种方法能直接处理字节，但又不用付出 O(n²) 的代价呢？

## MEGABYTE：第一次认真尝试

### 核心思路：多尺度解码

Meta 在 2023 年（NeurIPS）提出了 MEGABYTE，第一个认真挑战「字节级 LLM」的方案。

**想法很直觉：** 文本存在多尺度结构。单个字符变化很快（高频），但词和句子的语义变化慢（低频）。不同尺度的信息应该用不同大小的模型处理。

**架构：**
- 把字节流切成固定大小的 **patch**（比如每 8 个字节一组）
- **全局模型**（大）：处理 patch 之间的关系——相当于「高层思考」
- **局部模型**（小）：在每个 patch 内部逐字节生成——相当于「打字」

<svg viewBox="0 0 750 230" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:750px;margin:24px auto;display:block;">
  <defs>
    <marker id="arr6" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <rect x="0" y="0" width="750" height="230" rx="12" fill="#13131a" stroke="#23232e" stroke-width="1"/>
  <text x="375" y="20" text-anchor="middle" fill="#6b6b78" font-size="12" font-family="system-ui">MEGABYTE 多尺度架构</text>
  
  <!-- Byte stream -->
  <text x="30" y="48" fill="#6b6b78" font-size="10" font-family="system-ui">原始字节流：</text>
  <rect x="30" y="55" width="690" height="25" rx="4" fill="#1e1e2a" stroke="#23232e" stroke-width="1"/>
  <text x="375" y="72" text-anchor="middle" fill="#ededf0" font-size="10" font-family="monospace">T h e   c a t   s a t   o n   t h e   m a t .</text>
  
  <!-- Patches -->
  <text x="30" y="97" fill="#6b6b78" font-size="10" font-family="system-ui">固定 Patch (每 8 字节)：</text>
  <rect x="30" y="103" width="170" height="25" rx="4" fill="#22d3ee11" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="115" y="119" text-anchor="middle" fill="#22d3ee" font-size="10" font-family="monospace">T h e   c a t  </text>
  <rect x="210" y="103" width="170" height="25" rx="4" fill="#34d39911" stroke="#34d399" stroke-width="1.5"/>
  <text x="295" y="119" text-anchor="middle" fill="#34d399" font-size="10" font-family="monospace">s a t   o n  </text>
  <rect x="390" y="103" width="170" height="25" rx="4" fill="#fbbf2411" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="475" y="119" text-anchor="middle" fill="#fbbf24" font-size="10" font-family="monospace">t h e   m a t .</text>
  
  <!-- Global model -->
  <line x1="115" y1="128" x2="115" y2="148" stroke="#6e8eff" stroke-width="1" marker-end="url(#arr6)"/>
  <line x1="295" y1="128" x2="295" y2="148" stroke="#6e8eff" stroke-width="1" marker-end="url(#arr6)"/>
  <line x1="475" y1="128" x2="475" y2="148" stroke="#6e8eff" stroke-width="1" marker-end="url(#arr6)"/>
  
  <rect x="60" y="150" width="470" height="35" rx="8" fill="#6e8eff22" stroke="#6e8eff" stroke-width="2"/>
  <text x="295" y="165" text-anchor="middle" fill="#6e8eff" font-size="12" font-family="system-ui">全局 Transformer（大模型）</text>
  <text x="295" y="180" text-anchor="middle" fill="#6b6b78" font-size="9" font-family="system-ui">处理 patch 间关系 — 语义级理解</text>
  
  <!-- Local models -->
  <line x1="115" y1="185" x2="115" y2="200" stroke="#a78bfa" stroke-width="1" marker-end="url(#arr6)"/>
  <line x1="295" y1="185" x2="295" y2="200" stroke="#a78bfa" stroke-width="1" marker-end="url(#arr6)"/>
  <line x1="475" y1="185" x2="475" y2="200" stroke="#a78bfa" stroke-width="1" marker-end="url(#arr6)"/>
  
  <rect x="55" y="202" width="120" height="22" rx="4" fill="#a78bfa22" stroke="#a78bfa" stroke-width="1"/>
  <text x="115" y="216" text-anchor="middle" fill="#a78bfa" font-size="9" font-family="system-ui">局部模型(小)</text>
  
  <rect x="235" y="202" width="120" height="22" rx="4" fill="#a78bfa22" stroke="#a78bfa" stroke-width="1"/>
  <text x="295" y="216" text-anchor="middle" fill="#a78bfa" font-size="9" font-family="system-ui">局部模型(小)</text>
  
  <rect x="415" y="202" width="120" height="22" rx="4" fill="#a78bfa22" stroke="#a78bfa" stroke-width="1"/>
  <text x="475" y="216" text-anchor="middle" fill="#a78bfa" font-size="9" font-family="system-ui">局部模型(小)</text>
  
  <!-- Key insight box -->
  <rect x="570" y="140" width="160" height="70" rx="6" fill="#34d39911" stroke="#34d399" stroke-width="1"/>
  <text x="650" y="160" text-anchor="middle" fill="#34d399" font-size="10" font-family="system-ui">优势：</text>
  <text x="650" y="176" text-anchor="middle" fill="#6b6b78" font-size="9" font-family="system-ui">✓ 无需 tokenizer</text>
  <text x="650" y="190" text-anchor="middle" fill="#6b6b78" font-size="9" font-family="system-ui">✓ 100万+ 字节上下文</text>
  <text x="650" y="204" text-anchor="middle" fill="#6b6b78" font-size="9" font-family="system-ui">⚠ patch 大小固定</text>
</svg>

**结果：** MEGABYTE 在语言建模上接近（但未超越）tokenized 模型，但在图像密度估计上达到 SOTA，而且能处理超过 100 万字节的序列。

**关键局限：** patch 大小是固定的。"The " 和 "Pneumono..." 用同样大小的 patch，计算资源分配不合理。

## Byte Latent Transformer：自适应的革命

### 关键洞察：不是所有字节同样重要

2024 年 12 月，Meta 的 Byte Latent Transformer（BLT）解决了 MEGABYTE 的核心局限：**让 patch 大小根据文本复杂度动态变化。**

**核心直觉：**

想象你在读一本书。"It was a" 这几个字你的眼睛一扫而过——太常见了，不需要细想。但遇到 "endoplasmic reticulum" 时，你需要停下来一个字母一个字母地辨认。

BLT 让模型也这么做：**简单的部分一口吞（大 patch），复杂的部分细嚼慢咽（小 patch）。**

### 怎么判断「复杂度」？用熵

BLT 用一个小型的字节级模型来估计每个位置的**熵**（entropy）——信息论中衡量「不确定性」的指标。

- 熵低 = 高度可预测 = "the cat sat on the " → 用大 patch（比如 8-12 字节一组）
- 熵高 = 不确定、复杂 = "Pneumonoultramicr..." → 用小 patch（2-3 字节一组）

<iframe src="/assets/blt-dynamic-patching-animation.html" width="100%" height="560"
  style="border:1px solid #23232e; border-radius:12px; background:#0a0a0f;"
  loading="lazy"></iframe>

<p style="color:#6b6b78; font-size:0.85em; text-align:center; margin-top:8px;">
  ↑ 点击"下一步"观察动态 patching 过程 | 点击右上角对比 BPE 的固定切分
</p>

### BLT 的三段式架构

```
字节流 → [局部编码器] → patches → [全局 Transformer] → patches → [局部解码器] → 字节
```

1. **局部编码器**：轻量模型，把字节分组为可变大小的 patch，每个 patch 产生一个向量
2. **全局 Transformer**：大模型，在 patch 级别做推理——序列长度大大缩短
3. **局部解码器**：轻量模型，把 patch 向量展开回字节序列

**类比：** 全局模型像是在看一幅马赛克画——每个马赛克块（patch）的大小根据那个区域的细节复杂度而变。空白天空 = 大块。人脸五官 = 小块。

### 性能：真的能用

BLT 的关键贡献不是提出想法（很多人想过字节级模型），而是**第一次在大规模上证明它可以匹配 tokenized 模型**：

- 在 Llama 3 相同计算预算下，BLT 匹配其性能
- 在字符级理解（CUTE benchmark）上大幅超越 Llama 3
- 推理效率更高（简单文本跳过更快）
- 天然鲁棒：拼写错误、Unicode 变体不再影响理解

### 2026 更新：Fast BLT

2026 年 5 月，Meta + Stanford 发布 Fast BLT，进一步优化推理：
- 内存带宽降低 >50%
- 无需 tokenizer 的开销完全可忽略
- 字节级模型的实用性到达临界点

## 动态 Tokenization：中间路线

完全抛弃 tokenizer 是最激进的路线。还有一条折中路线：**保留词表，但让切分动态适应输入。**

### Retrofitting Dynamic Tokenization（ACL 2025）

这篇论文的思路：
1. 保留预训练 LLM 的词表和权重
2. 加一个小网络来动态决定「在哪里切分」
3. 用 LoRA 微调适应新的切分方式

好处：不用从头训练模型，可以给现有 LLM「升级」tokenization 能力。

### zip2zip（2025）

更激进的想法：在**推理时**根据当前输入动态调整 tokenization。类似在线压缩——看到当前上下文后，决定接下来的文本怎么切分最高效。

## Tokenization Is More Than Compression

EMNLP 2024 的一篇重要论文挑战了传统观念。他们训练了 64 个语言模型（350M-2.4B 参数），系统性地改变 tokenization 的各个因素。

**关键发现：** BPE 的成功**不只是因为压缩率高**。

- 预分词规则（空格处理、标点处理）对下游性能的影响，比词表大小或压缩率的影响更大
- 两个压缩率相同的 tokenizer，下游性能可能差距巨大
- Tokenization 影响的是模型的「认知结构」，不只是「输入长度」

**翻译成人话：** tokenizer 不只是一个压缩工具，它在塑造模型怎么「思考」。怎么切词决定了模型把什么当作「原子单位」——这直接影响它能学到什么样的模式。

## 水印：Token 概率分布上的隐写

一个意想不到的 tokenization 应用：**AI 文本水印**。

### 原理

生成每个 token 时，模型有一个词表概率分布。水印系统做一个微妙的操作：

1. 用一个密钥把词表分成「绿色」和「红色」两组
2. 稍微提升绿色 token 的概率
3. 人类读不出区别，但统计检测可以发现绿色 token 出现频率异常高

<svg viewBox="0 0 700 180" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <rect x="0" y="0" width="700" height="180" rx="12" fill="#13131a" stroke="#23232e" stroke-width="1"/>
  <text x="350" y="20" text-anchor="middle" fill="#6b6b78" font-size="12" font-family="system-ui">Token 级别水印机制</text>
  
  <!-- Normal distribution -->
  <text x="175" y="45" text-anchor="middle" fill="#6b6b78" font-size="10" font-family="system-ui">正常生成</text>
  <rect x="30" y="55" width="290" height="60" rx="6" fill="#1e1e2a" stroke="#23232e" stroke-width="1"/>
  <!-- Bars -->
  <rect x="45" y="80" width="20" height="30" rx="2" fill="#6e8eff" opacity="0.8"/>
  <rect x="75" y="70" width="20" height="40" rx="2" fill="#6e8eff" opacity="0.8"/>
  <rect x="105" y="60" width="20" height="50" rx="2" fill="#6e8eff" opacity="0.8"/>
  <rect x="135" y="75" width="20" height="35" rx="2" fill="#6e8eff" opacity="0.8"/>
  <rect x="165" y="85" width="20" height="25" rx="2" fill="#6e8eff" opacity="0.8"/>
  <rect x="195" y="90" width="20" height="20" rx="2" fill="#6e8eff" opacity="0.8"/>
  <rect x="225" y="95" width="20" height="15" rx="2" fill="#6e8eff" opacity="0.8"/>
  <rect x="255" y="100" width="20" height="10" rx="2" fill="#6e8eff" opacity="0.8"/>
  <rect x="285" y="103" width="20" height="7" rx="2" fill="#6e8eff" opacity="0.8"/>
  <text x="175" y="125" text-anchor="middle" fill="#6b6b78" font-size="9" font-family="system-ui">词表概率分布 (均匀选择)</text>
  
  <!-- Arrow -->
  <text x="350" y="80" text-anchor="middle" fill="#fbbf24" font-size="20" font-family="system-ui">→</text>
  <text x="350" y="100" text-anchor="middle" fill="#fbbf24" font-size="9" font-family="system-ui">+水印</text>
  
  <!-- Watermarked distribution -->
  <text x="525" y="45" text-anchor="middle" fill="#6b6b78" font-size="10" font-family="system-ui">水印生成</text>
  <rect x="380" y="55" width="290" height="60" rx="6" fill="#1e1e2a" stroke="#23232e" stroke-width="1"/>
  <!-- Green tokens (boosted) -->
  <rect x="395" y="72" width="20" height="38" rx="2" fill="#34d399" opacity="0.9"/>
  <rect x="425" y="60" width="20" height="50" rx="2" fill="#34d399" opacity="0.9"/>
  <rect x="455" y="68" width="20" height="42" rx="2" fill="#34d399" opacity="0.9"/>
  <rect x="485" y="75" width="20" height="35" rx="2" fill="#34d399" opacity="0.9"/>
  <!-- Red tokens (suppressed) -->
  <rect x="525" y="95" width="20" height="15" rx="2" fill="#fb7185" opacity="0.6"/>
  <rect x="555" y="98" width="20" height="12" rx="2" fill="#fb7185" opacity="0.6"/>
  <rect x="585" y="100" width="20" height="10" rx="2" fill="#fb7185" opacity="0.6"/>
  <rect x="615" y="103" width="20" height="7" rx="2" fill="#fb7185" opacity="0.6"/>
  <rect x="645" y="105" width="20" height="5" rx="2" fill="#fb7185" opacity="0.6"/>
  <!-- Labels -->
  <text x="450" y="125" text-anchor="middle" fill="#34d399" font-size="9" font-family="system-ui">绿色 (概率↑)</text>
  <text x="590" y="125" text-anchor="middle" fill="#fb7185" font-size="9" font-family="system-ui">红色 (概率↓)</text>
  
  <!-- Detection -->
  <rect x="30" y="140" width="640" height="30" rx="6" fill="#34d39911" stroke="#34d399" stroke-width="1"/>
  <text x="350" y="158" text-anchor="middle" fill="#34d399" font-size="11" font-family="system-ui">检测：统计文本中绿色 token 的比例 → z-score 显著性检验 → 判定是否 AI 生成</text>
</svg>

Google 的 SynthID（2024 年开源）就是基于这个原理的生产级系统。NeurIPS 2024 论文形式化了三方权衡：可检测性 ↔ 文本质量 ↔ 鲁棒性（改写后还能检测到）。

水印的有趣之处在于：**它利用了 token 级生成的离散性质。** 在连续空间生成中（如 diffusion），这种精确的概率操控要难得多。

## 总结：Tokenization 的三个时代

回顾整个系列，tokenization 正在经历三个时代：

**第一代：固定规则**（2014-2018）
- 手工规则 + 频率统计
- BPE、WordPiece、Unigram
- 训练一次，永远不变

**第二代：学习优化**（2019-2024）
- Tokenizer 与模型联合优化
- 自适应词表大小
- 多语言公平性改进
- 仍然是固定的预处理步骤

**第三代：动态/消除**（2024-未来）
- Byte Latent Transformer：消除固定词表
- 动态 Patching：让模型自己决定粒度
- Fast BLT：性能可行性到达临界点
- 最终愿景：模型直接处理原始信号，tokenization 成为模型内部的可学习操作

## 最终思考

这个系列从一个简单的问题开始：「为什么 AI 不直接读文字？」

六篇文章之后，我们看到 tokenization 远不是一个「设置好就忘掉」的预处理步骤。它是：
- 模型的**视网膜**——决定能看到什么
- AI 系统的**安全边界**——但也是攻击面
- 多模态统一的**桥梁**——万物皆可 token
- 正在被消除的**瓶颈**——字节级模型正在崛起

下一代模型可能不再有固定的 tokenizer。但「如何将连续世界离散化为可处理的符号」这个根本问题，将以新的形式继续存在。

---

*「理解 Tokenization」系列完结。*
