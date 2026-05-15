---
title: "Token 的阴暗面：当切词出错时会发生什么"
date: 2025-05-15
level: 3
series: "理解 Tokenization"
series_order: 4
series_total: 6
tags: [tokenization, security, glitch-tokens, adversarial, math]
summary: "LLM 数不清字母、算不对数学、甚至会被'幽灵 token'搞崩溃——这些诡异行为的根源都是 tokenization。本篇揭示 token 切分带来的安全漏洞、推理失败和生产事故。"
---

> 你有没有想过，为什么 GPT-4 坚持说 9.11 大于 9.9？为什么 LLM 数不清 "strawberry" 里有几个 r？为什么有一个叫 "SolidGoldMagikarp" 的词能让模型精神崩溃？答案都指向同一个地方：tokenization。

## Tokenization 不只是预处理

前三篇我们讲了 tokenization 的基础：BPE 怎么切词、不同算法的取舍、词表大小的影响。听起来像是一个「设置好就忘掉」的预处理步骤。

但事实远非如此。**Tokenization 是 LLM 的视网膜——它决定了模型能「看到」什么。** 如果视网膜有盲区，模型就会在这些盲区上犯系统性的错误。更危险的是，攻击者可以利用这些盲区来绕过安全防线。

这一篇，我们来看 tokenization 的阴暗面。

## 第一幕：数学灾难——为什么 LLM 算不对数

### 9.11 > 9.9？模型的数字困惑

2024 年初，一个看似简单的问题在社交媒体上引发热议：GPT-4 声称 9.11 大于 9.9。

这不是模型「不聪明」——而是它**根本看不到你看到的东西**。

当你写 `9.11` 时，你看到的是数字 9.11（九点一一）。但模型看到的是三个 token：`9` + `.` + `11`。而 `9.9` 被切成 `9` + `.` + `9`。

模型的「推理」变成了：小数点后面，`11` 和 `9` 哪个大？11 > 9，所以 9.11 > 9.9。

**这不是推理错误，是感知错误。** 模型从未接收到「9.11 是一个数值」这个信息——它收到的是三个独立的符号碎片。

### 大数的崩溃

问题在小数字上还不明显。但一旦数字超过 4-5 位，灾难性的切分就开始了：

| 你写的 | 模型看到的 token | 后果 |
|--------|-----------------|------|
| 42 | `[42]` | ✓ 单 token，没问题 |
| 127 | `[127]` | ✓ 还行 |
| 1287 | `[12][87]` | ⚠ 数值结构丧失 |
| 31415 | `[31][415]` | ⚠ 不是按位切的！ |
| 1000000 | `[100][0000]` | ⚠ 百万 = 100 + 0000？ |

<svg viewBox="0 0 700 180" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <rect x="0" y="0" width="700" height="180" rx="12" fill="#13131a" stroke="#23232e" stroke-width="1"/>
  <!-- Human view -->
  <text x="100" y="30" text-anchor="middle" fill="#6b6b78" font-size="12" font-family="system-ui">你看到的</text>
  <rect x="30" y="40" width="140" height="40" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="100" y="65" text-anchor="middle" fill="#ededf0" font-size="18" font-family="monospace">31415926</text>
  <text x="100" y="95" fill="#34d399" font-size="11" text-anchor="middle" font-family="system-ui">一个完整的数字 (π×10⁷)</text>
  <!-- Arrow -->
  <line x1="180" y1="60" x2="240" y2="60" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arr)"/>
  <text x="210" y="80" text-anchor="middle" fill="#6b6b78" font-size="10" font-family="system-ui">BPE 切分</text>
  <!-- Model view -->
  <text x="440" y="30" text-anchor="middle" fill="#6b6b78" font-size="12" font-family="system-ui">模型看到的</text>
  <rect x="250" y="40" width="55" height="40" rx="6" fill="#1e1e2a" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="277" y="65" text-anchor="middle" fill="#ededf0" font-size="16" font-family="monospace">314</text>
  <rect x="315" y="40" width="55" height="40" rx="6" fill="#1e1e2a" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="342" y="65" text-anchor="middle" fill="#ededf0" font-size="16" font-family="monospace">159</text>
  <rect x="380" y="40" width="40" height="40" rx="6" fill="#1e1e2a" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="400" y="65" text-anchor="middle" fill="#ededf0" font-size="16" font-family="monospace">26</text>
  <text x="340" y="100" fill="#fb7185" font-size="11" text-anchor="middle" font-family="system-ui">三个毫无数学关联的碎片</text>
  <!-- Consequence -->
  <rect x="250" y="115" width="400" height="50" rx="8" fill="#fb718511" stroke="#fb7185" stroke-width="1"/>
  <text x="450" y="137" text-anchor="middle" fill="#fb7185" font-size="12" font-family="system-ui">后果：模型无法进行进位、对齐、位值运算</text>
  <text x="450" y="155" text-anchor="middle" fill="#6b6b78" font-size="11" font-family="system-ui">加法 31415926 + 1 可能得到完全错误的结果</text>
</svg>

DeepMind 的研究证实：**token 边界与数位边界的不对齐，是 LLM 数学推理失败的主要驱动力。** 不是模型笨——是它看不见数字的结构。

### "strawberry" 里有几个 r？

这是 2024 年最广为人知的 LLM 失败案例。ChatGPT 反复声称 "strawberry" 里有 2 个 r。

原因很简单：模型看到的不是 s-t-r-a-w-b-e-r-r-y 这 10 个字母，而是：

```
["str", "aw", "berry"]
```

三个不透明的 token。模型需要「知道」`str` 里面包含一个 r、`berry` 里面包含两个 r——但这种子 token 内部结构的知识，在训练中并不总能被可靠地学到。

**这揭示了一个根本性限制：LLM 在 token 级别操作，而字符级别的任务（数字母、反转字符串、回文检测）需要比 token 更细的粒度。**

<iframe src="/assets/token-dark-side-animation.html" width="100%" height="560"
  style="border:1px solid #23232e; border-radius:12px; background:#0a0a0f;"
  loading="lazy"></iframe>

<p style="color:#6b6b78; font-size:0.85em; text-align:center; margin-top:8px;">
  ↑ 点击标签切换演示 | "下一步"逐步观察 | 注意数字如何被错误切分
</p>

## 第二幕：Glitch Token——词表里的幽灵

### SolidGoldMagikarp 事件

2023 年 2 月，研究者发现了一个惊人的现象：GPT 的词表中存在一些「幽灵 token」——它们在训练数据中出现过（所以被 BPE 加入了词表），但在后续微调阶段几乎没有被见到过。

这些 token 包括：`SolidGoldMagikarp`、`attRot`、`TheNitromeFan`、`StreamerBot` 等看起来莫名其妙的字符串。

当你强制模型处理这些 token 时，会发生以下诡异行为：
- 模型**拒绝重复**这些词（"I cannot say that word"）
- 模型**陷入循环**，反复输出相同内容
- 模型**声称自己不是 AI**
- 模型产生**完全不连贯**的胡言乱语
- 模型表现出**「存在恐惧」**（"I'm afraid"）

### 为什么会这样？

想象一下：你的大脑中有一个「概念槽位」，但这个槽位从未被真正训练过。当信号被路由到这个未初始化的区域时，输出就是随机的、不可预测的。

<svg viewBox="0 0 700 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <rect x="0" y="0" width="700" height="200" rx="12" fill="#13131a" stroke="#23232e" stroke-width="1"/>
  <!-- Normal token path -->
  <text x="180" y="25" text-anchor="middle" fill="#34d399" font-size="12" font-family="system-ui">正常 Token</text>
  <rect x="30" y="35" width="90" height="35" rx="6" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="75" y="57" text-anchor="middle" fill="#ededf0" font-size="13" font-family="monospace">"hello"</text>
  <line x1="120" y1="52" x2="160" y2="52" stroke="#34d399" stroke-width="1.5" marker-end="url(#arr)"/>
  <rect x="160" y="35" width="100" height="35" rx="6" fill="#34d39922" stroke="#34d399" stroke-width="1"/>
  <text x="210" y="57" text-anchor="middle" fill="#34d399" font-size="11" font-family="system-ui">训练充分的嵌入</text>
  <line x1="260" y1="52" x2="300" y2="52" stroke="#34d399" stroke-width="1.5" marker-end="url(#arr)"/>
  <rect x="300" y="35" width="100" height="35" rx="6" fill="#34d39922" stroke="#34d399" stroke-width="1"/>
  <text x="350" y="57" text-anchor="middle" fill="#34d399" font-size="11" font-family="system-ui">有意义的输出</text>
  <!-- Glitch token path -->
  <text x="180" y="110" text-anchor="middle" fill="#fb7185" font-size="12" font-family="system-ui">Glitch Token</text>
  <rect x="30" y="120" width="130" height="35" rx="6" fill="#1e1e2a" stroke="#fb7185" stroke-width="1.5"/>
  <text x="95" y="142" text-anchor="middle" fill="#ededf0" font-size="11" font-family="monospace">"SolidGoldMagikarp"</text>
  <line x1="160" y1="137" x2="200" y2="137" stroke="#fb7185" stroke-width="1.5" marker-end="url(#arr)"/>
  <rect x="200" y="120" width="120" height="35" rx="6" fill="#fb718522" stroke="#fb7185" stroke-width="1" stroke-dasharray="4,3"/>
  <text x="260" y="137" text-anchor="middle" fill="#fb7185" font-size="11" font-family="system-ui">未训练的嵌入 ⚠️</text>
  <line x1="320" y1="137" x2="360" y2="137" stroke="#fb7185" stroke-width="1.5" marker-end="url(#arr)"/>
  <rect x="360" y="120" width="120" height="35" rx="6" fill="#fb718522" stroke="#fb7185" stroke-width="1"/>
  <text x="420" y="132" text-anchor="middle" fill="#fb7185" font-size="11" font-family="system-ui">🔥 不可预测行为</text>
  <text x="420" y="148" text-anchor="middle" fill="#6b6b78" font-size="10" font-family="system-ui">循环/拒绝/乱码</text>
  <!-- Root cause -->
  <rect x="30" y="170" width="640" height="24" rx="6" fill="#fbbf2411" stroke="#fbbf24" stroke-width="1"/>
  <text x="350" y="185" text-anchor="middle" fill="#fbbf24" font-size="11" font-family="system-ui">根因：Tokenizer 在数据 A 上训练（包含该词），模型在数据 B 上微调（不包含该词）→ embedding 区域未初始化</text>
</svg>

**根因分析：**

1. **Tokenizer** 在大规模语料（如 Reddit、GitHub）上训练 → 学到了这些用户名/特殊字符串
2. **模型** 在筛选后的微调数据上训练 → 这些 token 几乎从未被见到
3. 结果：词表里有这个 token 的「位置」，但 embedding 空间中对应的区域几乎没有被优化过

这就像一栋大楼里有一间房间，门牌号存在，但里面从未装修——进去之后什么都可能发生。

### 后续发展

- **GlitchProber**（2024）：用 PCA 分析中间层激活，发现 glitch token 在 embedding 空间中远离正常聚类
- **梯度方法挖掘**（arXiv 2410.15052）：用梯度优化系统性地发现未训练 token
- 现代模型（GPT-4o、Claude 3）已经通过更好的训练覆盖修复了大部分 glitch token

## 第三幕：Token Smuggling——安全绕过攻击

### 安全过滤器的盲区

大多数 LLM 的安全系统在两个层面工作：
1. **输入过滤**：检查用户输入是否包含有害内容（关键词匹配、分类器）
2. **输出过滤**：检查模型输出是否包含有害内容

问题是：**输入过滤通常在原始文本上操作，而模型在 token 上操作。** 如果攻击者能让同一段文本在两个层面上有不同的「解读」，就可以绕过过滤。

### 攻击向量

**1. Unicode 同形字攻击**

用视觉上相同但编码不同的字符替换：
- 拉丁字母 `a`（U+0061）→ 西里尔字母 `а`（U+0430）
- 视觉上完全相同，但 tokenizer 会产生不同的切分
- 安全过滤器的正则表达式匹配失败

**2. 零宽字符注入**

在关键词中插入零宽字符（U+200B、U+FEFF）：
- 人眼看不到任何区别
- 但 tokenizer 会在零宽字符处断开，生成不同的 token 序列
- 过滤器检测不到完整的违规词

**3. 非标准切分利用**（ACL 2025 论文 "Adversarial Tokenization"）

研究发现：即使文本被「异常」tokenize（不是标准 BPE 的贪婪切分），LLM 仍然能理解语义。攻击者利用这一点：
- 用非标准的编码方式写出有害指令
- 安全分类器无法识别（因为关键词被打散）
- 但模型能「拼回来」理解意图

<svg viewBox="0 0 700 220" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <rect x="0" y="0" width="700" height="220" rx="12" fill="#13131a" stroke="#23232e" stroke-width="1"/>
  <text x="350" y="22" text-anchor="middle" fill="#6b6b78" font-size="12" font-family="system-ui">Token Smuggling 攻击流程</text>
  <!-- Attacker input -->
  <rect x="20" y="40" width="150" height="45" rx="8" fill="#1e1e2a" stroke="#fb7185" stroke-width="1.5"/>
  <text x="95" y="58" text-anchor="middle" fill="#fb7185" font-size="11" font-family="system-ui">攻击者输入</text>
  <text x="95" y="75" text-anchor="middle" fill="#ededf0" font-size="10" font-family="monospace">ha​rm​ful（含零宽字符）</text>
  <!-- Safety filter -->
  <line x1="170" y1="62" x2="210" y2="62" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arr)"/>
  <rect x="210" y="40" width="140" height="45" rx="8" fill="#34d39922" stroke="#34d399" stroke-width="1.5"/>
  <text x="280" y="58" text-anchor="middle" fill="#34d399" font-size="11" font-family="system-ui">安全过滤器</text>
  <text x="280" y="75" text-anchor="middle" fill="#34d399" font-size="10" font-family="system-ui">✓ 未检测到违规词</text>
  <!-- Tokenizer -->
  <line x1="350" y1="62" x2="390" y2="62" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arr)"/>
  <rect x="390" y="40" width="120" height="45" rx="8" fill="#1e1e2a" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="450" y="58" text-anchor="middle" fill="#fbbf24" font-size="11" font-family="system-ui">Tokenizer</text>
  <text x="450" y="75" text-anchor="middle" fill="#ededf0" font-size="10" font-family="monospace">["ha","rm","ful"]</text>
  <!-- Model -->
  <line x1="510" y1="62" x2="550" y2="62" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arr)"/>
  <rect x="550" y="40" width="130" height="45" rx="8" fill="#fb718522" stroke="#fb7185" stroke-width="1.5"/>
  <text x="615" y="58" text-anchor="middle" fill="#fb7185" font-size="11" font-family="system-ui">LLM</text>
  <text x="615" y="75" text-anchor="middle" fill="#fb7185" font-size="10" font-family="system-ui">理解语义 → 执行 ⚠️</text>
  <!-- Normal path comparison -->
  <text x="95" y="115" text-anchor="middle" fill="#6b6b78" font-size="11" font-family="system-ui">对比：正常输入</text>
  <rect x="20" y="125" width="150" height="45" rx="8" fill="#1e1e2a" stroke="#6b6b78" stroke-width="1"/>
  <text x="95" y="143" text-anchor="middle" fill="#6b6b78" font-size="11" font-family="system-ui">正常输入</text>
  <text x="95" y="160" text-anchor="middle" fill="#ededf0" font-size="10" font-family="monospace">harmful</text>
  <line x1="170" y1="147" x2="210" y2="147" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arr)"/>
  <rect x="210" y="125" width="140" height="45" rx="8" fill="#fb718522" stroke="#fb7185" stroke-width="1.5"/>
  <text x="280" y="143" text-anchor="middle" fill="#fb7185" font-size="11" font-family="system-ui">安全过滤器</text>
  <text x="280" y="160" text-anchor="middle" fill="#fb7185" font-size="10" font-family="system-ui">🚫 拦截！检测到违规词</text>
  <!-- Key insight -->
  <rect x="20" y="185" width="660" height="28" rx="6" fill="#6e8eff11" stroke="#6e8eff" stroke-width="1"/>
  <text x="350" y="203" text-anchor="middle" fill="#6e8eff" font-size="11" font-family="system-ui">核心漏洞：安全过滤器和 LLM 对「同一输入」的理解不一致——过滤器看字符串，模型看语义</text>
</svg>

### RAG 毒化与 Glitch Token DoS

OWASP 发现了另一种攻击路径：将 glitch token 注入 RAG（检索增强生成）的知识库。当模型检索到包含这些 token 的文档段落时，可能产生不连贯输出——实现拒绝服务攻击。

## 第四幕：Tokenization 意识——模型在「自我觉察」

### 模型知道自己被切词了吗？

2024 年的一项研究发现了一个令人惊讶的现象：**LLM 在第一层就已经发展出了「tokenization 意识」**——它学会了编码「输入是如何被切分的」这个元信息。

这意味着模型并非完全对自己的 tokenization 无知。它在某种程度上「知道」当前 token 是否是一个完整单词、是否是一个词的开头/中间/结尾。

但这种意识是有限的——它不足以可靠地完成字符级任务。就像你能感觉到自己在呼吸，但不能精确控制每一块肋间肌。

## 这意味着什么

Tokenization 的这些「阴暗面」揭示了几个深层问题：

1. **感知瓶颈**：模型的推理能力受限于它的「感知分辨率」——你不能在看不见的东西上做推理
2. **安全的脆弱性**：任何基于字符串匹配的安全措施，都可能被 tokenization 层面的操作绕过
3. **训练与词表的不对称**：词表来自数据集 A，模型训练在数据集 B，对不上的部分就是安全漏洞
4. **Tokenization ≠ 中性预处理**：它主动塑造了模型的认知边界

## 下一篇预告

如果文本的 tokenization 已经这么复杂，那图像、音频、机器人动作、甚至 DNA 序列呢？下一篇我们将看到，「token」这个概念已经远远超越了文字——万物皆可 token，这正在催生 AI 的大统一时代。
