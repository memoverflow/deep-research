---
title: "万物皆可 Token：图像、声音、动作的统一语言"
date: 2025-05-15
level: 3
series: "理解 Tokenization"
series_order: 5
series_total: 6
tags: [tokenization, multimodal, VQ-VAE, EnCodec, robotics, DNA]
summary: "Token 不只属于文字。图像被切成视觉词汇、音乐被编码为 codec token、机器人动作被离散化为 bin——万物正在被统一为 token 序列，由同一个 Transformer 处理。"
---

> 当 GPT-4o 能看图、听声音、生成图片时，它用的不是三个不同的模型——而是把一切都变成了 token。图片是 token，音频是 token，甚至机器人抓杯子的动作也是 token。这是 AI 正在经历的「大统一」。

## 一个疯狂的想法：如果一切都是 token？

Transformer 的成功建立在一个简单的框架上：**给定一个 token 序列，预测下一个 token。** 这个框架在文本上证明了自己——那为什么要局限于文本？

如果我们能把图像、音频、视频、机器人动作、甚至 DNA 都变成「token 序列」，那么同一个 Transformer 架构就能处理一切。不需要为每种模态设计专门的网络——统一的 next-token prediction 就够了。

这不是科幻。这正在发生。

## 图像：从像素到视觉词汇

### 问题：像素太多、太连续

一张 256×256 的图片有 65,536 个像素，每个像素 3 个通道。如果把每个像素当作一个「token」，序列长度就是 196,608——Transformer 处理不了这么长的序列。

而且像素值是连续的（0-255），不是离散的。Transformer 的 embedding 层需要有限的词表。

我们需要一种方法：把图片**压缩**成短序列，并且**离散化**为有限词汇表中的索引。

### VQ-VAE：学习「视觉单词」

2017 年，VQ-VAE（Vector Quantized Variational Autoencoder）提出了一个优雅的方案：

**核心直觉：** 就像文本由有限的词汇组成，图像也可以由有限的「视觉单词」拼贴而成。

具体做法：
1. **编码器**把图片压缩成小的特征图（比如 32×32）
2. 每个位置的特征向量，在一个「码本」（codebook）中找最近邻
3. 用码本索引替代原始向量——**离散化完成**
4. **解码器**从码本索引重建图片

<svg viewBox="0 0 750 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:750px;margin:24px auto;display:block;">
  <defs>
    <marker id="arr5" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <rect x="0" y="0" width="750" height="200" rx="12" fill="#13131a" stroke="#23232e" stroke-width="1"/>
  <!-- Image input -->
  <rect x="20" y="55" width="80" height="80" rx="4" fill="#1e1e2a" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="60" y="90" text-anchor="middle" fill="#22d3ee" font-size="10" font-family="system-ui">256×256</text>
  <text x="60" y="105" text-anchor="middle" fill="#22d3ee" font-size="10" font-family="system-ui">原图</text>
  <text x="60" y="145" text-anchor="middle" fill="#6b6b78" font-size="9" font-family="system-ui">196K 像素</text>
  <!-- Arrow -->
  <line x1="105" y1="95" x2="145" y2="95" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arr5)"/>
  <!-- Encoder -->
  <rect x="150" y="65" width="90" height="55" rx="8" fill="#6e8eff22" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="195" y="88" text-anchor="middle" fill="#6e8eff" font-size="12" font-family="system-ui">编码器</text>
  <text x="195" y="105" text-anchor="middle" fill="#6b6b78" font-size="9" font-family="system-ui">CNN/ViT</text>
  <!-- Arrow -->
  <line x1="245" y1="95" x2="285" y2="95" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arr5)"/>
  <!-- Feature map -->
  <rect x="290" y="60" width="65" height="65" rx="4" fill="#1e1e2a" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="322" y="88" text-anchor="middle" fill="#fbbf24" font-size="9" font-family="system-ui">32×32</text>
  <text x="322" y="102" text-anchor="middle" fill="#fbbf24" font-size="9" font-family="system-ui">特征图</text>
  <!-- Arrow to codebook -->
  <line x1="358" y1="95" x2="398" y2="95" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arr5)"/>
  <text x="378" y="82" text-anchor="middle" fill="#6b6b78" font-size="9" font-family="system-ui">最近邻</text>
  <!-- Codebook -->
  <rect x="402" y="50" width="100" height="90" rx="8" fill="#34d39922" stroke="#34d399" stroke-width="1.5"/>
  <text x="452" y="70" text-anchor="middle" fill="#34d399" font-size="11" font-family="system-ui">码本 (Codebook)</text>
  <text x="452" y="88" text-anchor="middle" fill="#ededf0" font-size="9" font-family="monospace">c₁ = [0.2, 0.8, ...]</text>
  <text x="452" y="102" text-anchor="middle" fill="#ededf0" font-size="9" font-family="monospace">c₂ = [0.5, 0.1, ...]</text>
  <text x="452" y="116" text-anchor="middle" fill="#ededf0" font-size="9" font-family="monospace">...</text>
  <text x="452" y="130" text-anchor="middle" fill="#ededf0" font-size="9" font-family="monospace">c₈₁₉₂ = [...]</text>
  <!-- Arrow -->
  <line x1="505" y1="95" x2="545" y2="95" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arr5)"/>
  <!-- Token sequence -->
  <rect x="550" y="65" width="180" height="55" rx="8" fill="#a78bfa22" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="640" y="82" text-anchor="middle" fill="#a78bfa" font-size="11" font-family="system-ui">视觉 Token 序列</text>
  <text x="640" y="100" text-anchor="middle" fill="#ededf0" font-size="10" font-family="monospace">[42, 891, 7, 456, ...]</text>
  <text x="640" y="115" text-anchor="middle" fill="#6b6b78" font-size="9" font-family="system-ui">1024 个 token（32×32）</text>
  <!-- Key insight -->
  <rect x="20" y="165" width="710" height="28" rx="6" fill="#34d39911" stroke="#34d399" stroke-width="1"/>
  <text x="375" y="182" text-anchor="middle" fill="#34d399" font-size="11" font-family="system-ui">196K 像素 → 1024 个离散 token | 压缩 192× | 可以用 Transformer 自回归生成！</text>
</svg>

DALL-E 1 正是用这个思路工作的：先用 dVAE（discrete VAE）把图片变成 1024 个视觉 token，然后用一个 Transformer 联合建模「文本 token + 视觉 token」——预测下一个 token 就能生成图片。

### 进化：VQGAN → MAGVIT-v2

VQ-VAE 的图片质量有限。后续改进沿着两个方向：

**VQGAN（2021）：** 加入 GAN 对抗训练，让重建图片更清晰、更真实。不再只优化像素级误差，而是让判别器检查「这看起来像真图吗？」

**MAGVIT-v2（2024）：** 一篇标题就很直白的论文——*"Language Model Beats Diffusion — Tokenizer is Key"*。核心创新是 Lookup-Free Quantization（LFQ），把码本大小推到了 2¹⁸ = 262,144。更大的视觉词汇表 = 更精细的图像描述 = 更高质量的生成。

**结论：Token-based 生成（LM 方法）已经能匹配甚至超越 Diffusion 模型——关键在于足够好的图像 tokenizer。**

## 音频：声音的离散密码

### 问题：声音是高维时序信号

一秒 24kHz 的音频有 24,000 个采样点。直接当 token 序列？太长了。而且音频有层次结构：音色、节奏、旋律分别在不同的时间尺度上运作。

### EnCodec & SoundStream：神经音频编解码器

Meta 的 EnCodec 和 Google 的 SoundStream 用了一个巧妙的层级方案——**残差向量量化**（Residual Vector Quantization, RVQ）：

1. 编码器把音频压缩成低帧率表示（~75 帧/秒）
2. 第一个量化器捕获最重要的信息（粗结构）
3. 第二个量化器处理残差（细节）
4. 第三个量化器处理剩余残差（更细的细节）
5. ...最多 32 层

每一层都有自己的 1024 大小码本。最终，1 秒音频 = 75 × N 个 token（N = 使用的层数）。

**类比：** 这就像画一幅肖像：第一层是大色块和轮廓，第二层加细节，第三层加纹理。每层都是独立的「描述序列」。

### 从 Token 到音乐生成

有了音频 token，音乐生成就变成了语言模型的任务：

- **MusicGen（Meta, 2023）：** 文本描述 → 音频 codec token 序列。发明了「delay pattern」解决多层 RVQ 的联合生成问题。
- **AudioLM（Google, 2023）：** 先生成语义 token（粗粒度），再生成声学 token（细粒度）——两级 token 对应不同抽象层次。
- **VALL-E（Microsoft, 2023）：** 3 秒语音 → 提取说话人的 codec token 特征 → 生成任意文本的语音 token。

<iframe src="/assets/multimodal-tokenization-animation.html" width="100%" height="580"
  style="border:1px solid #23232e; border-radius:12px; background:#0a0a0f;"
  loading="lazy"></iframe>

<p style="color:#6b6b78; font-size:0.85em; text-align:center; margin-top:8px;">
  ↑ 点击左侧模态切换 | 观察不同信号如何统一为 token 序列
</p>

## 机器人：动作也能变 Token

### 从连续控制到离散预测

机器人的动作是连续的：手臂在三维空间中的位置 (x, y, z)、旋转角度 (roll, pitch, yaw)、夹爪开合——共 7-8 个连续维度。

怎么变成 token？

**RT-2（Google, 2023）的方法简单粗暴：**
- 每个维度独立用 256 个 bin 离散化
- x = 0.37 → 映射到 bin 94
- 把 bin 号当作 token 输出
- 模型自回归预测：先预测 x 的 bin，再预测 y 的 bin...

这让 Google 的机器人能直接复用视觉-语言预训练的知识来控制机械臂。同一个 Transformer 既能理解图片、回答问题，又能输出机器人动作。

### FAST：频率域压缩

但简单 binning 对快速精细动作不够用。Physical Intelligence 在 2025 年提出 FAST（Frequency-space Action Sequence Tokenization）：

1. 不是逐帧离散化，而是对一段**动作序列**做 DCT 变换（离散余弦变换）
2. 在频率域上量化——低频分量（大动作）用少量 token，高频分量（细微抖动）可选择性保留
3. 压缩率高，且保留动作的连续性

**结论：** 从 RT-2 的朴素 binning 到 FAST 的频域压缩，再到 OAT（2026）的学习型有序 token——动作 tokenization 正在快速进化。

## DNA：生命的语言，也是 Token

### 基因组作为字符序列

DNA 由四种碱基组成：A、C、G、T。最简单的做法：每个碱基 = 一个 token。但人类基因组有 30 亿个碱基——这个序列长度太恐怖了。

**k-mer 方法**（DNABERT）：把每 k 个碱基当作一个「词」。6-mer 就相当于有 4⁶ = 4096 个「DNA 单词」。

**BPE 方法**（DNABERT-2）：直接把 NLP 的 BPE 搬到 DNA 上。让算法自动发现频繁出现的碱基组合——这些往往对应有生物学意义的 motif（转录因子结合位、基因调控元件等）。

**HyenaDNA：** 回到字符级，但用超长上下文（100 万 token）+ 次二次复杂度的 Hyena 架构来处理整条染色体。

**蛋白质？** ESM（Meta）把 20 种氨基酸作为 token，在 2.5 亿蛋白质序列上做 masked language modeling——居然能预测蛋白质 3D 结构。

<svg viewBox="0 0 700 180" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <rect x="0" y="0" width="700" height="180" rx="12" fill="#13131a" stroke="#23232e" stroke-width="1"/>
  <text x="350" y="22" text-anchor="middle" fill="#6b6b78" font-size="12" font-family="system-ui">Tokenization 的跨域统一</text>
  <!-- Domain boxes in a row -->
  <rect x="25" y="40" width="95" height="55" rx="6" fill="#22d3ee11" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="72" y="60" text-anchor="middle" fill="#22d3ee" font-size="11" font-family="system-ui">文本</text>
  <text x="72" y="78" text-anchor="middle" fill="#6b6b78" font-size="9" font-family="system-ui">BPE 子词</text>
  
  <rect x="135" y="40" width="95" height="55" rx="6" fill="#34d39911" stroke="#34d399" stroke-width="1.5"/>
  <text x="182" y="60" text-anchor="middle" fill="#34d399" font-size="11" font-family="system-ui">图像</text>
  <text x="182" y="78" text-anchor="middle" fill="#6b6b78" font-size="9" font-family="system-ui">VQ 码本索引</text>
  
  <rect x="245" y="40" width="95" height="55" rx="6" fill="#fbbf2411" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="292" y="60" text-anchor="middle" fill="#fbbf24" font-size="11" font-family="system-ui">音频</text>
  <text x="292" y="78" text-anchor="middle" fill="#6b6b78" font-size="9" font-family="system-ui">RVQ codec</text>
  
  <rect x="355" y="40" width="95" height="55" rx="6" fill="#a78bfa11" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="402" y="60" text-anchor="middle" fill="#a78bfa" font-size="11" font-family="system-ui">动作</text>
  <text x="402" y="78" text-anchor="middle" fill="#6b6b78" font-size="9" font-family="system-ui">256-bin / DCT</text>
  
  <rect x="465" y="40" width="95" height="55" rx="6" fill="#fb718511" stroke="#fb7185" stroke-width="1.5"/>
  <text x="512" y="60" text-anchor="middle" fill="#fb7185" font-size="11" font-family="system-ui">DNA</text>
  <text x="512" y="78" text-anchor="middle" fill="#6b6b78" font-size="9" font-family="system-ui">BPE / k-mer</text>
  
  <rect x="575" y="40" width="100" height="55" rx="6" fill="#fb923c11" stroke="#fb923c" stroke-width="1.5"/>
  <text x="625" y="60" text-anchor="middle" fill="#fb923c" font-size="11" font-family="system-ui">结构化数据</text>
  <text x="625" y="78" text-anchor="middle" fill="#6b6b78" font-size="9" font-family="system-ui">语法感知</text>
  
  <!-- Arrows converging down -->
  <line x1="72" y1="95" x2="350" y2="125" stroke="#6e8eff" stroke-width="1" opacity="0.5"/>
  <line x1="182" y1="95" x2="350" y2="125" stroke="#6e8eff" stroke-width="1" opacity="0.5"/>
  <line x1="292" y1="95" x2="350" y2="125" stroke="#6e8eff" stroke-width="1" opacity="0.5"/>
  <line x1="402" y1="95" x2="350" y2="125" stroke="#6e8eff" stroke-width="1" opacity="0.5"/>
  <line x1="512" y1="95" x2="350" y2="125" stroke="#6e8eff" stroke-width="1" opacity="0.5"/>
  <line x1="625" y1="95" x2="350" y2="125" stroke="#6e8eff" stroke-width="1" opacity="0.5"/>
  
  <!-- Unified -->
  <rect x="260" y="120" width="180" height="45" rx="8" fill="#6e8eff22" stroke="#6e8eff" stroke-width="2"/>
  <text x="350" y="140" text-anchor="middle" fill="#6e8eff" font-size="13" font-family="system-ui">统一的 Token 序列</text>
  <text x="350" y="157" text-anchor="middle" fill="#6b6b78" font-size="10" font-family="system-ui">→ 同一个 Transformer 架构处理</text>
</svg>

## 结构化数据：JSON 也要优化 Token

不只是感官模态。结构化数据（JSON、表格、SQL）也面临 tokenization 效率问题。

标准 BPE 会把 JSON 的 `{`, `}`, `:` 等语法符号和值内容一起切分，浪费 token。**TOON 格式**（Token-Oriented Object Notation）通过重新设计序列化格式，在保持 JSON 语义的前提下减少 40% token 消耗。

Google 的 **TaPas** 把表格展平成 token 序列，加上行/列位置编码——让模型能「看到」表格结构。

## 大统一的里程碑：Emu3

2024 年，百度的 Emu3 发表在 Nature 上，展示了真正的统一：

- **文本、图像、视频** 全部用同一个词表的 token 表示
- 训练目标只有一个：next-token prediction
- 没有 diffusion，没有专门的视觉模块
- 性能匹配各自领域的专用模型

他们还发现了一个有趣的现象：跨模态的信息传递集中在单个 `[EOI]`（End of Image）token 上——这一个 token 充当了「模态桥梁」。

## 这意味着什么

「万物皆可 token」不只是工程技巧——它背后是一个深刻的认知：

1. **通用接口**：Token 序列成了所有模态的通用表示层。任何信号只要能离散化为 token，就能接入 Transformer 的生态系统
2. **涌现能力**：当不同模态的 token 在同一个模型中训练时，跨模态的迁移学习会自然涌现（RT-2 用网络知识帮助机器人理解指令）
3. **Tokenizer 决定上限**：正如 MAGVIT-v2 的标题所言——tokenizer 才是视觉生成的关键瓶颈，不是模型大小

但也有根本性的局限：**离散化必然有损。** 你不能用有限词汇完美描述连续信号。这引出了下一篇的主题——如果连 tokenizer 本身都是问题，我们能不能干脆扔掉它？

## 下一篇预告

Byte Latent Transformer 说：不需要 tokenizer。MEGABYTE 说：直接处理原始字节。动态 patching 说：让模型自己决定在哪里「切词」。最后一篇，我们来看 tokenizer 的终结——或者说，它的进化。
