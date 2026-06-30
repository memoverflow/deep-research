---
title: "理解 AI 图像生成（三）：Latent Diffusion 与 Stable Diffusion——在压缩空间里做魔法"
date: 2025-06-30
level: 3
series: "理解 AI 图像生成"
series_order: 3
series_total: 5
tags: [latent-diffusion, stable-diffusion, vae, unet, compression, efficiency]
summary: "Latent Diffusion 的核心洞察：先用 VAE 把 512×512 图片压缩到 64×64 潜在空间，再在这个小空间里做扩散——计算量骤降 48 倍，让消费级 GPU 也能生图"
---

# Latent Diffusion 与 Stable Diffusion：在压缩空间里做魔法

> 一张 512×512 的图有 786,432 个像素值。一个 64×64×4 的潜在表示只有 16,384 个值。在后者上做扩散，计算量降低了 48 倍——但生成质量几乎不损失。这就是 Stable Diffusion 能在你的笔记本上跑的秘密。

## 问题：像素空间太大了

DDPM 直接在像素空间做扩散。对于 256×256 图片这还行，但到 512×512 甚至 1024×1024 时，每一步去噪都要处理巨大的张量。更致命的是 attention 机制——它的计算复杂度和空间分辨率的平方成正比。

512×512 的 attention 计算量是 64×64 的 $64$ 倍。如果直接在高分辨率像素空间做扩散，生成一张图可能需要几十分钟——这对实际应用来说不可接受。

## 核心思想：感知压缩 + 潜在空间扩散

Rombach 等人（2022）的关键洞察：**图像中大量信息是像素级冗余——邻近像素高度相关、高频细节对语义无贡献。先把这些冗余压缩掉，在"纯语义"的潜在空间做扩散，最后再解压回像素。**

这把问题拆成了两个阶段：
1. **感知压缩**（VAE）：学会把图片压缩/解压——只做一次
2. **语义生成**（扩散模型）：在压缩空间里做创造性的生成——跑很多步

<svg viewBox="0 0 720 250" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:720px;margin:24px auto;display:block;">
  <defs>
    <marker id="larr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Pixel space -->
  <rect x="20" y="60" width="140" height="130" rx="10" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="90" y="90" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">像素空间</text>
  <text x="90" y="115" text-anchor="middle" fill="#888" font-size="10" font-family="system-ui">512 × 512 × 3</text>
  <text x="90" y="135" text-anchor="middle" fill="#ff6b6b" font-size="10" font-family="system-ui">786,432 维</text>
  <text x="90" y="170" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">大量像素冗余</text>
  <!-- Encoder -->
  <line x1="165" y1="125" x2="210" y2="125" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#larr)"/>
  <text x="188" y="113" text-anchor="middle" fill="#22d3ee" font-size="9" font-family="system-ui">VAE</text>
  <text x="188" y="140" text-anchor="middle" fill="#22d3ee" font-size="9" font-family="system-ui">编码</text>
  <!-- Latent space -->
  <rect x="215" y="75" width="280" height="100" rx="10" fill="rgba(52,211,153,0.08)" stroke="#34d399" stroke-width="2"/>
  <text x="355" y="100" text-anchor="middle" fill="#34d399" font-size="12" font-weight="bold" font-family="system-ui">潜在空间（Latent Space）</text>
  <text x="355" y="120" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">64 × 64 × 4 = 16,384 维</text>
  <text x="355" y="140" text-anchor="middle" fill="#34d399" font-size="10" font-family="system-ui">↓ 48× 压缩，纯语义信息</text>
  <text x="355" y="160" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">⟲ 扩散过程在这里进行（20-50步）</text>
  <!-- Decoder -->
  <line x1="500" y1="125" x2="545" y2="125" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#larr)"/>
  <text x="523" y="113" text-anchor="middle" fill="#22d3ee" font-size="9" font-family="system-ui">VAE</text>
  <text x="523" y="140" text-anchor="middle" fill="#22d3ee" font-size="9" font-family="system-ui">解码</text>
  <!-- Output -->
  <rect x="550" y="60" width="140" height="130" rx="10" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="620" y="90" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">生成图片</text>
  <text x="620" y="115" text-anchor="middle" fill="#888" font-size="10" font-family="system-ui">512 × 512 × 3</text>
  <text x="620" y="135" text-anchor="middle" fill="#34d399" font-size="10" font-family="system-ui">高质量输出</text>
  <text x="620" y="170" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">细节由 VAE 恢复</text>
  <!-- Bottom note -->
  <text x="360" y="225" text-anchor="middle" fill="#888" font-size="10" font-family="system-ui">VAE 只运行 1 次 | 扩散模型运行 20-50 次 → 压缩空间的节省被放大数十倍</text>
</svg>

## VAE：感知压缩引擎

### 不是普通的 VAE

Latent Diffusion 的 VAE 不是教科书上那个生成模糊图片的基础 VAE。它用了三个改进：

1. **感知损失（Perceptual Loss）**：不是逐像素比较，而是在预训练网络的特征空间比较——保留语义结构而非像素精度
2. **对抗损失（Adversarial Loss）**：加了一个 patch-based 判别器，让解码输出更锐利
3. **轻度 KL 正则化**：保证潜在空间平滑可采样，但不过度约束

结果是一个"感知上无损"的压缩器：人眼看不出压缩前后的区别，但数据量减少了 48 倍。

### 下采样因子的选择

Stable Diffusion 使用 **f=8** 的下采样因子：

| 阶段 | 分辨率 | 通道数 | 总维度 |
|------|--------|--------|--------|
| 输入图片 | 512×512 | 3 (RGB) | 786,432 |
| 编码后潜在 | 64×64 | 4 | 16,384 |
| 压缩比 | - | - | **48×** |

为什么是 4 通道而不是 3？因为 VAE 需要额外的维度来编码图像中的信息——4 通道是实验发现的最优平衡点（3 通道信息有损，8 通道冗余）。

## U-Net：在潜在空间做去噪

扩散过程的"主引擎"是一个在潜在空间运行的 U-Net。它接收：
- 噪声潜在表示 $z_t$（64×64×4）
- 时间步 $t$（通过 sinusoidal embedding 编码）
- 条件信号（文本 embedding，通过 cross-attention 注入）

U-Net 的编码器-解码器结构特别适合去噪：
- **编码器**：逐步降低分辨率（64→32→16→8），捕获全局语义
- **解码器**：逐步恢复分辨率（8→16→32→64），重建细节
- **跳跃连接**：编码器的细节信息直接传给解码器，防止信息丢失

在低分辨率层（8×8, 16×16）加入 self-attention 块，让模型建立全局关联（比如"左边的人和右边的人应该在同一个场景里"）。

## 为什么 Latent Diffusion 如此高效

让我们算一笔账：

- 扩散模型跑 **50 步**，每步处理 64×64×4 的张量
- 如果在像素空间，每步处理 512×512×3 的张量
- Attention 的计算复杂度是 $O(n^2)$，其中 n 是 spatial tokens 数量

在 64×64 空间：n = 4096 tokens
在 512×512 空间：n = 262,144 tokens

Attention 计算量比：$(262144/4096)^2 = 4096$ 倍！

这就是为什么 Stable Diffusion 能在消费级 GPU（8GB VRAM）上跑，而像素空间的扩散模型可能需要一整个 A100 集群。

## Stable Diffusion 的完整生成流程

```
输入: 文本提示词 "a cat sitting on a windowsill, sunset"

1. CLIP 文本编码器 → 77×768 的 token embedding 序列
2. 采样随机噪声 z_T ~ N(0, I)，大小 64×64×4
3. 重复 50 次 (DDIM 去噪):
   - U-Net(z_t, t, text_embedding) → 预测噪声 ε
   - z_{t-1} = 去噪一步(z_t, ε)
4. VAE 解码器(z_0) → 512×512×3 的图片

输出: 一张在窗台上的猫，夕阳背景
```

整个过程：文本编码（1次）+ U-Net（50次）+ VAE 解码（1次）。在 RTX 3090 上大约 3-5 秒。

## 从 SD 1.5 到 SD-XL 的演进

| 版本 | U-Net 参数 | 文本编码器 | 原生分辨率 | 关键改进 |
|------|-----------|-----------|-----------|---------|
| SD 1.5 | 860M | CLIP ViT-L/14 | 512×512 | 开源革命的起点 |
| SD-XL | ~2.6B | OpenCLIP ViT-bigG + CLIP ViT-L | 1024×1024 | 更大 U-Net + 双编码器 |
| SD 3 | ~3B | CLIP + OpenCLIP + T5-XXL | 1024×1024 | MMDiT + Rectified Flow |

SD-XL 的关键改进还包括**尺寸/裁剪条件化**——训练时告诉模型"这张图原始尺寸是多少、裁剪了多少"，让模型学会生成符合目标尺寸的构图，避免了常见的"图片被截断"问题。

## 下一篇预告

Latent Diffusion 解决了"怎么高效生成"，但还有一个关键问题：**怎么让模型听懂你的文字描述？** 当你写"一只戴着墨镜的柯基在月球上冲浪"，模型怎么知道把这些概念组合在一起？答案涉及 CLIP 文本编码、Cross-Attention 机制、以及一个让文本控制效果提升 10 倍的"小技巧"——Classifier-Free Guidance。下一篇详解。
