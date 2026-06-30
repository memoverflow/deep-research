---
title: "理解 AI 图像生成（五）：主流模型全景——SD、DALL-E、Midjourney、Flux 与架构未来"
date: 2025-06-30
level: 3
series: "理解 AI 图像生成"
series_order: 5
series_total: 5
tags: [stable-diffusion, dall-e, midjourney, flux, dit, flow-matching, mmdit, architecture]
summary: "从 U-Net 到 DiT，从 DDPM 到 Flow Matching——2024-2025 年的架构革命正在重塑 AI 生图的全部技术栈"
---

# 主流模型全景：SD、DALL-E、Midjourney、Flux 与架构未来

> 2024 年的 AI 生图领域发生了一场安静的革命：U-Net 被 Transformer 取代，DDPM 被 Flow Matching 取代。这些看似内部的技术变动，正在根本性地改变生图模型的能力上限。

## 架构革命：从 U-Net 到 DiT

### U-Net 的局限

Stable Diffusion 1.5 和 XL 都使用 U-Net 作为去噪骨干网络。U-Net 的优势是归纳偏置（locality、skip connections 天然适合图像），但它有几个结构性限制：

1. **缩放困难**：U-Net 的参数增长不如 Transformer 那样"优雅"，无法简单地堆叠更多层来持续提升
2. **全局推理有限**：卷积操作天然局部化，全局理解依赖少数低分辨率层的 attention
3. **多模态融合笨拙**：文本条件只能通过 cross-attention "侧面注入"，不如 token 级混合自然

### DiT：Diffusion Transformer

Peebles & Xie（2023）提出 DiT——用纯 Vision Transformer 替代 U-Net：

1. 将潜在表示切成 patches（如 2×2）
2. 每个 patch 线性投影为一个 token
3. 标准 Transformer self-attention 处理所有 token
4. 时间步和类别条件通过 Adaptive Layer Normalization（AdaLN）注入

DiT 展现了**清晰的 scaling law**：模型越大，FID 越低，且改善持续到测试的最大规模。这和 LLM 的 scaling behavior 完全一致——暗示图像生成也可以"暴力出奇迹"。

### MMDiT：多模态 Diffusion Transformer

Stable Diffusion 3 进一步引入 **MMDiT**：

- 图像 token 和文本 token **共享同一个 attention 机制**
- 但各自有独立的权重集合（不完全共享参数）
- Joint attention 让文本和图像在每一层都能深度交互

这比 U-Net 的 cross-attention（文本只通过 K/V "旁观"图像生成）强大得多——现在文本和图像是"平等参与者"。

<svg viewBox="0 0 700 220" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="farr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="350" y="20" text-anchor="middle" fill="#ededf0" font-size="12" font-weight="bold" font-family="system-ui">架构演进</text>
  <!-- UNet -->
  <rect x="30" y="50" width="180" height="150" rx="8" fill="#1e1e2a" stroke="#ff6b6b" stroke-width="1.5"/>
  <text x="120" y="75" text-anchor="middle" fill="#ff6b6b" font-size="11" font-weight="bold" font-family="system-ui">U-Net (SD 1.5/XL)</text>
  <text x="120" y="100" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">Conv + 少量 Attention</text>
  <text x="120" y="118" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">文本通过 Cross-Attn 侧注</text>
  <text x="120" y="140" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">✓ 训练高效（归纳偏置）</text>
  <text x="120" y="158" text-anchor="middle" fill="#ff6b6b" font-size="9" font-family="system-ui">✗ 缩放困难</text>
  <text x="120" y="176" text-anchor="middle" fill="#ff6b6b" font-size="9" font-family="system-ui">✗ 全局推理有限</text>
  <!-- Arrow -->
  <line x1="215" y1="125" x2="255" y2="125" stroke="#6e8eff" stroke-width="2" marker-end="url(#farr)"/>
  <!-- DiT -->
  <rect x="260" y="50" width="180" height="150" rx="8" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="350" y="75" text-anchor="middle" fill="#a78bfa" font-size="11" font-weight="bold" font-family="system-ui">DiT (2023)</text>
  <text x="350" y="100" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">纯 Transformer</text>
  <text x="350" y="118" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">Patch tokens + Self-Attn</text>
  <text x="350" y="140" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">✓ Scaling Law 明确</text>
  <text x="350" y="158" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">✓ 全局推理强</text>
  <text x="350" y="176" text-anchor="middle" fill="#a78bfa" font-size="9" font-family="system-ui">条件仍为 AdaLN</text>
  <!-- Arrow -->
  <line x1="445" y1="125" x2="485" y2="125" stroke="#6e8eff" stroke-width="2" marker-end="url(#farr)"/>
  <!-- MMDiT -->
  <rect x="490" y="50" width="180" height="150" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="580" y="75" text-anchor="middle" fill="#34d399" font-size="11" font-weight="bold" font-family="system-ui">MMDiT (SD3/Flux)</text>
  <text x="580" y="100" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">多模态 Joint Attention</text>
  <text x="580" y="118" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">图文 token 平等交互</text>
  <text x="580" y="140" text-anchor="middle" fill="#34d399" font-size="9" font-family="system-ui">✓ Scaling + 全局 + 深度融合</text>
  <text x="580" y="158" text-anchor="middle" fill="#34d399" font-size="9" font-family="system-ui">✓ 文字渲染能力飞跃</text>
  <text x="580" y="176" text-anchor="middle" fill="#34d399" font-size="9" font-family="system-ui">✓ + Flow Matching</text>
</svg>

## 训练范式：从 DDPM 到 Flow Matching

### DDPM 的路径是"曲线"

DDPM 的前向过程按固定噪声调度加噪，反向过程沿着弯曲的 SDE 轨迹去噪。轨迹弯曲意味着需要更多步才能准确追踪。

### Flow Matching 的路径是"直线"

Flow Matching 学习一个 ODE（普通微分方程），让样本沿**直线路径**从噪声流向数据：

$$\frac{dx}{dt} = v_\theta(x, t)$$

网络学习一个速度场 $v_\theta$，直接指示"从当前位置到目标的方向"。

**Rectified Flow** 进一步"矫直"路径：通过迭代的 reflow 过程，让轨迹越来越直——最终直线路径只需 10-20 步就能完成高质量生成。

### 实际影响

| 维度 | DDPM | Flow Matching |
|------|------|---------------|
| 轨迹形状 | 弯曲 SDE | 直线 ODE |
| 所需步数 | 20-50 | 10-20 |
| 理论框架 | 随机微分方程 | 最优传输 |
| 训练目标 | 预测噪声 | 预测速度 |
| 确定性 | 可选(DDIM) | 天然确定性 |

## 主流模型对比

### Stable Diffusion 1.5（2022.08）

开源社区的基石。860M 参数 U-Net，CLIP 文本编码器，512×512。虽然"过时"，但拥有最庞大的生态系统——数以万计的 fine-tune、LoRA、ControlNet 适配器。对于特定风格（动漫、写实人像），社区模型往往优于更新的通用大模型。

### Stable Diffusion XL（2023.07）

参数量翻 3 倍，双文本编码器（OpenCLIP + CLIP），原生 1024×1024。引入尺寸/裁剪条件化，显著改善构图。仍是 U-Net 架构。

### Stable Diffusion 3 / 3.5（2024）

**革命性换代**：
- 架构：U-Net → **MMDiT**（多模态 Diffusion Transformer）
- 训练：DDPM → **Rectified Flow**
- 文本：单编码器 → **三编码器**（CLIP + OpenCLIP + T5-XXL）

T5-XXL（一个纯语言大模型）的加入是关键——它让模型真正"读懂"复杂的长提示词，而不只是匹配关键词。文字渲染能力（在图片中画出正确的文字）首次达到可用水平。

### DALL-E 3（2023.10）

OpenAI 的策略和别人不同——他们在**数据**上做文章。关键创新：训练一个 captioning 模型为每张训练图片生成超详细的描述（几百字），然后让扩散模型在这些高质量合成 caption 上训练。

结果：prompt following（对文字的遵循度）大幅超越同期竞品。集成 ChatGPT 做 prompt 改写——用户随便写几个字，ChatGPT 帮你扩展成详细的图片描述。

### Midjourney v6（2023.12）

闭源、架构不公开，但一直以来以**极强的美学质量**著称。Midjourney 的核心竞争力不是技术架构最新，而是：
- 精心策划的训练数据（偏向高美学质量的图片）
- 大量 human-in-the-loop 的美学偏好调优
- 对构图、光影、色彩有极强的"审美默认值"

v6 改善了文字渲染和字面 prompt 理解（早期版本的"艺术诠释"风格让一些人又爱又恨）。

### Flux（Black Forest Labs, 2024.08）

由 Stable Diffusion 原作者 Robin Rombach 离开 Stability AI 后创建的 Black Forest Labs 开发。代表当前 SOTA：

- **架构**：Rectified Flow Transformer（DiT 变体 + Flow Matching）
- **三个版本**：Pro（闭源 API）、Dev（开放权重研究用）、Schnell（蒸馏快速版）
- **关键优势**：文字渲染、人体解剖学正确性、复杂提示词理解

Flux.2（2025）进一步集成了 Mistral-3 24B 视觉语言模型，实现了更深度的"理解"能力。

### Imagen 3（Google DeepMind, 2024）

Google 的闭源模型。强调安全性和负责任部署，在写实照片生成上表现优异。2025 年升级为 Imagen 4。

## 全景对比

| 模型 | 架构 | 训练范式 | 开放性 | 核心优势 |
|------|------|---------|--------|---------|
| SD 1.5 | U-Net | DDPM | 开源 | 生态最大 |
| SD-XL | U-Net(大) | DDPM | 开源 | 高分辨率 |
| SD 3 | MMDiT | Flow Matching | 开源 | 文字渲染 |
| DALL-E 3 | unCLIP+DM | DDPM变体 | 闭源API | Prompt following |
| Midjourney v6 | 未知 | 未知 | 闭源 | 美学质量 |
| Flux | DiT+Flow | Rectified Flow | 部分开源 | 综合最强 |
| Imagen 3 | DM+Super-Res | 未公开 | 闭源 | 写实照片 |

## 未来方向

### 1. 统一模型

Flux.2 集成了视觉语言模型，暗示未来方向：**一个模型同时理解文字、生成图像、理解图像**。不再是"CLIP编码文字→扩散生图"的管线，而是端到端的多模态生成器。

### 2. 实时生成

Consistency Models、LCM（Latent Consistency Model）等蒸馏方法让 1-4 步就能生成可用图片。结合 Flow Matching 的直线轨迹，"实时文生图"（<100ms 延迟）正在成为现实。

### 3. 视频统一

Sora、Kling、Runway Gen-3 等视频模型表明，图像和视频生成正在统一——同一个 DiT 架构加上时间维度就能生成连贯视频。2025 年的主旋律是"图像→视频→3D"的统一。

### 4. 可控性持续提升

更精细的空间控制（3D-aware ControlNet）、更准确的物体关系理解、更好的计数能力——这些仍是活跃研究方向。

## 全系列总结

五篇文章，我们走完了 AI 图像生成的完整技术栈：

1. **进化史**：GAN→VAE→DDPM→自回归→Flow Matching，每一代解决前一代的痛点
2. **扩散原理**：加噪有解析解、去噪预测噪声、MSE 训练、DDIM 加速
3. **Latent Diffusion**：VAE 压缩 48 倍，让消费级 GPU 也能生图
4. **条件控制**：CLIP 编码语义 + Cross-Attention 注入 + CFG 放大信号 + ControlNet 精确控制
5. **主流模型**：U-Net→DiT 架构革命 + DDPM→Flow Matching 范式转移

AI 生图的核心叙事是**从复杂走向简洁**：GAN 的对抗博弈太不稳定，扩散模型用简单的去噪取代；DDPM 的弯曲路径太慢，Flow Matching 用直线路径取代；U-Net 的局部归纳偏置限制了扩展，Transformer 用全局 attention 取代。

技术在快速迭代，但基础原理是稳定的。理解了这些原理，你就能理解任何新模型——因为它们都在同一个数学框架内创新。
