---
title: "理解 AI 图像生成（一）：从 GAN 到 Diffusion——生图技术的十年进化史"
date: 2025-06-30
level: 3
series: "理解 AI 图像生成"
series_order: 1
series_total: 5
tags: [image-generation, gan, vae, diffusion, flow-matching, dall-e, stable-diffusion]
summary: "从 GAN 的对抗博弈到 Diffusion 的逐步去噪再到 Flow Matching 的直线传输——AI 生图的每一次范式转移都在解决前一代的核心痛点"
---

# 从 GAN 到 Diffusion：AI 生图技术的十年进化史

> 2014 年，AI 生成的图片像素模糊、勉强能看出是人脸。2024 年，AI 生成的照片已经能骗过绝大多数人。这十年发生了什么？

## 故事从一场博弈开始

2014 年，Ian Goodfellow 在蒙特利尔的一个酒吧里和朋友讨论生成模型时灵光一闪：**如果让两个神经网络互相博弈会怎样？**

一个网络（生成器）负责生成假图片，另一个网络（判别器）负责区分真假。生成器的目标是骗过判别器，判别器的目标是识破生成器。两者在对抗中不断进化——就像伪造者和鉴定师的军备竞赛。

这就是 **GAN（Generative Adversarial Network，生成对抗网络）**。它在2014-2020年间统治了图像生成领域。

## GAN 时代（2014-2020）：辉煌与痛苦

### 关键变体的演进

**DCGAN（2015）**：首次将 CNN 引入 GAN，用转置卷积替代全连接层。这是 GAN 第一次能稳定地生成像样的图片。

**ProGAN（2017）**：NVIDIA 提出渐进式训练——从 4×4 开始逐步增长到 1024×1024。先学粗糙结构再学细节，就像画家先构图再精修。这让 GAN 首次生成了 1024 分辨率的逼真图片。

**StyleGAN（2018-2019）**：同样来自 NVIDIA，引入了"风格映射网络"，让生成过程可以在不同层级独立控制——粗糙层控制姿态，中间层控制五官，精细层控制肤色和发色。StyleGAN 生成的人脸达到了肉眼难辨的程度（thispersondoesnotexist.com 就是它的作品）。

<svg viewBox="0 0 700 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="garr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Timeline -->
  <line x1="50" y1="100" x2="650" y2="100" stroke="#3a3a4a" stroke-width="2"/>
  <!-- GAN era -->
  <rect x="50" y="85" width="250" height="30" rx="4" fill="rgba(255,107,107,0.15)" stroke="#ff6b6b" stroke-width="1"/>
  <text x="175" y="75" text-anchor="middle" fill="#ff6b6b" font-size="11" font-weight="bold" font-family="system-ui">GAN 时代</text>
  <!-- Diffusion era -->
  <rect x="310" y="85" width="200" height="30" rx="4" fill="rgba(167,139,250,0.15)" stroke="#a78bfa" stroke-width="1"/>
  <text x="410" y="75" text-anchor="middle" fill="#a78bfa" font-size="11" font-weight="bold" font-family="system-ui">Diffusion 时代</text>
  <!-- Flow era -->
  <rect x="520" y="85" width="130" height="30" rx="4" fill="rgba(52,211,153,0.15)" stroke="#34d399" stroke-width="1"/>
  <text x="585" y="75" text-anchor="middle" fill="#34d399" font-size="11" font-weight="bold" font-family="system-ui">Flow Matching</text>
  <!-- Markers -->
  <circle cx="80" cy="100" r="4" fill="#ff6b6b"/><text x="80" y="135" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">2014</text><text x="80" y="148" text-anchor="middle" fill="#ededf0" font-size="8" font-family="system-ui">GAN</text>
  <circle cx="160" cy="100" r="4" fill="#ff6b6b"/><text x="160" y="135" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">2017</text><text x="160" y="148" text-anchor="middle" fill="#ededf0" font-size="8" font-family="system-ui">ProGAN</text>
  <circle cx="210" cy="100" r="4" fill="#ff6b6b"/><text x="210" y="135" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">2018</text><text x="210" y="148" text-anchor="middle" fill="#ededf0" font-size="8" font-family="system-ui">StyleGAN</text>
  <circle cx="330" cy="100" r="5" fill="#a78bfa"/><text x="330" y="135" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">2020</text><text x="330" y="148" text-anchor="middle" fill="#ededf0" font-size="8" font-family="system-ui">DDPM</text>
  <circle cx="410" cy="100" r="4" fill="#a78bfa"/><text x="410" y="135" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">2022</text><text x="410" y="148" text-anchor="middle" fill="#ededf0" font-size="8" font-family="system-ui">SD/DALL-E2</text>
  <circle cx="480" cy="100" r="4" fill="#a78bfa"/><text x="480" y="135" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">2023</text><text x="480" y="148" text-anchor="middle" fill="#ededf0" font-size="8" font-family="system-ui">SD-XL</text>
  <circle cx="570" cy="100" r="5" fill="#34d399"/><text x="570" y="135" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">2024</text><text x="570" y="148" text-anchor="middle" fill="#ededf0" font-size="8" font-family="system-ui">SD3/Flux</text>
</svg>

### GAN 的两大致命问题

**模式崩塌（Mode Collapse）**：生成器找到少数能骗过判别器的"安全答案"后就不再探索了。就像一个学生发现老师只考几种题型后，就只背那几个答案——多样性丧失殆尽。训练在人脸数据集上的 GAN 可能只会生成几种固定的面部特征组合。

**训练不稳定**：生成器和判别器的平衡极其脆弱。判别器太强，生成器得不到有效梯度（"不管画什么都是零分，没有方向"）；判别器太弱，生成器学不到真正的数据结构。就像跷跷板——稍有偏差就会崩溃。

## VAE：结构化的潜在空间

**VAE（Variational Autoencoder，变分自编码器）** 走了一条更"温和"的路线：

1. **编码器**把图像压缩到一个连续的潜在空间
2. **解码器**从潜在空间中采样点来重建图像
3. KL 散度正则化保证潜在空间平滑可插值

VAE 训练稳定、潜在空间有良好结构（可以在两张脸之间平滑过渡），但生成的图像通常**模糊**。原因是数学上的必然：VAE 优化像素级重建误差，面对不确定性时倾向于取平均——就像把多张可能的图片叠加，结果自然模糊。

GAN 用对抗损失直接惩罚"不真实"，所以输出锐利但不稳定。VAE 用重建损失保证"不偏离"，所以输出稳定但模糊。这是**模式寻找 vs 模式覆盖**的经典 trade-off。

## 扩散模型为何胜出（2020-2023）

2020 年，Jonathan Ho 等人的 DDPM 论文让扩散模型走向实用。2021 年，"Diffusion Models Beat GANs on Image Synthesis"正式宣告扩散模型在图像质量上超越 GAN。

三个核心优势：

**1. 训练稳定性**：GAN 的对抗训练容易崩溃。扩散模型的训练目标是简单的 MSE 去噪损失——没有对抗动态，训练过程"无聊但可靠"。

**2. 模式覆盖**：模式崩塌在扩散模型中不存在。它天然覆盖数据分布的所有模式，多样性极高。

**3. 可扩展性**：扩散模型的性能随计算和数据的增长表现出优异的 scaling behavior——这和 LLM 的发展路线完美契合。

## 自回归图像模型：用语言模型的方式生图

### DALL-E 1（2021）

OpenAI 走了一条完全不同的路：**像生成文本一样生成图像。**

1. 用离散 VAE 把图片编码为 32×32 的 token 序列（每个 token 从 8192 的词汇表中选）
2. 把文本 token 和图像 token 拼接成一个长序列
3. 用 120 亿参数的 GPT 模型自回归地预测下一个 token

DALL-E 1 证明了语言模型范式可以迁移到图像领域，但推理速度极慢（需要逐个生成数千个 token），且存在错误累积。

### Parti（2022）

Google 的 Parti 把自回归方法推到 200 亿参数，在复杂场景生成上表现出色。但同样受制于生成速度——最终，扩散模型因为推理效率更高而在实际应用中胜出。

## Flow Matching（2023-2024）：最新范式

扩散模型也有痛点：需要很多采样步骤（20-50 步），噪声调度要精心设计，理论框架（SDE）复杂。

**Flow Matching** 提出了更优雅的替代：**为什么不直接学习从噪声到数据的最短路径？**

核心思想：
- 定义一个连续的流，把噪声分布直线映射到数据分布
- 网络学习一个"速度场"，指导样本沿着最优传输路径移动
- **Rectified Flow** 让路径尽可能直——直线路径意味着只需很少步数就能到达

到 2024 年底，**Flow Matching 已取代 DDPM 成为主流范式**。Stable Diffusion 3 和 Flux 都采用了 Rectified Flow。

## 总结：每一代解决前一代的痛点

| 技术 | 核心思想 | 优势 | 痛点 |
|------|----------|------|------|
| GAN (2014) | 对抗博弈 | 锐利输出 | 训练崩溃、模式崩塌 |
| VAE (2013) | 概率编码解码 | 训练稳定、可插值 | 输出模糊 |
| DDPM (2020) | 逐步去噪 | 稳定+高质量+多样性 | 采样慢 |
| 自回归 (2021) | 逐 token 生成 | Scaling law 有效 | 推理极慢 |
| Flow Matching (2023) | 直线最优传输 | 少步骤+简洁+高效 | 相对较新 |

## 下一篇预告

我们知道了扩散模型胜出的原因，但还没真正理解它**怎么工作**。一个模型怎么能从纯噪声中"凭空"生成一张逼真的图片？答案是：它不是凭空创造，而是像修复师一样——一层一层地揭开覆盖在画作上的灰尘。下一篇，我们深入扩散模型的数学原理。
