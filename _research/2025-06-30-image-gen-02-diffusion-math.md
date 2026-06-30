---
title: "理解 AI 图像生成（二）：Diffusion Model 原理——为什么「加噪再去噪」能生成图片"
date: 2025-06-30
level: 3
series: "理解 AI 图像生成"
series_order: 2
series_total: 5
tags: [diffusion, ddpm, ddim, score-matching, noise-schedule, denoising]
summary: "扩散模型的核心是一个优雅的对称：前向过程把画作变成灰尘（有解析公式），反向过程训练一个修复师逐层清除灰尘（用简单的 MSE 损失）"
---

# Diffusion Model 原理：为什么「加噪再去噪」能生成图片

> 想象你有一幅名画。一个破坏者用 1000 层灰尘逐渐覆盖它，直到变成纯噪点。现在训练一个修复师，给他任何一个被覆盖了 k 层灰尘的版本，他能精确判断"最上面这层灰尘长什么样"并移除它。反向运行 1000 次，名画从噪声中重现。

## 前向过程：把数据变成噪声

### 逐步加噪的马尔可夫链

给定一张干净图片 $x_0$，前向过程在 $T=1000$ 步中逐步添加高斯噪声：

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} \cdot x_{t-1}, \beta_t \cdot I)$$

每一步做两件事：
1. 把原始信号乘以 $\sqrt{1-\beta_t}$（略微缩小）
2. 加入方差为 $\beta_t$ 的高斯噪声

$\beta_t$ 是**方差调度表（variance schedule）**——一个预设的序列，控制每一步加多少噪声。

### 关键性质：一步跳到任意时刻

这是让扩散模型训练变得可行的**最重要的数学性质**。

定义 $\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t = \alpha_1 \cdot \alpha_2 \cdots \alpha_t$（累积乘积）。

那么从 $x_0$ 直接跳到任意时刻 $t$ 的公式是：

$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

翻译成人话：**任意时刻的噪声版本 = 原图 × 信号保留系数 + 纯噪声 × 噪声系数。** 两个系数的平方和恰好等于 1（保持总方差不变）。

当 $\bar{\alpha}_t \approx 1$（早期步骤）：图片几乎没变
当 $\bar{\alpha}_t \approx 0$（后期步骤）：已经是纯噪声

<svg viewBox="0 0 700 180" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <!-- Clean image -->
  <rect x="30" y="50" width="80" height="80" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="2"/>
  <text x="70" y="95" text-anchor="middle" fill="#34d399" font-size="11" font-family="system-ui">x₀ 清晰</text>
  <text x="70" y="145" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">ᾱ≈1.0</text>
  <!-- Step 1 -->
  <rect x="160" y="50" width="80" height="80" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="200" y="90" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">x₂₅₀</text>
  <text x="200" y="105" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">轻微噪声</text>
  <text x="200" y="145" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">ᾱ≈0.7</text>
  <!-- Step 2 -->
  <rect x="290" y="50" width="80" height="80" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="330" y="90" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">x₅₀₀</text>
  <text x="330" y="105" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">明显噪声</text>
  <text x="330" y="145" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">ᾱ≈0.3</text>
  <!-- Step 3 -->
  <rect x="420" y="50" width="80" height="80" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="460" y="90" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">x₇₅₀</text>
  <text x="460" y="105" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">几乎全噪声</text>
  <text x="460" y="145" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">ᾱ≈0.05</text>
  <!-- Pure noise -->
  <rect x="550" y="50" width="80" height="80" rx="8" fill="#1e1e2a" stroke="#ff6b6b" stroke-width="2"/>
  <text x="590" y="90" text-anchor="middle" fill="#ff6b6b" font-size="10" font-family="system-ui">x_T</text>
  <text x="590" y="105" text-anchor="middle" fill="#ff6b6b" font-size="9" font-family="system-ui">纯噪声</text>
  <text x="590" y="145" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">ᾱ≈0.0</text>
  <!-- Arrows -->
  <text x="130" y="85" fill="#6e8eff" font-size="14" font-family="system-ui">→</text>
  <text x="255" y="85" fill="#6e8eff" font-size="14" font-family="system-ui">→</text>
  <text x="385" y="85" fill="#6e8eff" font-size="14" font-family="system-ui">→</text>
  <text x="515" y="85" fill="#6e8eff" font-size="14" font-family="system-ui">→</text>
  <text x="350" y="30" text-anchor="middle" fill="#888" font-size="11" font-family="system-ui">前向过程：逐步加噪（有解析公式，可一步跳到任意 t）</text>
</svg>

**为什么这个性质如此重要？** 因为训练时你不需要跑完 1000 步前向过程。你随机采样一个 $t$，直接用公式算出 $x_t$，然后训练网络去噪。这让训练效率提高了几个数量级。

## 反向过程：学会去噪

### 预测噪声，而非预测图片

反向过程从纯噪声 $x_T$ 开始，逐步去噪到 $x_0$。关键的设计选择：**网络 $\varepsilon_\theta(x_t, t)$ 预测的是"当前时刻的噪声"，而不是直接预测干净图片。**

为什么预测噪声比预测图片更好？

1. **目标分布一致**：不管 $t$ 是多少，噪声 $\varepsilon$ 永远是标准正态分布。网络的目标分布不随时间变化，学习更稳定。
2. **和 Score Matching 等价**：预测噪声在数学上等价于估计 score function $\nabla_x \log p(x_t)$——指向"数据更可能存在的方向"的梯度。

知道了噪声 $\varepsilon$ 后，反向一步的均值是：

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \cdot \varepsilon_\theta(x_t, t) \right)$$

翻译成人话：**从当前噪声图中减去预测的噪声（适当缩放），得到稍微干净一点的版本。**

## 训练：极简的 MSE 损失

DDPM 论文最惊人的发现之一是：理论上应该用复杂的变分下界（VLB）来训练，但实践中**只用简单的 MSE 损失效果更好**：

$$\mathcal{L} = \mathbb{E}_{t, x_0, \varepsilon} \left[ \| \varepsilon - \varepsilon_\theta(x_t, t) \|^2 \right]$$

训练算法极其简洁：

```
repeat:
    1. 从数据集采样一张图 x₀
    2. 随机选一个时间步 t ~ Uniform(1, T)
    3. 采样噪声 ε ~ N(0, I)
    4. 计算 xₜ = √ᾱₜ · x₀ + √(1-ᾱₜ) · ε
    5. 训练网络最小化 ||ε - ε_θ(xₜ, t)||²
```

这就是全部。没有对抗训练、没有复杂的概率推导——本质上就是有监督学习：给定噪声图片和时间步，预测其中的噪声成分。数据无限（你随时可以对同一张图加不同的噪声），过拟合风险低。

## DDIM：从 1000 步到 50 步

### 问题：采样太慢

DDPM 的反向过程是随机的马尔可夫链——每一步都有新噪声注入，必须走完全部 1000 步。生成一张图需要 1000 次神经网络前向传播，这在实际应用中不可接受。

### DDIM 的关键洞察

Song et al. (2020) 发现可以把反向过程重新定义为**非马尔可夫**的。当把随机性设为零（$\sigma_t = 0$）时，过程变成完全确定性的——给定同一个初始噪声永远生成同一张图。

确定性过程意味着你可以**跳步**：不需要走 1000→999→998→...→0，可以走 1000→900→800→...→0，只用 50-100 步就能生成高质量图片。

**额外好处：** 确定性映射创建了噪声和图片之间的双射——你可以在潜在空间做插值（两张脸之间平滑过渡）。

## Score-Based / SDE 框架统一

Song et al. (ICLR 2021) 用随机微分方程（SDE）统一了所有扩散方法：

$$dx = f(x,t)dt + g(t)dw \quad \text{(前向 SDE)}$$

反向 SDE 需要知道 score function $\nabla_x \log p_t(x)$，而这恰好是噪声预测网络（重缩放后）在估计的东西！

这个统一框架揭示了：
- DDPM = Variance Preserving SDE 的离散化
- DDIM = 概率流 ODE（同分布、无随机性、可跳步）
- 预测噪声 ≡ 预测 score ≡ 预测干净图（只是缩放因子不同）

## 噪声调度：线性 vs 余弦

### 线性调度（DDPM 原版）

$\beta_t$ 从 $10^{-4}$ 线性增长到 0.02。问题：信息被破坏得太快——到 t≈700 时已经是纯噪声了，后面 300 步浪费了。

### 余弦调度（Improved DDPM, 2021）

通过余弦函数定义 $\bar{\alpha}_t$，让信号衰减更均匀：

$$\bar{\alpha}_t = \cos^2\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)$$

**效果**：信息保留更久，每个时间步都有贡献，高分辨率图片质量显著提升。

现代系统（SD、DALL-E）基本都用余弦调度或其变体。

## 总结：扩散模型的数学之美

| 组件 | 作用 | 关键洞察 |
|------|------|----------|
| 前向过程 | 数据→噪声 | 有解析解，可一步跳到任意 t |
| 反向过程 | 噪声→数据 | 预测噪声比预测图片更稳定 |
| 训练 | 学习去噪 | 简单 MSE 损失胜过复杂 VLB |
| DDIM | 快速采样 | 确定性 + 跳步 = 10-50x 加速 |
| SDE 统一 | 理论框架 | score = 噪声预测（重缩放） |
| 噪声调度 | 控制加噪速率 | 余弦 > 线性 |

## 下一篇预告

扩散模型的数学很优雅，但有一个实际问题：在 512×512 像素空间里做 50 步去噪——每一步都是完整的 U-Net 前向传播——计算量依然巨大。Rombach 等人在 2022 年给出了答案：**何不先把图片压缩到一个小得多的潜在空间，然后在那里做扩散？** 这就是 Latent Diffusion Model——Stable Diffusion 的核心技术。下一篇，我们看看这个巧妙的"压缩再生成"策略。
