---
title: "理解 AI 图像生成（四）：条件控制——从文字到图像的桥梁"
date: 2025-06-30
level: 3
series: "理解 AI 图像生成"
series_order: 4
series_total: 5
tags: [clip, cross-attention, classifier-free-guidance, controlnet, ip-adapter, conditioning]
summary: "CLIP 把文字变成向量，Cross-Attention 让图像的每个位置「看到」文字，Classifier-Free Guidance 把微弱的条件信号放大 7-15 倍——三者配合才有了文生图"
---

# 条件控制：从文字到图像的桥梁

> 当你输入"一只戴墨镜的柯基在月球上冲浪"时，模型需要：理解"柯基"长什么样、"墨镜"怎么戴、"月球"的表面纹理、"冲浪"的姿态——然后把这些概念正确地组合在一张图里。这背后是三个精巧机制的配合。

## 第一步：CLIP 文本编码器——把文字变成向量

### CLIP 是什么

CLIP（Contrastive Language-Image Pre-training）是 OpenAI 在 2021 年发布的模型。它在 4 亿张图片-文字配对上训练，学会了把文字描述和图像映射到同一个向量空间：

- 语义相似的文字和图像，向量距离近
- 语义不同的，向量距离远

当你输入一段 prompt，CLIP 的文本编码器把它变成一个 **77×768 的 embedding 序列**（每个 token 一个 768 维向量）。这个序列携带了丰富的语义信息——不仅有每个词的含义，还有词与词之间的关系。

### 为什么用 CLIP 而不是其他文本模型

因为 CLIP 的特殊训练方式让它学会了"视觉相关的语义"。普通的语言模型（如 BERT）理解语法和逻辑，但不知道"赛博朋克风格"长什么样。CLIP 见过几亿张图片-文字对，它知道：

- "watercolor painting" 对应柔和的笔触和淡雅色彩
- "cyberpunk" 对应霓虹灯、暗色调、未来感
- "Studio Ghibli style" 对应特定的动画美学

这种**视觉-语言对齐**是纯语言模型无法提供的。

## 第二步：Cross-Attention——让图像"看到"文字

### 机制

U-Net 在多个分辨率层（64×64、32×32、16×16）插入了 **cross-attention 层**：

- **Query (Q)**：来自 U-Net 的空间特征图（"图像的每个位置在问：我该放什么？"）
- **Key (K) 和 Value (V)**：来自 CLIP 文本 embedding（"文字提供答案"）

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) \cdot V$$

<svg viewBox="0 0 680 220" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:680px;margin:24px auto;display:block;">
  <defs>
    <marker id="carr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- UNet features -->
  <rect x="30" y="50" width="150" height="120" rx="8" fill="#1e1e2a" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="105" y="75" text-anchor="middle" fill="#22d3ee" font-size="11" font-weight="bold" font-family="system-ui">U-Net 特征图</text>
  <text x="105" y="100" text-anchor="middle" fill="#888" font-size="10" font-family="system-ui">32×32 空间位置</text>
  <text x="105" y="120" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">每个位置问：</text>
  <text x="105" y="137" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">"我该画什么？"</text>
  <text x="105" y="160" text-anchor="middle" fill="#22d3ee" font-size="9" font-family="system-ui">→ 投影为 Query (Q)</text>
  <!-- Text embeddings -->
  <rect x="30" y="185" width="150" height="30" rx="6" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="105" y="205" text-anchor="middle" fill="#a78bfa" font-size="10" font-family="system-ui">CLIP 文本: 77×768 → K, V</text>
  <!-- Cross attention -->
  <rect x="250" y="70" width="160" height="80" rx="8" fill="rgba(110,142,255,0.1)" stroke="#6e8eff" stroke-width="2"/>
  <text x="330" y="95" text-anchor="middle" fill="#6e8eff" font-size="11" font-weight="bold" font-family="system-ui">Cross-Attention</text>
  <text x="330" y="115" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">softmax(QKᵀ/√d) · V</text>
  <text x="330" y="135" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">空间位置选择性关注文本</text>
  <!-- Arrows -->
  <line x1="185" y1="100" x2="245" y2="100" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#carr)"/>
  <line x1="185" y1="200" x2="250" y2="130" stroke="#a78bfa" stroke-width="1.5" marker-end="url(#carr)"/>
  <!-- Output -->
  <line x1="415" y1="110" x2="460" y2="110" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#carr)"/>
  <rect x="465" y="70" width="180" height="80" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="555" y="95" text-anchor="middle" fill="#34d399" font-size="11" font-weight="bold" font-family="system-ui">条件化特征</text>
  <text x="555" y="115" text-anchor="middle" fill="#888" font-size="10" font-family="system-ui">左上角区域关注"月球"</text>
  <text x="555" y="133" text-anchor="middle" fill="#888" font-size="10" font-family="system-ui">中心区域关注"柯基"+"冲浪"</text>
</svg>

### 直觉

每个空间位置可以独立地"关注"文本中不同的部分。图片左上角的位置可能主要关注"月球表面"这几个token，而图片中心的位置关注"柯基"和"冲浪"。这种选择性注意力让模型能把正确的概念放在正确的位置。

Cross-attention 在多个分辨率层都有——低分辨率层（8×8, 16×16）负责全局布局（"月球在上，柯基在中"），高分辨率层（32×32, 64×64）负责细节填充（"墨镜的形状"、"皮毛的纹理"）。

## 第三步：Classifier-Free Guidance——放大文本控制

### 问题：条件信号太弱

仅靠 cross-attention，模型确实会考虑文本信息，但生成的图片往往只是"大致相关"——像是写了命题作文但只沾了个边。模型更倾向于生成"好看的通用图片"而非"严格匹配描述的图片"。

### CFG 的天才设计

**训练时**：以 10-20% 的概率随机丢弃文本条件（用空 embedding 替代）。这让模型同时学会两种模式：
- 有条件生成：$\varepsilon_\theta(z_t, t, c)$——"按文字画"
- 无条件生成：$\varepsilon_\theta(z_t, t, \varnothing)$——"随便画"

**推理时**：两种都算，然后做外推：

$$\varepsilon_{\text{guided}} = \varepsilon_{\text{uncond}} + w \cdot (\varepsilon_{\text{cond}} - \varepsilon_{\text{uncond}})$$

其中 $w$ 是 guidance scale（通常 7-15）。

### 直觉解释

$\varepsilon_{\text{cond}} - \varepsilon_{\text{uncond}}$ 表示"文本条件指向的方向"——是"有文字引导"和"无文字引导"之间的差异。乘以 $w > 1$ 就是**放大这个方向**：不只是"考虑文字"，而是"强烈遵循文字"。

**比喻**：想象你在一个岔路口。$\varepsilon_{\text{uncond}}$ 说"随便走"，$\varepsilon_{\text{cond}}$ 说"往右偏一点"。CFG 做的是"不是往右偏一点，而是大步往右走"。$w=7$ 意味着把那个微弱的"右偏"信号放大 7 倍。

$w$ 太低（1-3）：图片漂亮但不听话，和描述关系不大
$w$ 太高（>20）：严格遵循文字但色彩饱和度爆炸、细节失真

**这就是为什么 "guidance scale" 是 Stable Diffusion 里最重要的用户参数之一。**

## ControlNet：精确的空间控制

### 问题：文字无法精确控制构图

"一个人站在左边，一棵树在右边"——这种描述对 Stable Diffusion 来说非常难以精确执行。文字是模糊的空间指令。

### ControlNet 的方案

ControlNet（Zhang et al., 2023）让你用**空间条件图**来精确控制生成：

- **Canny 边缘图**：控制物体轮廓
- **深度图**：控制前后景关系
- **人体姿态骨架**：控制人物动作
- **语义分割图**：控制区域内容

架构极其巧妙：
1. 复制 U-Net 编码器的 12 个 block（可训练副本）
2. 原始 U-Net 锁定不动
3. 副本接收空间条件输入
4. 副本的输出通过**零初始化卷积**加回到原模型

"零初始化"保证训练开始时 ControlNet 的影响为零——模型从预训练的完美状态开始，逐渐学习新条件，永远不会破坏原有能力。

### IP-Adapter：用图片当提示词

IP-Adapter（Ye et al., 2023）让你用一张参考图片来控制风格或内容——"画一张和这张风格类似的图"。

它用 CLIP 图像编码器提取参考图特征，通过独立的 cross-attention 层（和文本的 cross-attention 分开）注入到 U-Net 中。仅 22M 参数，却能实现精确的风格迁移。

## 总结：条件控制的层级

| 控制层级 | 机制 | 精度 | 灵活性 |
|----------|------|------|--------|
| 全局语义 | CLIP 文本 + Cross-Attention | 中等 | 最高（自然语言） |
| 文本强度 | Classifier-Free Guidance | 可调 | 一个参数控制 |
| 空间结构 | ControlNet | 像素级 | 需要条件图 |
| 风格参考 | IP-Adapter | 高 | 需要参考图 |

## 下一篇预告

我们已经理解了 AI 生图的完整技术栈。最后一篇，我们把目光投向更大的画面：当前主流模型（SD 1.5/XL/3、DALL-E 3、Midjourney、Flux）各有什么特点？从 U-Net 到 DiT（Diffusion Transformer）的架构革命意味着什么？Flow Matching 如何让新一代模型只用 10 步就能生成高质量图片？下一篇：主流模型全景与未来方向。
