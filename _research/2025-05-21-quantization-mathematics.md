---
title: "量化的数学：如何把 700 亿参数塞进一张显卡"
date: 2025-05-21
level: 3
series: "LLM 原理深度解析"
series_order: 13
series_total: 37
sources_count: 12
tags: [quantization, GPTQ, AWQ, GGUF, compression, inference]
summary: "从量化的基本数学出发，深入 GPTQ（二阶信息补偿）、AWQ（激活感知缩放）、GGUF（K-Quant 层级结构）三种方法的核心思想与数学直觉"
---

# 量化的数学：如何把 700 亿参数塞进一张显卡

> 一个 70B 参数的模型用 FP16 存储需要 140GB 显存——比任何消费级 GPU 都大。量化是唯一让普通人跑大模型的途径。但"把数字变小"远没有听起来那么简单。

## 故事从这里开始

假设你要搬家，所有行李必须塞进一辆小轿车。你有两个选择：把所有东西随意压缩（可能压坏贵重物品），或者仔细观察哪些东西可以压、哪些必须完好保留。

大语言模型的量化就是这个问题。一个 70B 参数的模型，每个参数用 16 位浮点数存储，总共需要 140GB 内存。但如果我们能把每个参数从 16 位压缩到 4 位——内存直接降到 35GB，一张 RTX 4090 就能装下。

问题是：你不能随便压。神经网络中有些权重对输出影响巨大，量化一旦"压坏"了它们，模型就会胡言乱语。三种主流量化方法——GPTQ、AWQ、GGUF——各自用了截然不同的数学策略来解决"压谁、怎么压、压完怎么补"的问题。

## 量化的基本数学：把连续变离散

### 问题是什么

神经网络的每个权重都是一个浮点数，比如 0.0234375。FP16 格式用 16 位存储，能表达大约 65,000 个不同的数值。但如果我们只用 4 位，就只有 16 个可能的值。怎么把连续空间里的 65,000 个点"映射"到 16 个格点上，并且尽量少丢信息？

### 核心直觉：缩放 + 取整

最简单的量化就像温度转换。假设一组权重的范围是 [-0.5, +0.5]，而 INT4 的范围是 [-8, +7]。我们需要一个"刻度尺"把两个范围对齐：

$$s = \frac{w_{\max} - w_{\min}}{2^b - 1}$$

其中 $b$ 是目标位宽。量化过程就是：先除以缩放因子 $s$，然后四舍五入到最近的整数：

$$q = \text{Round}\left(\frac{w}{s}\right)$$

反量化（推理时恢复近似值）：

$$\hat{w} = q \cdot s$$

这个过程丢失的信息就是**舍入误差**。对单个权重来说，最大误差是 $\frac{s}{2}$——刻度尺的半格。

<svg viewBox="0 0 650 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:650px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Original weights -->
  <rect x="10" y="60" width="140" height="60" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="80" y="85" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">原始权重 (FP16)</text>
  <text x="80" y="105" text-anchor="middle" fill="#8888aa" font-size="11" font-family="system-ui">0.0234, -0.187, ...</text>
  <!-- Arrow 1 -->
  <line x1="150" y1="90" x2="200" y2="90" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>
  <text x="175" y="80" text-anchor="middle" fill="#8888aa" font-size="10" font-family="system-ui">÷ scale</text>
  <!-- Quantized -->
  <rect x="200" y="60" width="140" height="60" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="270" y="85" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">整数 (INT4)</text>
  <text x="270" y="105" text-anchor="middle" fill="#8888aa" font-size="11" font-family="system-ui">2, -3, 7, -1, ...</text>
  <!-- Arrow 2 -->
  <line x1="340" y1="90" x2="390" y2="90" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>
  <text x="365" y="80" text-anchor="middle" fill="#8888aa" font-size="10" font-family="system-ui">× scale</text>
  <!-- Dequantized -->
  <rect x="390" y="60" width="160" height="60" rx="8" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="470" y="85" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">恢复权重 (FP16)</text>
  <text x="470" y="105" text-anchor="middle" fill="#8888aa" font-size="11" font-family="system-ui">0.0250, -0.188, ...</text>
  <!-- Error annotation -->
  <text x="470" y="145" text-anchor="middle" fill="#ff6b6b" font-size="11" font-family="system-ui">↑ 舍入误差 ≤ scale/2</text>
  <!-- Memory savings -->
  <rect x="200" y="150" width="140" height="30" rx="6" fill="#1a2a1a" stroke="#34d399" stroke-width="1"/>
  <text x="270" y="170" text-anchor="middle" fill="#34d399" font-size="11" font-family="system-ui">内存减少 75%</text>
</svg>

### 为什么"简单取整"不够好

如果每个权重独立地四舍五入（Round-to-Nearest, RTN），低位宽时精度急剧下降。原因很直觉：INT4 只有 16 个可能的值，相当于把原本精细的刻度尺换成了只有 16 格的粗尺。对于单个权重，误差可能不大。但一个 Transformer 层的矩阵乘法涉及数千个权重同时作用——这些小误差会**累积**。

更要命的是，权重的重要性极不均匀。有些权重改变 0.001 就让输出剧烈变化（"显著权重"），有些改变 0.1 也无所谓。RTN 对所有权重一视同仁，这就像搬家时把古董花瓶和旧报纸用同样的力度压缩。

三种高级量化方法的核心差异，就在于它们如何识别和保护"重要权重"。

## GPTQ：用二阶信息精确补偿

### 问题是什么

既然简单取整会在每一步引入误差，那能不能在量化一个权重之后，**立刻调整还没量化的其他权重**来"抵消"这个误差？

这就像做木工时的策略：切第一块板时切短了 1 毫米，那切第二块时就故意长 1 毫米，拼起来刚好严丝合缝。关键是：怎么知道第二块该补多少？

### 核心直觉：Hessian 告诉你"补偿系数"

GPTQ 的祖先是 1990 年代的"最优脑外科手术"（Optimal Brain Surgeon, OBS）——一种用二阶导数信息来决定删除哪个权重的方法。GPTQ 把这个想法搬到了量化领域。

想象一下误差函数的"地形图"。一阶导数（梯度）告诉你"坡的方向"，二阶导数（Hessian 矩阵）告诉你"坡的弯曲程度"。Hessian 描述了当你改变某个权重时，误差会如何随其他权重的变化而变化。

对于线性层 $Y = WX$，量化误差的目标函数是：

$$\text{Error} = \|WX - W_Q X\|_F^2$$

这个目标函数的 Hessian 恰好等于输入数据的协方差矩阵：

$$H = 2XX^T$$

为什么这个信息如此宝贵？因为 $H^{-1}$（Hessian 的逆）精确告诉你：当第 $i$ 个权重被"扰动"（量化引入了误差）时，应该按什么比例调整第 $j$ 个权重来最小化总误差。

### 算法步骤：逐列量化 + 即时补偿

GPTQ 对一个层的权重矩阵逐列处理：

1. **计算 Hessian**：用少量校准数据（128-256 条样本）跑一遍前向传播，得到每层输入 $X$，计算 $H = 2XX^T$，并求其逆 $H^{-1}$
2. **选择量化顺序**：按列处理（实际实现中以 128 列为一个 block）
3. **量化一列**：对第 $i$ 列的每个权重取整，得到量化误差 $\delta_i = w_i - \text{quant}(w_i)$
4. **补偿后续列**：用 Hessian 逆的对应行更新所有未量化的列：
   $$w_j \leftarrow w_j - \frac{\delta_i \cdot [H^{-1}]_{ij}}{[H^{-1}]_{ii}}, \quad \forall j > i$$
5. **重复**直到所有列处理完毕

这个补偿公式的含义是："按照输入数据的统计结构，把第 $i$ 列的量化误差，最优地分摊到后续所有列上。"

### 为什么比简单取整好那么多

GPTQ 的核心优势在于：它**不是独立地量化每个权重**，而是把整个层视为一个优化问题。通过 Hessian 信息，它知道哪些权重对输出最敏感（$H^{-1}$ 对角线大的权重），并在量化这些敏感权重后，精确地调整其他权重来消除误差的传播。

代价是什么？需要校准数据和 Hessian 计算。对于 70B 模型，用 GPTQ 量化到 4-bit 大约需要 4-8 小时 GPU 时间。但这是一次性的——量化完成后推理无额外开销。

## AWQ：激活感知的权重缩放

### 问题是什么

GPTQ 的方法很精确但计算量不小。有没有更轻量的策略？

AWQ 的出发点是一个关键观察：**不是所有权重都同样重要，而重要性可以通过激活值来判断**。

### 核心直觉：大激活 × 小误差 = 大损失

考虑一个线性层的计算 $y = wx$。量化误差是：

$$\text{Err} = |Q(w) \cdot x - w \cdot x| = \Delta \cdot \text{RoundErr} \cdot |x|$$

这里 $\Delta$ 是量化步长，RoundErr 是舍入带来的相对误差（均匀分布，期望为 0.25）。关键点：**误差与输入激活 $|x|$ 成正比**。

想象一下：如果某个通道的激活值特别大（比如 10.0），那这个通道的权重即使只有微小的量化误差，也会被放大 10 倍传递到输出。反之，激活值小的通道（比如 0.01），即使权重误差较大也没什么影响。

AWQ 的策略很直白：**保护大激活通道对应的权重**。

### 数学推导：缩放的魔法

如何"保护"重要权重又不需要混合精度（这对硬件不友好）？AWQ 的答案是**缩放**。

对权重矩阵按通道乘以缩放因子 $s$，同时对激活除以 $s$，数学上等价：

$$y = wx = (w \cdot s) \cdot (x / s)$$

量化后变成：

$$\hat{y} = Q(w \cdot s) \cdot (x / s)$$

现在分析误差。缩放后的量化步长变为 $\Delta' = \frac{\max|ws|}{2^{b-1}-1}$。误差变化比例为：

$$\frac{\text{Err(scaled)}}{\text{Err(original)}} = \frac{\Delta'}{\Delta} \cdot \frac{1}{s}$$

关键洞察：如果对重要权重通道（大激活对应的）乘以大的 $s$，虽然 $\Delta'$ 可能增大，但 $1/s$ 的缩小效果更强——前提是缩放因子选择得当，使得 $\frac{\Delta'}{\Delta} < s$。

### 实际优化：网格搜索一个参数

AWQ 最优雅的地方是简化了优化问题。它把每个通道的缩放因子参数化为：

$$s_j = \bar{x}_j^{\alpha}$$

其中 $\bar{x}_j$ 是第 $j$ 个通道的平均激活幅度（从校准数据统计得来），$\alpha \in [0, 1]$ 是唯一需要搜索的超参数。

- $\alpha = 0$：所有通道等比缩放（退化为普通量化）
- $\alpha = 1$：缩放完全跟随激活大小

最优的 $\alpha^*$ 通过在 [0, 1] 上网格搜索得到，评估标准是校准数据上的重建误差。整个过程只需几分钟。

### 工程精妙之处

AWQ 的激活缩放 $x/s$ 不需要额外计算：它可以**融合到前一层的 LayerNorm** 中。因为 LayerNorm 的仿射变换 $\gamma x + \beta$ 本身就是逐通道的，把 $\gamma$ 除以 $s$ 就完成了。所以推理时，AWQ 量化模型的速度和普通 4-bit 模型完全相同——零额外开销。

<svg viewBox="0 0 700 280" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Title -->
  <text x="350" y="25" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">AWQ vs GPTQ：两种保护重要权重的策略</text>
  
  <!-- GPTQ path -->
  <text x="180" y="55" text-anchor="middle" fill="#6e8eff" font-size="12" font-family="system-ui">GPTQ：量化后补偿</text>
  <rect x="30" y="70" width="120" height="45" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="90" y="90" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">量化第 i 列</text>
  <text x="90" y="105" text-anchor="middle" fill="#8888aa" font-size="10" font-family="system-ui">产生误差 δᵢ</text>
  <line x1="150" y1="92" x2="190" y2="92" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <rect x="190" y="70" width="140" height="45" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="260" y="90" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">用 H⁻¹ 补偿后续列</text>
  <text x="260" y="105" text-anchor="middle" fill="#8888aa" font-size="10" font-family="system-ui">wⱼ -= δᵢ·H⁻¹ᵢⱼ/H⁻¹ᵢᵢ</text>
  <line x1="330" y1="92" x2="370" y2="92" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <rect x="370" y="70" width="120" height="45" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="430" y="90" text-anchor="middle" fill="#34d399" font-size="11" font-family="system-ui">误差被分摊</text>
  <text x="430" y="105" text-anchor="middle" fill="#8888aa" font-size="10" font-family="system-ui">精确但计算重</text>
  
  <!-- AWQ path -->
  <text x="180" y="155" text-anchor="middle" fill="#a78bfa" font-size="12" font-family="system-ui">AWQ：量化前缩放</text>
  <rect x="30" y="170" width="120" height="45" rx="8" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="90" y="190" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">统计激活大小</text>
  <text x="90" y="205" text-anchor="middle" fill="#8888aa" font-size="10" font-family="system-ui">x̄ⱼ = E[|xⱼ|]</text>
  <line x1="150" y1="192" x2="190" y2="192" stroke="#a78bfa" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <rect x="190" y="170" width="140" height="45" rx="8" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="260" y="190" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">缩放重要通道</text>
  <text x="260" y="205" text-anchor="middle" fill="#8888aa" font-size="10" font-family="system-ui">w' = w·s, x' = x/s</text>
  <line x1="330" y1="192" x2="370" y2="192" stroke="#a78bfa" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <rect x="370" y="170" width="120" height="45" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="430" y="190" text-anchor="middle" fill="#34d399" font-size="11" font-family="system-ui">重要权重更精确</text>
  <text x="430" y="205" text-anchor="middle" fill="#8888aa" font-size="10" font-family="system-ui">快速 + 零开销</text>
  
  <!-- Comparison note -->
  <rect x="150" y="240" width="400" height="30" rx="6" fill="#1a1a2e" stroke="#3a3a4a" stroke-width="1"/>
  <text x="350" y="260" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">核心区别：GPTQ 在量化后用数学补偿，AWQ 在量化前用缩放预防</text>
</svg>

## GGUF / K-Quants：层级量化与重要性矩阵

### 问题是什么

GPTQ 和 AWQ 都是 GPU 推理场景的方案。但很多人想在 CPU 甚至手机上跑模型——llama.cpp 生态（GGUF 格式）就是为这个场景设计的。它面对独特的挑战：CPU 没有 GPU 那样高效的低精度矩阵乘法硬件，需要更灵活的量化粒度。

### 核心直觉：量化的量化——双层分级

GGUF 的传统量化（legacy quants）给每 32 个权重分配一对 FP16 的缩放因子（scale + zero-point）。对于 16B 参数模型，光是这些元数据就占约 2GB。

K-Quants 的核心创新是**双层量化**（double quantization）：**连量化的缩放因子都量化掉**。

具体结构：
- **超级块（super-block）**：包含 8 个普通块，有一对 FP16 的"超级缩放因子"
- **普通块（block）**：包含 32 个权重 + 一个 INT8 的"块缩放因子"（被超级缩放因子所量化）
- **权重**：INT4/INT5/INT6，用块缩放因子反量化

反量化路径：`INT4 权重 × INT8 块缩放 × FP16 超级缩放 → FP16 近似值`

这把元数据开销砍掉了一半（从 2GB 降到 ~1GB），同时因为超级块的连续内存布局，CPU 缓存命中率也更高。

### 混合精度：不是所有层都一样

K-Quants 的另一个精妙之处是**按层分配精度**。不是整个模型都用 4-bit，而是：

- 注意力权重和输出层：用 Q5 或 Q6（高精度）
- FFN 中间层：用 Q4（低精度）
- LayerNorm 参数：保持 FP16

文件名中的 S/M/L 后缀就是控制这个分配策略的激进程度。Q4_K_M 意味着"基础 4-bit，中等精度分配"。

### 重要性矩阵（Importance Matrix）：借鉴 AWQ 的思想

GGUF 的最新进化是引入了重要性矩阵（imatrix）。核心思想和 AWQ 类似——通过观察模型在校准数据上的行为，给每个权重评估重要性分数。

对权重矩阵 $W$（维度 $N \times M$），在校准数据上做前向传播得到 $y = Wx$，每行的重要性：

$$I_i = y_i^2$$

每个权重的重要性综合了行级重要性和权重本身的大小：

$$I_{ij} = y_i^2 + \sqrt{\sigma^2 + w_{ij}^2}$$

有了重要性矩阵后，量化时不是简单地让所有权重的重建误差相等，而是**让重要权重的重建误差更小**。实现方式是：使用不同的反量化常数（$S'$, $Z'$），通过最小化加权重建误差来选择：

$$L = \sum_{ij} I_{ij} \cdot (w_{ij} - \hat{w}_{ij})^2$$

妙处在于：这个优化**对推理零开销**。重要性矩阵只影响量化时的常数选择，不影响文件格式或推理速度。

## 三种方法的哲学对比

| 维度 | GPTQ | AWQ | GGUF K-Quants |
|------|------|-----|---------------|
| 核心策略 | 量化后补偿（二阶信息） | 量化前缩放（一阶信息） | 分级量化 + 重要性加权 |
| 数学工具 | Hessian 逆矩阵 | 激活统计 + 网格搜索 | 双层量化 + 加权最优化 |
| 目标硬件 | GPU | GPU | CPU / Apple Silicon |
| 量化速度 | 慢（4-8 小时/70B） | 快（几分钟） | 中等 |
| 推理额外开销 | 无 | 无 | 无 |
| 典型精度(4-bit) | 优秀 | 略优于 GPTQ | S/M/L 可调 |
| 代表实现 | AutoGPTQ, ExLlama | AutoAWQ, vLLM | llama.cpp, Ollama |

### 什么时候用什么

- **有 GPU + 追求速度**：AWQ（量化快、推理和 GPTQ 一样快、精度略好）
- **有 GPU + 已有 GPTQ 模型**：GPTQ（生态成熟，HuggingFace 上最多）
- **CPU / 笔记本 / 手机**：GGUF Q4_K_M 或 Q5_K_M（llama.cpp 生态，灵活）
- **极致压缩（2-3 bit）**：GGUF IQ2/IQ3 + imatrix（重要性矩阵在极低位宽收益最大）

## 量化的物理极限在哪里

一个有趣的理论问题：模型能被压缩到多低而不损失能力？

经验表明，4-bit 量化对大多数模型损失很小（perplexity 增加 < 1%）。到 3-bit 就开始明显下降，2-bit 几乎不可用——除非用了 imatrix 这样的高级策略。

背后的直觉是信息论的：模型的参数并非都承载等量的信息。大量参数接近零且相互冗余——这些可以安全地压缩。但少数"关键权重"编码了核心知识，压缩它们就像删除书中的关键章节。

GPTQ、AWQ、GGUF 三种方法，本质上都是在回答同一个问题：**哪些权重是"关键章节"，怎么在压缩空间中给它们分配更多的精度预算。**

---

## 这意味着什么

量化不是简单的"数字变小"。它是一个优化问题：在有限的比特预算下，如何分配精度使得模型行为的改变最小。三种主流方法代表了三种哲学：

- GPTQ 说："我量化完之后用精确的数学来修复破坏"
- AWQ 说："我先搞清楚哪里脆弱，然后在量化前就做好保护"  
- GGUF 说："我用分层结构加重要性评估，把有限的比特精打细算地分配"

当你下次在 HuggingFace 上看到 `model-7B-Q4_K_M.gguf` 或 `model-7B-AWQ` 时，你知道背后不是简单的四舍五入——而是几十年优化理论和信息论的结晶。

## 延伸阅读

- Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers" (2022) — GPTQ 原论文
- Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (2023) — AWQ 原论文  
- llama.cpp K-Quants PR #1684 — K-Quants 的实现讨论
- iuliaturc/gguf-docs — 非官方 GGUF 量化格式文档
