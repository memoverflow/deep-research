---
title: "混合精度训练：用一半的比特做同样的事"
date: 2025-05-17
level: 3
series: "LLM 原理深度解析"
series_order: 5
series_total: 43
tags: [mixed-precision, FP16, BF16, FP8, 数值计算, 训练优化]
summary: "为什么把 32 位浮点砍成 16 位甚至 8 位，模型照样能训好？深入浮点数的比特世界，理解 loss scaling 的数学必要性，以及 BF16 为何成为 LLM 训练的事实标准。"
---

# 混合精度训练：用一半的比特做同样的事

> 训练一个大模型需要数千块 GPU 跑数周。如果有一种方法能让训练快一倍、省一半显存，而且结果几乎一样——你会不会觉得太好了以至于不真实？混合精度训练就是这种"免费午餐"，但它背后的数值原理远比看起来精妙。

## 故事从这里开始

2017 年，NVIDIA 的 Paulius Micikevicius 团队发表了一篇看似简单的论文：把神经网络训练中的数字从 32 位浮点（FP32）换成 16 位浮点（FP16），训练速度翻倍，显存砍半，而且模型精度不受损。

听起来像是作弊。毕竟，你把每个数字的精度砍掉一半——相当于把尺子上的刻度从毫米变成了厘米——怎么可能测量结果还一样准？

答案在于一个关键洞察：**神经网络训练对精度的需求是不均匀的。** 有些运算（比如矩阵乘法）天生就不需要那么高的精度，而有些运算（比如梯度累加）则需要精心保护。混合精度的"混合"二字，就是让对的精度出现在对的地方。

但要理解为什么这能工作，我们得先钻进浮点数的比特世界。

## 浮点数的隐秘世界

### 你的 GPU 怎么表示一个小数？

我们每天使用的十进制小数（比如 3.14159）在计算机里并不是这样存储的。计算机用的是**浮点表示法**——本质上是科学计数法的二进制版本。

回忆一下科学计数法：$6.022 \times 10^{23}$。这里有三个部分：
- **符号**：正还是负
- **有效数字**（mantissa/significand）：6.022，决定精度
- **指数**（exponent）：23，决定数值范围

浮点数完全相同，只是换成了二进制：

$$(-1)^{sign} \times 1.mantissa \times 2^{exponent - bias}$$

关键来了：**给定总位数固定，指数位和尾数位是跷跷板关系。** 指数位多 → 能表示的数值范围大（从极小到极大），但精度粗糙；尾数位多 → 精度细腻（相邻两个可表示数之间的间距小），但范围受限。

这就是不同浮点格式设计哲学的核心分歧。

### 四种格式的比特账本

<svg viewBox="0 0 680 320" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:680px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Title -->
  <text x="340" y="28" text-anchor="middle" fill="#ededf0" font-size="15" font-family="system-ui" font-weight="bold">浮点格式比特分配对比</text>
  
  <!-- FP32 -->
  <text x="55" y="65" text-anchor="middle" fill="#94a3b8" font-size="12" font-family="system-ui">FP32</text>
  <rect x="90" y="50" width="20" height="24" rx="3" fill="#ef4444" opacity="0.3" stroke="#ef4444" stroke-width="1"/>
  <text x="100" y="66" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">1</text>
  <rect x="112" y="50" width="100" height="24" rx="3" fill="#f59e0b" opacity="0.3" stroke="#f59e0b" stroke-width="1"/>
  <text x="162" y="66" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">8 位指数</text>
  <rect x="214" y="50" width="290" height="24" rx="3" fill="#6e8eff" opacity="0.3" stroke="#6e8eff" stroke-width="1"/>
  <text x="359" y="66" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">23 位尾数</text>
  <text x="540" y="66" text-anchor="start" fill="#64748b" font-size="10" font-family="system-ui">范围 ±3.4×10³⁸</text>
  
  <!-- FP16 -->
  <text x="55" y="115" text-anchor="middle" fill="#94a3b8" font-size="12" font-family="system-ui">FP16</text>
  <rect x="90" y="100" width="20" height="24" rx="3" fill="#ef4444" opacity="0.3" stroke="#ef4444" stroke-width="1"/>
  <text x="100" y="116" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">1</text>
  <rect x="112" y="100" width="62" height="24" rx="3" fill="#f59e0b" opacity="0.3" stroke="#f59e0b" stroke-width="1"/>
  <text x="143" y="116" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">5 位指数</text>
  <rect x="176" y="100" width="125" height="24" rx="3" fill="#6e8eff" opacity="0.3" stroke="#6e8eff" stroke-width="1"/>
  <text x="238" y="116" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">10 位尾数</text>
  <text x="540" y="116" text-anchor="start" fill="#64748b" font-size="10" font-family="system-ui">范围 ±65,504</text>
  
  <!-- BF16 -->
  <text x="55" y="165" text-anchor="middle" fill="#94a3b8" font-size="12" font-family="system-ui">BF16</text>
  <rect x="90" y="150" width="20" height="24" rx="3" fill="#ef4444" opacity="0.3" stroke="#ef4444" stroke-width="1"/>
  <text x="100" y="166" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">1</text>
  <rect x="112" y="150" width="100" height="24" rx="3" fill="#f59e0b" opacity="0.3" stroke="#f59e0b" stroke-width="1"/>
  <text x="162" y="166" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">8 位指数</text>
  <rect x="214" y="150" width="87" height="24" rx="3" fill="#6e8eff" opacity="0.3" stroke="#6e8eff" stroke-width="1"/>
  <text x="257" y="166" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">7 位尾数</text>
  <text x="540" y="166" text-anchor="start" fill="#64748b" font-size="10" font-family="system-ui">范围 ±3.4×10³⁸</text>
  
  <!-- FP8 E4M3 -->
  <text x="55" y="215" text-anchor="middle" fill="#94a3b8" font-size="12" font-family="system-ui">FP8</text>
  <text x="55" y="228" text-anchor="middle" fill="#94a3b8" font-size="10" font-family="system-ui">E4M3</text>
  <rect x="90" y="204" width="20" height="24" rx="3" fill="#ef4444" opacity="0.3" stroke="#ef4444" stroke-width="1"/>
  <text x="100" y="220" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">1</text>
  <rect x="112" y="204" width="50" height="24" rx="3" fill="#f59e0b" opacity="0.3" stroke="#f59e0b" stroke-width="1"/>
  <text x="137" y="220" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">4 位</text>
  <rect x="164" y="204" width="37" height="24" rx="3" fill="#6e8eff" opacity="0.3" stroke="#6e8eff" stroke-width="1"/>
  <text x="182" y="220" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">3 位</text>
  <text x="540" y="220" text-anchor="start" fill="#64748b" font-size="10" font-family="system-ui">范围 ±448</text>

  <!-- FP8 E5M2 -->
  <text x="55" y="265" text-anchor="middle" fill="#94a3b8" font-size="12" font-family="system-ui">FP8</text>
  <text x="55" y="278" text-anchor="middle" fill="#94a3b8" font-size="10" font-family="system-ui">E5M2</text>
  <rect x="90" y="254" width="20" height="24" rx="3" fill="#ef4444" opacity="0.3" stroke="#ef4444" stroke-width="1"/>
  <text x="100" y="270" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">1</text>
  <rect x="112" y="254" width="62" height="24" rx="3" fill="#f59e0b" opacity="0.3" stroke="#f59e0b" stroke-width="1"/>
  <text x="143" y="270" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">5 位</text>
  <rect x="176" y="254" width="25" height="24" rx="3" fill="#6e8eff" opacity="0.3" stroke="#6e8eff" stroke-width="1"/>
  <text x="188" y="270" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">2</text>
  <text x="540" y="270" text-anchor="start" fill="#64748b" font-size="10" font-family="system-ui">范围 ±57,344</text>

  <!-- Legend -->
  <rect x="90" y="295" width="12" height="12" rx="2" fill="#ef4444" opacity="0.3" stroke="#ef4444" stroke-width="1"/>
  <text x="106" y="305" fill="#94a3b8" font-size="10" font-family="system-ui">符号</text>
  <rect x="150" y="295" width="12" height="12" rx="2" fill="#f59e0b" opacity="0.3" stroke="#f59e0b" stroke-width="1"/>
  <text x="166" y="305" fill="#94a3b8" font-size="10" font-family="system-ui">指数（决定范围）</text>
  <rect x="290" y="295" width="12" height="12" rx="2" fill="#6e8eff" opacity="0.3" stroke="#6e8eff" stroke-width="1"/>
  <text x="306" y="305" fill="#94a3b8" font-size="10" font-family="system-ui">尾数（决定精度）</text>
</svg>

来翻译成人话：

- **FP32**（32 位）：8 位指数 + 23 位尾数。范围极大（±3.4×10³⁸），精度极高（约 7 位有效十进制数字）。这是"标准尺子"。
- **FP16**（16 位）：5 位指数 + 10 位尾数。范围很小（最大只有 65,504），精度中等（约 3.3 位有效十进制数字）。
- **BF16**（16 位）：8 位指数 + 7 位尾数。范围和 FP32 一样大，但精度更粗（约 2.4 位有效十进制数字）。
- **FP8 E4M3**（8 位）：最大值只有 448，精度极粗。用于前向传播。
- **FP8 E5M2**（8 位）：范围稍大（57,344），精度更粗（只有 4 个可区分的值在任意两个 2 的幂之间）。用于反向传播的梯度。

### 为什么精度损失不会杀死模型？

这是最反直觉的地方。你把精度从 7 位有效数字砍到 3 位，按理说应该损失巨大——为什么实际上几乎无影响？

关键洞察：**神经网络本身就是一个噪声极大的系统。** 

想想看：每个 mini-batch 的梯度本身就是对真实梯度的有噪声估计。Dropout 会随机砍掉一半神经元。数据增强会故意加入扰动。在这样一个充满随机性的系统里，计算过程中多一点舍入误差，就像往大海里多倒了一杯水——几乎察觉不到。

但"几乎"不等于"完全"。有些情况下，精度不够真的会出问题。这就是为什么我们需要"混合"——在需要高精度的地方保留 FP32，在不需要的地方用 FP16/BF16 加速。

## 混合精度的三板斧

2017 年 NVIDIA 提出的混合精度训练框架，核心就是三个简单的技巧。但每个技巧都解决了一个精确的数值问题。

### 第一板斧：FP32 主权重（Master Weights）

**问题是什么？**

假设你有一个权重值为 1.0 的参数，学习率是 0.0001，梯度是 0.01。那么更新量是：

$$\Delta w = lr \times gradient = 0.0001 \times 0.01 = 0.000001$$

现在问题来了：在 FP16 里，1.0 附近两个相邻可表示数字之间的最小间距是 $2^{-10} \approx 0.001$。而你的更新量是 0.000001——比最小间距还小三个数量级！

这意味着如果你直接在 FP16 权重上加这个更新：

$$1.0 + 0.000001 = 1.0 \quad \text{(在 FP16 中)}$$

更新直接被吃掉了，就像没发生过一样。这就是浮点数的**"吞噬"（swamping）**现象：一个大数加上一个极小的数，小数完全消失。

**解决方案：** 保留一份 FP32 的"主权重"。所有前向和反向计算用 FP16 做（快），但权重更新在 FP32 副本上进行（准），然后把更新后的 FP32 权重四舍五入到 FP16 给下一轮计算用。

翻译成人话：考试的时候用铅笔草稿（快但粗糙），但正式答题卡用钢笔抄写（慢但精确）。两者配合，既不丢分也不慢。

代价是多存了一份 FP32 权重——但这和激活值占用的显存相比微不足道（激活值占大头，权重只是小头）。

### 第二板斧：损失缩放（Loss Scaling）

**问题是什么？**

FP16 能表示的最小正规数（normal number）是 $2^{-14} \approx 6.1 \times 10^{-5}$。虽然非正规数（subnormal/denormal）可以延伸到约 $6 \times 10^{-8}$，但精度极度退化且硬件处理很慢。

而在实际的深度网络训练中，**大量梯度值小于 $2^{-14}$。** NVIDIA 的论文展示了一个真实案例：某网络训练过程中，梯度值的分布有相当一部分落在 FP16 的"盲区"（值太小，变成零）。这些梯度不是不重要——它们只是恰好数值很小而已。一旦变成零，模型就失去了学习方向。

**解决方案的直觉：** 既然问题是"梯度太小，FP16 装不下"，那就在反向传播之前把所有梯度**等比放大**！

具体做法：
1. 前向传播计算出 loss 后，乘以一个缩放因子 S（比如 1024 或 65536）
2. 用放大后的 loss 做反向传播——由链式法则，所有梯度都会被放大 S 倍
3. 更新权重前，把梯度除以 S，恢复原始大小

这就像天文学家观察极暗的星星——先用望远镜把光放大（loss scaling），记录清楚后再按比例缩小回真实亮度。信号本身没变，只是在"运输"过程中被放大以避免丢失。

**动态损失缩放：** 静态的缩放因子需要手动调——太大会导致梯度溢出（变成 inf），太小则保护不够。PyTorch 的 `GradScaler` 实现了动态策略：
- 以一个大值（如 65536）开始
- 如果连续 N 步（默认 2000）没出现 inf/nan，就把 S 翻倍（继续试探上限）
- 一旦出现 inf/nan，把 S 减半，跳过这一步更新

这是一种"贪心试探"——在不溢出的前提下，尽可能把梯度放大到 FP16 能承受的最大范围，最大化利用有限的精度空间。

### 第三板斧：精度累积（FP32 Accumulation）

**问题是什么？**

矩阵乘法的核心操作是点积：把两组向量对应元素相乘，然后把所有乘积加起来。如果两个 FP16 向量各有 1024 个元素，你需要做 1024 次乘法然后累加。

每次乘法的结果本身就有舍入误差。当你把 1024 个这样的结果加起来，误差会累积。更糟糕的是，随着部分和（partial sum）越来越大，后面加入的小乘积又会被"吞噬"。

**解决方案：** 硬件层面，Tensor Core 做 FP16×FP16 的乘法，但把乘积累加到 FP32 的寄存器里。最终结果可以保留为 FP32 或转回 FP16。

这不需要程序员操心——它是内置在 GPU 硬件中的。NVIDIA 从 Volta 架构开始的 Tensor Core 就支持这种"FP16 乘、FP32 加"的模式。这也是混合精度能在不损失精度的情况下获得速度提升的硬件基础。

## BF16：Google 的"暴力美学"

### 为什么 FP16 还是不够好？

FP16 的混合精度训练需要 loss scaling，需要仔细选择哪些操作用 FP16、哪些保留 FP32，需要 dynamic loss scaler 来自动调节……工程上麻烦不说，还时不时会出问题。

HuggingFace 在实现大型 T5 模型时就踩过坑：即使用了混合精度和 loss scaling，模型的 attention score 在很深的层中会累积到超出 FP16 范围（> 65504），直接变成 inf。

根本原因是 FP16 的**范围太小**：最大值只有 65,504。在很深的网络里，中间值完全有可能超过这个上限。

### BF16 的设计哲学

Google 在设计 TPU 时做了一个大胆的决定：把 FP32 的 32 位"暴力截断"到 16 位——保留全部 8 位指数，只砍尾数（从 23 位砍到 7 位）。

这就是 BF16（Brain Floating Point，因为最早用在 Google Brain 项目中）。

翻译成人话：**"我宁可看不清细节，也不能看不到东西。"**

- FP16 的选择：5 位指数 + 10 位尾数 → 看得清但视野小（范围窄）
- BF16 的选择：8 位指数 + 7 位尾数 → 看不太清但视野大（范围和 FP32 一样）

对于深度学习训练来说，BF16 的设计哲学被证明是正确的：

1. **不需要 loss scaling。** 因为 BF16 的范围和 FP32 完全相同（±3.4×10³⁸），梯度几乎不可能下溢或上溢。训练代码直接删掉 GradScaler，省心。
2. **与 FP32 转换极简。** BF16 就是 FP32 的前 16 位（截断后 16 位尾数），所以 FP32↔BF16 转换只需要截断或补零——没有复杂的指数重映射。
3. **精度够用。** 7 位尾数意味着约 2-3 位有效十进制数字。对于梯度估计这种本身就有大噪声的信号，这完全够了。

### BF16 的代价

BF16 也不是没有缺点：

- **精度确实粗。** 在 1.0 附近，两个相邻 BF16 数之间的间距是 $2^{-7} = 1/128 \approx 0.0078$。而 FP16 是 $2^{-10} = 1/1024 \approx 0.001$。如果你的应用需要高精度（比如金融计算），BF16 不合适。
- **推理时不如 FP16。** 推理时数值范围通常可控（不会有梯度爆炸的风险），这时 FP16 的更高精度反而是优势。这也是为什么很多人用 BF16 训练、FP16 推理。

<svg viewBox="0 0 620 240" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:620px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Title -->
  <text x="310" y="24" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">FP16 vs BF16：精度与范围的权衡</text>
  
  <!-- Number line representation -->
  <!-- FP16 section -->
  <text x="50" y="70" fill="#22d3ee" font-size="13" font-family="system-ui" font-weight="bold">FP16</text>
  <line x1="100" y1="67" x2="580" y2="67" stroke="#3a3a4a" stroke-width="1"/>
  <!-- Dense ticks in limited range -->
  <line x1="100" y1="60" x2="100" y2="74" stroke="#22d3ee" stroke-width="1.5"/>
  <line x1="120" y1="62" x2="120" y2="72" stroke="#22d3ee" stroke-width="1"/>
  <line x1="140" y1="62" x2="140" y2="72" stroke="#22d3ee" stroke-width="1"/>
  <line x1="160" y1="62" x2="160" y2="72" stroke="#22d3ee" stroke-width="1"/>
  <line x1="180" y1="62" x2="180" y2="72" stroke="#22d3ee" stroke-width="1"/>
  <line x1="200" y1="62" x2="200" y2="72" stroke="#22d3ee" stroke-width="1"/>
  <line x1="220" y1="62" x2="220" y2="72" stroke="#22d3ee" stroke-width="1"/>
  <line x1="240" y1="62" x2="240" y2="72" stroke="#22d3ee" stroke-width="1"/>
  <line x1="260" y1="62" x2="260" y2="72" stroke="#22d3ee" stroke-width="1"/>
  <line x1="280" y1="62" x2="280" y2="72" stroke="#22d3ee" stroke-width="1"/>
  <line x1="300" y1="60" x2="300" y2="74" stroke="#22d3ee" stroke-width="1.5"/>
  <!-- End marker - cliff -->
  <rect x="300" y="55" width="280" height="24" rx="4" fill="#ef4444" opacity="0.1" stroke="#ef4444" stroke-width="1" stroke-dasharray="4,3"/>
  <text x="440" y="71" text-anchor="middle" fill="#ef4444" font-size="11" font-family="system-ui">溢出区（> 65504 → inf）</text>
  
  <text x="100" y="90" text-anchor="middle" fill="#64748b" font-size="10" font-family="system-ui">0</text>
  <text x="300" y="90" text-anchor="middle" fill="#64748b" font-size="10" font-family="system-ui">65504</text>
  <text x="200" y="52" text-anchor="middle" fill="#22d3ee" font-size="10" font-family="system-ui">精度高（刻度密）</text>

  <!-- BF16 section -->
  <text x="50" y="150" fill="#34d399" font-size="13" font-family="system-ui" font-weight="bold">BF16</text>
  <line x1="100" y1="147" x2="580" y2="147" stroke="#3a3a4a" stroke-width="1"/>
  <!-- Sparse ticks across full range -->
  <line x1="100" y1="140" x2="100" y2="154" stroke="#34d399" stroke-width="1.5"/>
  <line x1="140" y1="142" x2="140" y2="152" stroke="#34d399" stroke-width="1"/>
  <line x1="180" y1="142" x2="180" y2="152" stroke="#34d399" stroke-width="1"/>
  <line x1="220" y1="142" x2="220" y2="152" stroke="#34d399" stroke-width="1"/>
  <line x1="260" y1="142" x2="260" y2="152" stroke="#34d399" stroke-width="1"/>
  <line x1="300" y1="142" x2="300" y2="152" stroke="#34d399" stroke-width="1"/>
  <line x1="340" y1="142" x2="340" y2="152" stroke="#34d399" stroke-width="1"/>
  <line x1="380" y1="142" x2="380" y2="152" stroke="#34d399" stroke-width="1"/>
  <line x1="420" y1="142" x2="420" y2="152" stroke="#34d399" stroke-width="1"/>
  <line x1="460" y1="142" x2="460" y2="152" stroke="#34d399" stroke-width="1"/>
  <line x1="500" y1="142" x2="500" y2="152" stroke="#34d399" stroke-width="1"/>
  <line x1="540" y1="142" x2="540" y2="152" stroke="#34d399" stroke-width="1"/>
  <line x1="580" y1="140" x2="580" y2="154" stroke="#34d399" stroke-width="1.5"/>
  
  <text x="100" y="170" text-anchor="middle" fill="#64748b" font-size="10" font-family="system-ui">0</text>
  <text x="580" y="170" text-anchor="middle" fill="#64748b" font-size="10" font-family="system-ui">3.4×10³⁸</text>
  <text x="340" y="132" text-anchor="middle" fill="#34d399" font-size="10" font-family="system-ui">精度低（刻度疏）但覆盖全范围</text>

  <!-- Summary -->
  <text x="310" y="205" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">FP16 = 在小范围内看得清楚（高精度 + 窄范围）</text>
  <text x="310" y="225" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">BF16 = 视野覆盖全局但细节模糊（低精度 + 宽范围）</text>
</svg>

## 为什么用 16 位还能快一倍？

到目前为止我们一直在讲精度——但混合精度的初衷是**速度**。为什么位数减半就能快一倍？

### 两个加速来源

**1. 内存带宽翻倍。** GPU 计算的瓶颈往往不是算力，而是数据搬运。FP16 比 FP32 体积小一半，所以同样的内存带宽能搬运双倍数据。对于带宽受限的操作（大部分 element-wise 操作、softmax、LayerNorm 等），这直接意味着约 2 倍加速。

**2. Tensor Core 的乘法吞吐量。** NVIDIA 的 Tensor Core 专为低精度矩阵乘法设计。在 A100 上：
- FP32 矩阵乘法：19.5 TFLOPS
- FP16/BF16 Tensor Core：312 TFLOPS（16 倍！）
- FP8 Tensor Core（H100）：~2× FP16

这不是"快一点"，而是数量级的差异。矩阵乘法恰好是 Transformer 中计算量最大的操作（QKV projection、attention、FFN），用低精度做这些操作能带来巨大的实际加速。

### 显存节省

对于大模型训练，显存往往比算力更稀缺。混合精度的显存收益：
- **激活值（activations）** 占训练显存的大头（尤其长序列时）。全部 FP16 存储 → 减半。
- **梯度** 也用 FP16 存储 → 减半。
- **权重** 需要 FP32 master copy + FP16 工作副本 → 额外 50%（但相比激活值的节省，这是划算的买卖）。

总体来说，混合精度训练通常能节省 30-50% 的显存，让你能训更大的模型或用更大的 batch size。

## FP8：极限压缩的新战场

### 为什么不满足于 16 位？

随着模型越来越大（GPT-4 据传超过万亿参数，DeepSeek-V3 有 671B 参数），即使 BF16 的显存和计算效率也不够了。如果能再压缩一半——从 16 位到 8 位——那就是再次翻倍的加速和节省。

但 8 位只有 256 个可能的值。你需要用 256 个数来表示神经网络中所有可能出现的数值——这听起来近乎不可能。

### 两种 FP8：各有所长

NVIDIA、ARM、Intel 联合定义的 FP8 标准包含两种格式：

**E4M3**（4 位指数 + 3 位尾数）：
- 最大值：448
- 在任意两个相邻 2 的幂之间，只有 8 个可表示的值
- 适合**前向传播**——权重和激活值通常分布集中，精度需求相对温和

**E5M2**（5 位指数 + 2 位尾数）：
- 最大值：57,344（范围比 E4M3 大 100 多倍）
- 在任意两个相邻 2 的幂之间，只有 4 个可表示的值
- 适合**反向传播**——梯度的动态范围大（有些梯度非常大，有些非常小），需要更宽的范围

这种"前向用 E4M3、反向用 E5M2"的分工，是 FP8 训练能工作的关键设计。

### Per-Tensor Scaling：FP8 的"显微镜"

FP8 只有 256 个可表示值，动态范围极其有限。如果一个 tensor 里同时存在 0.001 和 100 这样的值，8 位根本无法同时精确表示两者。

解决方案是**逐 tensor 缩放（per-tensor scaling）**：
1. 计算 tensor 中的绝对值最大元素 $|x|_{max}$
2. 计算缩放因子 $s = \frac{FP8\_{max}}{|x|\_{max}}$（比如 $448 / |x|_{max}$）
3. 把整个 tensor 乘以 $s$ 后量化为 FP8
4. 计算时反向乘以 $1/s$ 恢复

这就像用显微镜观察——先调焦距让目标充满整个视野（最大化利用有限的分辨率），而不是用一个固定倍率看所有东西。

DeepSeek-V3 更进一步，使用了**逐块缩放（per-block scaling）**——把 tensor 切成 32 个元素一块，每块有独立的缩放因子。这样即使同一 tensor 内有数量级差异的值，也能分别保护。这就是 OCP 标准中的 MXFP8（Microscaling FP8）。

### 实际效果

在 NVIDIA H100 GPU 上，FP8 训练相比 BF16：
- 计算速度提升约 1.6-2×
- 显存再减少约 30-40%
- 精度损失在 0.1-0.3% 以内（配合良好的 scaling 策略）

DeepSeek-V3 用 FP8 训练了 14.8 万亿 token 的 671B MoE 模型，总成本仅 557 万美元（2048 块 H800）——很大程度上得益于 FP8 带来的效率提升。

## 实践中怎么选？

### 决策树

如果你在 2025 年训练一个模型，选择通常是这样的：

**训练时：**
- 有 H100/B200？→ 尝试 FP8 训练（Transformer Engine 库支持）
- 有 A100/H100？→ BF16 是最安全的选择（不需要 loss scaling，省心）
- 只有 V100？→ FP16 + dynamic loss scaling（V100 不支持 BF16 Tensor Core）

**推理时：**
- 追求精度？→ FP16（比 BF16 精度高）
- 追求速度 + 显存？→ FP8 或 INT8/INT4 量化

### PyTorch 中的实战代码

```python
# BF16 混合精度（推荐，最简单）
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(input)
    loss = criterion(output, target)
loss.backward()
optimizer.step()
# 不需要 GradScaler！BF16 的范围足够大

# FP16 混合精度（需要 GradScaler）
scaler = torch.cuda.amp.GradScaler()
with torch.autocast(device_type='cuda', dtype=torch.float16):
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()  # 放大梯度
scaler.step(optimizer)          # 先 unscale，再检查 inf，再 step
scaler.update()                 # 动态调整缩放因子
```

注意 BF16 的代码有多简洁——直接用，不需要任何额外机制。这就是为什么它成了 LLM 训练的事实标准。

## 数值精度的陷阱与教训

### Softmax 的 FP16 溢出

还记得我们上一篇讲的 softmax 数值稳定性吗？在 FP16 下这个问题更加尖锐。attention score 在深层网络中容易超过 65504（FP16 的最大值），导致 softmax 输入变成 inf，输出变成 nan，然后整个训练崩溃。

这就是为什么即使在混合精度训练中，softmax 通常保持在 FP32 做。PyTorch 的 `autocast` 有一个白名单和黑名单机制：
- **白名单**（用低精度）：matmul, conv, linear
- **黑名单**（强制 FP32）：softmax, layer_norm, cross_entropy, exp, log
- **灰色地带**：根据输入精度决定

### 累加器精度的隐形损失

即使 Tensor Core 在 FP32 累加器中计算点积，一个微妙的问题依然存在：当矩阵维度非常大（比如 hidden_dim = 12288），数千次 FP16 乘法的舍入误差在 FP32 累加过程中仍会积累。

对于大多数模型这不是问题，但一些对数值极其敏感的操作（比如 T5 中某些大尺寸的 attention + residual 组合）会因此产生 NaN。这也是为什么一些库（如 DeepSpeed）提供了"全 FP32 通信"选项——在分布式训练的 AllReduce 阶段保持 FP32，防止梯度累加的精度丢失。

## 这意味着什么

混合精度训练的故事，本质上是一个**关于信息论的故事**：神经网络训练过程中，不同阶段、不同操作对数值精度的需求是不同的。把精度资源（比特数）按需分配——就像带宽按需分配一样——就能在几乎不损失质量的前提下大幅提升效率。

从 FP32 到 FP16/BF16 到 FP8，再到未来可能的 FP4，每一步压缩都需要新的数值技巧来弥补精度损失：
- FP16 需要 loss scaling + master weights
- BF16 靠大范围绕开了 loss scaling 的需求
- FP8 需要 per-tensor/per-block scaling
- FP4 需要更精细的 2D 量化和 block floating point

这条路还在继续——每一代新 GPU 都在降低支持的最低精度。最终极限在哪里？理论上，只要缩放策略足够精细，即使 1 位（二值网络）都能训练——只不过需要的补偿机制越来越复杂。

混合精度不仅是一个工程优化技巧，更是理解"深度学习到底需要多少精度"这个基础问题的窗口。答案令人惊讶地乐观：远比我们直觉认为的少。

## 下一篇预告

我们讲了训练时的数值格式选择。但训练结束后呢？当你要把一个 70B 参数的模型部署到单张消费级显卡上时，还需要更激进的压缩——这就是**量化（Quantization）**的领域。GPTQ、AWQ、GGUF 这些方法是如何在 4 位甚至 2 位精度下保持模型质量的？它们的数学保证是什么？下一篇见。
