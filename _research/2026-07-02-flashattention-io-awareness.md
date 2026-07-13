---
title: "FlashAttention 的秘密：真正的瓶颈不是算力，是搬数据"
date: 2026-07-02
level: 3
series: "LLM 原理深度解析"
series_order: 24
series_total: 39
tags: [flashattention, gpu, io-complexity, attention, kernel优化]
summary: "FlashAttention 没有发明新的注意力算法，也没有减少一次浮点运算，却让 Transformer 训练提速 2-4 倍——秘密藏在 GPU 内部两块内存之间的搬运账单里。"
---

> Transformer 里那个 O(N²) 的注意力矩阵，真正让它变慢的从来不是"算得多"，而是"搬得多"。FlashAttention 一个精巧的重排,就把这笔账单砍掉了 90%。

## 故事从这里开始

假设你在一个巨大的仓库里打包快递。仓库中央有一张小小的工作台，只能放下几个箱子；仓库另一头是堆满货物的库房，走一趟很远。你的任务是把几百个箱子按某种规则重新分类打包。

一个笨办法是：每次只搬一个箱子到工作台，处理完，搬回库房，再去搬下一个。你会发现，大部分时间都花在"走路"上，而不是"打包"本身。

聪明的办法是：每次搬一批箱子到工作台，把这一批能做的事情都做完，再一次性搬回库房。走路的次数骤减，虽然你打包的动作总量没变，但整个任务快了很多倍。

这正是 2022 年 Tri Dao 等人在斯坦福提出的 FlashAttention 解决的问题——只不过"仓库"是 GPU 的显存(HBM),"工作台"是 GPU 芯片上那一小块高速缓存(SRAM),而"箱子"是 Transformer 里那张巨大的注意力矩阵。

这篇文章会告诉你:为什么一个不改变数学结果、不减少计算量、甚至还多算了一点点的算法,反而能让 BERT 训练提速 15%,GPT-2 提速 3 倍,还让模型第一次能处理 64K 长度的序列。答案跟"聪明的算法"关系不大,跟"理解硬件"关系很大。

## 第一部分:注意力为什么这么"贵"

### 问题是什么:表面的复杂度陷阱

如果你学过 Transformer,一定见过这张图:序列里每个 token 都要跟其他所有 token 计算一次相关性分数,得到一张 N×N 的矩阵(N 是序列长度)。这就是所谓的"二次复杂度"——序列长度翻倍,计算量翻两倍不止,是翻四倍。

这个二次项一直被当作"计算量太大"的罪魁祸首。过去几年里,大量论文尝试用各种近似方法砍掉这张矩阵——稀疏注意力只算部分位置,低秩近似用小矩阵代替大矩阵,线性注意力干脆换一套数学框架。这些方法在理论上确实减少了浮点运算(FLOPs)的数量。

但一个尴尬的事实是:这些"理论上更快"的方法,实测的运行速度往往并没有明显提升,有时甚至更慢。

这就很奇怪了。如果减少了计算量,为什么没有变快?这说明我们诊断错了病因。

### 直觉:真正的瓶颈藏在哪里

要理解真正的瓶颈,得先搞清楚 GPU 内部的"地理结构"。一块 GPU 上有两种完全不同性质的存储:

**HBM(高带宽内存)** ——这是你在 `nvidia-smi` 里看到的"显存",比如 A100 有 80GB。它容量很大,但离计算核心比较"远",数据搬运有延迟。

**SRAM** ——这是芯片内部紧贴计算单元的小缓存,一块 A100 的每个流处理器(SM)上大约只有 192KB。它容量小得可怜,但速度极快,大约比 HBM 快一个数量级。

一个直观的类比:HBM 像城市另一头的大型仓库,东西多但取货要跑一趟;SRAM 像你办公桌上的抽屉,东西少但伸手就能拿到。

标准的注意力计算是怎么做的?它老老实实按照数学公式的顺序执行:先算 Q 和 K 的乘积得到分数矩阵,把这张 N×N 的矩阵**完整写入** HBM;然后从 HBM 里把它读出来做 softmax,再写回去;最后再读出来跟 V 做乘法,得到最终输出。

问题就出在这里:这张 N×N 矩阵对于长序列来说非常巨大(序列长度 4096 时,单层就要占用约 64MB),而它被反复地写入、读出 HBM——每一次读写都是一次"跑去仓库搬箱子"的往返。真正拖慢速度的,不是乘法加法算得慢,是这些数据在 HBM 和计算核心之间来回搬运花的时间。

用行话说,这个操作是"内存受限"(memory-bound)的,而不是"计算受限"(compute-bound)的。GPU 的计算单元大部分时间在等数据,而不是在算数据。这就好比一个手脚极快的工人,却要花大部分时间在仓库和工作台之间跑腿,他的速度自然被跑腿的时间拖累,不是被打包动作拖累。

而之前那些"减少 FLOPs"的近似注意力方法,恰恰没有解决这个问题——它们减少了计算量,但没有减少内存搬运量,所以速度提升有限,甚至因为算法结构更复杂反而更慢。

FlashAttention 的洞察就是:**如果瓶颈是搬运数据,那就应该优化的对象是"内存访问次数"(IO),而不是"浮点运算次数"(FLOPs)。**

<svg viewBox="0 0 640 260" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrowA" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="320" y="24" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">GPU 内存的两个世界</text>

  <rect x="30" y="50" width="230" height="160" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="145" y="75" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui" font-weight="bold">HBM (显存)</text>
  <text x="145" y="100" text-anchor="middle" fill="#9a9ab0" font-size="11" font-family="system-ui">容量大: 40-80 GB</text>
  <text x="145" y="120" text-anchor="middle" fill="#9a9ab0" font-size="11" font-family="system-ui">带宽: ~2 TB/s</text>
  <text x="145" y="140" text-anchor="middle" fill="#9a9ab0" font-size="11" font-family="system-ui">"仓库"，远但装得多</text>
  <text x="145" y="170" text-anchor="middle" fill="#ff8c6e" font-size="11" font-family="system-ui">N² 矩阵在这里</text>
  <text x="145" y="188" text-anchor="middle" fill="#ff8c6e" font-size="11" font-family="system-ui">被反复读写</text>

  <rect x="380" y="80" width="230" height="100" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="495" y="105" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui" font-weight="bold">SRAM (片上缓存)</text>
  <text x="495" y="128" text-anchor="middle" fill="#9a9ab0" font-size="11" font-family="system-ui">容量小: ~192 KB / SM</text>
  <text x="495" y="148" text-anchor="middle" fill="#9a9ab0" font-size="11" font-family="system-ui">带宽: 极快 (10x+)</text>
  <text x="495" y="168" text-anchor="middle" fill="#9a9ab0" font-size="11" font-family="system-ui">"工作台"，快但装不下多少</text>

  <line x1="260" y1="130" x2="380" y2="130" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrowA)"/>
  <line x1="380" y1="150" x2="260" y2="150" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrowA)"/>
  <text x="320" y="120" text-anchor="middle" fill="#6e8eff" font-size="10" font-family="system-ui">标准注意力: 反复搬运整张矩阵</text>

  <text x="320" y="235" text-anchor="middle" fill="#9a9ab0" font-size="12" font-family="system-ui">瓶颈不是"算得慢"，是"两地之间跑腿太多次"</text>
</svg>

## 第二部分:FlashAttention 的核心技巧

### 问题是什么:如何在小抽屉里做大仓库的事

既然 SRAM 太小,装不下整张 N×N 矩阵,一个自然的想法是:把矩阵切成小块,每次只搬一小块到 SRAM 里处理。这就是"分块"(tiling)。

但这里有个数学上的麻烦。Softmax 需要"归一化"——要先知道一整行所有分数的最大值和总和,才能算出每个位置最终的权重。如果你把一行切成好几小块分别处理,每一块在被处理的那一刻并不知道整行的最大值和总和是多少,算出来的结果就是错的。

这就好比你要给一群人按身高比例分蛋糕,但你被要求一批一批地看人,每看一批就要分一次蛋糕——可你还没见到所有人,怎么知道"比例"该怎么算?

### 直觉:边看边修正的账本

FlashAttention 用的技巧叫"在线 softmax"(online softmax),思路其实很像我们记账时"边收边调整"的做法。

想象你要计算一群数字的加权平均,但这些数字是一批一批送过来的,你不能等所有数字到齐才开始算。你的策略是:每来一批新数字,先看这批里的最大值,和你手头记录的"历史最大值"比较,取较大的那个作为新的最大值;然后用这个新最大值,把你之前算出的"部分总和"和"部分加权结果"都重新按比例缩放一遍,再把新这批的贡献加进去。

这样一步步修正下来,当所有批次都处理完,你手里的结果跟"一次性看到全部数据再算"得到的结果完全一致——没有任何精度损失,只是分批做而已。这就是在线 softmax 的全部秘密:用一个可以增量更新的"最大值 + 累加和"账本,替代"必须先看完整行才能算"的要求。

有了这个数学工具,FlashAttention 就可以放心地把注意力计算分块进行了:

1. 把 Q(查询)、K(键)、V(值)矩阵沿序列长度方向切成一小块一小块。
2. 每次把一小块 Q 和一小块 K、V 一起搬进 SRAM。
3. 在 SRAM 里,把这一小块内该做的矩阵乘法、softmax 的部分统计量更新、输出的部分累加,全部在片上完成。
4. 处理完所有小块的组合后,更新一次输出到 HBM——而不是每算一步就往返一次。

整个过程中,那张巨大的 N×N 分数矩阵**从来没有被完整地写入过 HBM**。它只是以小碎片的形式短暂地存在于 SRAM 里,用完就丢,下一小块进来时覆盖掉。

### 技术细节:数字说话

我们可以精确地算一算这省下了多少搬运量。

标准注意力需要把 QK^T 结果(N×N)写入 HBM,再读出来做 softmax,再写回去,再读出来跟 V 相乘。粗略估计,总的读写量大约是 `4N² + 4Nd` 字节量级(d 是每个注意力头的维度)。当序列长度 N 远大于 d 时,这个式子基本被 N² 项主导——**内存访问量是序列长度的平方**。

FlashAttention 的分块策略下,K、V 的每个小块会被外层循环重复读取多次(因为要跟每一批 Q 配对),但从来不需要写出完整的 N×N 矩阵。经过对块大小的优化推导,可以证明 FlashAttention 的 HBM 访问量是:

`O(N²d² / M)`

其中 M 是 SRAM 的容量。这个公式的意思翻译成人话就是:**内存访问量除以标准注意力的内存访问量,大约等于 `d/M` 这个比例**——SRAM 越大、每个注意力头维度越小,省下的搬运就越多。论文中还证明了,在合理的 SRAM 容量范围内,这个算法达到的内存访问量已经是理论最优,不存在更省的分块方案。

用具体数字感受一下:在 A100 上(SRAM 约 192KB,d=128),序列长度从 1024 涨到 65536 时,标准注意力需要搬运的数据量从约 8MB 涨到约 34GB;而 FlashAttention 只需要搬运约 1MB 到约 4GB——**减少的倍数几乎跟序列长度成正比,序列越长,优势越大。**

这里有个反直觉的地方:如果你去数 FlashAttention 实际执行的浮点运算次数,会发现它比标准注意力**还要多一点**——因为在线 softmax 需要额外的"重新缩放"操作。但这完全不重要,因为决定实际运行时间的不是"算了多少次乘法",而是"等数据搬运花了多少时间"。这正是"算术强度"(arithmetic intensity,即每搬运一字节数据能配上多少次运算)这个概念要说明的东西:当算术强度低于硬件的某个临界点时,GPU 的计算单元在空转等数据,这时候你该优化的是搬运,不是运算;一旦算术强度被拉高过了临界点,才轮到优化运算本身。FlashAttention 做的,正是把注意力计算从"内存受限"这一侧,推过临界点,推到了"计算受限"这一侧——GPU 终于可以把它标称的算力真正用起来,而不是花大部分时间对着仓库大门发呆。

<svg viewBox="0 0 640 300" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrowB" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="320" y="24" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">分块 + 在线 Softmax：一次只搬一小块</text>

  <text x="60" y="55" fill="#9a9ab0" font-size="11" font-family="system-ui">Q 分块 (外循环)</text>
  <rect x="60" y="65" width="60" height="30" rx="6" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="90" y="85" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">Q₁</text>
  <rect x="130" y="65" width="60" height="30" rx="6" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="160" y="85" text-anchor="middle" fill="#9a9ab0" font-size="11" font-family="system-ui">Q₂</text>
  <rect x="200" y="65" width="60" height="30" rx="6" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="230" y="85" text-anchor="middle" fill="#9a9ab0" font-size="11" font-family="system-ui">Q₃</text>

  <text x="60" y="130" fill="#9a9ab0" font-size="11" font-family="system-ui">K,V 分块 (内循环)</text>
  <rect x="60" y="140" width="60" height="30" rx="6" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="90" y="160" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">K₁V₁</text>
  <rect x="130" y="140" width="60" height="30" rx="6" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="160" y="160" text-anchor="middle" fill="#9a9ab0" font-size="11" font-family="system-ui">K₂V₂</text>
  <rect x="200" y="140" width="60" height="30" rx="6" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="230" y="160" text-anchor="middle" fill="#9a9ab0" font-size="11" font-family="system-ui">K₃V₃</text>

  <rect x="380" y="95" width="220" height="90" rx="8" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="490" y="120" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui" font-weight="bold">SRAM 工作台</text>
  <text x="490" y="142" text-anchor="middle" fill="#9a9ab0" font-size="10" font-family="system-ui">Q₁ · K₁V₁ → 局部结果</text>
  <text x="490" y="160" text-anchor="middle" fill="#9a9ab0" font-size="10" font-family="system-ui">累加最大值/总和 → 修正</text>

  <line x1="90" y1="95" x2="440" y2="110" stroke="#6e8eff" stroke-width="1.2" marker-end="url(#arrowB)"/>
  <line x1="90" y1="170" x2="440" y2="150" stroke="#34d399" stroke-width="1.2" marker-end="url(#arrowB)"/>

  <rect x="380" y="215" width="220" height="45" rx="8" fill="#1e1e2a" stroke="#ff8c6e" stroke-width="1.5"/>
  <text x="490" y="242" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">最终输出 O (只写一次到 HBM)</text>
  <line x1="490" y1="185" x2="490" y2="215" stroke="#ff8c6e" stroke-width="1.5" marker-end="url(#arrowB)"/>

  <text x="320" y="290" text-anchor="middle" fill="#9a9ab0" font-size="11" font-family="system-ui">N×N 完整矩阵从未出现在 HBM 中</text>
</svg>

## 第三部分:反向传播的"以算代存"

### 问题是什么:训练时更大的内存黑洞

前面讲的是前向计算——给定输入,算出注意力的输出。但训练神经网络还需要反向传播来算梯度,而反向传播传统上需要用到前向计算时产生的中间结果,比如那张巨大的 softmax 之后的分数矩阵 P。

如果说前向传播时把 N×N 矩阵写进 HBM 已经很浪费了,那反向传播时还要把它**存下来留着用**,内存开销就更加恐怖——对于长序列,这张矩阵可能占用几十甚至上百 GB,根本存不下。

### 直觉:不存,重新算一遍反而更快

FlashAttention 的答案听起来有点反直觉:**干脆不存这张矩阵,反向传播需要用的时候,现场重新算一遍。**

这就像你参加一场需要频繁翻查笔记的考试,笔记本太厚背不动,于是你选择只记住几个关键的"锁"(比如每道题的答案范围),等真正需要细节时,凭这几个关键数字,现场把当时的推导过程重新推一遍——反而比翻找一本厚厚的笔记本更快,因为"随身携带的东西"变得很小很快。

具体来说,前向传播时,FlashAttention 只需要在 HBM 里保留两个很小的统计量(每行的最大值和归一化的累加和),丢弃掉那张巨大的分数矩阵本身。反向传播时,利用这两个小小的统计量和原始的 Q、K、V,在 SRAM 里把当时那一小块的注意力分数**重新计算一遍**,然后立刻用它去算梯度,用完就丢。

多花的这一点计算量,远远比不上因为不用存储 / 读取那张巨大矩阵而省下的搬运时间——这跟前面"多算一点 FLOPs 换来少搬很多数据"是同一个思路的延伸。

## 第四部分:进化的脉络——FA1 到 FA3

FlashAttention 不是一次性完工的作品,而是随着 GPU 硬件的进化持续被重新打磨。搞清楚这条演化线,能帮你理解"IO 感知"这个思路本身有多大的生命力。

**FlashAttention(2022)** 首次证明了 IO 感知的分块 + 在线 softmax + 反向重计算这一套组合,能带来 2-4 倍的实测加速,并把 Path-X(序列长度 16K)、Path-256(序列长度 64K)这类此前根本训不动的超长序列任务,变成了"第一次能跑赢随机猜测"的成绩。但它在 A100 上也只能达到理论峰值算力的 25%-40%——因为算法内部还有很多非矩阵乘法的琐碎操作(比如缩放、边界检查),而 GPU 的张量核心专门为矩阵乘法优化,处理这些"杂活"效率很低。

**FlashAttention-2(2023)** 针对这些瓶颈做了三处外科手术式的改进:一是重新设计在线 softmax 的计算顺序,减少非矩阵乘法运算的比例(因为在 A100 上,矩阵乘法的算力是普通浮点运算的 16 倍,一次"杂活"运算相当于浪费了 16 倍的机会成本);二是当序列很长、批次很小时,额外沿着序列长度维度做并行划分,让 GPU 上上百个流处理器都能被喂饱工作,而不是闲置;三是重新分配每个线程组内部的分工,让 Q、K、V 的切分方式减少线程之间互相等待、互相同步写共享内存的开销。这一套优化把利用率从 25%-40% 拉到了 50%-73%,端到端训练 GPT 类模型时能达到 72% 的模型算力利用率。

**FlashAttention-3(2024)** 瞄准的是 Hopper 架构(H100)带来的全新硬件能力:一种叫 TMA 的专用硬件单元可以异步地在后台搬运数据,不占用计算单元的时间;新一代张量核心 WGMMA 吞吐量更高;同时硬件原生支持 FP8 这种更低精度但速度翻倍的数值格式。FlashAttention-3 的核心思路是让"搬数据"和"做计算"这两件事真正并行发生——一部分线程专门负责提前把下一块数据搬进 SRAM,另一部分线程专心做矩阵乘法和 softmax,两者同时进行,互不等待。这让 H100 上的利用率从 FlashAttention-2 的 35% 提升到了 75%,速度比 FA2 快 1.5-2 倍;配合 FP8 精度,还能达到接近 1.2 PFLOPS 的吞吐量,并且用一种叫"非连贯处理"的技巧把低精度带来的误差降低到普通 FP8 注意力的三分之一左右。

这条演化线传达的信息很清楚:IO 感知不是一次性的技巧,而是一种持续应对硬件变化的方法论——每一代 GPU 带来新的内存结构、新的并行原语、新的数值格式,"如何最大化利用片上高速缓存、最小化跨内存搬运"这个问题就要被重新解一次。

## 这意味着什么

FlashAttention 教给我们一个在深度学习系统工程里经常被忽视的道理:**理论上的算法复杂度(FLOPs)和实际运行时间之间,常常隔着一层硬件的现实。**过去几年里,大量论文追逐"减少浮点运算次数"这个目标去设计近似注意力算法,却在真实 GPU 上跑不出理论应有的加速——因为它们诊断的病灶是错的,真正卡住脖子的是内存带宽,不是算力。

这也解释了为什么 FlashAttention 能够做到"精确"注意力(exact attention)而不是近似——它完全没有改变数学公式,只是改变了执行这些数学运算的**顺序和存储策略**。这提醒我们一个更普遍的经验:很多性能问题的答案不在"换一个更聪明的算法",而在"用现有的算法,更懂硬件地去执行它"。

这套 IO-aware 的思路如今已经渗透进几乎所有主流推理框架和训练库的底层实现,也启发了后续一整批工作——从针对推理场景的 FlashDecoding,到针对稀疏模式的 block-sparse 变体,到跨设备的分布式长上下文注意力。理解了 FlashAttention 的这套逻辑,再去看后续的各种"XX-Attention"优化,你会发现万变不离其宗:先问自己"数据在哪两块内存之间来回跑",再决定要不要动手优化。

---

*本文是"LLM 原理深度解析"系列第 24 篇。系列此前已覆盖 Attention 数学本质、位置编码(RoPE/ALiBi/YaRN)、归一化、优化器、采样策略、State Space Models、GQA/MQA/MLA 等主题,可在博客首页查看完整列表。*
