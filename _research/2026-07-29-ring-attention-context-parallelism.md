---
title: "Ring Attention：把一条无限长的序列,切成一个环"
date: 2026-07-29
level: 3
series: "LLM 原理深度解析"
series_order: 55
series_total: 60
tags: [Ring Attention, 上下文并行, 分布式训练, FlashAttention, Striped Attention, 长上下文]
summary: "当一条序列长到连拆开算都装不进一张显卡时,Ring Attention 用一个环形拓扑把计算和通信叠在一起,让上下文长度随显卡数量线性增长,几乎不花额外代价。"
---

> 2023 年底,几个 UC Berkeley 的研究者提出了一个听起来像变魔术的说法:只要给的 GPU 够多,上下文长度可以做到"几乎无限"。这篇文章讲清楚这个魔术是怎么变的——以及它其实一点都不神秘,只是把一个老问题重新摆对了位置。

## 故事从这里开始

假设你有一台显存 80GB 的 GPU,想训练一个隐藏维度 1024 的模型,处理一条 1 亿 token 的超长序列——可能是一部完整的电影转成的帧序列,也可能是一整个代码仓库拼起来的文本。听起来只是"数字大一点"的事,但算一下账你会傻掉:光是保存每一层的输出(激活值),批量为 1 时就需要超过 1000GB 内存。而你手上这张卡撑死给你 80GB。

这不是"模型太大装不下"的问题——模型本身可能只有几十亿参数,权重轻松放得进显存。真正的杀手是**序列长度带来的中间结果**。Transformer 里每一层的自注意力,都要让每个 token 去看序列里所有其他 token,这个"全体互相看一眼"的操作,朴素实现下内存开销是序列长度的平方。序列长 10 倍,内存爆炸 100 倍。

过去几年,FlashAttention 之类的工作已经把"平方级内存"这个坎迈过去了——用分块计算 + 在线归一化的技巧,让注意力的内存开销从平方降到了线性。这已经很了不起,但还不够。因为线性增长这件事本身也扛不住:序列一旦冲到几百万 token,哪怕是线性的内存开销,单张卡也装不下。

于是问题变成了:**我们有很多张卡,但每张卡还是只能装一小段序列。怎么让这些卡合作,算出一个"看起来好像整条序列都在同一张卡上"的完整注意力结果?**

这就是 Ring Attention 要回答的问题。它的作者是 Hao Liu、Matei Zaharia 和 Pieter Abbeel(UC Berkeley),论文题目是《Ring Attention with Blockwise Transformers for Near-Infinite Context》。它给出的答案非常直接:把参与计算的显卡组成一个**环**,让数据像接力赛一样绕着这个环传递,同时让传递的时间"藏"在计算的时间背后,几乎不花钱。

<svg viewBox="0 0 640 220" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow0" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="320" y="24" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui">单卡内存墙:序列越长,激活值内存越爆炸</text>
  <rect x="40" y="60" width="220" height="90" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="150" y="95" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">单张 GPU</text>
  <text x="150" y="118" text-anchor="middle" fill="#a0a0b8" font-size="12" font-family="system-ui">显存 80GB</text>
  <text x="150" y="138" text-anchor="middle" fill="#ff8080" font-size="12" font-family="system-ui">1 亿 token → 需要 1000GB+</text>
  <line x1="260" y1="105" x2="360" y2="105" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow0)"/>
  <text x="310" y="95" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">拆分</text>
  <rect x="360" y="55" width="240" height="40" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="480" y="80" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">GPU 1: token 0~250k</text>
  <rect x="360" y="100" width="240" height="40" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="480" y="125" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">GPU 2: token 250k~500k</text>
  <rect x="360" y="145" width="240" height="40" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5" stroke-dasharray="4,3"/>
  <text x="480" y="170" text-anchor="middle" fill="#a0a0b8" font-size="12" font-family="system-ui">GPU 3, 4 … N</text>
  <text x="320" y="205" text-anchor="middle" fill="#a0a0b8" font-size="12" font-family="system-ui">但每个 token 都要"看到"其他所有 token 才能算注意力——怎么办？</text>
</svg>

## Ring Attention 的核心直觉

### 问题是什么

单卡装不下整条序列,那就分片——每张卡拿一段。但注意力这件事的本质是"每个 query token 要跟序列里所有的 key、value 交互"。如果 GPU 1 只有前 25 万个 token,它算不出这些 token 对第 50 万个 token 的注意力,因为它压根没有那部分数据。

朴素的解法是:每张卡把自己那一段 key/value 广播给所有其他卡,大家凑齐全部数据再各算各的。问题是这样通信量爆炸——N 张卡,每张卡都要收 N-1 份别人的数据,而且这些数据还必须等到齐才能开始算,通信和计算完全串行,谁都别想快。

### 直觉:接力赛而不是广播

Ring Attention 的想法换了个思路:不广播,而是**传递**。想象 N 张卡围成一个圆桌,每张卡面前放着自己的一份 Query(负责算的那一块),以及自己的一份 Key/Value。

计算开始后,每张卡先用自己手里的 Key/Value 给自己的 Query 算一部分注意力结果(不完整,只是"看到了圆桌上离自己最近的那一段")。同时——这是关键——它把自己刚用完的 Key/Value 传给右边的邻居,同时接收左边邻居传来的下一份 Key/Value。等这次数据传递完成时,正好可以拿新收到的 Key/Value 继续往下算。这样转一圈(N 步),每张卡都跟所有其他卡的 Key/Value 打过照面,注意力结果也就攒齐了。

这里最巧妙的地方在于:**传递数据和计算数据是同时发生的**。GPU 在用手头这一份 Key/Value 做矩阵乘法的那几毫秒里,通信线路正忙着把下一份数据送过来。只要计算花的时间比通信花的时间长,通信就完全"隐身"——它不会拖慢整体速度,因为它一直在计算的背后偷偷进行。

论文里打了个很形象的类比:这就像一条流水线上的接力棒,棒子(Key/Value 块)一直在传,但传棒子这个动作本身不占用任何"额外"的时间,因为传棒子的时候前一个人还在跑。

### 技术细节:为什么结果依然是精确的

有一个前提条件必须满足,否则这套接力赛会算错:注意力的分块计算结果,**跟分块顺序无关,只要统计量(比如 softmax 的归一化因子)被正确地累积就行**。这正是 FlashAttention 引入的"在线 softmax"技巧提供的性质——你可以先算一部分 attention score,记住当前的最大值和累积和,等下一块数据来了再更新这两个统计量,数学上完全等价于一次性算完整个 softmax。

也就是说,Ring Attention 并不是在"近似"注意力,它算出来的结果跟单卡上跑一遍完整注意力**一模一样**,只是把计算过程拆成了绕圈的多轮迭代。这一点很重要——它跟很多"稀疏注意力""近似注意力"的方案不是一路,后者是牺牲精度换效率,Ring Attention 换的是通信路径,不动数学。

论文里给出了一个决定"通信是否真的能被计算完全掩盖"的条件。设每张卡的算力是 F(FLOPS)、卡间带宽是 B,块大小(每次处理的 token 数)为 c,隐藏维度为 d。算一个注意力块需要大约 4dc² 次浮点运算,而传输对应的 key/value 块需要 4cd 字节的通信量。要让计算时间盖住通信时间,需要:

$$\frac{4dc^2}{F} \geq \frac{4cd}{B}$$

化简一下就是 **c ≥ F/B**——块大小只要大于"算力除以带宽"这个比值,通信就能被完全隐藏,不增加任何额外耗时。这个不等式翻译成人话就是:**你的显卡算得越快、卡间连线越慢,你就需要一次处理更大的数据块,才能让传输跟得上计算的节奏**。反过来,如果卡间连接是 NVLink 这种超高带宽,那小一点的块也能藏住通信。这也解释了为什么 Ring Attention 在 TPU(网格式互联、带宽极高)上表现格外好——它天然匹配这种环状/网格状的通信拓扑。

<svg viewBox="0 0 600 340" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:500px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="300" y="24" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui">Ring Attention: 环形拓扑</text>
  <circle cx="200" cy="90" r="45" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="200" y="94" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">GPU 0</text>
  <circle cx="400" cy="90" r="45" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="400" y="94" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">GPU 1</text>
  <circle cx="400" cy="230" r="45" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="400" y="234" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">GPU 2</text>
  <circle cx="200" cy="230" r="45" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="200" y="234" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">GPU 3</text>
  <path d="M 245 90 L 355 90" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)" fill="none"/>
  <path d="M 400 135 L 400 185" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)" fill="none"/>
  <path d="M 355 230 L 245 230" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)" fill="none"/>
  <path d="M 200 185 L 200 135" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)" fill="none"/>
  <text x="300" y="80" text-anchor="middle" fill="#a0a0b8" font-size="11" font-family="system-ui">发 KV 块→</text>
  <text x="440" y="165" text-anchor="middle" fill="#a0a0b8" font-size="11" font-family="system-ui">发 KV 块↓</text>
  <text x="300" y="255" text-anchor="middle" fill="#a0a0b8" font-size="11" font-family="system-ui">←发 KV 块</text>
  <text x="160" y="165" text-anchor="middle" fill="#a0a0b8" font-size="11" font-family="system-ui">↑发 KV 块</text>
  <text x="300" y="300" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">每张卡：算自己的 Query × 手头的 KV 块 → 同时收发下一块</text>
  <text x="300" y="322" text-anchor="middle" fill="#a0a0b8" font-size="11" font-family="system-ui">只要"计算时间 ≥ 通信时间"，通信就是免费的</text>
</svg>

## 效果:内存开销降到了什么程度

Ring Attention 论文给出了一张对比表,展示了不同方案下每层激活值的内存开销(单位:字节,b 是批量大小,h 是隐藏维度,n 是头数,s 是序列长度,c 是块大小):

- **朴素 Transformer**:注意力开销 2bns²——跟序列长度的平方成正比,序列翻倍内存翻四倍
- **FlashAttention 类"内存高效注意力"**:降到 2bsh + 4bch——已经是线性了,但还留了一块跟块大小相关的额外项
- **Blockwise Parallel Transformer(把 FFN 也分块算)**:进一步压到 2bsh
- **Ring Attention**:每层激活只需要 6bch——这里的 c 是块大小,是一个**跟总序列长度完全脱钩的常数**

最后这一点是整篇论文真正的杠杆所在:当内存开销跟块大小 c 挂钩、而不是跟总序列长度 s 挂钩的时候,你想要更长的上下文,不需要单卡内存跟着涨,只需要**多加几张卡**——序列长度可以随设备数量线性扩展,单卡负担保持不变。论文里给出的实测数字相当震撼:在 TPUv4-1024 上,Ring Attention 能训练出比之前最好的内存高效方案长 500 倍以上的序列,轻松突破 1 亿 token,而且不做任何近似,不增加额外的通信和计算开销。

这也是后来 Google 的 Gemini 系列、以及 Berkeley 团队自己拿它训练"百万 token 上下文的世界模型"(处理长视频+语言)敢把上下文标到 100 万甚至更长的底层支撑之一。

## 一个隐藏的坑:因果掩码打破了完美的负载均衡

Ring Attention 讲的故事听起来天衣无缝,但发布后不久,MIT 的一组研究者(Brandon 等人)发现了一个微妙但重要的漏洞——论文叫《Striped Attention: Faster Ring Attention for Causal Transformers》。

### 问题是什么

在生成式语言模型里,注意力是"因果"的——每个 token 只能看它之前的 token,不能看之后的。这意味着有大约一半的 Query-Key 交互天生就是无效的(会被掩码盖掉,softmax 之后权重为 0)。在单卡上,FlashAttention 之类的实现会聪明地跳过这些无效交互,直接省掉一半计算量。

但 Ring Attention 沿用环形拓扑时,这个优化悄悄失效了。原因在于:序列被**连续切段**分配到各个 GPU 上——GPU 0 拿最前面一段,GPU N-1 拿最后面一段。当计算进行到某一轮迭代时,持有"早期 token"的那张卡几乎要跟所有传过来的 Key/Value 块做全量计算(因为早期 token 前面几乎没有更早的东西可以掩码掉),而持有"晚期 token"的那张卡,很多时候整块计算结果都会被因果掩码完全盖掉——白算了。

这就是**负载不均衡**:每一轮迭代里,有的卡在做"完全必要"的满负荷计算,有的卡在做"完全没用"的空转计算。而整个环的推进速度取决于最慢的那张卡——也就是说,不管你怎么优化单卡的因果掩码技巧,Ring Attention 的每轮耗时都等于"当作没有掩码、全量计算"时的耗时。相当于你本该省下一半计算量,却因为分配方式的问题一分没省。

### 核心直觉:洗牌,而不是切段

Striped Attention 的修复思路极其简洁:**不要把序列连续地切给每张卡,而是把序列打乱重排,让每张卡拿到分散在整条序列各处、间隔均匀的 token**。

打个比方:原来的分法像是把一本 1600 页的书按顺序撕成 4 份,第 1 张卡拿第 1~400 页,第 2 张卡拿第 401~800 页……这样"最靠前"的那张卡几乎跟所有后面的内容都有因果关系,负担很重;"最靠后"的那张卡则大部分工作都是徒劳。Striped Attention 换了个撕法:每隔 4 页撕一张给同一张卡,第 1 张卡拿第 1、5、9、13…页,第 2 张卡拿第 2、6、10、14…页。这样一来,每张卡手里都混合了"早期"和"晚期"的内容,不管跟哪个 Key/Value 块交互,大约都有一半会被因果掩码挡住、一半是有效的——负载天然就均匀了。

这个重排利用了一个数学性质:自注意力计算对 token 的排列是等变的(permutation equivariant)——只要你把 Query、Key、Value 按同一种方式打乱,算出来的结果经过逆重排之后跟原顺序算出来的完全一样。所以这不是近似,依然是精确注意力,只是内部换了个记账方式。

### 技术细节:实测收益

作者在 8 张 A100 80GB 上跑十亿参数级别的因果语言模型,序列长度 25.6 万 token 时,Striped Attention 相比原始 Ring Attention 拿到了最高 1.45 倍的端到端吞吐提升;在 16 张 TPUv4 芯片、序列长度超过 50 万 token 时,提升幅度到了 1.65 倍。这个数字直观地反映了负载不均衡问题有多严重——修好它几乎相当于把原来"浪费掉"的那部分算力捞回来了一半以上。

<svg viewBox="0 0 620 260" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:600px;margin:24px auto;display:block;">
  <text x="310" y="22" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui">连续切分 vs 交错切分（4 张卡, 16 个 token）</text>
  <text x="150" y="50" text-anchor="middle" fill="#a0a0b8" font-size="12" font-family="system-ui">Ring Attention（连续切分）</text>
  <g font-family="system-ui" font-size="11">
    <rect x="20" y="60" width="80" height="30" rx="6" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.2"/>
    <text x="60" y="80" text-anchor="middle" fill="#ededf0">0 1 2 3</text>
    <rect x="105" y="60" width="80" height="30" rx="6" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.2"/>
    <text x="145" y="80" text-anchor="middle" fill="#ededf0">4 5 6 7</text>
    <rect x="190" y="60" width="80" height="30" rx="6" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.2"/>
    <text x="230" y="80" text-anchor="middle" fill="#ededf0">8 9 10 11</text>
    <rect x="275" y="60" width="90" height="30" rx="6" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.2"/>
    <text x="320" y="80" text-anchor="middle" fill="#ededf0">12 13 14 15</text>
  </g>
  <text x="60" y="105" text-anchor="middle" fill="#ff8080" font-size="10" font-family="system-ui">GPU0: 几乎全有效</text>
  <text x="320" y="105" text-anchor="middle" fill="#6e8eff" font-size="10" font-family="system-ui">GPU3: 大多被掩码</text>
  <text x="150" y="145" text-anchor="middle" fill="#a0a0b8" font-size="12" font-family="system-ui">Striped Attention（交错切分）</text>
  <g font-family="system-ui" font-size="11">
    <rect x="20" y="155" width="80" height="30" rx="6" fill="#1e1e2a" stroke="#34d399" stroke-width="1.2"/>
    <text x="60" y="175" text-anchor="middle" fill="#ededf0">0 4 8 12</text>
    <rect x="105" y="155" width="80" height="30" rx="6" fill="#1e1e2a" stroke="#34d399" stroke-width="1.2"/>
    <text x="145" y="175" text-anchor="middle" fill="#ededf0">1 5 9 13</text>
    <rect x="190" y="155" width="80" height="30" rx="6" fill="#1e1e2a" stroke="#34d399" stroke-width="1.2"/>
    <text x="230" y="175" text-anchor="middle" fill="#ededf0">2 6 10 14</text>
    <rect x="275" y="155" width="90" height="30" rx="6" fill="#1e1e2a" stroke="#34d399" stroke-width="1.2"/>
    <text x="320" y="175" text-anchor="middle" fill="#ededf0">3 7 11 15</text>
  </g>
  <text x="190" y="200" text-anchor="middle" fill="#34d399" font-size="11" font-family="system-ui">每张卡的负载都约 50% 有效 —— 均衡</text>
  <text x="310" y="235" text-anchor="middle" fill="#a0a0b8" font-size="11" font-family="system-ui">A100 上最高 1.45×，TPUv4 上最高 1.65× 端到端提速</text>
</svg>

## Ring Attention 不是唯一答案:跟 DeepSpeed Ulysses 的分工之争

聊到"怎么把长序列拆到多张卡上",还有另一条路线值得放在一起理解——微软的 **DeepSpeed Ulysses**。这两个方案都属于"序列并行"(Sequence Parallelism)/"上下文并行"(Context Parallelism)大类,但切法完全不同,各有取舍。

Ring Attention 的切法是前面讲的:**按 token 切分序列**,每张卡拿一段 token,Key/Value 绕环传递。它的通信模式是"点对点"——每张卡只跟环上的邻居交换数据,这个特性让它在带宽有限、但拓扑规整(比如 TPU 的 mesh 网络)的场景下特别有优势,而且它天然兼容注意力头数很少的场景(比如用了 GQA/MQA 之后头数很少,按头切分空间不大)。

DeepSpeed Ulysses 走的是另一条路:**按注意力头切分**。它先把序列均匀分给各张卡(每张卡拿一段连续 token),但在算注意力之前,先做一次 all-to-all 通信,把数据重新组织成"每张卡拿全部序列长度、但只负责一部分注意力头"的布局。算完之后再做一次 all-to-all 换回来。这样一来,每张卡在算注意力的那一刻,看到的是**完整的序列**,只是头数少了——直接复用单卡上原封不动的 FlashAttention 实现,不需要改造注意力核心逻辑。

两者的权衡很清楚:Ulysses 的实现更简单,几乎不用动模型代码,但受限于"头数"这个资源——头数最多能切成多少份,GPU 并行度就封顶在那里(比如只有 32 个头,最多切 32 份)。Ring Attention 没有这个头数上限,理论上想加多少卡就能加多少卡,序列长度可以无限地线性扩展,但需要真正改造注意力计算的内部循环,工程复杂度更高。后来出现的一些工程方案(比如 Unified Sequence Parallelism, USP)干脆把两者结合起来用,兼顾两边的优点。

## 这意味着什么

回头看整个故事:长上下文这件事,从"塞不进内存"到"塞进内存但很慢"到"塞进内存、不慢、还精确",走的其实是同一条思路的三次迭代——先用分块计算把内存开销从平方降到线性(FlashAttention/BPT),再用环形通信把"分片之后怎么凑齐全局信息"这个问题变成一个可以跟计算完美重叠的流水线(Ring Attention),最后再抠掉那些因为掩码结构被浪费掉的计算量(Striped Attention)。

这三步叠在一起给出的结论很朴素但很有力:上下文长度不再是一个"模型架构的天花板",而变成了一个**你愿意花多少张卡**的工程问题。这也是为什么 2024 年之后,百万级 token 上下文窗口从论文里的实验数字变成了产品里能用的功能——Gemini 1.5 的 100 万 token、后来更长的窗口,背后都离不开这类上下文并行技术在真实训练集群里落地。

当然,"线性扩展"不等于"免费扩展"——你依然需要更多显卡、更快的互联网络,通信-计算重叠的假设一旦在带宽不够、网络拥堵的真实集群里被打破,理论上的零开销就会打折扣。这也是为什么后续还有大量工程工作在研究更精细的负载均衡策略、更适配的通信拓扑,以及如何把上下文并行和张量并行、专家并行组合到同一个训练系统里——这本身已经是一门独立的系统工程学问了。

---

*本文是"LLM 原理深度解析"系列第 55 篇。系列聚焦 Transformer 与大语言模型的核心原理,从注意力机制到训练、推理、对齐的方方面面,用讲故事的方式把公式背后的直觉讲清楚。*
