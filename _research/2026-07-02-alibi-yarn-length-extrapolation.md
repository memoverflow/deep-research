---
title: "训练时只见过 4K，推理时却要处理 100K：ALiBi 和 YaRN 如何让模型「越活越长」"
date: 2026-07-02
level: 3
series: "LLM 原理深度解析"
series_order: 22
series_total: 43
tags: [ALiBi, YaRN, NTK-aware, position-encoding, length-extrapolation, RoPE]
summary: "为什么一个在 4096 长度上训练的模型，一旦输入超过这个长度就开始胡言乱语？ALiBi 用一个近乎粗暴的线性惩罚绕开了这个问题，而 YaRN 则把 RoPE 的频率结构拆开重新校准——这两条路线揭示了「长度外推」背后一个被长期低估的真相：模型不是不会数数，而是从没见过那些数字。"
---

# 训练时只见过 4K，推理时却要处理 100K：ALiBi 和 YaRN 如何让模型「越活越长」

> 如果你曾经把一段很长的文档丢给某个开源模型，看着它读到中途开始满嘴胡话、前言不搭后语——大概率不是模型"变笨"了，而是它撞上了一堵看不见的墙：位置编码墙。这篇文章讲两种绕开这堵墙的思路：ALiBi 的"简单粗暴"和 YaRN 的"精细手术"。

## 故事从这里开始

假设你在教一个学生做加法，但你的训练材料只包含 1 到 1000 以内的数字。学生练得很熟，1+1、500+300 都算得又快又准。有一天你问他 "8888 + 7777 等于多少"，他愣住了——不是因为加法原理变了，而是因为他从来没有在这个数字范围里"感受"过数字的排列规律。

这几乎就是 Transformer 在处理超长序列时发生的事情。

回到 2017 年 Transformer 刚诞生的时候，Vaswani 等人在原始论文里留了一句颇有信心的话：这个架构"也许可以外推到比训练时更长的序列"。这个猜测听起来很合理——毕竟 sinusoidal（正弦）位置编码本身是一个连续的数学函数，任意位置都能算出一个值，理论上没有"越界"这回事。

但现实很快打了脸。2021 年，Ofir Press、Noah Smith 和 Mike Lewis 三人做了一个非常直白的实验：把一个用长度 512 或 1024 训练的语言模型,拿到更长的验证序列上测困惑度（perplexity，越低表示模型对下一个词的预测越准）。结果是灾难性的——不管是 sinusoidal、还是当时被认为更先进的 RoPE（旋转位置编码）、还是 T5 的相对位置偏置，三种方法在超过训练长度之后困惑度都会急剧飙升，模型仿佛突然"失忆"了。

这就是这篇文章要讲的核心问题：**为什么模型在训练时表现完美的位置编码,一旦输入变长就会集体崩溃？又有哪些方法能让模型"越活越长"，在没见过的长度上依然保持理智?**

我们会讲两条几乎完全相反的解决路线：
1. **ALiBi**——干脆不用位置编码这种东西，换成一个几何直觉极其简单的"距离惩罚"。
2. **NTK-aware / YaRN**——保留 RoPE，但重新校准它内部的频率结构，像给一台精密仪器做"频段矫正"。

## 第一部分：为什么"训练时没见过的位置"会让模型崩溃

### 问题是什么

先把"位置编码"这件事讲清楚。Transformer 的自注意力机制本质上是"集合"操作——如果你把输入的词打乱顺序，attention 计算出来的结果结构不会变。这在处理语言时是致命缺陷，因为"猫追狗"和"狗追猫"含义完全不同。所以模型必须有某种方式，让每个 token 知道自己在序列中的第几个位置。

最直接的做法(也是 Transformer 论文原始的做法)：给每个位置分配一个向量，位置 1 用向量 A，位置 2 用向量 B……以此类推,加到词嵌入上。sinusoidal 编码用一堆不同频率的正弦/余弦函数生成这些向量，好处是理论上位置可以无限延伸(正弦函数定义域是全体实数)。

但问题出在训练过程本身。模型在训练时,所有样本都被切成固定长度的片段——比如 1024 个 token。模型的参数在这 1024 个位置对应的向量组合上被反复调整、优化，学会了"位置 500 附近通常长什么样"、"位置 1000 是快到片段结尾了要收尾"这类隐含规律。可是位置 1500、2000 对应的向量,模型在训练中从来没有实际"用力"学习过——它们理论上存在，但从未被梯度下降真正触碰、校准过。

这就跟前面学生做加法的比喻一样：加法规则(位置编码的数学定义)对任意大的数字都成立,但学生的直觉、熟练度只在他真正练习过的范围内可靠。

RoPE 情况类似但更隐蔽。RoPE 不是把位置向量加到词嵌入上,而是把 query 和 key 向量按不同频率"旋转"一个跟位置相关的角度,两个向量的点积就自然编码了它们的相对距离。这个设计非常优雅,理论上也没有"位置上限"。但当你把位置索引推到训练时从未出现过的数值,某些注意力头的分数会突然出现远超训练时正常分布的异常值——就像一台精密仪器被输入了超出量程的信号,针不是慢慢偏,而是直接打表。Meta 团队在 Position Interpolation 论文里专门分析了这个现象,并证明这正是直接外推 RoPE 会失败的根本原因。

### 直觉：核心想法

那怎么解决?历史上出现了两条完全不同的哲学:

**路线一(ALiBi):不给模型任何"绝对位置"的概念,只给"相对距离"的感觉,而且这个感觉设计得极其简单、线性,简单到不管数字多大都不会"失真"。**

**路线二(NTK-aware / YaRN):保留 RoPE 精巧的旋转结构,但意识到问题出在"频率被拉得太开",于是想办法把频率"压缩"回模型熟悉的范围,同时尽量少破坏模型已经学到的东西。**

我们先讲第一条路,因为它更直观。

<svg viewBox="0 0 640 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <rect x="10" y="20" width="600" height="50" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="20" y="50" fill="#ededf0" font-size="13" font-family="system-ui">训练时：模型只在位置 0 ~ 1024 之间被反复校准和优化</text>

  <rect x="10" y="90" width="600" height="50" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="20" y="120" fill="#ededf0" font-size="13" font-family="system-ui">推理时：输入长度 8000，位置 1024~8000 从未被"校准"</text>

  <line x1="310" y1="70" x2="310" y2="90" stroke="#ff6e6e" stroke-width="2" marker-end="url(#arrow1)"/>
  <text x="330" y="83" fill="#ff6e6e" font-size="12" font-family="system-ui">越界 → 困惑度飙升</text>
</svg>

## 第二部分：ALiBi——干脆不学位置，只学"离多远"

### 问题是什么

sinusoidal、RoPE、T5 相对位置偏置,这些方法都需要给每个可能的位置分配一个具体的表示(向量或者旋转角度)。这些表示是模型训练时的产物——本质上是在有限的位置范围内"精心调制"出来的。一旦超出这个范围,调制失效。

Press、Smith 和 Lewis 三人在 2021 年提出一个反直觉的问题：**如果我们压根不精心调制任何东西,只是给注意力分数加一个简单到无法出错的惯性,会怎样?**

### 直觉：核心想法

想象你在一个大礌堂里说话,声音会随着距离自然衰减——这是物理规律,不需要"学习",不管礌堂有多大,规律永远成立(声音强度和距离成反比,不会因为礌堂突然变长就"失灵")。

ALiBi(Attention with Linear Biases)的想法几乎就是这个物理直觉的数学翻版：不添加任何位置嵌入,而是直接在注意力分数上加一个**跟距离成正比的惩罚**。距离越远,惩罚越大,模型天然就会更关注近处的 token。这个惩罚的计算方式是纯粹的减法,不涉及任何"训练时学到的表示"——不管两个 token 相距 10 个位置还是 10000 个位置,公式都同样成立,不会有"超出范围"这种情况。

这就是为什么 ALiBi 能外推：它压根没有"位置的合法范围"这个概念,只有"越远惩罚越大"这个永远成立的规则。

### 技术细节（选读）

具体来说,ALiBi 在做完 query-key 点积之后,加上一个静态、不参与训练的偏置项：

```
softmax(q_i · K^T + m · [-(i-1), ..., -2, -1, 0])
```

翻译回人话：对于第 i 个 query,它跟第 j 个 key 的原始注意力分数(点积)会被减去 `m × (i - j)`——距离越远,减去的值越大,softmax 之后这个位置获得的注意力权重自然就越低。

这里的 `m` 是每个注意力头(head)专属的斜率(slope),训练前就固定死,不参与梯度更新。论文给出的设置方式很讲究：对于 n 个头,斜率是从 `2^(-8/n)` 开始、以同一个值为比值的等比数列。比如 8 个头时,斜率依次是 1/2, 1/4, 1/8, ..., 1/256。

为什么要用等比数列,而不是让每个头的斜率一样?因为不同的头需要负责不同"视野"——斜率大的头(比如 1/2)惩罚衰减得很快,基本只关注紧挨着的几个 token,像个"近视眼";斜率小的头(比如 1/256)惩罚衰减得很慢,能看到很远的地方,像个"望远镜"。多个头组合在一起,模型就同时拥有了从"极近"到"较远"的多尺度感受野,这是 ALiBi 论文里特别强调的"多尺度归纳偏置"。

作者还试过让斜率变成可学习参数,结果反而更差(还拖慢训练速度 3%)——这个反直觉的结果说明,有时候"精心设计但固定不变的规则"比"让模型自己去学"更稳健,尤其是当这个规则本身就足够简单、足够符合物理直觉的时候。

实测效果相当惊艳：一个用长度 1024 训练、带 ALiBi 的 13 亿参数模型,在长度 2048 上测试时,困惑度和一个直接用 2048 长度训练的 sinusoidal 模型打平——但前者训练速度快 11%,内存少用 11%。而且 ALiBi 模型在两倍训练长度附近表现最好,一直到长度 10000(接近训练长度的 10 倍)依然保持不错的性能,而三种传统方法早就崩了。

<svg viewBox="0 0 640 260" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="320" y="20" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">困惑度 vs 输入长度（示意）</text>
  <line x1="60" y1="230" x2="600" y2="230" stroke="#3a3a4a" stroke-width="1.5"/>
  <line x1="60" y1="230" x2="60" y2="40" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="30" y="235" fill="#9a9aae" font-size="11">低</text>
  <text x="30" y="45" fill="#9a9aae" font-size="11">高</text>
  <text x="55" y="250" fill="#9a9aae" font-size="11">训练长度</text>
  <text x="580" y="250" fill="#9a9aae" font-size="11">10x</text>

  <path d="M 60 200 L 200 200 L 300 90 L 400 60 L 500 50 L 600 45" fill="none" stroke="#ff6e6e" stroke-width="2"/>
  <text x="420" y="42" fill="#ff6e6e" font-size="11">sinusoidal / RoPE (崩溃)</text>

  <path d="M 60 205 L 200 205 L 300 195 L 400 200 L 500 205 L 600 210" fill="none" stroke="#6e8eff" stroke-width="2.5"/>
  <text x="420" y="225" fill="#6e8eff" font-size="11">ALiBi (稳定)</text>

  <line x1="200" y1="230" x2="200" y2="40" stroke="#3a3a4a" stroke-width="1" stroke-dasharray="4,4"/>
</svg>

## 第三部分：如果不想放弃 RoPE 怎么办？——NTK-aware 与 YaRN 的精细手术

### 问题是什么

ALiBi 很优雅,但它有一个现实约束：你必须从头用 ALiBi 训练模型。而 2023 年之后,业界几乎所有主流开源大模型(LLaMA、Mistral、Qwen 等)都已经用 RoPE 预训练好了,重新训练成本太高。于是一个更实际的问题浮现出来：**能不能不推倒重来,而是对已经训练好的 RoPE 模型"动手术",让它支持更长的上下文?**

最早的朴素尝试是"位置插值"(Position Interpolation, PI)：把新序列的位置索引统一按比例压缩,让它们仍然落在模型训练时见过的范围内。比方说模型原本训练长度是 2048,现在想支持 8192,就把位置索引都除以 4——位置 8000 在计算时会被当成"位置 2000"来处理。这样位置永远不会"越界",效果确实比直接硬来好很多。但 PI 有个明显的缺点：它把所有维度的旋转频率**一视同仁**地压缩,这就好比把一整支管弦乐队的所有乐器统一调低了音调——听起来是熟悉的旋律范围了,但细节的音色关系被打乱了。PI 因此仍需要相当多的微调(通常 1-10 亿 token 级别)才能恢复效果,而且微调后短序列上的表现反而会稍微变差。

### 直觉：核心想法

这里需要先理解 RoPE 内部的一个关键结构：它不是用"一个"频率来编码位置,而是把隐藏维度切成很多组,每组对应一个不同的旋转频率——有些维度转得很快(高频,编码短距离的细粒度差异,就像钟表的秒针),有些维度转得很慢(低频,编码长距离的整体关系,就像时针)。

Neural Tangent Kernel(NTK)理论里有一个结论：如果输入维度低,又缺乏高频成分,深度网络很难学好高频细节。这给了研究者一个洞见：PI 均匀压缩所有频率,恰恰破坏了那些负责细粒度局部关系的高频维度——这些维度本该继续用它熬练出来的"快节奏"去编码相邻 token 之间的细微差异,不该被强行拖慢。

于是"NTK-aware"缩放(最早由 Reddit 社区一位研究者提出)换了个思路：不去压缩位置索引,而是直接调整 RoPE 公式里的 base(基数,原来通常是 10000),让高频维度基本保持原速(继续负责局部细节的外推),低频维度承担更多压缩(负责长距离关系的插值)。用一句话总结这套思路的口号就是:**"高频外推、低频内插"**。

但这个方法用一个统一公式改 base,粒度还是太粗。"NTK-by-parts"进一步精细化：按每个维度对应的"波长"分组处理——波长远小于训练长度的维度(高频)完全不缩放,保持外推;波长远大于训练长度的维度(低频)完全按比例压缩,进行插值;中间地带用一个平滑的斜坡函数(ramp function)过渡,由两个超参数 α、β 控制过渡区间的起止。这就像给管弦乐队里的每种乐器分别调音,而不是笨拙地把所有音调统一往下压。

YaRN(Yet another RoPE extensioN method,EleutherAI 团队提出)在 NTK-by-parts 基础上,又加了一个额外的观察：**在做完位置频率调整之后,给 attention 的 softmax 加一个"温度"参数,可以进一步压低困惑度。**这有点像调好一件乐器之后,再微调一下整体的音量平衡。EleutherAI 通过实验拟合出一个经验公式:

```
√(1/t) = 0.1 × ln(s) + 1
```

其中 s 是上下文扩展的倍数(比如从 4K 扩到 32K,s=8)。这个公式神奇的地方在于:它对 LLaMA 2 的 7B、13B、70B 三种规模都近似适用,说明它捕捉到了某种跟具体模型大小无关的通用规律,而不是死记硬背某一个模型的"最佳参数"。

### 技术细节（选读）

把上面的逐步演化串起来,用统一记号表示。设 f_q(x_m, m, θ_d) 是把 query 向量在第 d 个频率维度上按角度 m·θ_d 旋转,其中 θ_d = b^(-2d/|D|)。所有改进方法本质上都是把这个公式改成:

```
f'(x_m, m, θ_d) = f(x_m, g(m), h(θ_d))
```

- **Position Interpolation**: g(m) = m/s, h(θ_d) = θ_d (只改位置索引,不改频率)
- **NTK-aware**: 保持 g(m) = m,但改变 base b,间接改变所有 θ_d(粗粒度地统一调整频率)
- **NTK-by-parts / YaRN**: 按波长分组,不同 d 对应的 θ_d 用不同的缩放系数处理——高频维度(d 小,θ_d 大,波长短)几乎不缩放,低频维度(d 大,θ_d 小,波长长)充分缩放,中间用斜坡函数插值

YaRN 还在 attention 分数上引入温度 t:
```
softmax(q_m^T k_n / (t·√|D|))
```
这个调整可以直接"烧"进位置编码的实现里(相当于额外缩放 q、k 向量),不需要修改 attention 计算的其他部分。

还有一个非常实用的推理时技巧叫**Dynamic Scaling**：实际推理场景中(比如自回归生成),序列长度是从 1 逐 token 增长到最大值的。如果缩放因子 s 提前固定死,会出现两个问题——短序列时性能有一个莫名其妙的下降(因为按最大长度设计的缩放对短序列是"过度矫正"),长序列刚超过设计阈值时又会突然崩溃。Dynamic Scaling 的解法是让 s 随当前实际长度 l' 实时计算:

```
s = l'/L  (如果 l'/L > 1，否则 s = 1)
```

EleutherAI 特别指出：**"dynamic NTK" 在完全不做任何微调的情况下效果出奇地好**——这意味着哪怕你手上只有一个原始 RoPE 模型,不改任何权重,光是在推理时套用这个动态缩放公式,就能获得相当可观的长度外推能力。这也是为什么很多开源推理框架(如 vLLM、Hugging Face transformers)都内置了 dynamic NTK scaling 选项。

<svg viewBox="0 0 640 280" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <rect x="20" y="20" width="180" height="55" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="110" y="42" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">Position</text>
  <text x="110" y="60" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">Interpolation</text>

  <rect x="230" y="20" width="180" height="55" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="320" y="42" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">NTK-aware</text>
  <text x="320" y="60" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">(改 base)</text>

  <rect x="440" y="20" width="180" height="55" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="530" y="42" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">NTK-by-parts</text>
  <text x="530" y="60" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">(按波长分组)</text>

  <line x1="200" y1="47" x2="230" y2="47" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow3)"/>
  <line x1="410" y1="47" x2="440" y2="47" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow3)"/>

  <line x1="530" y1="75" x2="530" y2="105" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow3)"/>

  <rect x="330" y="105" width="200" height="65" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="430" y="130" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">+ Attention 温度缩放</text>
  <text x="430" y="150" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">= YaRN</text>

  <line x1="430" y1="170" x2="430" y2="200" stroke="#34d399" stroke-width="1.5" marker-end="url(#arrow3)"/>

  <rect x="280" y="200" width="300" height="60" rx="8" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="430" y="225" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">+ Dynamic Scaling</text>
  <text x="430" y="245" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">（推理时实时调整，无需微调）</text>
</svg>

## 第四部分：ALiBi vs RoPE 系方案——谁才是真正的赢家？

理论上讲两条路都能"外推"，但业界后来的选择很有意思：主流的顶尖开源和商业大模型（LLaMA 系列、GPT 系列、Qwen 等）几乎清一色选择了 RoPE 及其扩展方法，而不是 ALiBi。这背后有几个现实原因。

第一,ALiBi 的"偏向近处"归纳偏置(inductive bias towards recency)虽然带来了外推能力,但也天然限制了模型对远距离信息的利用效率——它假设越远的 token 越不重要,可这对某些任务(比如需要精确检索文档开头某个具体事实)未必成立。RoPE 没有这种强假设,理论上灵活性更高。

第二,产业界的实际约束是：大部分昂贵的预训练已经用 RoPE 完成了,重新用 ALiBi 训练意味着推倒重来,而 YaRN 这类方法只需要极少量的微调(甚至像 dynamic NTK 那样完全不需要微调)就能给现有模型续命,性价比高得多。

第三,后续研究(如 2023 年底的长度外推综述)也提出了一个值得警惕的观察：很多所谓的"外推方法"其实并不是让模型真正学会了处理陌生位置,而是通过插值巧妙地避免让模型看到任何"越界"的输入——本质上是把问题"绕开"而不是"解决"。这提醒我们,评估这些技术时要分清"真正的泛化能力"和"精心设计避免了分布外输入"这两件不同的事。

## 这意味着什么

回头看整篇故事,核心矛盾其实是同一个：模型在训练阶段对某个数值范围产生了"熟练度",而现实需求却经常要求它处理这个范围之外的输入。ALiBi 选择从根源上消灭"数值范围"这个概念,用一个永远成立的物理规律替代;NTK-aware/YaRN 选择尊重已经训练好的模型,像做外科手术一样精细调整它内部的频率结构,让"熟练度"平滑地延伸到更大的范围。

这两条路线没有绝对的胜负,而是分别对应了两种不同的工程现实：从零开始设计新模型时,ALiBi 式的"简单规则胜过复杂学习"值得认真考虑;而在庞大的存量 RoPE 模型基础上做升级时,YaRN 式的"精细手术+极少量微调"几乎是唯一现实的选择。

更深一层看,这场关于"长度外推"的探索也揭示了深度学习里一个常见的模式：很多看似深刻的能力缺陷,追根究底只是"训练分布覆盖不到"这个朴素的问题——模型不是缺乏智能,而是缺乏经验。而聪明的工程解法,往往不是让模型变得更聪明,而是想办法让输入"看起来"依然落在它熟悉的经验范围内。

## 系列小结

这是"LLM 原理深度解析"系列关于位置编码这条线索的一次收尾式深挖——从更早的位置编码通用介绍、到 RoPE 的旋转几何,再到今天这篇关于长度外推的两条路线。如果你从头读到这里,应该已经对"模型如何知道 token 的位置"这件事,有了从直觉到数学再到工程实现的完整理解。这个系列后续还会继续拆解 LLM 训练、推理、对齐中的更多底层原理,敬请期待。
