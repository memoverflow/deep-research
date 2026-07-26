---
title: "原生稀疏注意力：当模型自己决定该看哪里"
date: 2026-07-28
level: 3
series: "LLM 原理深度解析"
series_order: 54
series_total: 54
tags: [attention, sparse-attention, NSA, MoBA, long-context, DeepSeek, transformer]
summary: "从 H2O 的事后剪枝到 DeepSeek 的 NSA、Kimi 的 MoBA，稀疏注意力经历了一次范式转变：不再是训练完再砍，而是让模型在训练时就学会该看哪里。"
---

> 一个模型读 6 万字的书要花掉大部分时间在"回头看"上——而它回头看的方式，最近被 DeepSeek 和 Kimi 各自重新发明了一次。

## 故事从这里开始

假设你在读一本 300 页的推理小说，读到第 250 页的时候，侦探突然说"记得第 12 页那个不起眼的细节吗？"。这时候你会怎么做？

你不会把整本书从头到尾重新读一遍去印证这句话。你的大脑会做一件很聪明的事：先粗略地扫一眼整本书的目录和你大概记得的章节脉络（"哦，凶器是在第一部分提到的"），锁定一个大致范围，然后再回头精读那几页，同时手指还停留在当前这一页，随时能看到刚读过的上下文。

这其实就是人类阅读长文本时天然采用的三层策略：**粗看全局、精读重点、盯紧眼前**。

现在把这套策略搬到 Transformer 里，问题就变得很具体了。标准的 self-attention 机制在生成第 250 页对应的那个词时，会让模型把第 1 页到第 249 页的每一个词都算一遍相关性——不管这个词是不是真的重要。序列长度是 N，这个"回头看"的计算量就是 N²。当 N 从几千涨到几十万，这个平方增长就成了真正意义上的"读不动"。DeepSeek 在他们的论文里给出了一个很扎心的数字：处理 6.4 万 token 长度的上下文做解码时，**attention 计算能吃掉整个模型 70%-80% 的推理延迟**。也就是说，模型大部分时间不是在"思考"，而是在"回头翻书"。

这篇文章要讲的，就是 2025 年两家公司——DeepSeek 和月之暗面（Kimi 的开发商）——针对这个问题给出的两份几乎同时发表、思路却截然不同的答案：**NSA（Native Sparse Attention）** 和 **MoBA（Mixture of Block Attention）**。它们共同的关键词是"稀疏"（不看所有词，只看该看的词），但更关键的突破是"原生"（native）——这个稀疏模式不是训练完之后硬砍上去的，而是模型自己在训练过程中学出来的。

要理解这个"原生"到底有多重要，我们得先看看之前的路走到了哪个死胡同。

<svg viewBox="0 0 700 260" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="350" y="24" text-anchor="middle" fill="#ededf0" font-size="15" font-family="system-ui" font-weight="600">稀疏注意力的三个时代</text>

  <rect x="20" y="55" width="180" height="90" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="110" y="80" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">静态规则稀疏</text>
  <text x="110" y="100" text-anchor="middle" fill="#a8a8b8" font-size="11" font-family="system-ui">Longformer / BigBird</text>
  <text x="110" y="116" text-anchor="middle" fill="#a8a8b8" font-size="11" font-family="system-ui">Sliding Window / Sink</text>
  <text x="110" y="132" text-anchor="middle" fill="#8888aa" font-size="10" font-family="system-ui">预先规定"谁看谁"</text>

  <line x1="200" y1="100" x2="250" y2="100" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>

  <rect x="260" y="55" width="180" height="90" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="350" y="80" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">事后推理剪枝</text>
  <text x="350" y="100" text-anchor="middle" fill="#a8a8b8" font-size="11" font-family="system-ui">H2O / Quest / MInference</text>
  <text x="350" y="116" text-anchor="middle" fill="#a8a8b8" font-size="11" font-family="system-ui">训练完后动态砍 KV</text>
  <text x="350" y="132" text-anchor="middle" fill="#8888aa" font-size="10" font-family="system-ui">偏离预训练轨迹</text>

  <line x1="440" y1="100" x2="490" y2="100" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>

  <rect x="500" y="55" width="180" height="90" rx="8" fill="#1e1e2a" stroke="#4ade80" stroke-width="1.5"/>
  <text x="590" y="80" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">原生可训练稀疏</text>
  <text x="590" y="100" text-anchor="middle" fill="#a8a8b8" font-size="11" font-family="system-ui">NSA / MoBA (2025)</text>
  <text x="590" y="116" text-anchor="middle" fill="#a8a8b8" font-size="11" font-family="system-ui">训练时就学会该看哪里</text>
  <text x="590" y="132" text-anchor="middle" fill="#4ade80" font-size="10" font-family="system-ui">硬件对齐 + 端到端可训练</text>

  <text x="110" y="175" text-anchor="middle" fill="#8888aa" font-size="10" font-family="system-ui">~2020</text>
  <text x="350" y="175" text-anchor="middle" fill="#8888aa" font-size="10" font-family="system-ui">~2023</text>
  <text x="590" y="175" text-anchor="middle" fill="#8888aa" font-size="10" font-family="system-ui">2025</text>
  <line x1="30" y1="190" x2="670" y2="190" stroke="#3a3a4a" stroke-width="1"/>
</svg>

## 第一站：为什么"先训练再砍"这条路走不通

在 NSA 和 MoBA 出现之前，业界解决长上下文效率问题的主流思路其实很朴素：**先按老办法训练一个全注意力（Full Attention）模型，等模型训好了，再想办法在推理的时候少算一点**。这类方法统称为"推理阶段稀疏"（inference-time sparsity），H2O、Quest、MInference 都是这条路线上的代表作。

这个思路听起来很合理——毕竟你不用改训练流程，只要在部署阶段做点手脚就能省钱省时间。但 DeepSeek 团队在 NSA 论文里花了很大篇幅论证，这条路存在两个几乎是结构性的坑。

### 坑一：理论上稀疏了，实际上没提速——"效率的幻觉"

第一个坑很反直觉：你以为算得少了就该跑得快，但很多时候并没有。

原因之一叫"阶段局限性稀疏"（phase-restricted sparsity）。像 H2O 这类方法，只在自回归解码（一个字一个字往外蹦的那个阶段）的时候做稀疏，但在预填充（prefilling，也就是把输入的长 prompt 一次性喂进模型算一遍）阶段，它反而需要先老老实实把完整的 attention map 算出来，才能判断出哪些 token 是"重磅选手"（Heavy Hitter）值得保留。而 MInference 恰恰反过来，只优化预填充阶段。结果就是：不管你选哪个方法,总有一个阶段的成本跟全注意力一样高。如果你的任务恰好是"预填充为主"（比如长文档摘要）或者"解码为主"（比如长链条推理），你总能踩中那个没被优化到的阶段。

原因之二更微妙，跟现代模型普遍采用的 GQA（Grouped-Query Attention，分组查询注意力）架构有关。GQA 的设计初衷是让多个 query head 共享同一份 key/value，以减少显存搬运。但像 Quest 这样的稀疏方法，是**每个 attention head 各自独立**去挑选自己认为重要的 KV 子集的。问题是，在 GQA 架构下，同一个组里所有 head 最终要读取的 KV，是所有 head 各自选择结果的**并集**——你这个 head 觉得第 3、5、7 块重要，那个 head 觉得第 2、5、9 块重要，最后你们组要读的仍然是 {2,3,5,7,9} 这五块，跟没稀疏化省不了多少。理论上算力（FLOPs）确实降低了，但真正决定推理速度的内存搬运量（memory access）却没怎么变——而这恰恰是自回归解码阶段真正的瓶颈所在。

### 坑二：训练完再砍，模型会"学坏"——可训练性的迷思

第二个坑更根本：**如果一个模型是在全注意力下训练出来的，它内部形成的信息流动方式已经"适应"了能看到所有 token 这件事**。事后再把某些连接砍掉，等于强迫模型偏离它当初被优化出来的那条轨迹。

DeepSeek 论文里引用了一个很有说服力的实证结果：**只保留 attention 分数最高的 20% 的连接，只能覆盖总注意力质量（attention mass）的 70%**。换句话说，剩下的 30% 的"注意力预算"，分散在你以为不重要、但其实是模型运转所必需的那 80% 连接里。而更麻烦的是，模型内部存在一种被称为"检索头"（retrieval heads）的特殊结构——这些注意力头专门负责从长文本里精确定位并"拷贝"出某个具体细节（就像我们开头例子里侦探提到的"第 12 页的细节"）。这些检索头往往依赖一些看起来不起眼、稀疏但关键的远距离连接，一旦被剪枝算法误判为"不重要"而删掉,模型的检索能力就会毫无预警地垮掉。

还有一类问题出在"能不能训练"这个层面上。有些稀疏方法内部用了 k-means 聚类（比如 ClusterKV）或者 SimHash 哈希（比如 MagicPIG）这类**离散、不可求导**的操作来决定要保留哪些 token。这些操作在计算图里制造了断点，梯度没法从"选择了哪些 token"这一步往回传——这意味着这类方法从设计上就没法参与端到端训练，模型永远学不会怎么更好地做这个选择。

把这两个坑连起来看,你会发现一个共同的病根：**这些方法都是把"稀疏"当成推理阶段的补丁,而不是训练阶段的原生能力**。这正是 NSA 这个名字里"Native"（原生）二字的份量所在——DeepSeek 和 Kimi 几乎同时得出了同一个结论：**稀疏模式必须是训练出来的,而不是事后砍出来的**。

## 第二站：DeepSeek 的答案——NSA 的三双眼睛

NSA 的核心直觉,回到我们开头那个读侦探小说的类比会格外清楚：一个善于处理长文档的读者，脑子里同时运转着三套不同粒度的注意力机制。

### 直觉：三个分支各管一段

想象你在读一份 300 页的合同,律师助理需要在其中找出对客户不利的条款。他会怎么工作?

- **第一套机制**是"翻目录"：先把全文按章节压缩成摘要式的印象——"第一部分讲付款条件、第二部分讲违约责任、第三部分讲仲裁条款"——用这个粗略但覆盖全局的印象快速判断"我大概该重点看哪一段"。
- **第二套机制**是"精读重点段"：一旦锁定了可能有问题的那几段（比如"违约责任"那部分），就回去把这几段的每一个字都仔仔细细读一遍，不能有遗漏。
- **第三套机制**是"盯紧当前句"：不管前面锁定了哪些重点,你手指划到哪一句，那一句以及紧邻的上下文你都会本能地、无条件地看一眼——这是最基础的局部连贯性,不需要"决定"要不要看。

NSA 就是把这三套机制原原本本地做成了三条并行的计算分支：**压缩（Compression）、选择（Selection）、滑动窗口（Sliding Window）**。每个 query token 在算 attention 的时候,同时问这三条分支要答案,再用一个学出来的门控权重把三份答案加权融合。

<svg viewBox="0 0 720 380" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:720px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="360" y="24" text-anchor="middle" fill="#ededf0" font-size="15" font-family="system-ui" font-weight="600">NSA：三条并行分支 + 门控融合</text>

  <rect x="290" y="45" width="140" height="40" rx="8" fill="#1e1e2a" stroke="#4ade80" stroke-width="1.5"/>
  <text x="360" y="70" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">当前 Query qₜ</text>

  <line x1="330" y1="85" x2="140" y2="130" stroke="#6e8eff" stroke-width="1.2" marker-end="url(#arrow2)"/>
  <line x1="360" y1="85" x2="360" y2="130" stroke="#6e8eff" stroke-width="1.2" marker-end="url(#arrow2)"/>
  <line x1="390" y1="85" x2="580" y2="130" stroke="#6e8eff" stroke-width="1.2" marker-end="url(#arrow2)"/>

  <rect x="30" y="135" width="210" height="100" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="135" y="158" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">① 压缩分支</text>
  <text x="135" y="176" text-anchor="middle" fill="#a8a8b8" font-size="10" font-family="system-ui">整块 KV 聚合成一个</text>
  <text x="135" y="190" text-anchor="middle" fill="#a8a8b8" font-size="10" font-family="system-ui">压缩表示（粗看全局）</text>
  <text x="135" y="212" text-anchor="middle" fill="#8888aa" font-size="10" font-family="system-ui">block 长度 l=32</text>

  <rect x="255" y="135" width="210" height="100" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="360" y="158" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">② 选择分支</text>
  <text x="360" y="176" text-anchor="middle" fill="#a8a8b8" font-size="10" font-family="system-ui">复用①的分数选出</text>
  <text x="360" y="190" text-anchor="middle" fill="#a8a8b8" font-size="10" font-family="system-ui">Top-n 重要 block 精读</text>
  <text x="360" y="212" text-anchor="middle" fill="#8888aa" font-size="10" font-family="system-ui">n=16, 无额外算分开销</text>

  <rect x="480" y="135" width="210" height="100" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="585" y="158" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">③ 滑动窗口分支</text>
  <text x="585" y="176" text-anchor="middle" fill="#a8a8b8" font-size="10" font-family="system-ui">固定看最近 w 个 token</text>
  <text x="585" y="190" text-anchor="middle" fill="#a8a8b8" font-size="10" font-family="system-ui">（局部连贯性,不用选）</text>
  <text x="585" y="212" text-anchor="middle" fill="#8888aa" font-size="10" font-family="system-ui">w=512, 独立 KV 防抢跑</text>

  <line x1="135" y1="235" x2="330" y2="290" stroke="#6e8eff" stroke-width="1.2" marker-end="url(#arrow2)"/>
  <line x1="360" y1="235" x2="360" y2="290" stroke="#6e8eff" stroke-width="1.2" marker-end="url(#arrow2)"/>
  <line x1="585" y1="235" x2="390" y2="290" stroke="#6e8eff" stroke-width="1.2" marker-end="url(#arrow2)"/>

  <rect x="270" y="295" width="180" height="55" rx="8" fill="#1e1e2a" stroke="#facc15" stroke-width="1.5"/>
  <text x="360" y="318" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">门控融合 gᶜ</text>
  <text x="360" y="336" text-anchor="middle" fill="#a8a8b8" font-size="10" font-family="system-ui">MLP + sigmoid 学出权重</text>

  <line x1="360" y1="350" x2="360" y2="368" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <text x="360" y="378" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">输出 oₜ</text>
</svg>

### 技术细节:公式背后到底在算什么

先说压缩分支。它把连续的一段 key（比方说 32 个 token 一组）通过一个带位置信息的小型 MLP 压成一个"代表"：

$$\tilde{K}^{cmp}_t = \{\varphi(k_{id+1:id+l}) \mid 0 \le i \le \lfloor \frac{t-l}{d} \rfloor\}$$

**翻译回人话**：把一整块 32 个 key 向量,喂给一个小神经网络,吐出一个能代表这一整块"大意"的向量。这就跟我们把一章合同压缩成一句摘要一样。这个操作只在训练时执行一次,压缩出来的表示是可学习、可求导的——这正是"原生可训练"的第一处体现。

选择分支最巧妙的地方在于它**几乎不花额外成本**。既然压缩分支已经算出了 query 和每个压缩 block 之间的相关性分数（这本来就是它计算流程里的中间产物）：

$$p^{cmp}_t = \text{Softmax}(q_t^T \tilde{K}^{cmp}_t)$$

那这份分数就可以直接拿来当作"这个 block 到底重不重要"的排行榜依据,不需要再单独跑一遍额外的重要性打分。**翻译回人话**：律师助理翻目录时对每一段留下的印象分,直接拿来当"这段值不值得精读"的排序标准,不用另起炉灰重新评估一次。选出 Top-n 个 block 之后,这些 block 里原始、未压缩的 key/value 才会被真正取出来做精细的 attention 计算。

这里还有一个容易被忽略但极其重要的工程细节：**在 GQA 架构下,同一组里所有 query head 的重要性分数会先求和、再统一决定选哪些 block**：

$$p^{slc'}_t = \sum_{h=1}^{H} p^{slc,(h)}_t$$

**翻译回人话**：回想前面说的那个坑——如果每个 head 各自挑自己的 block,同一组内不同 head 挑的 block 不一样,最后要读的 KV 反而是所有人挑的并集,内存搬运量没降下来。NSA 直接把这个问题从设计上解决了：同一组内所有 head 先"开会"统一意见,再一起去读同一批 block。这样内存访问才能真正跟着算力一起降下来——这就是所谓"硬件对齐"(hardware-aligned)的含义所在,不是空谈优化,而是从算法设计阶段就替 GPU 的内存搬运模式考虑周全。

最后是滑动窗口分支——它最简单,但设计初衷值得说一说。为什么要单独开一条分支来处理"最近的几百个 token",而不是让压缩/选择分支自己顺便覆盖到? 论文给出的理由很有意思：**局部模式学得太快,会"抢跑"**。如果不隔离,模型很容易走捷径,只靠"看最近几个词"就能应付大部分预测任务,压缩和选择这两条本该学习远程依赖的分支反而得不到足够的梯度信号去学习该学的东西。给局部模式单独开一条路,相当于给它一个"专用通道",让它不去抢占理应属于远程理解能力的学习资源。

三条分支跑完各自的 attention,输出通过一个 MLP+sigmoid 学出的门控分数加权求和,就是 NSA 的最终输出。

### 结果说了什么

DeepSeek 用一个 27B 参数(3B 激活参数,结合了 GQA 和 MoE)的模型跑了 270B token 的完整预训练做对比。结果是:在 MMLU、GSM8K、MATH、HumanEval 这些常规评测上,NSA(平均分 0.456)略微超过 Full Attention(0.443);在专门测长文本理解能力的 LongBench 上,NSA 的表现同样能打平甚至超过全注意力,并且远远甩开 H2O、InfLLM、Quest 这些"事后剪枝"方法一个身位。效率方面,64k 长度序列下,解码阶段加速 11.6 倍,前向传播加速 9 倍,反向传播加速 6 倍——这意味着 NSA 不仅推理快,连**训练本身**也变快了,因为稀疏结构在反向传播时同样生效。

这一点值得多说一句:很多稀疏方法只优化推理,但 NSA 因为是"原生"训练出来的,训练阶段本身就省了计算量——这正是"原生可训练"这四个字带来的额外红利,不只是推理省钱,连训练这个更贵的阶段也一起省了。

## 第三站:Kimi 的答案——把 MoE 的哲学搬进 Attention

如果说 NSA 走的是"精心设计三种互补的注意力形态"这条路,月之暗面的 MoBA 走的是完全相反的一条路——**尽量少设计,让模型自己决定该看哪里**。

### 直觉:开一个"专家评审团"来决定该看哪一段

MoBA 论文里反复强调一个原则,叫"少结构"(less structure)。什么意思?我们可以换个类比来理解:如果你要审一篇很长的论文,你可以规定审稿人必须"先看摘要,再看最后一段,再看你手指划到的当前段"(这就是滑动窗口 + attention sink 这类静态规则),这种规定简单粗暴,但万一这篇论文的关键漏洞恰好藏在中间某一段,而不是开头结尾,你就会错过。

MoBA 的想法是:干脆不预设规则,而是像 Mixture-of-Experts(混合专家模型)里那样,搞一个"评审团投票"机制——把整篇论文切成若干段(block),对每一处需要判断的地方(每个 query token),都让一个门控网络去给每一段打个相关性分数,然后选出分数最高的那几段去精读。至于具体该看哪几段,不再是人为规定,而是**模型自己在训练中学会怎么打分、怎么选**。

这正是"把 MoE 的思路搬进 attention"这句话的准确含义——MoE 通常用在 FFN 层,让每个 token 只激活一部分"专家"神经元;MoBA 把同样的"选择性激活"思想用在了 attention 上,让每个 query 只激活一部分历史 KV block。

<svg viewBox="0 0 680 300" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:680px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="340" y="24" text-anchor="middle" fill="#ededf0" font-size="15" font-family="system-ui" font-weight="600">MoBA：无参数门控 Top-k 选择 block</text>

  <rect x="30" y="55" width="620" height="50" rx="6" fill="none" stroke="#3a3a4a" stroke-width="1"/>
  <rect x="35" y="60" width="140" height="40" rx="6" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.2"/>
  <text x="105" y="85" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">Block 1</text>
  <rect x="185" y="60" width="140" height="40" rx="6" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.2"/>
  <text x="255" y="85" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">Block 2</text>
  <rect x="335" y="60" width="140" height="40" rx="6" fill="#1e1e2a" stroke="#facc15" stroke-width="1.5"/>
  <text x="405" y="85" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">Block 3(当前块,必选)</text>
  <rect x="485" y="60" width="140" height="40" rx="6" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.2" stroke-dasharray="4,2"/>
  <text x="555" y="85" text-anchor="middle" fill="#666680" font-size="11" font-family="system-ui">Block 4(未来,禁止)</text>

  <text x="120" y="130" text-anchor="middle" fill="#8888aa" font-size="10" font-family="system-ui">mean-pool → 打分</text>
  <text x="255" y="130" text-anchor="middle" fill="#8888aa" font-size="10" font-family="system-ui">mean-pool → 打分</text>

  <line x1="105" y1="100" x2="230" y2="180" stroke="#4ade80" stroke-width="1.5" marker-end="url(#arrow3)"/>
  <text x="150" y="150" fill="#4ade80" font-size="10" font-family="system-ui">s₁ 高→选中</text>
  <line x1="255" y1="100" x2="280" y2="180" stroke="#666680" stroke-width="1.2" stroke-dasharray="3,2"/>
  <text x="300" y="150" fill="#666680" font-size="10" font-family="system-ui">s₂ 低→跳过</text>

  <rect x="180" y="190" width="280" height="60" rx="8" fill="#1e1e2a" stroke="#4ade80" stroke-width="1.5"/>
  <text x="320" y="215" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">当前 Query q</text>
  <text x="320" y="233" text-anchor="middle" fill="#a8a8b8" font-size="10" font-family="system-ui">只精读 Block 1 + Block 3(当前块)</text>

  <text x="340" y="278" text-anchor="middle" fill="#8888aa" font-size="10" font-family="system-ui">门控函数无可学习参数：s_i = ⟨q, mean_pool(K[I_i])⟩</text>
</svg>

### 技术细节:公式怎么读

MoBA 的核心公式其实非常克制:

$$\text{MoBA}(q,K,V) = \text{Softmax}(qK[I]^\top)V[I]$$

**翻译回人话**:跟标准 attention 长得一模一样,唯一区别是 K、V 不再是全部历史,而是被 I 这个索引集合筛过一遍的子集。所有的巧思都藏在 I 是怎么被决定的这个问题上。

决定 I 的门控分数是这样算的:

$$s_i = \langle q, \text{mean\_pool}(K[I_i]) \rangle$$

**翻译回人话**:把第 i 个 block 里所有的 key 向量取个平均,当作这个 block 的"代表",然后看当前 query 跟这个"代表"的点积有多大——点积大说明这个 block 跟当前问题相关性高。**这里最值得注意的一点是:这个打分函数没有任何可学习的权重参数**,它就是纯粹的均值池化 + 点积。这也是"parameter-less gating"(无参数门控)这个名字的来源——跟 MoE 里通常需要专门训练一个路由网络(router)不同,MoBA 的路由决策直接从已有的 K 向量里现算出来,不需要额外增加参数、额外训练成本。

选出 top-k 个分数最高的 block 之后:

$$g_i = \begin{cases} 1 & s_i \in \text{Topk}(\{s_j \mid j \in [n]\}, k) \\ 0 & \text{otherwise} \end{cases}$$

**翻译回人话**:分数进前 k 名的 block 就整块选中(gate=1),没进前 k 名的整块跳过(gate=0)——这是个硬性的、二元的选择,不是软性加权。

因果性的处理是这篇论文里工程细节做得最扎实的部分。语言模型是自回归的,不能让当前 token "看到"未来的信息,这条铁律必须严格遵守。MoBA 用两条规则守住了这条线:第一,直接禁止路由到任何完全处于未来的 block(把这些 block 的分数直接设为负无穷,gate 强制为 0);第二,更微妙的是"当前块"本身——因为当前块的 mean pooling 是对整块(包括当前 token 之后的部分)求平均,这本身就可能泄漏未来信息,所以 MoBA 强制规定当前块必须被选中(不管打分结果如何),同时在块内部再叠加一层标准的因果 mask,把块内部真正未来的那几个 token 的注意力权重清零。作者还打了个很精准的比方:这种"当前块必选"的规则,功能上跟现代 MoE 架构里的"共享专家"(shared experts)一样——共享专家是无条件参与计算的专家,当前块则是无条件参与计算的 KV 段。

### 一个意外的理论洞察:MoBA 是"更通用的祖先"

MoBA 论文里有一段论证,读起来有点像是"一统江湖"的宣言,但确实经得起推敲:**滑动窗口注意力和 attention sink,都只是 MoBA 的特例**。

怎么理解这句话?滑动窗口注意力可以看作是把 MoBA 的门控网络"钉死"成一个恒定规则——不管 query 是什么,门控永远只选"最近的那几个 block"。attention sink 则是把门控钉死成另一个恒定规则——永远选"最开头的 block + 最近的 block"。而 MoBA 的门控网络是**动态的、依赖内容的**,它可以在训练中学出比这些固定规则更灵活的策略,理论表达能力自然是这两种静态方案的超集。

这个论证也顺带回答了一个问题:MoBA 到底比 Longformer/Mistral 的滑动窗口方案强在哪?答案是:表达能力上的降维打击——静态方案只是动态方案在参数空间里的一个特殊取值点,而模型几乎不可能自己"恰好"找到那个最优的静态取值点,除非你从设计上就允许它去探索更广阔的选择空间。

### MoBA 的另一个杀手特性:平滑切换

因为 MoBA 不引入任何额外或减少的参数——它跟全注意力共享完全一样的参数量——这带来一个很实用的能力:**训练过程中,每一层可以随时在"全注意力模式"和"MoBA 模式"之间切换**,不需要重新初始化任何东西。这为渐进式的训练策略、混合精度架构的调试都提供了很大的灵活性,也是它已经能"无缝部署"进 Kimi 生产系统支持长上下文请求的重要原因之一。

## 两条路线,一个共同的答案

把 NSA 和 MoBA 放在一起看,会发现它们表面上长得很不一样——NSA 是三条精心设计的互补分支加门控融合,MoBA 是单一的、极简的 top-k block 路由——但它们在回答同一个更深的问题时给出了几乎一致的答案:

**稀疏,不能是训练完之后强加上去的规则,必须是训练过程本身塑造出来的能力。**

这句话背后有两层含义值得拆开来看。

第一层是**可学习性**。无论是 NSA 里那个学出 block 重要性的压缩表示,还是 MoBA 里那个内容驱动的路由打分,它们都参与了梯度反向传播——模型在训练时会亲身体验到"选错了 block 会导致 loss 变高"这个反馈,进而调整自己内部的表示方式,让稀疏选择本身变得更准。这跟"训练完之后拿一个固定规则去砍"是本质不同的两件事:前者是模型自己学出来的判断力,后者是外部强加的手术刀。

第二层是**硬件对齐**。这一点在 NSA 论文里体现得尤其明显——不管算法多精妙,如果最终的内存访问模式是碎片化的、不连续的,那所谓的"理论加速比"在真实 GPU 上就是一句空话。NSA 的 GQA 感知重要性共享、blockwise 而非 token-wise 的选择粒度,都是在回答一个很朴素的工程问题:"这套算法在 A100 上到底能不能跑出对应的实际加速?"而不只是在纸面上把 FLOPs 算得很好看。

顺着这个思路往下看会发现,长上下文这个领域正在经历一次静悄悄的路线收敛:早期百花齐放的静态规则(Longformer 的局部窗口、BigBird 的随机+局部+全局)证明了稀疏性确实存在且可利用;中期涌现的事后剪枝方法(H2O、Quest、MInference)证明了"完全不用改训练流程"这条捷径其实处处是坑;而 NSA 和 MoBA 代表的这一代方法,则共同指向了唯一靠得住的方向——把稀疏结构变成模型训练目标的一部分,让模型自己在海量数据里学会"什么时候该翻目录、什么时候该精读、什么时候该盯紧眼前这一句"。

## 这意味着什么

回到我们开头那本 300 页的推理小说。以前的做法是训练一个"眼里没有优先级,每一页都要平等对待"的读者,然后再想办法在他读完之后砍掉一些他没用上的记忆。NSA 和 MoBA 教给我们的东西是:**不如从一开始就训练一个懂得分配注意力的读者**——他自己会判断什么时候该粗看目录,什么时候该精读细节,什么时候该盯紧当前这一行。这个判断力不是被规定出来的,而是他在无数次阅读练习里,靠着"读错了会被纠正"这个反馈信号,慢慢磨出来的直觉。

这也解释了为什么这两篇论文都不满足于"跑得更快"这一个结果,而是反复强调"性能不降甚至略升"。因为如果稀疏性只是靠硬砍换来的速度,那多多少少是拿准确性去交换效率;而如果稀疏性是训练出来的能力,那模型完全可能因为"不需要在无关信息上分心"而学得更专注、更准——这正是 NSA 论文里 27B 模型在通用benchmark和长文本benchmark上双双小幅超过全注意力基线的原因所在。稀疏,从一个不得已的性能妥协,变成了一种值得追求的架构特性。

这条路走到今天还远没有终点。国内的 InfLLM v2(MiniCPM4 采用)延续了同样的"原生可训练稀疏"思路,并进一步把训练成本压到了 5B token 量级就能获得稀疏注意力能力;学界也在探索把 NSA 的三分支设计和 MoBA 的无参数路由结合起来的混合方案。可以确定的是,"事后砍枝"这条老路基本已经被判定为死胡同,而"训练时就学会该看哪里"正在成为长上下文 LLM 架构设计的新常识。

---

*参考资料*
- Yuan et al. (DeepSeek-AI), *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention*, arXiv:2502.11089
- Lu et al. (Moonshot AI), *MoBA: Mixture of Block Attention for Long-Context LLMs*, arXiv:2502.13189
- Zhang et al., *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs*, arXiv:2306.14048
- Beltagy et al., *Longformer: The Long-Document Transformer*, arXiv:2004.05150
- Jiang et al. (Mistral AI), *Mistral 7B*, arXiv:2310.06825
