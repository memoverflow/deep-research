---
title: "Mixture of Depths：条件计算与「按需思考」的 Transformer"
date: 2026-07-09
level: 3
series: "LLM 原理深度解析"
series_order: 30
series_total: 30
tags: [transformer, conditional-computation, mixture-of-depths, moe, efficiency, routing]
summary: "为什么每个 token 都要经过 Transformer 的每一层？Mixture of Depths 用一个巧妙的路由机制告诉我们：其实没必要，模型可以自己学会「该省力气的地方省力气」。"
---

> 你有没有想过，读一句话的时候，"的"、"了"、"是"这些字你几乎不用思考，但读到一个陌生的专业术语或者一个反转剧情的关键词时，你会突然放慢速度，多想一会儿？大脑天生就懂得"分配注意力"——可是我们训练出来的 Transformer，长期以来一直不懂这个道理。

## 故事从这里开始

想象一家工厂的流水线。传送带上跑着一批一批的零件，每个零件都要经过完全相同的 20 道工序，无论它是一颗简单的螺丝，还是一个复杂的精密齿轮。工厂老板从没问过一个问题：螺丝真的需要 20 道工序吗？

这正是所有主流 Transformer 语言模型的现状。一句话被切成一串 token 送进模型，每个 token——不管是无关紧要的"的"字，还是承载整句话核心信息的关键词——都会被推着走过模型的每一层，每一层都要做一次完整的自注意力计算和一次完整的 MLP 计算。层数是固定的，计算量是固定的，不管这个 token 到底"值不值得"这么多计算。

这听起来有点浪费,对吧?但在 2024 年之前,几乎没人认真挑战过这个默认设定,原因很现实：**如果计算量因输入而变化,你的 GPU 就麻烦了**。GPU 和 TPU 这类硬件天生喜欢"规规矩矩"的计算——形状固定的张量、提前知道大小的矩阵乘法。一旦你说"这批 token 算 3 层,那批 token 算 8 层",计算图就变成动态的,硬件的并行优势会被大打折扣,工程实现也会一团糟。

于是就出现了一个矛盾：直觉告诉我们不同的 token 应该获得不同的计算量,但硬件现实又逼着我们给每个 token 分配一模一样的计算量。2024 年 4 月,Google DeepMind 的一篇论文——《Mixture-of-Depths: Dynamically allocating compute in transformer-based language models》——找到了一个巧妙的折中方案,既尊重了硬件的脾气,又真正实现了"哪个 token 该多想、哪个该少想"的动态分配。这就是我们今天要讲的 **Mixture of Depths (MoD)**。

## 第一个问题：省计算,到底能不能一边省一边不掉分?

### 问题是什么

先说清楚这个问题的分量。Transformer 里最贵的两块计算是自注意力(attention)和前馈网络(MLP/FFN),它们的开销跟处理的 token 数量直接相关——自注意力甚至是平方级的:token 数翻倍,attention 的计算量变成 4 倍。

如果我们能让一部分 token"跳过"某些层的自注意力和 MLP 计算,直接原样通过(也就是只走残差连接,不做任何加工),那这部分的计算就完全省下来了。这不是异想天开——工程师们很早就在想这件事,这个思路有个专门的名字,叫"**条件计算**"(conditional computation),Bengio 在 2013 年就提出了这个概念:让模型自己决定什么时候该花计算力气,什么时候不需要。

但条件计算这么多年一直没能真正进入主流大模型的训练流程,问题恰恰出在开头说的那个矛盾上。早期的很多条件计算方案——比如让模型自己决定"要不要再多算一层"(Universal Transformer 那类工作)、或者让每个样本自己决定"要不要提前退出"(早退出 / early-exit 方法,比如 DeeBERT)——都会导致计算图在运行时才能确定,不同的输入需要不同数量的计算步骤。这种"运行时才知道要算多少"的模式,恰恰是现代硬件最不喜欢的东西。

### 直觉:核心想法

MoD 的解法非常巧妙,它换了一个问题的问法。不是问"这个 token 该不该多算几层",而是问:**"这一层,我允许固定数量的 token 进来做完整计算,那具体是哪些 token,由网络自己选。"**

这个转变听起来很微妙,但它彻底改变了游戏规则。假设一层原本要处理 2048 个 token,现在我们规定:这一层只允许 256 个 token(也就是 12.5%)进来做真正的自注意力和 MLP 计算,剩下的 1792 个 token 一律走"免费通道"——直接通过残差连接,原样传到下一层,不做任何加工。

关键在于:**256 这个数字是提前定好的、固定不变的**——所以计算图依然是静态的,张量形状永远已知,硬件照样能高效并行。唯一"动态"的地方,是这 256 个名额到底给哪些 token,这件事由一个叫"路由器"(router)的小模块,根据每个 token 当前的表示,实时决定。

这就好比一家餐厅规定"每天只接待 50 桌客人",但具体是哪 50 桌,取决于谁先到、谁的预订更重要——名额数量是固定的,分配方式是动态的。这个设计上的小小转折,把一个硬件不友好的问题,变成了一个硬件完全能接受的问题。

论文作者管这个策略叫 **Mixture of Depths**——名字来自于一个观察:不同的 token,实际上穿过了不同数量的"深度"(层数)。有些 token 一路被选中,走满了模型所有的层;有些 token 大部分时候都被路由器"打回"残差通道,只经历了寥寥几层真正的计算。同一批输入,不同的 token 走过了不同深度的旅程——这就是"深度上的混合"。

<svg viewBox="0 0 640 260" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="320" y="22" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">普通 Transformer vs Mixture of Depths</text>

  <!-- vanilla -->
  <text x="20" y="55" fill="#9a9ab0" font-size="12" font-family="system-ui">普通 Transformer（每个 token 都算）</text>
  <rect x="20" y="65" width="580" height="40" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="310" y="90" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">全部 token → Self-Attn + MLP（100% 容量）</text>

  <!-- MoD -->
  <text x="20" y="135" fill="#9a9ab0" font-size="12" font-family="system-ui">Mixture of Depths（路由器挑 12.5%）</text>
  <rect x="20" y="145" width="580" height="34" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="310" y="167" text-anchor="middle" fill="#9a9ab0" font-size="11" font-family="system-ui">全部 token（2048 个）</text>

  <line x1="310" y1="179" x2="200" y2="205" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>
  <line x1="310" y1="179" x2="440" y2="205" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>

  <rect x="80" y="210" width="240" height="38" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="200" y="234" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">Top-256（12.5%）→ Self-Attn + MLP</text>

  <rect x="340" y="210" width="240" height="38" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="460" y="234" text-anchor="middle" fill="#9a9ab0" font-size="11" font-family="system-ui">剩余 87.5% → 直接走残差，不计算</text>
</svg>

## 第二个问题：让谁走"免费通道"，靠什么标准来选？

### 问题是什么

好,思路有了:每一层放一个门槛,只让固定数量的 token 通过。但这马上带来一个新问题——**谁来决定放谁进去?**

如果这个决定权交给每个 token 自己("我想不想进这个块"),会出现经典的"抢名额"问题:万一某一批 token 全都抢着要进,名额超了怎么办?万一没人愿意进,名额空着又怎么办?这在 Mixture of Experts(专家混合模型)里早就是个头疼的老问题,叫"负载不均衡"。MoE 通常需要加一个专门的"辅助损失"来强行拉平各路径的负载,但这个辅助损失本身会带来训练上的额外复杂度和干扰。

### 直觉:核心想法

MoD 的作者们换了一个视角:不让 token 自己选路径,而是让**每条路径自己去挑它想要的 token**。这个思路在 MoE 文献里有个名字,叫"专家选择路由"(expert-choice routing),和更常见的"token 选择路由"(token-choice routing)刚好反过来。

这就像大学招生:如果让每个学生自己报名想去哪个学院(token-choice),某个热门学院可能爆满,某个冷门学院可能招不满人,还得靠强制调剂政策(辅助损失)来平衡。但如果反过来,让每个学院自己划定分数线,按分数从高到低招够名额为止(expert-choice)——那么名额数量永远是精确匹配的,不需要任何强制调剂。

具体到 MoD 这里:每一层有一个"路由器",它给每个 token 打一个分数(其实就是一个非常简单的线性投影,把 token 的向量映射成一个标量)。然后这一层只挑分数最高的那 256 个(也就是 top-k)进来做完整计算,剩下的全部走残差通道。因为固定要选出恰好 256 个,负载天生就是均衡的,完全不需要额外的平衡损失。

这里还有个细节值得琢磨:**路由器打的分数,不只是决定"谁能进",还会乘到该 token 经过这一层计算之后的输出上**。换句话说,分数越高的 token,不仅能"入场",它入场之后产生的更新在数值上也会被这个分数放大或缩小。这样一来,路由器的打分行为本身就直接暴露在梯度下降的路径上——语言建模的损失函数会反过来"教"路由器,让它学着把分数打给那些真正需要更新的 token。整个路由机制没有用到任何强化学习或者复杂的启发式规则,就是最朴素的端到端梯度下降。

<svg viewBox="0 0 640 220" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="320" y="22" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">Token-Choice vs Expert-Choice 路由</text>

  <text x="160" y="50" text-anchor="middle" fill="#9a9ab0" font-size="12" font-family="system-ui">Token-Choice（token 自己选）</text>
  <rect x="30" y="60" width="80" height="34" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="70" y="82" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">token A</text>
  <rect x="130" y="60" width="80" height="34" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="170" y="82" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">token B</text>
  <line x1="70" y1="94" x2="140" y2="130" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <line x1="170" y1="94" x2="140" y2="130" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <rect x="90" y="135" width="100" height="34" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="140" y="157" text-anchor="middle" fill="#9a9ab0" font-size="10" font-family="system-ui">路径（可能超载）</text>
  <text x="140" y="195" text-anchor="middle" fill="#9a9ab0" font-size="10" font-family="system-ui">⚠ 需要辅助平衡损失</text>

  <text x="480" y="50" text-anchor="middle" fill="#9a9ab0" font-size="12" font-family="system-ui">Expert-Choice（路径自己挑）</text>
  <rect x="440" y="135" width="100" height="34" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="490" y="157" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">路径（挑 top-k）</text>
  <line x1="470" y1="130" x2="410" y2="94" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <line x1="510" y1="130" x2="550" y2="94" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <rect x="370" y="60" width="80" height="34" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="410" y="82" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">token C</text>
  <rect x="510" y="60" width="80" height="34" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="550" y="82" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">token D</text>
  <text x="490" y="195" text-anchor="middle" fill="#9a9ab0" font-size="10" font-family="system-ui">✓ 天然负载均衡，无需额外损失</text>
</svg>

### 技术细节(选读)

用数学语言写一下这个规则。假设某一层的输入是一串 token 的向量表示 $X^l = \{x_i^l\}$,路由器对每个 token 算一个分数:

$$r_i^l = w_\theta^T x_i^l$$

这其实就是一个最简单的线性层,把向量压成一个数。设 $P_\beta(R^l)$ 是这一层所有路由分数里第 $\beta$ 百分位的分数(其中 $\beta = 1 - C/S$,$C$ 是设定的容量,$S$ 是总 token 数),那么每个 token 的输出规则是:

$$
x_i^{l+1} =
\begin{cases}
r_i^l \cdot f_i(\tilde{X}^l) + x_i^l, & \text{if } r_i^l > P_\beta(R^l) \\
x_i^l, & \text{if } r_i^l < P_\beta(R^l)
\end{cases}
$$

翻译回人话:如果这个 token 的分数排在前 $C$ 名以内,它就会被送进真正的自注意力+MLP 计算 $f$,然后计算结果乘上它自己的路由分数,再加回原来的残差;如果排不进前 $C$ 名,它就什么都不做,原样往下传。这里有个小心思:被路由进来的 token,它的输出结果里"混入"了自己的路由分数——这一步保证了路由器的打分行为始终处在梯度可以流动的路径上,损失函数的反馈能直接教会路由器"该给谁打高分"。

## 第三个问题：这个设计会不会有一个致命缺陷?

### 问题是什么

细心的读者应该已经嗅到了一丝不对——如果路由的标准是"这一层所有 token 分数里排前 $C$ 名",那问题来了:**在生成文本的时候,后面的 token 还没生成出来,你怎么知道前面这个 token 的分数排不排得进前 $C$ 名?**

这不是杞人忧天,是这个方案真实存在的一个漏洞。Top-k 这个操作,天生就是"非因果"的(non-causal)——要判断某个 token 排不排前几名,你得看到整个序列所有 token 的分数才能比较。训练的时候没问题,因为训练时整句话是一次性喂给模型的,可以一次性算完所有分数再排名。但真正生成文本(自回归采样)的时候,模型是一个字一个字往后蹦的,写到第 5 个字时,后面的字压根还不存在,拿什么去比较排名?

这个矛盾如果解决不了,MoD 就只能训练用,生成的时候用不了——那这篇论文的实用价值会大打折扣。

### 直觉:核心想法

解法其实很朴素:**训练一个"预言家"**。既然真正的排名要等看到全句才知道,那就干脆再训练一个小小的辅助模型,让它在只看到当前 token(看不到后面)的情况下,去预测"如果我能看到全句,这个 token 大概会不会排进前 $C$ 名"。

这就像考试前的模拟预测——你不知道最终全班的分数排名(因为大家还没考完),但一个经验丰富的老师能根据你平时的表现,提前预判"这个学生大概能进前 20%"。只要这个预判足够准,就可以在生成阶段直接依赖预判结果,而不需要等到"全班都考完"才能做决定。

论文里试了两种"预言家"的做法。第一种更简单:直接给主任务加一个小小的辅助损失,让路由器的输出去逼近"是否排进 top-k"这个 0/1 标签,这样训练完之后,路由器自己输出的分数往 0.5 这个中间值收拢——高于 0.5 大概率是会被选中的,低于 0.5 大概率会被跳过。生成的时候就不用管别的 token,直接看这个分数是不是超过 0.5 就行,完全因果、完全在线。这个办法对语言建模的核心损失有大约 0.2%~0.3% 的轻微拖累,但换来的是一个能在生成阶段直接用的路由信号。

第二种更"干净":单独训一个小型 MLP 预测器,输入和路由器一样,但训练目标完全独立于语言建模损失(而且输入要加一个"梯度截断",不让这个预测器的训练干扰主干网络)。这个额外的预测器完全不影响主任务的损失,而且经验上很快就能训练到 97%~99% 的准确率——也就是说,这个"要不要选中我"的预判问题,本身是个相当容易学会的小任务。

无论哪种方案,论文的实验结果都令人满意:从训练时的非因果 top-k 切换到生成时的因果预测器路由,模型性能几乎没有明显下降。这说明"预判排名"这件事对模型来说不难,MoD 真的可以在生成阶段落地使用,而不只是一个训练时的理论把戏。

## 第四个问题：省了计算,到底值不值?

### 问题是什么

到这里逻辑已经很完整了,但工程师最关心的问题永远是:省是省了,那模型效果掉了多少?值不值得这么折腾?

### 直觉与结果

论文用了"等 FLOPs 对比"(isoFLOP)的方法——固定训练总计算量,分别训练普通 Transformer 和 MoD Transformer,看谁在相同的计算预算下,损失(loss)更低。这是评估架构改进最公平的方式,避免了"我模型更大所以效果更好"这种耍赖式的对比。

结果相当亮眼。作者发现,在同样的训练总算力下,MoD 版本的"最优配置"往往参数更多(网络更大),但因为大量 token 被路由跳过,每次前向传播实际消耗的 FLOPs反而更少。这意味着 MoD 打开了一个此前被浪费掉的自由度:**你可以造一个更"胖"的模型,但让它在推理时表现得像一个更"瘦"的模型。**

具体数字:一个 2.2 亿参数的 MoD 模型,在和同等计算预算下最优的普通 Transformer 打平损失的情况下,训练时的单步速度快了超过 60%;在自回归采样(也就是实际生成文字)阶段,某些配置下能快up到 50%。这些不是理论上的"应该",是实测的。

而且最优的路由配置有一个明确的经验规律:**每隔一层设一次路由(不是每层都路由),容量设为 12.5%**——也就是每次路由,只有 12.5% 的 token 走完整计算,87.5% 的 token 走残差捷径。低于 12.5% 效果开始明显下滑;但从更高的容量往下降到 12.5% 的过程中,性能是逐步变好的。这说明"该省的地方要敢省",12.5% 这个比例不是拍脑袋定的,是在多组消融实验里跑出来的经验最优点。

还有一个特别有意思的旁证:研究者分析了训练好的模型的路由决策,发现有些 token 几乎每次都被选中走完整计算,有些 token 大多数时候都被跳过。初步分析表明,那些被频繁选中的 token,往往对应着模型输出概率分布"更混乱"(熵更高)、也就是更难预测的位置。换句话说,**模型好像真的学会了把计算力气花在"难题"上,而在"送分题"上偷懒**——这正是这篇论文最初想验证的那个直觉。

## 这跟"提前退出"(Early Exit)有什么不一样?

在 MoD 之前,想让模型"少算点"的思路其实已经有一批前人探索过,最有代表性的是"早退出"(early exit)方法——比如 DeeBERT。这类方法的做法是:让每个样本/token 在某一层做一次"我够自信了吗"的判断,如果够自信,就直接从这一层"跳车",后面所有层全部跳过,不再回来。

这和 MoD 有一个本质区别。早退出是"一去不回头"——一旦决定退出,后面的层永远见不到这个 token 了。而 MoD 允许一个 token 在某一层被跳过,却在下一层重新被选中,继续接受完整计算——**跳过是逐层独立决定的,不是一次性判死刑**。论文作者特别强调了这点,并且提出一个假设:这种"可以中途跳过、后面又重新参与"的灵活性,可能比早退出那种"一旦退出就永不回头"的刚性策略更有优势,因为语言里很多 token 的"难度"本身不是均匀分布在某一段连续的层上,而是在不同深度反复起伏的。

<svg viewBox="0 0 640 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="320" y="20" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">Early Exit vs Mixture of Depths</text>

  <text x="160" y="48" text-anchor="middle" fill="#9a9ab0" font-size="12" font-family="system-ui">Early Exit：退出后永不回头</text>
  <rect x="20" y="60" width="60" height="30" rx="6" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="50" y="80" text-anchor="middle" fill="#ededf0" font-size="10">层1</text>
  <line x1="80" y1="75" x2="115" y2="75" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow3)"/>
  <rect x="120" y="60" width="60" height="30" rx="6" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="150" y="80" text-anchor="middle" fill="#ededf0" font-size="10">层2</text>
  <line x1="150" y1="90" x2="150" y2="115" stroke="#ff6e6e" stroke-width="1.5" marker-end="url(#arrow3)"/>
  <text x="150" y="130" text-anchor="middle" fill="#ff8e8e" font-size="10">退出（层3-6永远跳过）</text>
  <rect x="220" y="60" width="60" height="30" rx="6" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1"/>
  <text x="250" y="80" text-anchor="middle" fill="#5a5a6a" font-size="10">层3</text>
  <rect x="300" y="60" width="60" height="30" rx="6" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1"/>
  <text x="330" y="80" text-anchor="middle" fill="#5a5a6a" font-size="10">层4</text>

  <text x="480" y="48" text-anchor="middle" fill="#9a9ab0" font-size="12" font-family="system-ui">MoD：跳过后还能重新加入</text>
  <rect x="400" y="60" width="55" height="30" rx="6" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="427" y="80" text-anchor="middle" fill="#ededf0" font-size="10">层1</text>
  <line x1="455" y1="75" x2="480" y2="75" stroke="#6e8eff" stroke-width="1" stroke-dasharray="3,3" marker-end="url(#arrow3)"/>
  <rect x="485" y="60" width="55" height="30" rx="6" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1"/>
  <text x="512" y="80" text-anchor="middle" fill="#9a9ab0" font-size="9">层2 跳过</text>
  <line x1="540" y1="75" x2="565" y2="75" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow3)"/>
  <rect x="570" y="60" width="55" height="30" rx="6" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="597" y="80" text-anchor="middle" fill="#ededf0" font-size="9">层3 重新参与</text>
</svg>

## MoD 和 MoE 能不能一起用?

论文最后还试了一件很自然的事:既然 MoD 借用了 MoE 的路由机制,那能不能把两者直接叠在一起?作者把这种结合叫 **Mixture-of-Depths-and-Experts (MoDE)**。

思路是把 MoD 的"跳过"这个选项,当成 MoE 里的一个特殊"专家"——一个什么都不做的"空专家"。这样一来,原本的 MoE 路由器不仅要在多个真正的专家之间选,还多了一个"什么都不选"的选项。作者发现,把"跳过"直接整合进专家选择机制里(而不是简单粗暴地缩小每个专家的处理容量、指望多余的 token 被自然丢弃),效果明显更好——因为整合式的做法让 token 显式地"学会主动选择跳过",而不是被动地"因为名额不够被挤掉"。这两者听起来像文字游戏,但训练动态上差异很大:主动选择意味着路由器的梯度信号是清晰的、有方向的;被动挤掉则更接近随机丢弃,学不到什么有用的规律。

这个结果也说明,MoD 不是一个孤立的技巧,而是可以嵌入到现有效率优化生态(MoE)里的一块拼图,两种稀疏性——"专家维度的稀疏"和"深度维度的稀疏"——可以叠加,收益是可以复合的。

## 这意味着什么

回过头看整个故事,MoD 解决的本质问题非常朴素:**Transformer 长期以来把"计算量"和"层数"绑得太死**——每个 token 无论难易都必须走完全部层数、做完全部计算。MoD 用一个几乎不增加多少复杂度的路由机制,把这两者解绑了:计算预算依然是提前定好、硬件友好的静态数字,但具体谁能用上这份预算,变成了一个由模型自己学习、由梯度下降驱动的动态选择。

它带来的实际收益也很具体:同样的训练算力预算下,能训出损失更低、同时前向传播更快的模型;生成阶段能省下高达 50% 的计算;还能和 MoE 结合,把"专家维度的稀疏"和"深度维度的稀疏"叠加起来。而它解决非因果 top-k 的方式——训一个小小的、几乎不额外增加成本的"预言家"来在推理时模拟排名——也提供了一个可以复用的工程范式:很多"训练时容易、推理时因果受限"的问题,或许都能用这种"训一个廉价的预测器去逼近理想排名"的思路来解。

当然,这项工作也留下了没解决的地方。作者自己也承认,路由决策的可解释性还有限——为什么某些 token 就是"天生难",这背后的机制并不完全清楚;此外,MoD 目前主要验证在预训练阶段的 isoFLOP 对比上,在真实的大规模产品级模型里(叠加各种其它效率优化手段之后)收益能保留多少,也还需要更多公开的大规模验证。但作为一个"用固定预算实现动态分配"的设计范式,MoD 提供的思路——用专家选择路由让路径挑 token、用可训练的小型预测器把非因果操作变成因果操作——已经被后续不少条件计算和推理加速的工作借鉴。

如果说 Mixture of Experts 教会了我们"模型的宽度可以是稀疏的",那 Mixture of Depths 教会我们的是:**模型的深度,同样可以是稀疏的**。两者合在一起,勾勒出一个越来越清晰的方向——未来的大模型,大概不会是一个"每个 token 都被同等对待"的铁板一块,而是一张动态的、根据输入内容实时重新分配计算资源的网络。

---

*参考资料:*
- Raposo, D., Ritter, S., Richards, B., Lillicrap, T., Humphreys, P.C., & Santoro, A. (2024). *Mixture-of-Depths: Dynamically allocating compute in transformer-based language models*. arXiv:2404.02258.
- Ainslie, J., Lei, T., de Jong, M., et al. (2023). *CoLT5: Faster Long-Range Transformers with Conditional Computation*. arXiv:2303.09752.
- Xin, J., Tang, R., Lee, J., Yu, Y., & Lin, J. (2020). *DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference*. arXiv:2004.12993.
