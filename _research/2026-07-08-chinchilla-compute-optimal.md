---
title: "Chinchilla 最优：一块钱的算力，该买大脑子还是买书？"
date: 2026-07-08
level: 3
series: "LLM 原理深度解析"
series_order: 29
series_total: 37
tags: [scaling-laws, chinchilla, compute-optimal, pretraining, llm-training]
summary: "DeepMind 用 400 多个模型的实验告诉我们：过去几年大家一直把 GPT-3、Gopher 这样的巨兽训练得太大、喂得太少了——而修正这个错误后诞生的 70B 参数 Chinchilla，靠着更多的数据，把四倍大的 Gopher 打得没有还手之力。"
---

> 同样一块钱的算力预算，你选择造一个更大的脑子，还是给它读更多的书？2022 年之前，几乎所有人都选错了答案。

## 故事从这里开始

2020 年前后，整个 AI 圈子陷入了一种近乎信仰的共识：**模型越大越好**。GPT-3 有 1750 亿参数，随后 Google 的 Megatron-Turing NLG 冲到 5300 亿，各家实验室仿佛在打一场"参数量竞赛"——谁的模型更大，谁就赢了。

但这里有一个几乎没人细想的问题：**这些巨大的模型,喂了它们多少数据?**

答案是:出奇地少。GPT-3 用了大约 3000 亿 token 训练它的 1750 亿参数;更极端的 Megatron-Turing NLG,530B 参数,也只用了差不多 2700 亿 token。换算一下,每个参数平均只"读"了不到 1 个 token 的数据。

这就好比你雇了一个绝顶聪明的博士生,却只给他读了一本薄薄的小说,然后期待他能通晓天下学问。他确实聪明,但没读过的书,他不会凑巧知道。

DeepMind 在 2022 年的一篇论文里问了一个简单却致命的问题:**如果算力(也就是钱)是固定的,你应该把它花在"更大的脑子"上,还是"更多的书"上?** 换句话说——给定一笔训练预算(总共能做多少次浮点运算),模型参数量和训练数据量,应该按什么比例分配,才能让最终的模型最聪明?

这篇论文的答案后来被称为"**Chinchilla 定律**"(因为他们据此训练出的验证模型叫 Chinchilla)。它的结论极其简单,却彻底改写了后续几年所有大模型的训练配方——包括 LLaMA、Falcon、几乎所有你听过的开源大模型的训练策略,起点都是这篇论文。

这篇文章,我们就来搞清楚:这个"最优比例"是怎么算出来的,它背后的道理是什么,以及——这个"最优"到底是对谁最优。因为剧透一下:后来的 LLaMA、GPT-4 级别模型,又故意"违反"了这个最优比例,而且是有意为之。这中间发生了什么,值得讲透。

<svg viewBox="0 0 640 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow0" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <rect x="20" y="20" width="280" height="70" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="160" y="48" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">GPT-3 / Gopher 时代</text>
  <text x="160" y="70" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">参数量 ↑↑↑    数据量 ↑</text>

  <rect x="340" y="20" width="280" height="70" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="480" y="48" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">Chinchilla (2022)</text>
  <text x="480" y="70" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">参数量 ↑    数据量 ↑ (同步)</text>

  <line x1="300" y1="55" x2="335" y2="55" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow0)"/>

  <rect x="60" y="130" width="200" height="55" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="160" y="155" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">同等算力预算</text>
  <text x="160" y="173" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">Gopher: 280B / 300B tok</text>

  <rect x="380" y="130" width="200" height="55" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="480" y="155" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">同等算力预算</text>
  <text x="480" y="173" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">Chinchilla: 70B / 1.4T tok</text>
</svg>

## 第一部分：算力这块饼要怎么切

### 问题是什么

先把问题说精确。训练一个 Transformer 语言模型,消耗的浮点运算量(FLOPs,记作 $C$)大致等于:

$$C \approx 6ND$$

这里 $N$ 是模型参数量,$D$ 是训练用的 token 数量。这个近似公式来自 Kaplan 等人 2020 年的工作,直觉很简单:每处理一个 token,模型大约要做 $6N$ 次浮点运算(前向 $2N$,反向传播大约 $4N$,粗略估计)。处理 $D$ 个 token,就是 $6ND$。

现在问题来了:如果你有固定的预算 $C$(比如说,你租得起的 GPU 小时数决定了你能做多少次运算),那么 $N$ 和 $D$ 该怎么分配,才能让最终模型的 loss(也就是"模型有多聪明"的反向指标)最小?

这不是一个抽象的理论问题。因为 $C \approx 6ND$ 意味着 $N$ 和 $D$ 是"跷跷板"关系——同样的预算下,模型越大,能喂的数据就越少;数据喂得越多,模型就必须越小。你无法同时把两头都做到最大。而在 2020-2022 年那段时间,几乎整个行业的默契选择是:**疯狂加大模型,数据量基本不变**。

这个选择对不对?没人真正验证过——直到 DeepMind 做了这件事。

### 直觉:核心想法

DeepMind 的做法,说白了就是"暴力实验 + 数学建模"两步走。他们训练了**超过 400 个**不同大小的模型(从 7000 万参数到 160 多亿参数),每个模型又用了不同数量的训练数据(50 亿到 5000 亿 token),然后观察一个规律:

> 对于任何一个固定的算力预算,当你把参数量从很小逐渐调到很大(同时数据量按 $C=6ND$ 自动调整变小),最终的 loss 会先降后升,中间存在一个明确的"谷底"。

想象你在爬一座山,山的一侧是"模型太小、学不动东西"的坡,另一侧是"数据太少、模型没读够书就被迫收工"的坡。两侧都不好,中间有一个山谷,那就是给定预算下的最优点。

这就是论文里著名的"IsoFLOP 曲线"(IsoFLOP profile)的意思——"Iso"是"相同"的意思,IsoFLOP 就是"固定总算力"。DeepMind 固定了 9 个不同的算力档位,每个档位里训练一系列不同大小的模型(数据量随之调整以保持总 FLOPs 恒定),画出 loss vs 模型大小的曲线,每条曲线都呈现出一个漂亮的谷底。

<svg viewBox="0 0 640 280" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="320" y="24" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">IsoFLOP 曲线示意：固定算力预算下，Loss 随模型大小变化</text>
  <line x1="60" y1="220" x2="580" y2="220" stroke="#3a3a4a" stroke-width="1.5" marker-end="url(#arrow1)"/>
  <line x1="60" y1="220" x2="60" y2="50" stroke="#3a3a4a" stroke-width="1.5" marker-end="url(#arrow1)"/>
  <text x="580" y="240" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">模型参数量 N (log)</text>
  <text x="30" y="50" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui" transform="rotate(-90 30 130)">Loss</text>

  <path d="M 90 190 Q 260 60 320 60 Q 380 60 550 190" fill="none" stroke="#6e8eff" stroke-width="2.5"/>
  <circle cx="320" cy="60" r="6" fill="#1e1e2a" stroke="#6e8eff" stroke-width="2"/>
  <text x="320" y="42" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">最优点 N_opt</text>

  <text x="150" y="205" fill="#ededf0" font-size="11" font-family="system-ui">模型太小</text>
  <text x="150" y="218" fill="#ededf0" font-size="10" font-family="system-ui">(学不动)</text>
  <text x="440" y="205" fill="#ededf0" font-size="11" font-family="system-ui">模型太大</text>
  <text x="440" y="218" fill="#ededf0" font-size="10" font-family="system-ui">(数据没喂够)</text>
</svg>

有了每个算力档位的最优点(谷底位置),再把这些"最优模型大小"和"最优 token 数"对算力做一次拟合,就能得到一条幂律关系,告诉你:算力翻倍,模型该放大多少倍,数据该增加多少倍。

### 技术细节(选读)

DeepMind 一共用了**三种独立的方法**来估计这个规律,这一点很值得说——因为如果三种完全不同的方法得出一致的结论,那结论就更可信。

**方法一:看训练曲线的最低点**。固定模型大小,让它一路训练下去,记录训练过程中(N, D, Loss)的每一个点,直接从整条训练曲线里找规律。

**方法二:上面讲的 IsoFLOP 曲线**。固定 9 个不同的算力预算(从 $6\times10^{18}$ 到 $3\times10^{21}$ FLOPs),每个预算下训练一批不同大小的模型,对 loss-vs-模型大小的曲线**拟合一条抛物线**,抛物线的顶点就是这个预算下的最优模型大小。之所以用抛物线拟合而不是更复杂的曲线,是因为在 log 坐标系下,loss 曲线在最优点附近近似对称,抛物线就够用了(后续有研究指出这个近似在某些尺度下会带来系统性偏差,我们后面会提到)。

**方法三:拟合一个参数化的损失函数**。这是最有意思的一种做法。DeepMind 假设最终 loss 可以写成:

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

这个公式翻译成人话是:模型的最终 loss,可以拆成三部分——

- $E$:自然语言文本本身固有的"不可约熵"。就算你有无穷大的模型和无穷多的数据,文本里天生的随机性(比如下一个词到底是"苹果"还是"香蕉",本身就有不确定性)也不可能被消除。这是理论上的天花板。
- $A/N^\alpha$:模型容量不足带来的额外损失。模型越大($N$ 越大),这一项越小——因为更大的模型理论表达能力更强,能更好地逼近真实的语言分布。
- $B/D^\beta$:数据不够、训练不充分带来的额外损失。数据越多($D$ 越大),这一项越小——因为模型见过的样本越多,越能"收敛"到它能力范围内的最优表现。

这三项加起来就是实际观测到的 loss。DeepMind 用超过 400 个实验点,通过最小化预测 loss 和实际 loss 之间的误差(用 Huber loss,一种对异常值不那么敏感的损失函数),拟合出了 $A, B, E, \alpha, \beta$ 这五个数字。

拟合出这些系数后,在约束 $C \approx 6ND$ 下求这个损失函数的最小值,就能推出一个漂亮的解析解:

$$N_{opt}(C) = G\left(\frac{C}{6}\right)^{a}, \qquad D_{opt}(C) = G^{-1}\left(\frac{C}{6}\right)^{b}$$

其中 $a = \beta/(\alpha+\beta)$,$b = \alpha/(\alpha+\beta)$,$G$ 是另一个由 $A, B, \alpha, \beta$ 决定的常数。翻译回人话:**最优的模型大小和最优的数据量,都随算力预算呈幂律增长,增长的"速度"(指数 $a$ 和 $b$)由损失函数里模型容量项和数据项的相对"贵贱"决定**。

三种方法算出来的 $a, b$ 惊人地一致:

| 方法 | $N_{opt} \propto C^a$ 中的 $a$ | $D_{opt} \propto C^b$ 中的 $b$ |
|---|---|---|
| 方法一(训练曲线最低点) | 0.50 | 0.50 |
| 方法二(IsoFLOP 抛物线) | 0.49 | 0.51 |
| 方法三(参数化损失拟合) | 0.46 | 0.54 |
| Kaplan 等人 2020 年之前的结论 | **0.73** | **0.27** |

看最后一行——这才是真正炸场的部分。

## 第二部分：为什么之前所有人都错了

### 问题是什么

Kaplan 等人 2020 年那篇早期的 scaling law 论文(也是那场"参数量竞赛"的理论依据),给出的结论是 $a=0.73, b=0.27$——意味着算力每增加一份,**模型大小应该长得远远快于数据量**。这正是 GPT-3、Gopher 那批模型的训练哲学的来源:多加参数,数据基本不变。

但 Chinchilla 三种方法都得出 $a \approx b \approx 0.5$——模型和数据应该**同步等比例增长**。这是一个方向性的分歧,不是误差范围内的小偏差。

DeepMind 认为差异的根源在于实验设计上的一个细节:**Kaplan 等人训练每个模型时都用了同样长度的学习率衰减计划(learning rate schedule),不管模型最终训练多久**。这导致小算力预算下的模型,如果训练步数少于学习率衰减计划设计的长度,就会因为学习率还没退火到位而"看起来"没训练充分——人为地压低了小模型的表现,让"加大模型"看起来比实际更有效。Chinchilla 团队修正了这一点,让每个模型的学习率衰减计划长度精确匹配它实际训练的 token 数,重新做了实验,得出了完全不同的结论。

### 直觉:核心想法

如果你把这个发现拍成一句话总结,那就是:

> **过去几年的大模型,是被养得"虎背熊腰",但脑子里的知识却营养不良。**

想象两个学生准备同一场考试,用的复习时间总量(算力)完全相等。学生 A 拼命练脑力(变成更大的模型),但复习的书却只翻了一遍;学生 B 脑子没那么"大",但把书翻了十几遍,吃透了每一个细节。DeepMind 的发现是:在同样的总复习时间下,学生 B 会考得更好。

这就是"Chinchilla 定律"最直观的解读:**模型太大而数据太少,是一种典型的资源错配**。你花了大价钱买来的额外参数,如果没有配上足够多的数据去"喂养"它,那些参数很大一部分其实是被浪费的——它们本可以塞进更多知识,但因为没读够书,能力被锁在了潜力之下。

DeepMind 用这个结论做了一次直接的验证:他们拿 Gopher 模型(280B 参数,用 300B token 训练)所消耗的总算力,重新按 Chinchilla 定律的比例分配,训练出一个只有 70B 参数、但用了 1.4T token(约 4 倍数据量)的模型——**Chinchilla**。结果 Chinchilla 用四分之一的参数量,在几乎所有下游评测上都**碾压**了 Gopher,在 MMLU 基准上更是拿到 67.5% 的准确率,比 Gopher 高出超过 7 个百分点。

同样的算力,换个分配比例,结果天差地别。这就是这篇论文震动整个行业的原因。

### 技术细节(选读)

推导出的经验规律给出了一个大致的比例:对于 DeepMind 当时的架构和数据分布,compute-optimal 的训练,大约需要每个参数配 **20 个 token** 左右的数据(具体数字是从上面的 $N_{opt}, D_{opt}$ 公式里代入具体的 $A,B,\alpha,\beta$ 算出来的,不是一个理论上的"魔法数字",换一批数据/架构这个比值会漂移,但作为经验法则被广泛引用)。

需要澄清一个常见误解:这个"20:1"规则并不是"越接近这个比例模型就一定越好"的铁律,它只回答了一个非常具体的问题——**如果你的目标是用给定的训练算力预算,让 loss 最小,那么在 N 和 D 之间怎么切分。** 这是一个纯粹关于**训练效率**的答案。它完全没有考虑另一件同样重要的事——这个模型训练完之后,还要被拿去用(推理),而推理也要花钱。这一点我们下一节细说,因为它正是后来 LLaMA 等模型"故意打破"Chinchilla 比例的原因。

## 第三部分：论文里被找出的一个漏洞

科学的可贵之处在于它允许被检验、被纠错。2024 年,一篇名为《Chinchilla Scaling: A replication attempt》的论文尝试完整复现 DeepMind 三种方法里的**第三种**(参数化损失拟合)。他们从论文的图表里手动重建了原始数据点,重新做拟合,结果发现:

- DeepMind 报告的第三种方法的具体系数,和第一、第二种方法的结果**不一致**,而这三种方法本应互相印证。
- 更严重的是,DeepMind 报告的置信区间"窄得不可思议"——按统计学计算,那么窄的置信区间需要超过 **60 万次**实验才能达到,而 DeepMind 实际训练的模型不到 500 个。这意味着原论文在计算置信区间时可能存在一个统计错误。

好消息是:复现团队用同样的第三种方法,自己重新推导出的系数,与方法一、二是**吻合**的——也就是说,"模型和数据应该大致等比例扩展"这个**核心结论本身是站得住脚的**,只是原论文报告第三种方法结果时,某个统计环节出了纰漏。

这段小插曲值得写进文章,因为它示范了科学应该有的样子:一个重要结论被更细致地检验,发现细节上的瑕疵,但核心洞见经得起独立复现的考验——比"发表了就是真理"要真实得多。

## 第四部分：LLaMA 为什么"故意"打破了这个比例

### 问题是什么

如果 Chinchilla 定律是对的,那为什么后来 Meta 的 LLaMA 系列,反而远远超过了这个比例训练?LLaMA 2 用了 2 万亿 token,LLaMA 3 更是用了 15 万亿 token——这远远超出"20 tokens/param"的建议范围(LLaMA-3 8B 版本按参数换算,相当于每个参数被喂了近 2000 个 token,是 Chinchilla 建议值的 100 倍)。

这不是 Meta 不懂 Chinchilla 定律——恰恰相反,这是他们**深刻理解**了这个定律的适用范围之后,故意选择偏离它。

### 直觉:核心想法

关键在于问对问题。Chinchilla 定律回答的是:"**给定固定的训练算力预算,怎么让训练出的模型 loss 最低?**"

但一个真正要被成千上万人每天调用的模型,它的总成本从来不只是训练那一次——它要被部署、被无数次调用,每一次调用(推理)都要消耗计算资源。如果这个模型要被调用十亿次,那么推理的总成本,可能远远超过训练成本本身。

于是就出现了另一个更现实的问题:"**给定我要训练+部署这个模型的总花费(训练+所有未来推理的总和),我该怎么选 N 和 D,才能让总花费最低,同时达到我要的模型质量?**"

这是一个不同的优化目标,答案自然也不同。而答案的方向很直觉:**推理成本只跟模型大小有关(模型越小,每次调用越便宜),跟训练时喂了多少数据无关**。所以,如果你预期这个模型会被大量调用,那你应该**故意训练一个比 Chinchilla-optimal 更小的模型**,但用**远超 Chinchilla 建议比例的数据**去"过度训练"(overtrain)它,把它的能力尽量榨到接近大模型的水平,同时享受小模型带来的低廉推理成本。

这就像雇人一样:如果你只需要这个人干一次活,那么找一个"刚好够用"的人成本最低。但如果你要雇他给你打一辈子工,那多花点培训成本,把一个便宜的人培养到接近专家水平,长期算下来反而更省钱——因为你每天都要付他工资,一个"更贵但技能一般"的专家长期算总账反而不划算。

2024 年 MosaicML(现 Databricks)团队的论文《Beyond Chinchilla-Optimal》把这个直觉严格量化了。他们把推理成本纳入 Chinchilla 的公式,重新推导最优分配,并训练了 47 个不同大小/token 数的模型来验证。结论是:**如果你预期这个模型会被调用大约 10 亿次以上,那么应该训练比 Chinchilla-optimal 更小的模型,训练更长时间。** 他们还发现一个有点反直觉的现象:即使把 token/参数比例推到极端(高达 10000:1,也就是每个参数配一万个 token),模型质量仍然在缓慢提升,并没有撞到明显的天花板——只是收益越来越小。

<svg viewBox="0 0 640 260" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="320" y="24" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">两种"最优"：只算训练 vs 算训练+推理总账</text>

  <rect x="30" y="50" width="260" height="180" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="160" y="75" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui" font-weight="bold">Chinchilla-optimal</text>
  <text x="160" y="95" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">目标：训练 loss 最小</text>
  <text x="160" y="115" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">N : D ≈ 1 : 20</text>
  <text x="160" y="145" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">适合：一次性训练</text>
  <text x="160" y="165" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">    不考虑后续调用量</text>
  <text x="160" y="200" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui" font-weight="bold">例：Chinchilla 70B</text>
  <text x="160" y="218" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">1.4T tokens</text>

  <rect x="350" y="50" width="260" height="180" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="480" y="75" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui" font-weight="bold">Inference-aware optimal</text>
  <text x="480" y="95" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">目标：训练+推理总花费最小</text>
  <text x="480" y="115" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">N : D ≫ 1 : 20</text>
  <text x="480" y="145" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">适合：会被大量部署调用</text>
  <text x="480" y="165" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">    的生产模型</text>
  <text x="480" y="200" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui" font-weight="bold">例：LLaMA 3 8B</text>
  <text x="480" y="218" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">15T tokens</text>
</svg>

### 技术细节(选读)

Sardana 等人的做法是,在 Chinchilla 原始的损失公式 $L(N,D) = E + A/N^\alpha + B/D^\beta$ 基础上,加入一个推理成本项。假设模型部署后预计要处理 $D_{inf}$ 个推理 token,那么总计算成本大致是:

$$C_{total} = C_{train}(N, D) + C_{inference}(N, D_{inf})$$

训练成本仍是熟悉的 $6ND$,推理成本大致与 $N \times D_{inf}$ 成正比(模型越大,处理每个 token 越贵;推理量越大,总成本越高)。现在的优化目标变成:在达到某个目标 loss(质量水准)的约束下,让 $C_{total}$ 最小,而不是单纯让 $C_{train}$ 最小。

由于推理项只依赖 $N$(不依赖已经用掉的训练数据量 $D$),这个新目标函数会把最优解往"更小的 $N$、更大的 $D$"方向推——因为你可以持续增加 $D$ 来降低 loss(边际收益递减但为正),却不需要为此付出任何额外的推理代价;而增加 $N$ 虽然也能降 loss,却会永久性地推高未来每一次推理的成本。$D_{inf}$(预期调用量)越大,这种"往小模型、长数据倾斜"的效应就越强——这正是为什么 GPT-4、LLaMA、Gemini 这些注定要被十亿级调用的产品级模型,都远远偏离了 Chinchilla 的 20:1 比例,选择了更小的模型体量加更极端的数据投喂。

## 第五部分：这个"最优",还有一个隐藏前提

最后补一个容易被忽略的细节:Chinchilla 的整套推导,隐含假设你有**无限多的新鲜(未重复)数据**可以用。但现实是,高质量的文本数据是有限的资源——互联网上能爬到的、值得训练用的干净文本,正在被越来越快地耗尽。

2023 年 Hugging Face 团队的《Scaling Data-Constrained Language Models》就专门研究了这个问题:当数据变得稀缺、你不得不重复使用同一批数据训练多个 epoch 时会发生什么?他们发现,重复训练最多 4 个 epoch,loss 几乎不受影响(和用等量的全新数据几乎一样好);但超过这个范围之后,继续重复数据的边际价值会迅速衰减到接近零——多花的算力,基本打了水漂。这意味着,当数据不够用的时候,原来那套"再加数据就能变强"的公式需要修正,数据的"新鲜度"本身也是一种隐藏的稀缺资源。

## 这意味着什么

回过头看,Chinchilla 定律真正教会我们的,不是一个死板的"20:1"魔法数字,而是一种思维方式:**训练大模型是一道资源分配的优化题,而"最优"永远取决于你优化的目标是什么**。

- 如果你只关心一次性训练出最强的模型、不考虑它以后要被怎么用,那答案是模型和数据大致同步扩大,比例大约在 20:1 附近。
- 如果你知道这个模型将来会被海量调用,那答案变成:故意造一个偏小的模型,拼命多喂数据,把训练算力"预付"成未来推理阶段的省钱红利。
- 如果你连"新鲜数据"这个资源本身都快用完了,那答案又要再打一次折——重复数据的边际价值远不如新数据。

这也是为什么今天几乎所有严肃的大模型技术报告,都会明确写出自己训练用了多少参数、多少 token、这个比例是不是"Chinchilla-optimal"——因为这已经成了整个行业衡量"这个训练配方是否经过深思熟虑"的通用语言。理解了 Chinchilla 定律,你才能真正读懂一份模型技术报告里那些看似枯燥的数字背后,团队做了什么样的权衡。

## 参考来源

1. Hoffmann et al. (2022). *Training Compute-Optimal Large Language Models*. arXiv:2203.15556
2. Besiroglu et al. (2024). *Chinchilla Scaling: A replication attempt*. arXiv:2404.10102
3. Sardana et al. (2024). *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws*. arXiv:2401.00448 (ICML 2024)
4. Muennighoff et al. (2023). *Scaling Data-Constrained Language Models*. arXiv:2305.16264 (NeurIPS 2023)
5. Weng, L. (2026). *Scaling Laws, Carefully*. lilianweng.github.io
