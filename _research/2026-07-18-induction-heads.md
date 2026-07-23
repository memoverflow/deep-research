---
title: "归纳头：当 Transformer 学会「见过一次就能用」的那个瞬间"
date: 2026-07-18
level: 3
series: "LLM 原理深度解析"
series_order: 42
series_total: 53
tags: [mechanistic-interpretability, induction-heads, in-context-learning, transformer-circuits, attention]
summary: "训练一个 Transformer 的过程中,会有一个几分钟内突然发生的"顿悟"——模型学会了"复制刚才见过的模式"这一件事,而这一件事很可能就是 In-Context Learning 能力的真正来源。这篇文章带你看清这个顿悟时刻,以及背后那个由两个 attention head 组成的小电路。"
---

> 训练日志里有一个诡异的现象:损失曲线本来在平滑下降,忽然在某个瞬间"卡"了一下,拱起一个小包,然后重新俯冲向下,比之前下降得更快。这个不起眼的小包,后来被证明和一种叫"归纳头"(induction head)的电路的诞生精确重合——而归纳头,可能就是"上下文学习"这个能力真正的物理载体。

## 故事从这里开始

假设你在给一个刚接触 Transformer 可解释性的朋友解释这样一件事:GPT 能在你的对话里"现学现用"。你在提示词里随口发明了一个新词——比如把"很厉害"叫做"绝绝子",模型接下来的回复居然真的开始用"绝绝子"这个词,用法还挺准。它没有被重新训练,权重一个字节都没变,却好像刚刚学会了一个新概念。

这个现象叫做 In-Context Learning(上下文学习,简称 ICL)。它古怪的地方在于:模型的所有知识理应都编码在权重里,权重在推理时是冻结的。那么这种"看一眼就会用"的能力,到底是从哪来的?

2022 年,Anthropic 的一支团队(Catherine Olsson、Nelson Elhage、Neel Nanda 等人)给出了一个大胆的答案:几乎全部这种能力,可能都来自 Transformer 内部一种极其具体、极其小巧的电路结构——他们称之为"归纳头"(induction head)。更让人意外的是,这个电路不是训练了很久才慢慢长出来的,而是在训练过程中的某一个"瞬间"忽然冒出来的,而且这个瞬间恰好和损失曲线上一个奇怪的"鼓包"精确重合。

这篇文章想讲清楚三件事:归纳头到底是什么、它为什么被怀疑是 ICL 的主要机制、以及这套说法后来遇到了哪些质疑和修正。这不是一个"已经完全解决"的故事,而是一个可解释性研究里少见地既漂亮又留有余地的中间结论。

<svg viewBox="0 0 640 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="320" y="24" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">序列: ... A B ... A ?</text>
  <rect x="20" y="50" width="50" height="40" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="45" y="75" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui">A</text>
  <rect x="90" y="50" width="50" height="40" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="115" y="75" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui">B</text>
  <rect x="160" y="50" width="50" height="40" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="185" y="75" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">...</text>
  <rect x="230" y="50" width="50" height="40" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="2"/>
  <text x="255" y="75" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui">A</text>
  <rect x="300" y="50" width="60" height="40" rx="8" fill="#1e1e2a" stroke="#8fd19e" stroke-width="2"/>
  <text x="330" y="75" text-anchor="middle" fill="#8fd19e" font-size="14" font-family="system-ui">→B?</text>
  <line x1="115" y1="90" x2="255" y2="90" stroke="#6e8eff" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#arrow1)"/>
  <text x="185" y="120" text-anchor="middle" fill="#6e8eff" font-size="12" font-family="system-ui">"上次 A 后面跟着 B，这次也这么猜"</text>
  <text x="320" y="160" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">这就是归纳头要完成的核心算法：</text>
  <text x="320" y="182" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">[A][B] ... [A] → 预测 [B]</text>
</svg>

## 第一部分:问题是什么——ICL 到底该怎么被"抓住"

在讲归纳头之前,得先讲清楚 Anthropic 这批人当时到底在追什么。

ICL 这个词很容易被用得很松散。有人说"给几个例子模型就会做题"是 ICL,有人说"上下文越长模型表现越好"也是 ICL。这两种说法其实指的不完全是同一件事。Anthropic 的团队选择了第二种更"可测量"的定义:**衡量模型在一个序列的靠后位置,是否比靠前位置预测得更准。**

具体做法是:拿模型在一大段文本上跑一遍,记录每个 token 位置上的 loss(预测有多准),然后比较"第 500 个 token 的 loss"和"第 50 个 token 的 loss"。如果模型有 ICL 能力,那么随着看到的上下文越来越多,它应该越来越会"顺着当前这段话的风格/规律"去预测,所以第 500 个位置的 loss 应该显著低于第 50 个位置的。这个差值,就是他们量化 ICL 强弱的核心指标。

这个选择很关键,因为它把一个模糊的"能力"变成了一条可以在训练过程中逐点画出来的曲线——而正是这条曲线,让他们撞见了那个诡异的"鼓包"。

### 直觉:训练日志里的那个鼓包

想象你在看着一个模型训练的实时曲线。损失(loss)一路平滑下降,你已经看惯了这种曲线的形状。但突然,在训练进行到某个特定阶段,曲线不再平滑——它会稍微向上拱一下,像是遇到了一个小小的减速带,然后重新俯冲下降,而且下降得比之前更快了。

这个现象在深度学习里并不新鲜,大家通常把这类现象叫做"phase change"(相变)或者"grokking"式的突变。但让人意外的是,这个鼓包在几乎所有 Transformer(只要层数大于 1)上都会出现,时间点惊人一致,而且——这是本文的核心——它和"模型忽然学会用归纳头做模式匹配"这件事,在时间上几乎完全对齐。

换句话说:损失曲线上那个不起眼的小坎,可能就是模型"顿悟"上下文学习的物理证据。这是一个非常诱人的假说——如果真的成立,就意味着你可以通过盯着损失曲线上的一个鼓包,精确定位模型"学会举一反三"的那一刻。

## 第二部分:归纳头到底是什么——两个 attention head 的接力赛

### 问题是什么

要理解归纳头,先要接受一件反直觉的事:**单独一个 attention head,做不了归纳这件事。**

想象归纳任务的样子:序列里出现了 `...A B... A`,模型需要预测下一个 token 是 `B`。要完成这件事,当前位置(第二个 A)必须"知道"三件事:1)它自己是什么 token(A);2)序列前面某个位置也出现过 A;3)那个位置后面跟着的是什么(B)。

单个 attention head 只能做一次"查询-匹配-取值"的操作。但这里需要的信息链条有两跳:先要找到"上一次 A 出现的位置",再要从那个位置"往后挪一格"去看跟着的是什么。这是一种典型的两步推理,而一层 attention 天生只能做一步。

### 直觉:接力赛里的两个人

这就是为什么 Anthropic 发现,归纳能力永远不会出现在只有一层 attention 的模型里——它必须由**两个不同层的 head 组成一个接力**才能完成。

第一个 head 叫"上一个 token 头"(previous token head)。它的工作极其简单:站在每个位置上,只往前看一格,把"我前一个位置是什么 token"这个信息复制到自己身上。你可以想象一个只会说"我刚才是谁"的哨兵,站在序列的每一个岗位上。

第二个 head 才是真正的"归纳头"。它站在当前位置(最新的那个 A),要去序列里的所有位置发问:"你们里面谁的'前一个 token 是 A' 这个标记是亮着的?"——而正是第一个 head 提前把这个标记贴好了。归纳头找到那个匹配的位置后,把那个位置"当时的内容"(也就是 B)取过来,作为对下一个词的预测。

这套组合在 Transformer 可解释性圈子里有一个专门术语,叫"K-composition"(key 的组合):归纳头的 query 是当前 token(A)的表征,而它要匹配的 key,不是那个位置原始的 token embedding,而是经过第一个 head 改写过的、混入了"我前一个词是什么"信息的 key。是这层间接性,让整套机制能work。

### 技术细节(选读)

用 Elhage et al. 2021 提出的"Transformer 电路数学框架"的语言,一个 attention head 可以拆成两个独立的子电路:

- **QK 电路**决定"往哪里看"(attention pattern 由谁决定注意力权重);
- **OV 电路**决定"看到了搬运什么信息"(实际写回 residual stream 的内容)。

对于归纳头来说,QK 电路的关键是:query 由当前 token 的 embedding 直接产生,而 key 却不是来自 token 本身,是来自前一层"上一个 token 头"往 residual stream 里写入的信息。这就是所谓 K-composition——第二层的 key,组合了第一层某个 head 的输出。

翻译回人话:归纳头之所以"聪明",不是因为它单独有多复杂,而是因为它站在了另一个更简单的 head 的肩膀上,读到了本来不该在那个位置出现的信息(前一个 token 是什么)。这种"跨层借用信息"的把戏,正是 Transformer 能做多步推理的秘密之一,归纳头只是其中最容易观察、最"干净"的一个案例。

<svg viewBox="0 0 640 260" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="320" y="24" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">归纳电路：两层 attention 的接力</text>

  <rect x="40" y="50" width="560" height="70" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="60" y="75" fill="#ededf0" font-size="13" font-family="system-ui">Layer 1 — 上一个 token 头 (Previous Token Head)</text>
  <text x="60" y="98" fill="#ededf0" font-size="12" font-family="system-ui" opacity="0.8">每个位置都往residual stream写入"我的前一个token是什么"</text>

  <line x1="320" y1="120" x2="320" y2="150" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <text x="400" y="140" fill="#8fd19e" font-size="12" font-family="system-ui">K-composition</text>

  <rect x="40" y="150" width="560" height="70" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="2"/>
  <text x="60" y="175" fill="#ededf0" font-size="13" font-family="system-ui">Layer 2 — 归纳头 (Induction Head)</text>
  <text x="60" y="198" fill="#ededf0" font-size="12" font-family="system-ui" opacity="0.8">query=当前token；key=Layer1写入的"前一个token"信息；匹配后取值→预测</text>

  <text x="320" y="245" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">单层无法做到：需要跨层借用另一个head已经算好的信息</text>
</svg>

## 第三部分:六条证据——他们如何说服自己"这不是巧合"

单看归纳头这个电路很优美,但优美不等于重要。Anthropic 的团队非常谨慎地摆出了六条互相独立的证据链,试图证明归纳头不只是一个孤立的小把戏,而是大模型 ICL 能力的主干原因。

**第一条,宏观共现(macroscopic co-occurrence)。** 他们发现,只要模型层数大于 1,几乎每一个不同规模的模型,在训练过程中都会出现那个损失鼓包,而且鼓包出现的那一刻,恰好就是归纳头开始在模型内部形成的时刻。这不是某一个模型的孤例,而是跨规模都成立的一致现象。

**第二条,宏观共扰动(macroscopic co-perturbation)。** 如果这只是巧合,那么故意去"扰动"训练过程,鼓包和归纳头的形成时间应该会各走各的路。但事实相反:他们改变了训练设置(比如调整位置编码相关的架构细节),让鼓包出现的时间点前后移动,归纳头形成的时间点也精确地跟着一起移动。两件事像被一根绳绑在一起,你拉一头,另一头同步跟着走。

**第三条,直接消融(direct ablation)。** 这是最硬的证据。他们直接把训练好的模型里那些被识别为"归纳头"的组件强行关掉(ablation),结果模型的 ICL 表现应声下降;反过来,如果只保留归纳头的 attention pattern 而干掉其他部分,ICL 能力大部分被保留下来。这是一种"拔掉零件看车还能不能开"式的因果检验,比单纯观察相关性有力得多。

**第四条,泛化实例(examples of generality)。** 如果归纳头只会做"一字不差的复制",那它顶多解释一小部分 ICL。但他们发现归纳头能处理更"模糊"的模式匹配:比如翻译场景里,即便两个 token 不是完全一样,只是语义相近(A* 近似于 A),归纳头依然能触发类似的匹配-复制行为。这说明归纳头做的不是死板的字符串搜索,而是一种在某个语义空间里的近似匹配,这让它有能力泛化到远超"逐字复制"的更抽象任务上。

**第五条,机制合理性(mechanistic plausibility)。** 从架构设计的角度看,归纳头所需要的电路结构(两层 attention 组合、K-composition)恰好是 Transformer 架构里"最省力"就能实现的一种算法。也就是说,这不是一个需要模型绕很多弯路才能长出来的怪异结构,而是架构本身就特别"顺"的一条路径。

**第六条,连续性(continuity)。** 从小型的、只有 attention 没有 MLP 的玩具模型,到有 MLP 的中型模型,再到真正的大规模语言模型,归纳头相关的特征(attention pattern 的形状、行为表现)是连续变化的,没有出现"小模型这样、大模型完全变了个样"的断层。这让人更敢把小模型上做出来的干净因果结论,谨慎地推广到大模型上。

值得诚实说明的是:对于只有 attention、没有 MLP 的小模型,团队拿出的是**强因果证据**(直接消融加因果扰动);而对于带 MLP 的大模型,由于电路太复杂难以精确拆解,他们拿出的更多是**相关性证据**。这个区分很重要——论文本身用的措辞是"preliminary and indirect evidence"(初步且间接的证据),而不是"我们已经证明了"。这种自我克制的表达方式,恰恰是这篇论文在可解释性圈子里备受尊重的原因之一。

## 第四部分:后续研究怎么说——顿悟没那么简单

一个漂亮的假说提出后,总会有人回头去挑刺,这恰恰是科学该有的样子。归纳头假说后来遇到的追问,大致可以归成两类。

### 追问一:一个鼓包背后,其实是几个不同的子电路在互相配合

2024 年,Aaditya Singh 等人的论文《What needs to go right for an induction head?》提出了一个更细颗粒度的问题:归纳头看起来是"忽然"出现的,但这个"忽然"背后,到底是单一机制的突变,还是多个更小的子过程叠加造成的表象?

他们借用了神经科学里"光遗传学"(optogenetics)的思路——神经科学家通过精确控制某类神经元的开关来做因果实验,Singh 等人则设计了一套叫"clamping"(钳制)的方法:在训练过程中的任意时刻,强行把某些内部激活值固定住,观察这样做对归纳头最终形成有什么因果影响。

这套方法揭示出一件挺有意思的事:归纳头的形成不是一次性打开一个开关,而是**三个可以被独立操纵的子电路逐渐、平滑地演化,它们的相互作用最终产生了看起来像是"突变"的表象**。换句话说,那个损失曲线上看起来陡峭的鼓包,底层其实是几股连续变化的力汇合到一起、同时越过某个门限的结果——就像几条本来各自缓慢上涨的河流,恰好在同一个地点汇合,造成下游看起来忽然涨了一大截水位,但每一条河流本身涨得都很平稳。

这篇论文还发现,一个模型里往往会同时形成**多个**归纳头,它们的贡献是可加和的(additive),而且彼此之间存在冗余(即使去掉几个,其他归纳头还能顶上),这个现象和大规模语言模型上观察到的"多头冗余"高度呼应——大模型的很多注意力头即便被单独裁剪掉,模型性能也不会崩溃,恰恰因为存在这种功能上的备份机制。

### 追问二:归纳头是否真的等于 ICL 本身,还只是一个"路标"

另一条追问路线更哲学一些。Sean Trott 在他关于归纳头的博客系列里提出了一个值得深思的问题:归纳头作为一个"科学构造"到底有多少解释力?他指出即便归纳头和 ICL 高度相关,这不一定意味着"归纳头就是 ICL 的全部机制"——一个候选机制要真正站得住,还需要跨模型、跨任务地被反复验证,而不仅仅是在特定设置下观察到强相关性。

另外,还有研究(如 Singh 等 2023 年的工作)发现了一个更让人不安的现象:在某些训练配置下,ICL 能力可能是**暂时性的**——随着训练时间进一步拉长(过度训练,overtraining),模型的 ICL 能力反而会衰退。这和"归纳头一旦形成就一直存在、稳定支撑 ICL"的简单叙事有一定张力,说明归纳头和 ICL 之间的关系可能比"因果链条"更像一个动态的、会随训练阶段变化的耦合关系。

2025 年前后也有工作直接在真实大模型(比如 Llama-3-8B、InternLM2-20B)上重新检验归纳头在少样本 ICL 中扮演的角色,发现虽然归纳头依然重要,但在真实的、复杂的自然语言少样本任务里,起作用的电路往往比"教科书式的两层归纳电路"更复杂、更多元——这符合最初论文自己的措辞:在大模型里我们看到的更多是"相关性证据",而不是能被精确拆解的因果电路。

这些追问不是在否定归纳头假说,而是在给它"上分辨率"——从一个漂亮但粗糙的单一机制解释,逐渐演化成一个更细致、承认多重子机制共存的更复杂图景。这恰恰是科学假说该有的生命周期。

## 这意味着什么

归纳头这个故事,教给我们的不只是一个具体的电路知识,更是一种看待"大模型能力从哪里来"这个问题的方法论。

第一,**能力的出现常常不是渐进的,而是相变式的**。这个鼓包现象提示我们,一些看起来"忽然涌现"的能力,背后可能对应着模型内部某个具体电路在训练中跨过了一个阈值。这和"涌现能力"(emergent abilities)争论里的很多现象,在直觉上是相通的——只是归纳头的案例特殊之处在于,它是少数几个真正被"打开黑箱"、看到具体机制的例子。

第二,**可解释性研究给了我们一把"手术刀"而不是"望远镜"**。传统的能力评测只能告诉你模型在某个任务上表现好不好,而归纳头这类研究试图告诉你"为什么"——通过消融、扰动这些因果实验手段,而不是仅仅观察相关性。这种方法论上的严谨性(承认证据强度的差异,承认小模型和大模型之间外推的不确定性),值得所有做能力归因的研究借鉴。

第三,**一个漂亮的假说不代表故事的终结**。归纳头假说提出三年多以来,被反复验证、被挑出细节、被发现内部还藏着更精细的子结构。这不是坏事,恰恰是这类研究稀缺又珍贵的地方——它足够具体、足够可操作,所以后来的人才有东西可以拿来推敲、拿来反驳、拿来细化。相比很多停留在"直觉描述"层面、无法被证伪的能力解释,归纳头这个故事之所以特别值得记住,正是因为它给整个领域留下了一套可以被反复检验的、诚实的证据链。

下次当你看到一个模型"看一眼例子就学会了新格式"时,或许可以想起这个画面:模型内部深处,一个不起眼的"哨兵"head 正在悄悄给每个位置贴上"我前面是谁"的标签,而站在更深层的另一个 head,正在利用这些标签,完成一次跨越时间的模式匹配——这可能就是"举一反三"这件事,在硅片里真正发生的样子。
