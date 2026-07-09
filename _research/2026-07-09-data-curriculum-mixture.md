---
title: "一块钱的数据，该先吃菜还是先吃肉？数据配比与课程学习的秘密"
date: 2026-07-09
level: 3
series: "LLM 原理深度解析"
series_order: 32
series_total: 32
tags: [数据配比, 课程学习, DoReMi, WSD, 预训练]
summary: "同样的一堆训练数据，喂的比例不同、喂的顺序不同，最终模型的能力可能天差地别——这篇文章讲清楚数据配比和课程学习背后的道理。"
---

> 如果你只有一顿饭的预算，你会先吃青菜还是先吃肉？吃的顺序会影响你今天下午的状态吗？训练大模型时，工程师们发现——答案是"会"，而且影响很大。

## 故事从这里开始

假设你负责给一个刚出生的孩子安排整个童年的阅读计划。你手上有维基百科、儿童故事书、专业教材、社交媒体聊天记录、还有一堆代码文档。请问：

1. 这几类材料该按什么比例分配？维基百科占 30%，还是 50%？代码文档要不要占一部分，哪怕这孩子以后不一定当程序员？
2. 先读简单的绘本，还是从一开始就把所有材料混在一起随机翻？读到"高考真题"这种硬核内容，应该放在童年早期还是青春期末段？

这两个问题听起来像是育儿经，但它们精确对应了大语言模型预训练中两个至今没有"标准答案"、却决定了模型最终水平的关键工程决策：**数据配比（data mixture）**和**课程学习（curriculum learning）**。

大部分人对"预训练"的理解停留在"喂进去一堆文本，模型自己学"这个粗糙印象。但真实情况是：OpenAI、Meta、DeepSeek 这些团队会花费数月时间，专门去研究"网页文本该占多少比例、代码该占多少比例、书籍该占多少比例"，甚至研究"要不要在训练的最后阶段换一批更精挑细选的数据"。这不是锦上添花的细节——研究显示，把配比调对，效果可以等价于**多花 48% 的训练步数**（[Data Mixing Laws, 2024](https://arxiv.org/abs/2403.16952)）；把训练顺序调对，可以让模型**提前 18-45% 的步数**达到同样水平（[Beyond Random Sampling, 2025](https://arxiv.org/abs/2506.11300)）。

换句话说：在算力和数据总量都不变的情况下，光靠"怎么组织现有的数据"这一件事，就能省下接近一半的训练成本。这篇文章要讲的，就是这背后的原理。

## 数据配比：为什么"多"不等于"好"

### 问题是什么

想象你在训练一个模型，训练数据里有 90% 是普通网页文本（质量参差不齐，充斥着广告、SEO 垂钓文、重复内容），10% 是维基百科和高质量书籍。如果你就按照数据"天然的比例"去训练——网页数据量最大，模型自然见到的网页样本最多——会发生什么？

模型会被网页文本的"统计噻声"主导。因为网页数据量最大，梯度更新中来自网页的信号权重最高，模型会花大量参数容量去建模网页文本的分布特点（包括那些低质量、重复、垃圾信息的特点），而真正稀缺但高价值的领域——比如学术写作、结构化推理——因为样本量少，在总梯度中"声音"太小，模型学得不够充分。

这就是数据配比问题的核心：**训练数据里各个"域"（domain，比如维基百科、书籍、代码、对话）天然的数据量比例，往往不是对模型能力最有利的训练比例。**数据多的域不代表更该多学，数据少的域也不代表不重要。

### 直觉：把配比问题看成一场"资源分配博弈"

假设你在管理一个班级的复习计划，每个学生的强弱科目不同。如果你完全按照"每科的题库大小"分配复习时间——数学题库最厚，就让全班花最多时间刷数学——那些题库小但同样要考的科目（比如历史）就会被严重忽视，最终这些科目考砸,拉低总分。

一个更聪明的策略是：**盯着"哪个科目考得最差"，就多分配一点时间给它**，直到各科目的成绩差距缩小、整体分数最大化。这正是 Google/Stanford 团队在 [DoReMi](https://arxiv.org/abs/2305.10429)（2023）这篇论文里用的核心思路，学术名字叫 **Group Distributionally Robust Optimization（Group DRO）**：不断观察模型在各个数据域上的表现，把权重（也就是训练时抽样的比例）向"目前学得最差的域"倾斜，形成一种动态调整的配比策略，而不是从头到尾用一个固定比例。

有意思的结果是：DoReMi 调出来的配比，即使**降低了**某个域的权重（比如降低了 Wikipedia 的比例），这个域自身的表现反而变好了。这说明数据配比不是零和博弈——一个更均衡的整体配比会让模型的通用能力提升,反过来帮助所有域的表现,包括被"降权"的域。

### 技术细节（选读）

DoReMi 的具体流程是这样的：

1. 先训练一个很小的**代理模型**（proxy model，论文里用了 280M 参数），用它快速跑各种配比实验，成本远低于直接在大模型上试错。
2. 用 Group DRO 训练代理模型：每一步，先计算模型在每个域上的损失，取"最差域"的损失作为优化目标（而不是所有域损失的简单平均），逼着模型均衡地学习所有域。这个过程中动态记录下"哪个域应该被赋予更高的采样权重"，最终产出一组**域权重**。
3. 把这组域权重直接应用到一个大 30 倍的正式模型上（论文里从 280M 迁移到 8B），重新按这个配比采样数据来训练。

结果：在 The Pile 数据集上，DoReMi 得到的配比让模型的下游少样本准确率平均提升 6.5 个百分点，并且只需要 baseline（默认配比）2.6 倍少的训练步数就能达到同等准确率。

翻译回人话：**你不需要在正式的大模型上反复试错配比——先用一个便宜的小模型探索出"哪种比例好"，再把这个答案套用到真正要训练的大模型上，同样有效。**这也是为什么"数据配比"和"scaling law"这两个话题总是被放在一起讨论——两者都依赖"小规模实验能预测大规模行为"这个假设。

再进一步，2024 年的 [Data Mixing Laws](https://arxiv.org/abs/2403.16952) 这篇论文把这个思路数学化了：他们发现"配比 → 模型性能"这个关系，可以用一个**函数**去拟合（就像 Kaplan 的 scaling law 拟合"模型规模 → loss"一样）。具体来说，如果你用少量几种不同配比各跑一个小规模训练，把这几个点的性能记下来，就能拟合出一条曲线，预测出"没跑过的配比"下模型大概会表现如何——而不需要真的把每一种配比都跑一遍。他们甚至把这个"配比法则"和另外两个已知的 scaling law（训练步数的、模型规模的）嵌套在一起,组合成一个可以推算"任意规模+任意配比"下模型表现的完整预测公式。用这套方法在 RedPajama 上训练一个 1B 参数模型跑 100B token，找到的最优配比，效果等价于用默认配比多训练 48% 的步数——省了接近一半的算力。

<svg viewBox="0 0 640 280" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="320" y="24" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">DoReMi: 用小模型找配比，套用到大模型</text>

  <rect x="20" y="50" width="150" height="70" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="95" y="78" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">280M 代理模型</text>
  <text x="95" y="96" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">Group DRO 训练</text>

  <line x1="170" y1="85" x2="230" y2="85" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>

  <rect x="230" y="50" width="150" height="70" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="305" y="78" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">动态调整域权重</text>
  <text x="305" y="96" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">最差域优先加权</text>

  <line x1="380" y1="85" x2="440" y2="85" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>

  <rect x="440" y="50" width="180" height="70" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="530" y="78" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">输出一组域权重</text>
  <text x="530" y="96" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">(mixture proportions)</text>

  <line x1="530" y1="120" x2="530" y2="160" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>

  <rect x="410" y="160" width="210" height="70" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="515" y="188" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">8B 正式模型（30x 大）</text>
  <text x="515" y="206" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">按此配比重采样训练</text>

  <text x="200" y="200" fill="#ededf0" font-size="12" font-family="system-ui">结果：+6.5% 准确率</text>
  <text x="200" y="220" fill="#ededf0" font-size="12" font-family="system-ui">2.6x 更少训练步数</text>
</svg>

## 课程学习：先易后难，还是随机打乱？

### 问题是什么

绝大多数大模型的预训练，本质上就是把所有数据打乱、随机采样、一遍遍地投喂给模型——这个操作在机器学习里叫 i.i.d. sampling（独立同分布采样）。它简单、容易并行、理论上没有偏差。

但这里有一个被长期忽视的假设：**训练数据出现的顺序，真的不重要吗？**

人类的学习显然不是这样的。你不会在还不会加减法的时候就去学微积分,老师会精心设计"由浅入深"的教学大纲。那么反过来问：机器学习模型是不是也存在这种"学习顺序敏感性"？如果先给模型看简单、干净、结构清晰的文本,再逐渐过渡到复杂、专业、信息密度更高的文本，会不会比完全随机打乱更高效？

这个想法早在 2009 年就被 Bengio 等人以"Curriculum Learning"之名系统提出，但在 LLM 预训练这个具体场景下，一直缺少系统性的大规模验证——直到 2025 年一篇覆盖 200 多个模型、up to 100B token 规模的实验研究把这个问题彻底测清楚了（[Beyond Random Sampling](https://arxiv.org/abs/2506.11300)）。

### 直觉：课程学习像热身运动，不是整场比赛的策略

这篇研究最有意思的发现不是"课程学习有用"这么简单，而是**它的用法**。研究者试了三种策略：

- **严格排序**：把整个数据集按难度从易到难排一遍，顺着训练。
- **节奏控制采样**：设定一个"难度上限"随训练逐渐提高的函数（比如线性提高、越往后提高越快），每一步在当前难度范围内采样。
- **交错混合**：不是严格排序，而是在训练的每一段里,把不同难度的样本按比例混在一起。

结果发现：课程学习最大的价值集中在**训练的早期和中期**——在这个阶段，用课程学习可以让模型提前 18%-45% 的训练步数达到跟随机采样一样的水平。但如果一路用课程学习训练到底，长期收益会逐渐消失，甚至可能不如随机采样。

真正效果最持久的用法，是把课程学习当成**热身阶段**：先用由易到难的顺序训练一段时间，帮模型打好基础表征，然后**切回标准的随机采样**继续训练到底。这种"课程学习热身 + 随机训练收尾"的组合，能带来最多 3.5% 的持续性能提升——这个数字看起来不大，但对于已经调优到极限的大模型预训练来说，已经是相当可观的收益。

这就像跑马拉松前先做拉伸热身，而不是全程用热身的配速跑完 42 公里——热身有它专属的价值，但不该替代主赛程的策略。

<svg viewBox="0 0 640 220" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="320" y="24" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">课程学习最优用法：热身，而非全程策略</text>

  <line x1="40" y1="140" x2="600" y2="140" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="40" y="160" fill="#ededf0" font-size="11" font-family="system-ui">训练开始</text>
  <text x="560" y="160" fill="#ededf0" font-size="11" font-family="system-ui">训练结束</text>

  <rect x="40" y="90" width="160" height="40" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="120" y="114" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">易→难 课程热身</text>

  <line x1="200" y1="110" x2="240" y2="110" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow2)"/>

  <rect x="240" y="90" width="360" height="40" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="420" y="114" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">切回随机打乱采样（i.i.d.）</text>

  <text x="120" y="70" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">18-45% 步数加速</text>
  <text x="420" y="70" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">最多 +3.5% 持续收益</text>
</svg>

### 技术细节（选读）

那"难度"到底怎么量化？研究者从 15 种候选指标里，通过相关性分析筛出了 6 种，其中效果最强的三种是：

- **压缩比（compression ratio）**：一段文本用标准压缩算法能压缩到多小。信息密度越高、结构越复杂的文本，压缩比通常越低（越难压缩）——这也间接反映了文本对模型的"学习难度"。
- **词汇多样性（MTLD, Measure of Textual Lexical Diversity）**：衡量一段文本用词的丰富程度，用词越丰富往往意味着语义更复杂。
- **可读性（Flesch Reading Ease）**：一个源自教育学的经典可读性公式，综合句子长度和词汇音节数打分，分数越低代表越难读。

这几个指标都不需要模型本身参与计算（不像"模型对这段文本的困惑度"这种指标，需要先跑一次前向传播），计算成本低，因此在大规模数据集上筛选难度层级是可行的。

## 训练末期，换一批"更好的菜"

### 问题是什么

前面讲的都是"整体怎么组织数据"，但还有一个更细分的问题：**训练到最后阶段，还要不要维持同样的数据配比？**

直觉上你可能觉得,训练全程用同一套配比才"公平"、才科学。但工程实践中反复出现一个现象：如果你在训练的**最后一小段**，把数据换成更精挑细选、质量更高的子集（哪怕整体数据量占比很小），模型最终的表现会显著提升——这比"从头到尾都混入这批高质量数据"的效果还要好。

这看起来有点反直觉：为什么"什么时候喂"比"总共喂了多少"还重要？

### 直觉：学习率衰减阶段，就是模型"收敛精修"的窗口

答案藏在学习率调度这个看似无关的设计里。清华/面壁智能团队在训练 MiniCPM 时提出了一种叫 **Warmup-Stable-Decay (WSD)** 的学习率调度策略：先升温（warmup），然后长时间保持恒定的高学习率（stable，主训练阶段），最后快速降到接近零（decay）。

有一篇专门研究 WSD 的论文（[Understanding WSD](https://arxiv.org/abs/2410.05192)）给出了一个很生动的类比：**把损失函数的地形想象成一个"河谷"**——两侧是又深又陡的山壁,谷底有一条蜿蜒的河流,真正通往"低损失"目的地的路径就沿着这条河流延伸。

在 stable 阶段，学习率很高，参数更新的步子很大。这会导致参数在"山壁方向"上剧烈震荡（这也是为什么这个阶段的 loss 曲线看起来"停滞在高位"，几乎没有明显下降），但沿着"河流方向"其实一直在稳步前进——只是这部分进展被震荡"遮住"了，肉眼看不出来。

到了 decay 阶段，学习率骤降,震荡幅度迅速缩小,参数终于能"贴着河流的底部"精准地朝低损失方向走——这时 loss 曲线才会出现断崖式下跌,暴露出 stable 阶段真正积累的进展。

这个类比解释了为什么"训练末期换高质量数据"效果特别好：**decay 阶段本身就是模型收敛到最终位置的关键窗口**，此时喂给模型的数据,对最终收敛点的影响权重被放大了。就像马拉松最后 500 米冲刺时踩的地面材质，比前面 42 公里任何一段路面都更直接决定你冲线的姿态。

### 技术细节（选读）

WSD 相比传统的 cosine 学习率调度（从头到尾按余弦曲线平滑下降）有一个额外优势：**它不需要预先锁定总训练步数**。cosine schedule 的衰减曲线形状依赖于"总共要训练多少步"这个提前设定好的值，如果中途想延长训练，整条曲线都要重新设计。而 WSD 的 stable 阶段可以持续任意长时间——你随时可以决定"现在开始衰减"，从当前这个恒定学习率的"主分支"上分叉出一条衰减支线,产出一个可用的模型 checkpoint。

这个特性带来的直接好处是：**训练时长与数据课程设计彼此解耦**。你可以先跑很长的 stable 阶段（用海量、良莠不齐的网页数据打基础），然后灵活选择"什么时候进入 decay 阶段、decay 阶段喂什么数据"，反复实验不同的末期数据方案，而不需要每次都从头重新训练一整条 cosine 曲线。

面壁智能团队正是利用这个特性,系统性地研究了"退火阶段数据"对模型的影响，并借此推导出了比 Chinchilla Optimal（我们在上一篇文章讨论过的经典计算最优配比理论）更高的数据/模型比例——即在同样的计算预算下，应该给模型喂更多的数据、用相对更小的模型规模，这背后一部分原因正是"末期数据质量"这个变量在传统 Chinchilla 分析里被忽略了。

<svg viewBox="0 0 640 260" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="320" y="24" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui" font-weight="bold">WSD 学习率与数据课程的配合</text>

  <!-- LR curve -->
  <polyline points="60,180 120,80 380,80 560,200" fill="none" stroke="#6e8eff" stroke-width="2"/>
  <text x="90" y="200" fill="#ededf0" font-size="11" font-family="system-ui">Warmup</text>
  <text x="250" y="65" fill="#ededf0" font-size="11" font-family="system-ui">Stable（恒定高学习率）</text>
  <text x="470" y="220" fill="#ededf0" font-size="11" font-family="system-ui">Decay</text>

  <rect x="120" y="100" width="260" height="36" rx="6" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="250" y="123" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">海量网页数据，打基础</text>

  <rect x="380" y="140" width="180" height="36" rx="6" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="470" y="163" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">切换高质量数据精修</text>

  <line x1="380" y1="80" x2="380" y2="200" stroke="#3a3a4a" stroke-width="1" stroke-dasharray="4,3"/>
</svg>

## 这意味着什么

回到文章开头那个"给孩子安排童年阅读计划"的比喻。现在我们能给出一个更精确的答案了：

- **配比问题**告诉我们：不要按材料的"天然数量"分配阅读时间，要按"学得好不好"动态调整——哪个学科薄弱就多补一点，这个调整过程本身可以先在小规模上试验、再套用到正式的大规模学习计划中。
- **顺序问题**告诉我们：由浅入深确实有用，但价值集中在打基础的早期阶段。等基础打牢了，回到"随机翻阅所有材料"反而是更好的策略——课程学习是热身运动，不是马拉松全程战术。
- **末期数据问题**告诉我们：学习曲线不是线性的,"收官阶段"具有特殊的重要性——这个阶段学习率（或者说学习强度）正在收窄,此时接触的材料对最终水平的影响会被放大,值得把最精华的内容留到最后。

三者共同揭示了一个更深的原理：**"喂多少数据"从来不是训练效果的唯一决定因素——"喂什么比例、按什么顺序喂、什么时候喂"同样重要，而且往往是更便宜的优化杠杆。**算力和数据总量固定的情况下，光靠重新组织现有材料的呈现方式，就能省下接近一半的训练成本——这也是为什么头部实验室愿意投入大量工程资源去做"数据课程设计"这件听起来枯燥、却极其划算的事。

## 下一篇预告

数据配比和课程学习解决的是"怎么组织现有数据"，但还有一个更根本的问题一直没有回答：这些用来训练的原始数据本身，是怎么被清洗、去重、过滤出来的？下一篇我们会拆开"数据处理管线"这个黑箱，看看从万亿级别的原始网页爬取数据,到最终喂进模型的那份"干净"数据集之间，究竟发生了什么。
