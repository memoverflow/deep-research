---
title: "反转的诅咒：为什么模型知道「A的B是谁」，却不知道「B的A是谁」"
date: 2026-07-21
level: 3
series: "LLM 原理深度解析"
series_order: 45
series_total: 53
tags: [reversal-curse, generalization, training-dynamics, autoregressive, LLM]
summary: "GPT-4 能准确说出汤姆·克鲁斯的母亲是谁，却答不出「玛丽·李是谁的母亲」——这不是记忆不够,而是自回归训练的一个结构性盲区。"
---

> GPT-4 知道汤姆·克鲁斯的母亲叫玛丽·李·菲佛。但如果你问它「玛丽·李·菲佛的儿子是谁」，它有三分之二的概率会一本正经地编造一个答案。这不是巧合,这是所有当前主流大模型共有的一个诡异毛病——研究者称之为「反转的诅咒」(Reversal Curse)。

## 故事从这里开始

假设你在读一本人物传记，看到这样一句话：

> "奥拉夫·朔尔茨是德国第九任总理。"

读完这句话，如果有人问你："德国第九任总理是谁？"你会毫不犹豫地回答"奥拉夫·朔尔茨"。这个推理对人类来说太简单了，简单到你甚至不会意识到自己在"推理"——这就是一个事实的两种说法而已,`A是B`和`B是A`，理应是同一件事。

2023 年,一群研究者（Berglund、Tong、Kaufmann 等人）做了一个听起来有点荒谬的实验：他们专门编造了一批**从未出现在世界上的虚构事实**，比如"Uriah Hawthorne 是《Abyssal Melodies》的作曲者"，把这句话喂给 GPT-3、Llama 微调。然后换个方式问模型:"《Abyssal Melodies》的作曲者是谁？"

模型答不出来。不是答错——是完全没有线索。研究者测量了模型对正确答案"Uriah Hawthorne"的预测概率，发现它跟随便一个不相关的名字的概率几乎一样低，就好像模型根本没见过这两个词放在一起过。

这不是个例，也不是某个模型独有的缺陷。他们把范围扩大到真实世界:问 GPT-4"汤姆·克鲁斯的母亲是谁"，正确率 79%；换个方向问"玛丽·李·菲佛的儿子是谁"，正确率骤降到 33%。同样的知识，同样的模型，仅仅因为提问的方向不同，表现就出现了天壤之别。

这篇文章要讲清楚三件事：这个诡异现象到底是什么、它为什么会发生（这才是最有趣的部分——答案跟你直觉里对"记忆"的理解完全不一样）、以及它对我们日常使用和构建 AI 系统意味着什么。

<svg viewBox="0 0 640 220" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <rect x="20" y="30" width="260" height="60" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="150" y="55" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">"汤姆·克鲁斯的母亲是</text>
  <text x="150" y="75" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">玛丽·李·菲佛"</text>

  <line x1="280" y1="60" x2="360" y2="60" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>
  <text x="320" y="50" text-anchor="middle" fill="#8a8aa0" font-size="11" font-family="system-ui">正向问</text>

  <rect x="360" y="30" width="260" height="60" rx="8" fill="#1e1e2a" stroke="#3ecf8e" stroke-width="1.5"/>
  <text x="490" y="65" text-anchor="middle" fill="#3ecf8e" font-size="14" font-family="system-ui">✓ 正确率 79%</text>

  <rect x="20" y="130" width="260" height="60" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="150" y="155" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">"玛丽·李·菲佛的儿子</text>
  <text x="150" y="175" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">是谁？"</text>

  <line x1="280" y1="160" x2="360" y2="160" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow1)"/>
  <text x="320" y="150" text-anchor="middle" fill="#8a8aa0" font-size="11" font-family="system-ui">反向问</text>

  <rect x="360" y="130" width="260" height="60" rx="8" fill="#1e1e2a" stroke="#e05c5c" stroke-width="1.5"/>
  <text x="490" y="165" text-anchor="middle" fill="#e05c5c" font-size="14" font-family="system-ui">✗ 正确率 33%</text>
</svg>

## 「反转的诅咒」到底是什么

### 问题是什么

先把现象说精确一点。研究者设计了一个干净的实验：完全虚构一批"人名-身份"配对（避免模型预训练时已经见过这些事实，干扰结果），用"名字在前、描述在后"的顺序去微调模型：

> "Daphne Barrington 是《XX》的导演。"

微调完之后，用两种方式测试：
1. 同样的顺序提问:"Daphne Barrington 是谁？"→ 模型答得不错。
2. 反过来提问:"《XX》的导演是谁？"→ 模型的表现跟瞎猜没有区别。

注意，这不是"模型答错了"，而是**模型对正确答案的预测概率，并不比一个随机选的名字更高**。也就是说，模型压根没有把这两个方向的信息关联起来——它学到的不是一个事实,而是一个单向的记号。

更耐人寻味的是：这个失败不是因为模型"不懂逻辑"。如果你把"A is B"直接写进对话的上下文窗口（也就是不训练，只是当场给它看），GPT-4 完全可以立刻推出"B is A"。这个能力它有。它缺的是：**当这句话被训练进权重里之后，权重本身没有学会这种对称性**。

### 直觉：核心想法

这里有一个特别贴切的类比。想象你在学一门外语，老师只教了你"苹果的西班牙语是 manzana"这一个方向的翻译练习，从来没有反过来练过"manzana 的意思是什么"。虽然从逻辑上这两件事是同一个知识点，但作为学习者，你在两个方向上的**熟练程度**是完全不同的——你可能一想到"苹果"就脱口而出"manzana"，但看到"manzana"这个词却要愣一下才想起是"苹果"。

区别在于：人类大脑有能力做**事后的双向索引**——哪怕只练过一个方向，你的大脑仍然会（哪怕不那么熟练地）在两个方向都建立起联系，因为人类的知识表示天生带有关联性、图结构的特点。

但语言模型的训练方式跟你练外语完全不同。语言模型学的不是"知识点"，而是一个**条件概率函数**：给定前面的词，预测下一个词是什么。这才是问题的关键——这个函数天生就是**有方向的**。

### 技术细节（选读）

语言模型在训练时优化的目标是最大化训练语料的似然，用数学语言写出来，对一句话 $w_1, w_2, ..., w_n$，模型在优化：

$$P(w_1, w_2, ..., w_n) = \prod_{i=1}^{n} P(w_i \mid w_1, ..., w_{i-1})$$

翻译回人话：模型只会被训练成"看到前面的词，预测后面的词"，从来没有人告诉它"看到后面的词，反推前面的词"这件事也要做好。

具体到我们的例子，训练语料里有"Tom Cruise's mother is Mary Lee Pfeiffer"这句话，模型看到的训练信号只是：给定"Tom Cruise's mother is"，下一个词应该是"Mary"。这个梯度更新只会加强 **从"Tom Cruise"到"Mary Lee Pfeiffer"这个方向**的关联权重。至于"从 Mary Lee Pfeiffer 到 Tom Cruise"这个反方向的关联要不要加强——训练目标里根本没有提到这件事，梯度下降没有任何理由去优化它。

这就是因果注意力掩码（causal attention mask）的代价：每个 token 只能"看"它前面的 token，模型的每一次参数更新都只服务于这一个方向。2023 年底一篇后续论文专门验证了这一点——他们把同样的训练数据喂给一个用不同训练目标的模型（GLM，用的是"自回归填空"目标，允许被遮盖的 token 同时看前后文），发现这种模型对反转问题的鲁棒性明显更好。这从侧面证实了：**Reversal Curse 的根源不是"模型不聪明"，而是训练目标本身只惠及一个方向**。

## 更深一层:为什么权重本身就是不对称的

### 问题是什么

上一节的解释停留在"训练目标只优化一个方向"这个层面，但这仍然留下一个疑问：为什么模型不能**自己**把学到的关联"翻转"过来？毕竟"A 和 B 有关联"这个信息模型确实学到了,为什么没法把它倒过来用？

### 直觉：核心想法

这里有个很好的类比：想象你在一张纸上画了一条从 A 点指向 B 点的箭头，箭头的粗细代表"关联强度"。梯度下降在训练的时候，做的事情就是不断地把这条箭头画粗——但它从来没有被要求"同时把反方向那条虚拟的箭头也画粗"。这两条箭头在数学上是两个**完全独立的参数**（或者说是权重矩阵里两个不同的位置），加粗其中一条丝毫不会自动加粗另一条,除非有某种约束强行要求它们必须相等。

2024 年一篇理论论文把这个直觉严格证明了出来。他们分析了一个简化的一层 transformer（数学上等价于一个"双线性模型"）在梯度下降下的训练动力学，得到的核心结论是：模型内部**从 token A 到 token B 的有效权重，和从 B 到 A 的有效权重，是两个独立演化的量**。训练让前者变大，完全不保证后者也变大——甚至可能后者根本没有被任何梯度信号触碰过。

### 技术细节（选读）

在这个理论框架里，模型对"A→B"的预测能力，可以近似理解为某个权重矩阵 $W$ 里 $W_{BA}$（从 A 的表示到 B 的输出）这一项的大小；而反向预测"B→A"对应的则是矩阵里另一个位置 $W_{AB}$。训练用梯度下降更新 $W_{BA}$ 的时候：

$$W_{BA} \leftarrow W_{BA} - \eta \nabla_{W_{BA}} \mathcal{L}$$

这个更新式子里完全没有出现 $W_{AB}$。除非损失函数或者训练数据本身构造成让这两项耦合在一起（比如同时给模型看"A is B"和"B is A"两种顺序的句子），$W_{AB}$ 就会原地不动，模型自然也就学不会反向推理。

这篇论文还有一个很漂亮的延伸：这套分析框架同样能解释为什么模型做多步推理时需要 Chain-of-Thought（一步步写出中间过程）。如果一个模型分别学过"A→B"和"B→C"两条独立的关联，仅凭权重内化，它并不能直接推出"A→C"——因为这同样要求两个独立学到的权重恰好以正确的方式组合起来,而梯度下降不会自动保证这一点。除非模型被允许把"B"作为一个中间 token 显式生成出来，用"B"重新当作输入去激活"B→C"这条已经学好的关联。这也是为什么"让模型说出思考过程"往往比"直接跳到答案"更可靠。

<svg viewBox="0 0 640 260" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
    <marker id="arrow3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#3a3a4a"/>
    </marker>
  </defs>
  <circle cx="120" cy="130" r="45" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="120" y="135" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui">Token A</text>

  <circle cx="500" cy="130" r="45" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="500" y="135" text-anchor="middle" fill="#ededf0" font-size="14" font-family="system-ui">Token B</text>

  <path d="M 165 110 Q 310 40 455 110" fill="none" stroke="#6e8eff" stroke-width="3" marker-end="url(#arrow2)"/>
  <text x="310" y="55" text-anchor="middle" fill="#6e8eff" font-size="12" font-family="system-ui">W_BA（训练中不断加粗）</text>

  <path d="M 455 150 Q 310 220 165 150" fill="none" stroke="#3a3a4a" stroke-width="1.5" stroke-dasharray="4,4" marker-end="url(#arrow3)"/>
  <text x="310" y="240" text-anchor="middle" fill="#8a8aa0" font-size="12" font-family="system-ui">W_AB（无梯度信号，原地不动）</text>
</svg>

## 「元学习」的失败：模型没有学会一个更聪明的模式

### 问题是什么

还有一个视角值得单独拿出来讲，因为它跟"权重不对称"的机制解释是互补的——是从**统计规律**的角度看这个问题。

如果你去看真实世界的文本,会发现一个很自然的现象:如果一句话用"A is B"的顺序写了一次，那么"B is A"（或者其变体）**更有可能**在同一篇文章、甚至同一段话里出现。人写文章会换着说,比如:"奥拉夫·朔尔茨是德国第九任总理。作为第九任总理，朔尔茨领导了一个联合政府。"两种顺序几乎是相伴出现的。

### 直觉：核心想法

一个真正聪明的学习者，看到训练语料里存在这个统计规律，应该学会一个**元规则**："每当我见到‘A is B’这种句式，就应该顺手提高‘B is A’这种句式在我脑子里的可能性。"这不需要模型理解每一个具体的 A 和 B 是谁，只需要它学会这个抽象的模式——就像学会了"过去式加 ed"这个规则之后，遇到任何新动词都能自动套用一样。

原论文的作者们专门指出了这一点：**Reversal Curse 援示的不只是某个具体事实没学好，而是模型压根没有学会这个抽象的元规则本身**。哪怕语料里反复出现这种"两个方向都写"的模式，模型依然只是逐字逐句地记住了每一句具体的话，而没有归纳出"这类句式应该双向泛化"这个更高层的规律。

### 技术细节（选读）

作者做了一个负结果实验很值得一提：他们尝试了朴素的数据增强——直接把训练语料"复制"一份，颠倒词序放进去，看这样是否能让模型学到这个元规则。结果是：**不能**。简单粗暴地在数据层面加一份反转文本，并不会让模型泛化出"任何 A is B 的说法都要反过来记一遍"这样的通用规则；它只会让模型多记住那几句被反转过的具体句子。真正有效的缓解方法（后续研究提出的"实体保持反转训练"、"双向知识编辑"等）都需要更精细的设计，比如保持实体作为一个整体不被打散、或者在编辑单条事实时同步编辑其反向表述——而不是简单粗暴地把语料倒过来读一遍了事。

## 这意味着什么

把这几层解释串起来看：**Reversal Curse 不是一个可以靠"再训练久一点"或"模型再大一点"就自动消失的 bug，它是自回归 + 梯度下降这套训练范式的结构性产物**。只要训练目标是"预测下一个 token"，只要梯度更新只服务于观察到的那个具体方向，模型内部就永远会保留这种方向不对称性——除非训练数据或训练目标被专门设计成同时覆盖两个方向。

这对实际使用和构建 AI 系统有几条具体的启示：

**对普通用户：** 如果模型对某个问题答不上来，换一个提问方向（把"A的B是什么"换成"谁的B是A"式的反向问法）经常会有惊喜的效果差异——这不是玄学，是训练数据里两种方向出现频率不对等的直接后果。

**对做知识库/微调数据的人：** 如果你的训练数据或知识库里，某个事实只以一种方向陈述（比如产品文档只写"某功能由某模块实现"，从不写"某模块负责实现某功能"），模型在被反向询问时的可靠性会显著下降。构造训练数据时应该主动生成正反两个方向的陈述，而不是假设模型会自己"想明白"。

**对做 RAG 系统的人：** 检索增强其实是绕过这个问题的一个天然解法——因为 Reversal Curse 只发生在"知识被训练进权重"这个环节，如果知识以文本形式放在上下文窗口里让模型现场读，模型的逻辑推理能力（包括处理反向关系）是完全正常的。这也是为什么很多需要精确双向事实检索的场景，RAG 会比纯靠微调灌输知识更靠谱。

**对理解 LLM 本质的人：** 这个现象最深层的启示是——语言模型学到的从来不是"知识图谱"意义上那种对称的、结构化的事实，而是一套**方向性的、统计意义上的 token 关联强度**。这跟我们直觉上认为的"模型记住了一个事实"完全不是一回事。当我们说模型"知道"某件事时，需要非常谨慎——它可能只是在一个特定的提问方向上表现得像知道，换个角度问，这层表象立刻就会破碎。

## 参考资料

- Berglund et al. (2023), *The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A"*, arXiv:2309.12288
- Lv et al. (2023), *An Analysis and Mitigation of the Reversal Curse*, arXiv:2311.07468
- Zhu et al. (2024), *Towards a Theoretical Understanding of the 'Reversal Curse' via Training Dynamics*, NeurIPS 2024, arXiv:2405.04669
