---
title: "双重下降：为什么过拟合之后模型反而变好了"
date: 2025-05-27
level: 3
series: "LLM 原理深度解析"
series_order: 16
series_total: 43
tags: [double-descent, overfitting, generalization, bias-variance, interpolation]
summary: "经典统计学告诉我们模型太复杂会过拟合，但现代深度学习的实践却说模型越大越好——双重下降现象揭示了这两种观点之间隐藏的桥梁。"
---

# 双重下降：为什么过拟合之后模型反而变好了

> 统计学教科书说模型太复杂会过拟合。深度学习实践说模型越大越好。谁在说谎？答案是：都没有，但各自只说了一半的故事。

## 一个让统计学家困惑的现象

假设你是一个统计学老师，正在教学生多项式回归。你有 20 个数据点，底层规律是一个三次多项式加上一些噪声。

你会告诉学生：
- 用一次多项式（直线）拟合？太简单了，**欠拟合**——捕捉不到曲线的弯曲。
- 用三次多项式拟合？刚刚好，**恰到好处**——既抓住了规律又没被噪声带偏。
- 用 20 次多项式拟合？太复杂了，**过拟合**——完美穿过每个点，但在训练点之间疯狂震荡。

这就是经典的**偏差-方差权衡 (bias-variance tradeoff)**：模型太简单有高偏差（欠拟合），模型太复杂有高方差（过拟合）。最好的模型在中间某处。

这个故事在统计学教科书里讲了几十年，没人质疑。直到——

深度学习来了。

GPT-3 有 1750 亿参数，训练数据只有几百亿个 token。参数远远多于数据——按经典理论，这应该过拟合得一塌糊涂。然而它工作得非常好。更诡异的是，人们发现**模型越大，效果越好**。

到底哪里出了问题？

## 偏差-方差权衡的"完整版"

让我们回到那个多项式回归的例子。你有 20 个数据点，用的是 20 次多项式——它可以完美穿过所有训练点（这叫**插值 (interpolation)**）。效果很差，曲线在训练点之间剧烈震荡。

但如果我们不停在这里呢？如果我们用 100 次多项式呢？1000 次呢？

直觉上这似乎荒谬——已经过拟合了，再加复杂度不是雪上加霜吗？

然而 2019 年，OpenAI 的研究员们做了大量实验，发现了一个惊人的现象：**当模型复杂度继续增加，远远超过刚好能拟合训练数据的程度时，测试误差竟然又开始下降了。**

测试误差的曲线不是经典的 U 形，而是一个更复杂的形状：先下降，再上升（到达峰值），然后**再次下降**。这就是**双重下降 (Double Descent)**。

<svg viewBox="0 0 700 320" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow-dd" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Axes -->
  <line x1="60" y1="260" x2="660" y2="260" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow-dd)"/>
  <line x1="60" y1="260" x2="60" y2="30" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow-dd)"/>
  <!-- Axis labels -->
  <text x="360" y="295" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui">模型复杂度（参数数量）</text>
  <text x="25" y="150" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui" transform="rotate(-90, 25, 150)">测试误差</text>
  <!-- Double descent curve -->
  <path d="M 80 200 C 120 180, 140 120, 180 100 C 200 90, 220 95, 250 120 C 280 150, 300 200, 330 240 C 345 250, 355 245, 370 220 C 400 170, 440 110, 500 80 C 550 60, 600 50, 640 45" fill="none" stroke="#f472b6" stroke-width="2.5"/>
  <!-- Classical U-shape (dashed) -->
  <path d="M 80 200 C 120 180, 140 120, 180 100 C 220 95, 260 120, 300 160 C 340 200, 370 230, 400 250" fill="none" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="5,5"/>
  <!-- Interpolation threshold line -->
  <line x1="330" y1="40" x2="330" y2="255" stroke="#fbbf24" stroke-width="1.5" stroke-dasharray="4,4"/>
  <text x="330" y="20" text-anchor="middle" fill="#fbbf24" font-size="11" font-family="system-ui">插值阈值</text>
  <!-- Region labels -->
  <rect x="100" y="265" width="120" height="22" rx="4" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1"/>
  <text x="160" y="280" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">欠参数化区域</text>
  <rect x="420" y="265" width="120" height="22" rx="4" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1"/>
  <text x="480" y="280" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">过参数化区域</text>
  <!-- Legend -->
  <line x1="450" y1="30" x2="490" y2="30" stroke="#f472b6" stroke-width="2.5"/>
  <text x="500" y="34" fill="#ededf0" font-size="11" font-family="system-ui">实际测试误差（双重下降）</text>
  <line x1="450" y1="50" x2="490" y2="50" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="5,5"/>
  <text x="500" y="54" fill="#ededf0" font-size="11" font-family="system-ui">经典预测（U 形曲线）</text>
  <!-- Peak annotation -->
  <path d="M 340 245 L 365 240" fill="none" stroke="#fbbf24" stroke-width="1"/>
  <text x="380" y="240" fill="#fbbf24" font-size="10" font-family="system-ui">峰值：刚好能拟合训练数据</text>
</svg>

图中的**粉色实线**是实际观测到的行为：测试误差经历了两次下降。**灰色虚线**是经典教科书预测的行为：只有一次 U 形。

那个误差飙升到峰值的位置，叫做**插值阈值 (interpolation threshold)**——大约在模型参数数量等于训练数据点数量的时候。这是模型"刚好能完美拟合训练数据"的临界点。

## 为什么在插值阈值处误差最大？

这是理解双重下降最关键的直觉。让我用一个比喻：

想象你是一个设计师，需要画一条曲线穿过纸上的若干个点。

**情况一：你手里的"自由度"远少于点的数量。** 你画不出穿过所有点的曲线，所以你画出一条"最接近"所有点的平滑曲线。虽然不完美，但因为你的工具有限，曲线自然是平滑的——这就是**欠拟合**区域，模型被迫保持简单。

**情况二：你的自由度刚好等于点的数量。** 你恰好有"唯一"一条曲线能穿过所有点。不管点里有没有噪声、有没有异常值，你被迫完美拟合它们**所有人**，包括那些被噪声污染的错误点。你没有任何"余地"来选择一条更平滑的路径——唯一的解就是那条疯狂震荡的曲线。这就是插值阈值：**被迫拟合噪声，且无路可选。**

**情况三：你的自由度远多于点的数量。** 现在有无数条曲线都能穿过所有点。你可以选择！梯度下降这类优化算法有一个隐含的倾向：它们会在所有完美拟合训练数据的解中，选择**最"简单"的那个**（在数学上是范数最小的解）。这就像在无数条能穿过所有点的路径中，选了那条最平滑、最不折腾的。

这就是双重下降背后的核心直觉：

> **在插值阈值处，模型被迫用唯一的方式拟合噪声数据——这是最糟糕的情况。一旦模型远远过参数化，它有无穷多种方式拟合数据，优化算法会帮它选一个平滑的解。**

## 三种"维度"的双重下降

OpenAI 2019 年的论文揭示了一个更深层的统一：双重下降不只发生在模型大小这一个维度上。它实际上沿着至少三个轴出现：

### 1. 模型维度双重下降 (Model-wise)

这是最经典的版本：固定训练数据，逐渐增加模型参数数量。在 ResNet18 上的 CIFAR-10 实验中，随着网络宽度增加，测试误差先降后升再降。

### 2. 训练时间双重下降 (Epoch-wise)

固定一个足够大的模型，观察它在训练过程中的表现。测试误差先下降（学到了有用的模式），然后上升（开始记住噪声），然后——如果你继续训练足够久——又开始下降！

这解释了一个实践中的困惑：为什么有时候训练更久反而效果更好？因为模型可能正在经历 epoch-wise 的双重下降。

### 3. 样本维度双重下降 (Sample-wise)

这个最反直觉：固定模型大小，增加训练数据量——有时候**更多数据反而让效果变差**！

为什么？因为增加数据会把插值阈值推向右边（需要更大的模型才能拟合更多数据）。如果你的模型大小恰好在新的插值阈值附近，你就掉进了那个"峰值陷阱"。

<svg viewBox="0 0 700 350" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow-3d" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Three panels -->
  <!-- Panel 1: Model-wise -->
  <rect x="20" y="40" width="200" height="180" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="120" y="30" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui" font-weight="bold">模型维度</text>
  <text x="120" y="60" text-anchor="middle" fill="#94a3b8" font-size="10" font-family="system-ui">固定数据，增加参数</text>
  <line x1="40" y1="200" x2="200" y2="200" stroke="#6e8eff" stroke-width="1"/>
  <line x1="40" y1="200" x2="40" y2="70" stroke="#6e8eff" stroke-width="1"/>
  <path d="M 50 160 C 65 140, 75 110, 90 100 C 100 105, 110 130, 125 160 C 135 175, 140 170, 150 150 C 160 130, 175 100, 200 85" fill="none" stroke="#f472b6" stroke-width="2"/>
  <text x="120" y="235" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">参数数量 →</text>

  <!-- Panel 2: Epoch-wise -->
  <rect x="250" y="40" width="200" height="180" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="350" y="30" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui" font-weight="bold">训练时间维度</text>
  <text x="350" y="60" text-anchor="middle" fill="#94a3b8" font-size="10" font-family="system-ui">固定模型，增加 epoch</text>
  <line x1="270" y1="200" x2="430" y2="200" stroke="#6e8eff" stroke-width="1"/>
  <line x1="270" y1="200" x2="270" y2="70" stroke="#6e8eff" stroke-width="1"/>
  <path d="M 280 170 C 295 130, 310 100, 325 95 C 335 100, 345 125, 360 150 C 370 160, 378 155, 385 140 C 395 120, 410 100, 430 90" fill="none" stroke="#34d399" stroke-width="2"/>
  <text x="350" y="235" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">训练轮数 →</text>

  <!-- Panel 3: Sample-wise -->
  <rect x="480" y="40" width="200" height="180" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="580" y="30" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui" font-weight="bold">样本维度</text>
  <text x="580" y="60" text-anchor="middle" fill="#94a3b8" font-size="10" font-family="system-ui">固定模型，增加数据</text>
  <line x1="500" y1="200" x2="660" y2="200" stroke="#6e8eff" stroke-width="1"/>
  <line x1="500" y1="200" x2="500" y2="70" stroke="#6e8eff" stroke-width="1"/>
  <path d="M 510 100 C 530 95, 545 100, 560 120 C 575 145, 585 160, 595 155 C 610 140, 630 110, 660 90" fill="none" stroke="#a78bfa" stroke-width="2"/>
  <text x="580" y="235" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">训练样本数 →</text>

  <!-- Unifying concept -->
  <rect x="150" y="270" width="400" height="50" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="350" y="292" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">统一视角：误差峰值出现在「有效模型复杂度 ≈ 样本数」处</text>
  <text x="350" y="310" text-anchor="middle" fill="#94a3b8" font-size="10" font-family="system-ui">无论你沿哪个轴移动，跨过插值阈值后误差都会重新下降</text>
</svg>

Nakkiran 等人提出了一个统一的解释框架：定义**有效模型复杂度 (Effective Model Complexity, EMC)** 为模型在给定训练过程下能够"刚好拟合"的最大样本数。当 EMC ≈ 训练样本数 $n$ 时，你正处于插值阈值，误差最大。无论你通过增加参数、增加训练时间还是减少数据来让 EMC 远远大于 $n$，误差都会重新下降。

## 噪声是"催化剂"

一个重要的实验发现：**双重下降在有标签噪声时最为明显。** 如果训练数据完全干净（没有错误标签），双重下降的峰值很小，甚至可能看不到。但一旦加入 10-20% 的标签噪声，峰值就变得非常显著。

为什么？回到之前的比喻：

- 如果所有点都"说的是真话"（无噪声），那么被迫完美拟合它们其实问题不大——你确实应该经过这些点。
- 但如果有些点"在撒谎"（标签噪声），你被迫完美拟合它们就是灾难——你在认真对待错误信息。

在插值阈值处，模型必须把所有计算能力都用来"相信"每一个训练样本——包括那些有噪声的。它没有任何余量来"质疑"异常值。而在过参数化区域，模型有足够的容量以一种"温和"的方式拟合所有点，不需要为了拟合噪声点而扭曲整个函数。

这也解释了为什么现代 LLM 的训练数据清洗如此重要——不是因为模型会过拟合（模型足够大，处于深度过参数化区域），而是因为噪声数据仍然会消耗模型容量、影响学到的表示质量。

## 隐式正则化：梯度下降的"偏好"

双重下降故事里最精彩的部分，是它揭示了**梯度下降并不是一个"中性"的优化器**。

当模型过参数化时（参数远多于数据），存在无穷多个能完美拟合训练数据的解。梯度下降不会随机选一个——它有一个隐含的偏好。

对于线性模型，这个偏好有精确的数学描述：从零初始化开始的梯度下降，收敛到所有插值解中**$\ell_2$ 范数最小**的那个。翻译成人话就是：在所有完美答案中，选权重最"小"的那个。

$$\hat{w} = \arg\min_w \|w\|_2 \quad \text{subject to} \quad Xw = y$$

这个最小范数解可以写成 $\hat{w} = X^{\dagger}y$（Moore-Penrose 伪逆）。

为什么最小范数意味着平滑？因为小权重意味着模型对输入的微小变化不会产生剧烈反应——函数变化是温和的。这本质上等价于加了一个无穷小的 L2 正则化（ridge regression，正则化系数 $\lambda \to 0^+$）。

这就是**隐式正则化 (implicit regularization)**：你不需要手动添加正则化项，梯度下降本身就帮你选了一个"好"的解。

但有一个关键条件：这种隐式正则化只在**过参数化区域**才有效。在插值阈值处（唯一解），没有选择的余地，隐式正则化无从发挥。

## 从线性模型到深度网络

你可能会问：以上分析都是关于线性模型的，深度神经网络也是这样吗？

答案是：深度网络的情况更复杂，但基本直觉相同。

### Neural Tangent Kernel 视角

在极宽（参数极多）的限制下，深度网络的训练动态可以用**神经切线核 (Neural Tangent Kernel, NTK)** 来近似描述。在 NTK 体制下，网络的行为类似于一个核方法，梯度下降会收敛到再生核希尔伯特空间 (RKHS) 中范数最小的插值函数。

这给了我们同样的故事：过参数化的网络通过隐式正则化选择了"平滑"的解。

### 但实际网络更有趣

实际的深度网络不完全在 NTK 体制中工作。它们会经历**特征学习 (feature learning)**——网络会主动学习好的数据表示，而不只是在固定特征上做线性拟合。

这意味着实际网络的隐式正则化比"最小范数"更强大：它们不只是选择平滑的函数，还会选择**与数据结构对齐**的解。这也是为什么深度学习在实践中表现远超核方法的原因之一。

## 实际影响：对训练 LLM 意味着什么

双重下降现象对大语言模型的训练有几个重要的实践启示：

### 1. "大即是好"有了理论解释

现代 LLM 参数量远超训练数据量，它们深处过参数化区域的"第二次下降"阶段。Scaling Laws 之所以成立——模型越大效果越好——部分原因就是这些模型远离了插值阈值，享受着充分的隐式正则化。

### 2. 早停不总是最优

传统做法是用验证集监控，一旦验证误差开始上升就停止训练。但 epoch-wise 双重下降告诉我们：如果你在第一次上升时就停下来，可能错过了后面的第二次下降。

当然，这不意味着永远不要早停——但你需要知道你的模型处于双重下降曲线的哪个位置。

### 3. 正则化改变了双重下降的形状

适当的显式正则化（如 weight decay）可以**消除**双重下降的峰值，让曲线变成单调下降。这是因为正则化相当于远离了"纯插值"状态，哪怕在参数等于数据点的地方，模型也不会被迫完美拟合每一个噪声。

研究表明，最优的正则化强度也会随着模型大小展现出双重下降行为：对于过参数化模型，最优 weight decay 通常非常小——这与"模型越大需要越强正则化"的直觉相反。

### 4. 数据质量在阈值附近格外重要

当你的模型大小恰好在插值阈值附近时，标签噪声的影响会被放大到最大。这在实践中意味着：如果你用的是中等大小的模型（不是极端过参数化），数据清洗的回报特别高。

## 更深的问题：为什么教科书错了那么久？

双重下降的发现暴露了统计学习理论的一个历史盲区：经典理论只研究了"欠参数化"那一半的故事。

原因很实际：在计算资源有限的时代，没有人会用 1000 个参数去拟合 20 个数据点——那太浪费了。经典统计学的全部智慧都是在"参数是稀缺资源"的假设下建立的。

但深度学习翻转了这个假设：参数是廉价的（GPU 提供了巨大的计算量），数据才是稀缺的。在这个新世界里，过参数化是常态，插值阈值只是通往更好泛化的一个需要跨过的"中间阶段"。

这并不意味着偏差-方差权衡是错的——它在经典区域内依然正确。双重下降是对偏差-方差权衡的**扩展**，把故事从 U 形补完成了一个先 U 后降的完整图景。

## 与 Grokking 的关系

如果你读过我们系列的前几篇文章，可能注意到双重下降和 **Grokking**（过拟合后突然泛化）有相似之处。确实如此：

- **双重下降** 关注的是：跨过插值阈值后，模型泛化为什么变好
- **Grokking** 关注的是：在训练后期（远超过拟合点），模型突然从"记忆"转变为"理解"

两者都涉及"过拟合不是终点"这个核心洞察。区别在于：双重下降通常是渐进的（测试误差平滑下降），而 Grokking 是突然的（测试准确率突然跳升）。Grokking 被认为与模型内部从"记忆电路"到"泛化电路"的相变有关，而双重下降更多与解空间的几何性质有关。

## 总结

双重下降现象告诉我们的核心故事是：

1. **经典智慧没有错，但只说了一半。** 在欠参数化区域，偏差-方差权衡完全成立。
2. **插值阈值是最危险的地方。** 模型"刚好够大"反而最糟糕——被迫拟合噪声，无路可选。
3. **过参数化带来了自由度。** 无穷多个解中，优化算法的隐式偏好帮你选了一个好的。
4. **梯度下降不是中性的。** 它有内建的"奥卡姆剃刀"——在所有完美解中倾向简单的。
5. **深度学习之所以有效，部分原因是它跳过了危险区。** 现代大模型远在插值阈值右边，享受着过参数化带来的"第二次下降"红利。

下次有人问你"为什么 GPT 有 1750 亿参数但没过拟合"，你可以回答：**因为它太大了，以至于过拟合变成了一件好事。** 它有足够的自由度以一种"优雅"的方式拟合数据——而梯度下降会帮它选那条最平滑的路。

## 进一步阅读

- Belkin et al. (2019). "Reconciling Modern Machine Learning Practice and the Classical Bias-Variance Trade-off." PNAS.
- Nakkiran et al. (2019). "Deep Double Descent: Where Bigger Models and More Data Can Hurt." arXiv:1912.02292.
- Bartlett et al. (2020). "Benign Overfitting in Linear Regression." PNAS.
- Neal (2019). "On the Bias-Variance Tradeoff: Textbooks Need an Update." MSc Thesis.
