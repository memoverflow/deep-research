---
title: "理解 RLHF 与对齐训练（五）：DPO 与 GRPO——跳过奖励模型的新范式"
date: 2025-06-30
level: 3
series: "理解 RLHF 与对齐训练"
series_order: 5
series_total: 5
tags: [rlhf, dpo, grpo, deepseek, kto, simpo, preference-optimization]
summary: "DPO 证明语言模型本身就是奖励模型，GRPO 证明不需要价值函数也能做 RL——对齐训练正在变得越来越优雅"
---

# DPO 与 GRPO：跳过奖励模型的新范式

> 2023 年，一个数学推导证明了一个惊人的事实：你的语言模型本身就是一个奖励模型。你根本不需要单独训练一个。2024 年，DeepSeek 又证明了：即使你选择用 RL，也不需要那个烦人的价值函数。

## DPO：一个优雅的数学恒等式

### 从"三步"变成"一步"

回顾 RLHF 的标准流程：
1. 收集偏好数据 → 训练奖励模型
2. 用 PPO 优化策略 → 最大化奖励 - KL 惩罚
3. 维护四个模型，调几十个超参数

DPO 的论文标题说明了一切："Your Language Model is Secretly a Reward Model"——你的语言模型**秘密地**就是一个奖励模型。

这意味着：你可以跳过步骤 1 和 2 的大部分复杂性，直接用偏好数据训练语言模型本身。不需要单独的 RM，不需要 PPO，不需要价值函数。

### 推导：从约束优化到闭式解

RLHF 的核心优化目标是：

$$\max_\pi \; \mathbb{E}_{x \sim D, y \sim \pi}[r(x, y)] - \beta \cdot \text{KL}(\pi \| \pi_{\text{ref}})$$

"最大化奖励，同时不要离参考模型太远。"

这个有约束的优化问题有一个**闭式解**：

$$\pi^*(y|x) = \frac{1}{Z(x)} \cdot \pi_{\text{ref}}(y|x) \cdot \exp\left(\frac{r(x,y)}{\beta}\right)$$

其中 $Z(x)$ 是归一化常数。翻译成人话：**最优策略就是参考模型乘以奖励的指数。** 奖励高的回答概率被放大，奖励低的被压缩。

现在，DPO 的关键一步来了。把上面的等式反过来，解出 $r(x,y)$：

$$r(x,y) = \beta \cdot \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \cdot \log Z(x)$$

这告诉我们：**奖励函数可以用最优策略和参考策略的对数概率比来表示。**

最妙的地方：当你把这个表达式代入 Bradley-Terry 偏好模型 $P(y_w > y_l) = \sigma(r(y_w) - r(y_l))$ 时，$Z(x)$ 项会**完美抵消**（因为它只和 $x$ 有关，不和 $y$ 有关）！

最终得到 DPO 的损失函数：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \cdot \left(\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right)\right]$$

### 直觉解释

这个损失函数在做什么？

- **增大好回答的概率**（相对于参考模型）
- **减小坏回答的概率**（相对于参考模型）
- $\beta$ 控制这个调整的力度

就是这样。一个简单的二分类损失函数。不需要奖励模型，不需要 PPO，不需要价值函数，不需要 rollout 采样。只需要偏好数据 + 策略模型 + 冻结的参考模型。

<svg viewBox="0 0 700 280" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arr7" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="350" y="25" text-anchor="middle" fill="#ededf0" font-size="13" font-weight="bold" font-family="system-ui">PPO vs DPO：复杂度对比</text>
  <!-- PPO side -->
  <rect x="30" y="45" width="280" height="210" rx="10" fill="#1e1e2a" stroke="#ff6b6b" stroke-width="1.5"/>
  <text x="170" y="70" text-anchor="middle" fill="#ff6b6b" font-size="12" font-weight="bold" font-family="system-ui">传统 RLHF (PPO)</text>
  <text x="170" y="100" fill="#ededf0" font-size="10" font-family="system-ui">• 策略模型 (训练中)</text>
  <text x="170" y="120" fill="#ededf0" font-size="10" font-family="system-ui">• 参考模型 (冻结)</text>
  <text x="170" y="140" fill="#ededf0" font-size="10" font-family="system-ui">• 奖励模型 (冻结)</text>
  <text x="170" y="160" fill="#ededf0" font-size="10" font-family="system-ui">• 价值模型 (训练中)</text>
  <text x="170" y="185" fill="#888" font-size="10" font-family="system-ui">需要: 在线生成 + RM 打分</text>
  <text x="170" y="203" fill="#888" font-size="10" font-family="system-ui">超参数: ε, β, lr, GAE λ, epochs...</text>
  <text x="170" y="221" fill="#ff6b6b" font-size="10" font-family="system-ui">GPU 占用: ~4x 模型大小</text>
  <text x="170" y="240" fill="#ff6b6b" font-size="10" font-family="system-ui">训练稳定性: 🔥 容易崩</text>
  <!-- DPO side -->
  <rect x="380" y="45" width="280" height="210" rx="10" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="520" y="70" text-anchor="middle" fill="#34d399" font-size="12" font-weight="bold" font-family="system-ui">DPO</text>
  <text x="520" y="100" fill="#ededf0" font-size="10" font-family="system-ui">• 策略模型 (训练中)</text>
  <text x="520" y="120" fill="#ededf0" font-size="10" font-family="system-ui">• 参考模型 (冻结)</text>
  <text x="520" y="140" fill="#34d399" font-size="10" font-family="system-ui">✗ 不需要奖励模型</text>
  <text x="520" y="160" fill="#34d399" font-size="10" font-family="system-ui">✗ 不需要价值模型</text>
  <text x="520" y="185" fill="#888" font-size="10" font-family="system-ui">只需: 偏好数据对 (y_w, y_l)</text>
  <text x="520" y="203" fill="#888" font-size="10" font-family="system-ui">超参数: β, lr (就两个)</text>
  <text x="520" y="221" fill="#34d399" font-size="10" font-family="system-ui">GPU 占用: ~2x 模型大小</text>
  <text x="520" y="240" fill="#34d399" font-size="10" font-family="system-ui">训练稳定性: ✓ 和 SFT 一样稳</text>
</svg>

### DPO 的局限

DPO 如此优雅，为什么没有完全取代 PPO？因为它有几个根本性的限制：

**1. 离线学习的局限**

DPO 只在固定的偏好数据集上训练。它不能像 PPO 那样"探索"——生成新的回答、观察反馈、调整策略。这意味着 DPO 的上限受限于偏好数据的质量和覆盖范围。

如果你的偏好数据来自一个很弱的模型生成的回答，DPO 只能学会在这些弱回答之间选择更好的——它无法发现真正好的、数据中没出现过的回答模式。

**2. 分布偏移**

当策略训练到一定程度后，它生成的回答分布已经和训练数据中的回答分布相差很大。此时 DPO 的梯度信号可能不再准确——你在用"旧世界"的评判标准来指导"新世界"的决策。

**3. 不适合推理任务**

对于数学推理这样的任务，正确答案的空间很稀疏。DPO 需要偏好数据中存在"好的推理过程"才能学到东西。但对于难题，在离线数据中很难找到正确的解法示例。

## GRPO：保留 RL 的探索，砍掉价值函数

### 问题：PPO 的价值函数真的必要吗？

回忆一下 PPO 需要价值模型的原因：为了计算 advantage（"这个行为比平均水平好多少"）。价值模型估计"从这个状态开始的期望奖励"，作为 baseline 减少方差。

但训练一个好的价值函数**非常难**——它本身需要精确估计复杂的语言空间中的奖励期望值。而且它额外占用一整个模型的显存。

DeepSeek 的 GRPO（Group Relative Policy Optimization）提出了一个巧妙的替代方案：**不需要学一个 baseline，直接从数据中算一个。**

### GRPO 的核心思想

对每个问题 $q$，生成一组 $G$ 个回答 $\{o_1, o_2, ..., o_G\}$。给每个回答打分 $r_i$。然后做**组内标准化**：

$$\hat{A}_i = \frac{r_i - \text{mean}(r_1, ..., r_G)}{\text{std}(r_1, ..., r_G)}$$

就是这样！不需要学习的价值函数。baseline 就是同一个问题的回答组的平均分数。

**直觉：** 如果你生成 8 个回答，其中一个得了 9 分而其他都只有 3-4 分，那个 9 分的 advantage 自然就很高。你不需要一个神经网络来告诉你"9 分比平均高"——直接算就行了。

然后用标准的 PPO clipping 目标来更新策略：

$$L_{\text{GRPO}} = \mathbb{E}_q \left[\frac{1}{G} \sum_i \min\left(\frac{\pi_\theta(o_i|q)}{\pi_{\text{old}}(o_i|q)} \cdot \hat{A}_i, \; \text{clip}(\cdot) \cdot \hat{A}_i\right)\right] - \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$$

<svg viewBox="0 0 700 300" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arr8" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <text x="350" y="25" text-anchor="middle" fill="#ededf0" font-size="13" font-weight="bold" font-family="system-ui">GRPO 工作流程</text>
  <!-- Prompt -->
  <rect x="20" y="120" width="80" height="40" rx="6" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="60" y="145" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">问题 q</text>
  <!-- Generate G responses -->
  <line x1="105" y1="140" x2="140" y2="140" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arr8)"/>
  <rect x="145" y="60" width="120" height="165" rx="8" fill="#1e1e2a" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="205" y="85" text-anchor="middle" fill="#22d3ee" font-size="10" font-weight="bold" font-family="system-ui">生成 G 个回答</text>
  <text x="205" y="110" text-anchor="middle" fill="#ededf0" font-size="10" font-family="monospace">o₁ → r₁ = 3.2</text>
  <text x="205" y="130" text-anchor="middle" fill="#ededf0" font-size="10" font-family="monospace">o₂ → r₂ = 7.8</text>
  <text x="205" y="150" text-anchor="middle" fill="#ededf0" font-size="10" font-family="monospace">o₃ → r₃ = 4.1</text>
  <text x="205" y="170" text-anchor="middle" fill="#ededf0" font-size="10" font-family="monospace">o₄ → r₄ = 2.9</text>
  <text x="205" y="195" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">RM 或可验证奖励</text>
  <text x="205" y="212" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">（数学正确性等）</text>
  <!-- Normalize -->
  <line x1="270" y1="140" x2="310" y2="140" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arr8)"/>
  <rect x="315" y="90" width="140" height="100" rx="8" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="385" y="115" text-anchor="middle" fill="#a78bfa" font-size="10" font-weight="bold" font-family="system-ui">组内标准化</text>
  <text x="385" y="140" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">mean = 4.5</text>
  <text x="385" y="158" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">std = 2.1</text>
  <text x="385" y="178" text-anchor="middle" fill="#34d399" font-size="10" font-family="system-ui">Â₂ = (7.8-4.5)/2.1</text>
  <text x="385" y="193" text-anchor="middle" fill="#34d399" font-size="10" font-weight="bold" font-family="system-ui">= +1.57 ✓ 高优势</text>
  <!-- PPO update -->
  <line x1="460" y1="140" x2="500" y2="140" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arr8)"/>
  <rect x="505" y="100" width="120" height="80" rx="8" fill="rgba(34,211,238,0.1)" stroke="#22d3ee" stroke-width="2"/>
  <text x="565" y="125" text-anchor="middle" fill="#22d3ee" font-size="10" font-weight="bold" font-family="system-ui">PPO Clip 更新</text>
  <text x="565" y="145" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">增大 o₂ 概率</text>
  <text x="565" y="160" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">减小 o₄ 概率</text>
  <text x="565" y="175" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">+ KL 惩罚</text>
  <!-- Bottom note -->
  <rect x="145" y="250" width="480" height="35" rx="6" fill="rgba(52,211,153,0.05)" stroke="#34d399" stroke-width="1"/>
  <text x="385" y="272" text-anchor="middle" fill="#34d399" font-size="11" font-family="system-ui">✓ 不需要价值模型 | ✓ 在线探索 | ✓ 天然的比较 baseline | ✓ 省 50% 显存</text>
</svg>

### DeepSeek-R1 的成功

GRPO 最耀眼的成就是 DeepSeek-R1。这个模型用 GRPO + 可验证奖励（数学正确性、代码执行结果），达到了和 OpenAI o1 相当的推理能力——而且**模型自发地学会了 chain-of-thought、自我验证和回溯**。

没有人教它"请先思考再回答"。RL 的探索过程自动发现了：先思考步骤再给最终答案的策略，能获得更高的奖励。这是 emergent behavior——从简单的奖励信号中涌现出复杂的推理行为。

### GRPO vs PPO vs DPO

| 维度 | PPO | DPO | GRPO |
|------|-----|-----|------|
| 在线探索 | ✓ | ✗ | ✓ |
| 价值函数 | 需要 | 不需要 | **不需要** |
| 奖励模型 | 需要 | 不需要 | 需要 |
| 计算成本 | 4 模型 | 2 模型 | 3 模型 |
| 训练稳定性 | 低 | 高 | 中 |
| 推理任务 | 强 | 弱 | **最强** |
| 数据需求 | 在线生成 | 离线偏好对 | 在线生成 |
| 超参数 | 多 | 少 | 中 |

## 偏好优化方法的寒武纪爆发

DPO 打开了一扇门，之后涌现出大量变体。每个方法都在解决 DPO 或 PPO 的某个具体痛点：

### KTO（Kahneman-Tversky Optimization）

**解决的问题：** DPO 需要配对数据（同一个问题的好回答和坏回答），收集成本高。

**方案：** 只需要二元标注——对每个 (问题, 回答) 标"好"或"坏"。基于行为经济学的前景理论（人对损失的敏感度是收益的 2 倍），用不对称损失来建模。

**适用场景：** 标注预算有限，只能打 thumbs up/down 的场景。

### IPO（Identity Preference Optimization）

**解决的问题：** DPO 可能过拟合——把坏回答的概率压到接近零。

**方案：** 在 Bradley-Terry 假设上加正则化，防止极端的概率比。更鲁棒，尤其当标注者之间存在分歧时。

### SimPO（Simple Preference Optimization）

**解决的问题：** DPO 还需要一个冻结的参考模型，占显存。

**方案：** 用序列的平均对数概率作为隐式奖励，加上 chosen/rejected 之间的 margin 约束。完全不需要参考模型——真正的单模型训练。

**结果：** 在 AlpacaEval 2 和 Arena-Hard 上持续优于 DPO。

### ORPO（Odds Ratio Preference Optimization）

**解决的问题：** SFT 和对齐是两个独立的阶段，能不能合并？

**方案：** 把 SFT 的 NLL loss 和基于 odds ratio 的偏好信号合并成一个 loss。一次训练、一个模型、一个目标函数，同时学会语言建模和偏好对齐。

## 方法选择指南

| 场景 | 推荐方法 | 原因 |
|------|----------|------|
| 追求极致性能，预算充足 | PPO | 在线探索 + 精细的 advantage 估计 |
| 快速对齐，计算有限 | DPO | 简单、稳定、两个前向传播搞定 |
| 数学/代码推理 | **GRPO** | 可验证奖励 + 在线探索 + 简单 |
| 只有 thumbs up/down 数据 | KTO | 不需要配对偏好 |
| 想最简单的部署 | SimPO | 不需要参考模型 |
| 想一步到位（SFT+对齐） | ORPO | 单阶段训练 |

## 全系列总结：从混乱到秩序

让我们回顾这整个旅程：

**第 1 篇**：我们理解了为什么预训练模型需要对齐——目标函数的错位导致有毒、无用、失控的行为。

**第 2 篇**：SFT 通过少量示范数据"激活"了模型的助手行为模式——用 13K 数据改变百亿参数模型的行为。

**第 3 篇**：奖励模型用 Bradley-Terry 模型把人类的比较判断（"A 比 B 好"）转化为可微分的标量信号——像 Elo 等级分一样从比赛胜负推断实力。

**第 4 篇**：PPO 用 clipping 护栏和 KL 保险绳，在最大化奖励和保持稳定之间走钢丝——强大但复杂，被称为"调参地狱"。

**第 5 篇**：DPO 证明整个 RM+PPO 可以被一个分类损失替代；GRPO 保留 RL 的探索优势但砍掉价值函数——对齐训练正在变得越来越简单和优雅。

对齐研究的核心洞察始终是同一个：**将人类模糊的、直觉性的偏好，通过巧妙的数学转换，变成可以大规模优化的训练信号。** 方法在变——从 PPO 到 DPO 到 GRPO——但核心问题不变：怎么让 AI 系统做我们想要它做的事？

这个问题的答案仍在演进。当你读到这篇文章时，可能已经有了更新的方法。但理解了这些基础原理，你就能快速理解任何新的对齐方法——因为它们都在回答同一个问题的不同侧面。
