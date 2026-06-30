---
title: "理解 RLHF 与对齐训练（四）：PPO——用强化学习微调语言模型的艺术与苦难"
date: 2025-06-30
level: 3
series: "理解 RLHF 与对齐训练"
series_order: 4
series_total: 5
tags: [rlhf, ppo, reinforcement-learning, clipping, kl-divergence, value-function]
summary: "PPO 是 RLHF 的执行引擎：用 clipping 机制当护栏、KL 惩罚当保险绳，在奖励最大化和模型稳定之间走钢丝"
---

# PPO：用强化学习微调语言模型的艺术与苦难

> PPO 原本是为 Atari 游戏和机器人设计的。把它搬到语言模型上，就像把赛车引擎装到自行车上——强大但需要极其精细的调校才不会翻车。

## 为什么需要强化学习？

在前面的文章中，我们建好了一个"评委"——奖励模型。它能对任何 (问题, 回答) 打分。

现在的问题是：**怎么用这个分数来改进模型？**

直觉上你可能想："对高分的回答做更多 SFT 不就行了？" 这叫 Best-of-N 或 Rejection Sampling——生成 N 个回答，选分最高的那个，用它做 SFT。这确实有效，但效率极低：大部分生成都被丢弃了，而且你无法利用"失败尝试"中的信息。

强化学习的优势在于：**它不只奖励好的行为，还能从坏的行为中学习**。PPO 在每一步生成时都获得梯度信号——不只是"最终答案好不好"，而是"每一步选择的方向对不对"。

## PPO 的核心思想：高速公路上的护栏

PPO（Proximal Policy Optimization）的核心创新可以用一个比喻来理解：

想象你在一条高速公路上开车。你的目标是尽快到达终点（最大化奖励），但高速公路两侧有护栏。护栏不会减慢你的速度，但它们防止你因为转向过猛而冲出路面。

在传统的策略梯度方法（如 REINFORCE）中，没有护栏。每一步更新可能很大，如果方向稍微偏了，模型会突然崩溃——输出变成乱码、无限重复、或退化为固定模板。

PPO 的护栏就是 **clipping 机制**。

### Clipping 机制的数学

PPO 计算新旧策略之间的概率比：

$$r(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$$

当 $r(\theta) = 1$ 时，新旧策略完全一样。$r(\theta) > 1$ 意味着新策略对这个 action 更自信，$r(\theta) < 1$ 意味着新策略更不看好这个 action。

PPO 的目标函数限制了 $r(\theta)$ 的变化范围：

$$L^{\text{CLIP}} = \mathbb{E}\left[\min\left(r(\theta) \cdot A_t, \; \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) \cdot A_t\right)\right]$$

其中 $\epsilon$ 通常是 0.1-0.2，$A_t$ 是优势函数（advantage，衡量这个 action 比平均水平好多少）。

**直觉解释：**

- 如果一个 action 是好的（$A_t > 0$），PPO 允许增加它的概率，但增幅上限是 $(1+\epsilon)$ 倍——不能一步到位全压上去
- 如果一个 action 是坏的（$A_t < 0$），PPO 允许降低它的概率，但降幅下限是 $(1-\epsilon)$ 倍——不能一步就完全排除

这就是护栏：**你可以在正确的方向加速，但每一步的转向幅度有限。**

<svg viewBox="0 0 700 300" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arr5" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Title -->
  <text x="350" y="25" text-anchor="middle" fill="#ededf0" font-size="14" font-weight="bold" font-family="system-ui">PPO Clipping: 策略更新的护栏</text>
  <!-- Axis -->
  <line x1="100" y1="250" x2="620" y2="250" stroke="#3a3a4a" stroke-width="1"/>
  <line x1="350" y1="50" x2="350" y2="250" stroke="#3a3a4a" stroke-width="1"/>
  <text x="360" y="270" fill="#888" font-size="11" font-family="system-ui">r(θ) = π_new / π_old</text>
  <text x="85" y="155" fill="#888" font-size="11" font-family="system-ui" transform="rotate(-90,85,155)">目标函数值</text>
  <!-- r=1 mark -->
  <line x1="350" y1="245" x2="350" y2="255" stroke="#ededf0" stroke-width="1.5"/>
  <text x="350" y="285" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">1.0</text>
  <!-- 1-eps mark -->
  <line x1="230" y1="245" x2="230" y2="255" stroke="#ff6b6b" stroke-width="1.5"/>
  <text x="230" y="285" text-anchor="middle" fill="#ff6b6b" font-size="10" font-family="system-ui">1-ε</text>
  <!-- 1+eps mark -->
  <line x1="470" y1="245" x2="470" y2="255" stroke="#ff6b6b" stroke-width="1.5"/>
  <text x="470" y="285" text-anchor="middle" fill="#ff6b6b" font-size="10" font-family="system-ui">1+ε</text>
  <!-- Unclipped line (A>0 case) -->
  <line x1="100" y1="220" x2="620" y2="90" stroke="#3a3a4a" stroke-width="1" stroke-dasharray="4,4"/>
  <text x="600" y="80" fill="#3a3a4a" font-size="10" font-family="system-ui">无 clip</text>
  <!-- Clipped objective (A>0) -->
  <line x1="100" y1="220" x2="470" y2="128" stroke="#34d399" stroke-width="2.5"/>
  <line x1="470" y1="128" x2="620" y2="128" stroke="#34d399" stroke-width="2.5"/>
  <!-- Clip zone highlight -->
  <rect x="470" y="50" width="150" height="200" fill="rgba(255,107,107,0.05)" stroke="none"/>
  <text x="545" y="70" text-anchor="middle" fill="#ff6b6b" font-size="10" font-family="system-ui">被 clip 的区域</text>
  <text x="545" y="85" text-anchor="middle" fill="#ff6b6b" font-size="9" font-family="system-ui">（梯度为 0，停止更新）</text>
  <!-- Labels -->
  <text x="280" y="145" fill="#34d399" font-size="11" font-family="system-ui">正常优化区域</text>
  <text x="280" y="162" fill="#888" font-size="9" font-family="system-ui">（A > 0 的情况）</text>
  <!-- Legend note -->
  <text x="120" y="50" fill="#888" font-size="10" font-family="system-ui">超过 1+ε 时梯度被截断</text>
  <text x="120" y="67" fill="#888" font-size="10" font-family="system-ui">→ 防止过度自信的更新</text>
</svg>

## RLHF 中的四模型系统

PPO 在 RLHF 中的完整实现需要同时维护**四个模型**：

| 模型 | 作用 | 是否更新 |
|------|------|----------|
| **策略模型** (Actor) | 生成回答 | ✓ 持续训练 |
| **参考模型** (Reference) | 计算 KL 惩罚 | ✗ 冻结的 SFT 模型 |
| **奖励模型** (Reward) | 给回答打分 | ✗ 已训练好，冻结 |
| **价值模型** (Critic) | 估计期望奖励 | ✓ 持续训练 |

### 为什么需要价值模型（Critic）？

策略梯度的一个基本问题是**方差太大**。如果一个回答得了 8 分，你怎么知道这是好还是坏？如果平均水平是 3 分，8 分就很好；如果平均水平是 9 分，8 分就是失败。

价值模型 $V(s)$ 就是用来估计"平均水平"的。它预测从当前状态开始，未来能获得的期望奖励。有了它，你可以计算**优势（Advantage）**：

$$A_t = R_t - V(s_t)$$

"这个行为比我预期的好多少"——这才是策略应该关注的信号。

### KL 惩罚：保险绳

除了 clipping 护栏，RLHF-PPO 还有第二道安全机制：KL 散度惩罚。

$$R_{\text{total}} = R_{\text{reward}}(x, y) - \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$$

KL 散度衡量新策略和参考模型（冻结的 SFT 模型）之间的"距离"。如果新策略开始生成和原模型完全不同的东西，KL 惩罚会拉住它。

**比喻：** Clipping 是每一步的护栏（限制单步更新幅度），KL 惩罚是全局的保险绳（限制累计偏离程度）。两者配合，确保模型在追求高奖励的同时不会"迷失自我"。

$\beta$ 通常设为 0.05-0.2。太小了，模型会 reward hack；太大了，模型完全不学习。

## 为什么 PPO 在 LLM 上是"调参地狱"？

### 问题 1：训练不稳定

四个模型必须同步训练。策略改进 → 价值函数跟不上 → 优势估计不准 → 策略更新方向错误 → 训练崩溃。

超参数之间的相互耦合也很严重：
- 学习率影响 clipping 是否被触发
- KL 系数影响策略能探索多远
- PPO epoch 数影响每 batch 数据的利用率
- GAE λ 影响长期 vs 短期信号的权重

改动任何一个，其他参数的最优值都会变化。

### 问题 2：奖励黑客

PPO 是一个强大的优化器。如果给它足够多的步数和足够小的 KL 约束，它**一定会**找到奖励模型的漏洞。典型症状：

- 训练初期：奖励上升，回答质量确实在改善
- 中期：奖励继续上升，但人类评估分数开始停滞
- 后期：奖励飙升，但回答变得冗长、谄媚、格式化——RM 给高分，人类觉得恶心

这就是那条著名的"overoptimization 曲线"：proxy reward 持续上升，但 gold reward（真实人类偏好）先升后降。

### 问题 3：模式坍塌

如果 KL 约束太弱，模型会收敛到一个狭窄的输出分布——对所有问题都用类似的模板回答。多样性和创造力消失，模型变成一个只会说"八股文"的应试机器。

### 问题 4：计算成本

对一个 70B 参数的模型做 PPO，需要同时在 GPU 上维护：
- 70B 策略模型
- 70B 参考模型（冻结但占内存）
- 70B 价值模型
- 6-70B 奖励模型

总计 200-300B 参数的显存占用。即使用混合精度和梯度累积，这也需要数百张 GPU 和数周的训练时间。

## PPO 的训练循环

每一步 PPO 更新的完整流程：

```
for each batch of prompts:
    1. 策略模型生成回答 (rollout)
    2. 奖励模型给回答打分
    3. 参考模型计算 KL 散度
    4. 计算总奖励 = RM 分数 - β·KL
    5. 价值模型估计每个 token 位置的 V(s)
    6. 用 GAE 计算 Advantage
    7. 做 K 轮 PPO 更新（通常 K=1-4）:
       - 计算概率比 r(θ)
       - 计算 clipped 目标
       - 更新策略模型
       - 更新价值模型
```

<svg viewBox="0 0 720 350" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:720px;margin:24px auto;display:block;">
  <defs>
    <marker id="arr6" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Prompt -->
  <rect x="20" y="140" width="100" height="40" rx="6" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="70" y="165" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">Prompt x</text>
  <!-- Policy -->
  <line x1="125" y1="160" x2="155" y2="160" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arr6)"/>
  <rect x="160" y="130" width="100" height="55" rx="8" fill="#1e1e2a" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="210" y="152" text-anchor="middle" fill="#22d3ee" font-size="11" font-weight="bold" font-family="system-ui">策略 π_θ</text>
  <text x="210" y="170" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">生成回答 y</text>
  <!-- Response -->
  <line x1="265" y1="157" x2="300" y2="157" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arr6)"/>
  <!-- RM scoring -->
  <rect x="305" y="60" width="110" height="45" rx="8" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="360" y="80" text-anchor="middle" fill="#a78bfa" font-size="10" font-weight="bold" font-family="system-ui">奖励模型 RM</text>
  <text x="360" y="95" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">→ r(x,y)</text>
  <!-- Ref model -->
  <rect x="305" y="120" width="110" height="45" rx="8" fill="#1e1e2a" stroke="#94a3b8" stroke-width="1.5"/>
  <text x="360" y="140" text-anchor="middle" fill="#94a3b8" font-size="10" font-weight="bold" font-family="system-ui">参考模型 π_ref</text>
  <text x="360" y="155" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">→ KL(π_θ||π_ref)</text>
  <!-- Value model -->
  <rect x="305" y="180" width="110" height="45" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="360" y="200" text-anchor="middle" fill="#34d399" font-size="10" font-weight="bold" font-family="system-ui">价值模型 V</text>
  <text x="360" y="215" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">→ baseline</text>
  <!-- Connections -->
  <line x1="300" y1="147" x2="305" y2="82" stroke="#6e8eff" stroke-width="1" marker-end="url(#arr6)"/>
  <line x1="300" y1="155" x2="305" y2="142" stroke="#6e8eff" stroke-width="1" marker-end="url(#arr6)"/>
  <line x1="300" y1="163" x2="305" y2="202" stroke="#6e8eff" stroke-width="1" marker-end="url(#arr6)"/>
  <!-- Compute advantage -->
  <line x1="420" y1="82" x2="480" y2="140" stroke="#6e8eff" stroke-width="1" marker-end="url(#arr6)"/>
  <line x1="420" y1="142" x2="480" y2="145" stroke="#6e8eff" stroke-width="1" marker-end="url(#arr6)"/>
  <line x1="420" y1="202" x2="480" y2="150" stroke="#6e8eff" stroke-width="1" marker-end="url(#arr6)"/>
  <rect x="485" y="120" width="110" height="55" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="540" y="140" text-anchor="middle" fill="#6e8eff" font-size="10" font-weight="bold" font-family="system-ui">计算 Advantage</text>
  <text x="540" y="157" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">A = R - β·KL - V</text>
  <text x="540" y="170" text-anchor="middle" fill="#888" font-size="9" font-family="system-ui">(GAE 平滑)</text>
  <!-- PPO update -->
  <line x1="600" y1="147" x2="635" y2="147" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arr6)"/>
  <rect x="640" y="115" width="65" height="65" rx="8" fill="rgba(34,211,238,0.1)" stroke="#22d3ee" stroke-width="2"/>
  <text x="672" y="140" text-anchor="middle" fill="#22d3ee" font-size="9" font-weight="bold" font-family="system-ui">PPO</text>
  <text x="672" y="155" text-anchor="middle" fill="#22d3ee" font-size="9" font-family="system-ui">Clip</text>
  <text x="672" y="168" text-anchor="middle" fill="#22d3ee" font-size="9" font-family="system-ui">更新</text>
  <!-- Feedback loop -->
  <path d="M 672 185 L 672 310 L 210 310 L 210 190" fill="none" stroke="#22d3ee" stroke-width="1.5" stroke-dasharray="4,4" marker-end="url(#arr6)"/>
  <text x="440" y="300" text-anchor="middle" fill="#22d3ee" font-size="10" font-family="system-ui">更新策略参数 → 生成新回答 → 循环</text>
</svg>

## 成功的 PPO 训练是什么样的

一个健康的 RLHF-PPO 训练曲线通常表现为：

- **奖励分数**：稳步上升，然后趋于平稳（不是飙升）
- **KL 散度**：缓慢增长，稳定在 5-15 nats 之间
- **人类偏好胜率**：持续提升（vs SFT baseline）
- **回答长度**：微增但不爆炸
- **困惑度（PPL）**：轻微上升但不崩溃

如果你看到奖励飙升但 KL 爆炸，或者奖励上升但回答长度翻倍——都是 reward hacking 的信号。

## 实践中的关键超参数

| 参数 | 典型值 | 含义 |
|------|--------|------|
| ε (clip) | 0.1-0.2 | 单步更新幅度限制 |
| β (KL) | 0.05-0.2 | 偏离参考模型的惩罚强度 |
| 学习率 | 1e-6 ~ 5e-6 | 比 SFT 更小 |
| PPO epochs | 1-4 | 每批数据重复利用次数 |
| Batch size | 32-512 prompts | 越大越稳定 |
| GAE λ | 0.95 | 优势估计的时间折扣 |
| 生成长度 | 256-2048 tokens | 回答的最大长度 |

## 下一篇预告

PPO 强大但复杂：四个模型、大量超参数、训练不稳定、计算成本高昂。自然的问题是：**能不能更简单？**

2023 年，一篇论文给出了惊人的回答：不需要奖励模型，不需要 PPO，不需要价值函数——只用一个简单的分类损失，就能达到几乎相同的效果。这就是 DPO（Direct Preference Optimization）。而 2024 年 DeepSeek 更进一步：保留 RL 的探索能力，但砍掉价值函数——这就是 GRPO。下一篇，我们看看这些"后 PPO"时代的方法如何优雅地简化对齐训练。
