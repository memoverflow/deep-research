---
title: "MoE 的生死难题：负载均衡与训练稳定性"
date: 2025-05-15
level: 3
series: "理解 Mixture of Experts"
series_order: 2
series_total: 3
tags: [MoE, load balancing, auxiliary loss, z-loss, expert death]
summary: "8 个 expert 摆好了，router 学会了——但如果所有 token 都挤向同一个 expert 怎么办？MoE 训练中最隐蔽也最致命的问题：负载失衡、expert 死亡、训练崩溃，以及解决它们的精巧方案。"
---

> 上一篇我们搭好了 MoE 的骨架：expert、router、top-k 选择。一切看起来很美——直到你真正开始训练。几千步之后，你会发现一个恐怖的现象：64 个 expert 中，只有 3 个在干活，其余 61 个的参数完全冻结，一个 token 都收不到。

## 一个故事：新开的美食街

想象你新开了一条美食街，8 个摊位（experts），一个导航员（router）站在入口指路。

第一天，导航员随机指路。碰巧，3 号摊位多接了几桌客人。厨师多练了几道菜，手艺变好了。第二天，导航员发现去 3 号的客人评价最高，于是更多地推荐 3 号。3 号越来越忙、越做越好。其他摊位门可罗雀，厨师都快忘了怎么炒菜。

一周后——3 号忙到翻桌，客人要排队两小时。7 号、8 号一整天零客人，厨师辞职了。你花了 8 个摊位的租金，但只有 1 个在运营。

这就是 MoE 训练中的 **routing collapse**（路由坍塌），也叫"富者越富"（rich-get-richer）。没有人为干预，MoE 模型几乎一定会退化成只用几个 expert 的 dense 模型。

## 为什么 Router 天然倾向失衡？

Router 的梯度来自语言模型的主损失（预测下一个 token）。这个损失只关心一件事：**预测准不准**。它完全不在乎工作分配公不公平。

训练初期：
1. 随机初始化 → Expert 3 碰巧比别人好一点点
2. Router 发现发给 Expert 3 的 token loss 更低 → 梯度推动 router 更多选 Expert 3
3. Expert 3 获得更多 token → 更多梯度更新 → 变得更好
4. 正反馈循环启动 → 指数级加剧失衡

这是一个**自我强化的正反馈环**。小到不可见的初始差异，经过几百步训练就会放大成巨大的不平衡。

<svg viewBox="0 0 600 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:600px;margin:24px auto;display:block;">
  <defs>
    <marker id="arr1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10z" fill="#f87171"/></marker>
  </defs>
  <!-- Central cycle -->
  <rect x="30" y="70" width="130" height="50" rx="8" fill="#1e1e2a" stroke="#f87171" stroke-width="1.5"/>
  <text x="95" y="92" text-anchor="middle" fill="#f87171" font-size="11" font-family="system-ui">Expert 3 稍好</text>
  <text x="95" y="108" text-anchor="middle" fill="#6b6b78" font-size="9" font-family="system-ui">随机初始化</text>

  <line x1="160" y1="95" x2="210" y2="95" stroke="#f87171" stroke-width="1.2" marker-end="url(#arr1)"/>

  <rect x="215" y="70" width="140" height="50" rx="8" fill="#1e1e2a" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="285" y="92" text-anchor="middle" fill="#fbbf24" font-size="11" font-family="system-ui">Router 多选 E3</text>
  <text x="285" y="108" text-anchor="middle" fill="#6b6b78" font-size="9" font-family="system-ui">loss 更低 → 梯度奖励</text>

  <line x1="355" y1="95" x2="405" y2="95" stroke="#f87171" stroke-width="1.2" marker-end="url(#arr1)"/>

  <rect x="410" y="70" width="160" height="50" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="490" y="92" text-anchor="middle" fill="#34d399" font-size="11" font-family="system-ui">E3 获得更多梯度</text>
  <text x="490" y="108" text-anchor="middle" fill="#6b6b78" font-size="9" font-family="system-ui">参数更新更频繁</text>

  <!-- Feedback arrow going back -->
  <path d="M490 120 L490 160 L95 160 L95 120" fill="none" stroke="#f87171" stroke-width="1.2" stroke-dasharray="4,3" marker-end="url(#arr1)"/>
  <text x="290" y="177" text-anchor="middle" fill="#f87171" font-size="10" font-family="system-ui">E3 变得更好 → 正反馈加剧</text>

  <!-- Dead experts -->
  <text x="95" y="30" text-anchor="middle" fill="#6b6b78" font-size="10" font-family="system-ui">E1, E5, E7: 零 token → 零梯度 → 参数冻结 💀</text>
</svg>

## 第一道防线：Auxiliary Loss（辅助损失）

既然主损失不管公平，那就**额外加一个损失**专门惩罚失衡。这就是 Switch Transformer（2022）提出的 load balancing loss：

$$L_{\text{balance}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

别急，我们一项一项拆：
- **N** = expert 数量（比如 8）
- **$f_i$** = 这批 token 中，实际被分到 Expert i 的比例（硬性计数）
- **$P_i$** = 所有 token 的 router 概率中，分给 Expert i 的平均概率（软性度量）
- **α** = 一个很小的系数，通常 0.01

### 为什么是 $f_i \times P_i$ 的形式？

想象完美均衡的情况：每个 expert 恰好收到 1/N 的 token（$f_i = 1/N$），router 对每个 expert 的平均概率也是 1/N（$P_i = 1/N$）。此时：

$$L = \alpha \cdot N \cdot N \cdot \frac{1}{N} \cdot \frac{1}{N} = \alpha$$

损失最小。但如果 Expert 3 霸占了所有 token（$f_3 = 1, P_3 \approx 1$），损失就爆炸了。

$f_i$ 和 $P_i$ 的乘积设计有一个精妙之处：$f_i$ 是 argmax 的结果（不可微分），但 $P_i$ 来自 softmax（可微分）。梯度通过 $P_i$ 这条路径反传，推动 router 把概率从拥挤的 expert 移向空闲的 expert。

### 一个微妙的矛盾

auxiliary loss 的 α 系数是个两难选择：

| α 太大（> 0.1） | α 太小（< 0.001） |
|---|---|
| 模型拼命追求均衡 | 依然可能坍塌 |
| 所有 expert 变得一模一样 | 几个 expert 死掉 |
| 失去 MoE 的意义——专业化 | 失去 MoE 的意义——利用率 |

实践中大多用 **α = 0.01** 作为起点。但这个矛盾——均衡性 vs 专业化——是 auxiliary loss 方案的根本局限。

## 第二道防线：Capacity Factor（容量因子）

在分布式训练中，GPU 不能动态调整内存大小。你必须**提前预设**每个 expert 最多处理多少 token。这就是 capacity factor 的作用：

$$\text{Expert Capacity} = \frac{\text{tokens\_per\_batch}}{\text{num\_experts}} \times \text{capacity\_factor}$$

举例：1024 个 token，8 个 expert，CF = 1.25：
$$\text{Expert Capacity} = \frac{1024}{8} \times 1.25 = 160$$

每个 expert 最多处理 160 个 token。如果完美均衡，每个 expert 处理 128 个，所以 CF = 1.25 给了 25% 的余量。

这就像设计停车场——预期 100 辆车，建 100 个车位意味着稍有波动就溢出。建 125 个留了缓冲，建 200 个浪费水泥。

| CF 值 | 含义 | 适用场景 |
|-------|------|---------|
| 1.0 | 零余量，稍有不均就溢出 | 只配合极强的均衡策略 |
| 1.25 | Switch Transformer 默认 | 大多数场景 |
| 1.5 | 宽裕，很少丢 token | 训练初期或小规模实验 |
| > 2.0 | 浪费——大部分 buffer 空着 | 不推荐 |

## 第三道防线：Token Dropping（丢弃溢出的 token）

当一个 expert 的容量满了，后续被分配到它的 token 怎么办？

**最简单的答案：丢掉。**

被"丢弃"的 token 跳过 MoE 层，直接通过残差连接（skip connection）继续前进。它不会凭空消失，但它错过了一层 expert 处理。在一个有 32 个 MoE 层的模型中，偶尔错过一层是可以容忍的——但如果经常发生，模型能力就会下降。

<svg viewBox="0 0 650 180" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:650px;margin:24px auto;display:block;">
  <defs>
    <marker id="arr2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10z" fill="#6e8eff"/></marker>
    <marker id="arr2r" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10z" fill="#f87171"/></marker>
  </defs>

  <!-- Expert box -->
  <rect x="200" y="30" width="180" height="70" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="290" y="55" text-anchor="middle" fill="#34d399" font-size="12" font-family="system-ui">Expert 3</text>
  <text x="290" y="75" text-anchor="middle" fill="#6b6b78" font-size="10" font-family="system-ui">容量: 160 tokens</text>
  <text x="290" y="90" text-anchor="middle" fill="#fbbf24" font-size="9" font-family="system-ui">已满 ✓ (160/160)</text>

  <!-- Token coming in (accepted) -->
  <rect x="30" y="45" width="80" height="30" rx="5" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1"/>
  <text x="70" y="64" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">Token A</text>
  <line x1="110" y1="60" x2="195" y2="60" stroke="#6e8eff" stroke-width="1.2" marker-end="url(#arr2)"/>
  <text x="152" y="52" text-anchor="middle" fill="#34d399" font-size="8" font-family="system-ui">✓ 第 45 个</text>

  <!-- Token coming in (dropped) -->
  <rect x="30" y="120" width="80" height="30" rx="5" fill="#1e1e2a" stroke="#f87171" stroke-width="1"/>
  <text x="70" y="139" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">Token B</text>
  <line x1="110" y1="135" x2="195" y2="105" stroke="#f87171" stroke-width="1.2" stroke-dasharray="4,3"/>
  <text x="160" y="130" text-anchor="middle" fill="#f87171" font-size="8" font-family="system-ui">✗ 容量满!</text>

  <!-- Residual path for dropped token -->
  <path d="M110 135 L440 135 L500 135" fill="none" stroke="#f87171" stroke-width="1.2" marker-end="url(#arr2r)"/>
  <text x="290" y="158" text-anchor="middle" fill="#f87171" font-size="9" font-family="system-ui">残差连接 → 跳过 MoE 层</text>

  <!-- Output -->
  <rect x="505" y="45" width="100" height="30" rx="5" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1"/>
  <text x="555" y="64" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">正常输出</text>

  <rect x="505" y="120" width="100" height="30" rx="5" fill="#1e1e2a" stroke="#f87171" stroke-width="1"/>
  <text x="555" y="139" text-anchor="middle" fill="#f87171" font-size="9" font-family="system-ui">降级输出</text>

  <!-- Output arrows -->
  <line x1="380" y1="65" x2="500" y2="60" stroke="#6e8eff" stroke-width="1.2" marker-end="url(#arr2)"/>
</svg>

### 不丢 Token 可以吗？——DeepSeek 的方案

DeepSeek-V3 做到了一个看似不可能的事：**零 token dropping**。

他们的思路是绕开 auxiliary loss 的矛盾。与其往梯度里加"均衡压力"，不如用一个完全独立于梯度的机制来调节：

```
routing_score(token, expert_i) = affinity(token, expert_i) + b_i
```

其中 $b_i$ 是给每个 expert 加的**偏置项**，但这个偏置项**不参与梯度计算**。它用一个简单的控制规则更新：

- Expert i 过载 → $b_i$ 减小 → router 少选它
- Expert i 空闲 → $b_i$ 增大 → router 多选它

这是一个纯粹的控制论方案——PID 控制器的思路。不污染主损失，不干扰专业化学习，只在 routing 决策层面轻轻推一把。结果是 256 个 expert 保持良好均衡，全程不用丢弃任何 token。

## 致命病症：Expert Death（专家死亡）

如果说 routing collapse 是"大多数 expert 变闲"，expert death 是它的终极形态：**一个 expert 彻底死掉，再也无法复活。**

死亡螺旋是这样的：

1. Expert 7 因初始化运气不好，收到的 token 比别人少
2. Token 少 → 梯度更新少 → 参数进步慢
3. 别的 expert 在进步，Expert 7 在原地踏步 → 相对性能越来越差
4. Router 发现送给 Expert 7 的 token loss 很高 → 彻底不选它了
5. Expert 7 收到 **零** token → **零** 梯度 → 参数完全冻结
6. 就算 router 偶尔"探索"一下送个 token 给 Expert 7，它的表现已经远远落后于其他训练了几万步的 expert → router 更加确信不该选它

这就是死亡。参数还在 GPU 内存里，但永远不会再被更新。在一个 64-expert 的模型中，没有干预的话，10-20% 的 expert 会这样死掉。

**检测方法**很简单：监控每个 expert 的 token 接收率。如果 `tokens_received[i] / total_tokens` 长期低于某个阈值（比如 0.01/N），这个 expert 正在死或已经死了。

## 隐形杀手：Router Logit 爆炸与 Z-Loss

这是一个更隐蔽的问题。训练过程中，router 的输出 logit 可能会**无限制地增长**。

为什么？因为 router 的"奖励"是选对 expert。如果 Expert 3 确实最适合某个 token，router 可以输出 `[0.1, 0.2, 47.3, 0.1, ...]` 也可以输出 `[0.1, 0.2, 1.3, 0.1, ...]`——两者都选了 Expert 3。但前者有个严重问题：softmax 被极度压缩。

当 logit 很大时：
- **数值不稳定**：`exp(47)` ≈ 2.5×10²⁰，在 bfloat16 下直接溢出
- **梯度悬崖**：softmax 几乎饱和时，logit 的微小扰动会导致概率剧烈跳变，训练变得极度不稳定

这就是 MoE 训练中臭名昭著的 **loss spike**——训练在平稳进行时突然出现一个巨大的损失跳跃，有时直接 NaN。本质是 router logit 默默增长到了临界点。

### Z-Loss：优雅的解法

ST-MoE（Zoph et al., 2022）提出了 router z-loss：

$$L_z = \frac{1}{B} \sum_{i=1}^{B} \left( \log \sum_{j=1}^{N} e^{x_{ij}} \right)^2$$

翻译成人话：**惩罚 softmax 分母的大小**。如果所有 logit 都很小，$\sum e^{x_{ij}}$ 就接近 N（每个 $e^x \approx 1$），$\log$ 后接近 $\log N$——一个温和的数值。如果 logit 爆炸，这个值会跟着爆炸，平方后惩罚就很重。

Z-loss 的妙处在于：它**不限制 router 选谁**，只限制 router **多大声地喊**。你可以偏好 Expert 3，但不需要用火箭筒表达这个偏好——用轻微的倾斜就够了。

典型的 z-loss 系数 $\lambda_z = 0.001$——非常小，对模型质量几乎没有影响，但足以防止 logit 失控。

## 其他稳定性技巧

### Jitter Noise（抖动噪声）

在 router 输入上乘以微小的随机噪声：

```
x_noisy = x × (1 + ε × uniform(-1, 1))    # ε ≈ 0.01
```

作用类似强化学习中的 ε-greedy 探索：偶尔让 token 被分配到"非最佳" expert，给每个 expert 学习的机会，防止 router 过早固化。

### Expert Choice Routing（让 expert 选 token）

一个颠覆性的思路（Zhou et al., 2022）：不让 token 选 expert，而是让**每个 expert 自己选它想处理的 token**。

- 计算一个 (experts × tokens) 的分数矩阵
- 每个 expert 取分数最高的 k 个 token
- 每个 expert 恰好处理 k 个 token → **天然完美均衡**

不需要 auxiliary loss，不需要 token dropping。代价是某些 token 可能被 0 个 expert 选中（跳过），或被多个 expert 选中（多次处理）。

## 现代 MoE 的稳定性"全家桶"

2024-2025 年的最佳实践是**多种机制叠加**：

<svg viewBox="0 0 600 260" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:600px;margin:24px auto;display:block;">
  <defs>
    <marker id="arr3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10z" fill="#6e8eff"/></marker>
  </defs>

  <!-- Layer 1: Prevention -->
  <rect x="20" y="20" width="560" height="45" rx="6" fill="#1a1a2e" stroke="#34d399" stroke-width="1.5"/>
  <text x="40" y="47" fill="#34d399" font-size="11" font-family="system-ui" font-weight="bold">预防层</text>
  <text x="140" y="47" fill="#ededf0" font-size="10" font-family="system-ui">Auxiliary Loss (α=0.01) 或 DeepSeek Bias 调节 + Jitter Noise (ε=0.01)</text>

  <!-- Layer 2: Containment -->
  <rect x="20" y="80" width="560" height="45" rx="6" fill="#1a1a2e" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="40" y="107" fill="#fbbf24" font-size="11" font-family="system-ui" font-weight="bold">遏制层</text>
  <text x="140" y="107" fill="#ededf0" font-size="10" font-family="system-ui">Capacity Factor (CF=1.25) → 硬性限制每个 expert 最多处理的 token 数</text>

  <!-- Layer 3: Damage control -->
  <rect x="20" y="140" width="560" height="45" rx="6" fill="#1a1a2e" stroke="#f87171" stroke-width="1.5"/>
  <text x="40" y="167" fill="#f87171" font-size="11" font-family="system-ui" font-weight="bold">止损层</text>
  <text x="140" y="167" fill="#ededf0" font-size="10" font-family="system-ui">Token Dropping → 容量溢出时丢弃 token，通过残差连接保底</text>

  <!-- Layer 4: Numerical stability -->
  <rect x="20" y="200" width="560" height="45" rx="6" fill="#1a1a2e" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="40" y="227" fill="#a78bfa" font-size="11" font-family="system-ui" font-weight="bold">数值层</text>
  <text x="140" y="227" fill="#ededf0" font-size="10" font-family="system-ui">Z-Loss (λ=0.001) → 防止 router logit 爆炸导致 NaN/loss spike</text>

  <!-- Arrow on the left -->
  <text x="10" y="140" fill="#6b6b78" font-size="18" font-family="system-ui">↕</text>
</svg>

这套"全家桶"的演进趋势很清晰：从 2020-2021 年的重手干预（大 α 值 + 激进 dropping），到 2024-2025 年的轻量外科手术式方案（DeepSeek 的 bias 调节，不需要 auxiliary loss，不丢 token）。核心洞察是：**MoE 不稳定的根源在于 router 学得太快（过早过度自信）或太不均匀（创造了失衡）。** 所有稳定性技巧都是在"减慢 router"（z-loss, jitter, dropout）、"强制均衡"（auxiliary loss, expert choice, capacity）或"解耦均衡与学习"（DeepSeek bias）。

## 回头看一眼实战中的数字

| 模型 | Expert 数 | 稳定性策略 | 是否 Drop Token |
|------|-----------|-----------|----------------|
| Switch Transformer | 128 | Aux loss (α=0.01) + CF=1.25 | 是 |
| ST-MoE (269B) | 64 | Aux loss + Z-loss (0.001) | 是 |
| DeepSeek-V2 | 160 | Bias 调节（无 aux loss） | 否 |
| DeepSeek-V3 | 256 | Bias 调节 + 极轻 seq-aux loss | 否 |

从 Switch Transformer 到 DeepSeek-V3，expert 数量从 128 增长到 256，但训练反而更稳定了。秘密不是"加更多约束"，而是"找到更优雅的平衡机制"。

## 下一篇预告

均衡和稳定性解决了，还有一个工程问题：256 个 expert 分布在几十张 GPU 上，token 在设备之间飞来飞去——这涉及 expert parallelism、all-to-all 通信、以及 DeepSeek 如何在 2048 张 H800 上高效训练一个 671B 参数的 MoE 模型。
