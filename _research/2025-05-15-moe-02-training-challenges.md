---
title: "MoE 训练的七大挑战：从负载均衡到专家并行"
date: 2025-05-15
level: 3
series: "理解 Mixture of Experts"
series_order: 2
series_total: 3
tags: [MoE, load-balancing, training, expert-parallelism]
summary: "MoE 模型训练比 dense 模型难得多。Expert 会坍塌、会饿死、router 会僵化。这篇讲所有让 MoE 训练稳定的技巧——以及 DeepSeek-V3 如何用一个极简方案解决了核心矛盾。"
---

> MoE 的设计很优雅——但如果你真的去训练一个，会发现它像一个精密但脆弱的生态系统：任何微小的不平衡都会被放大到灾难级别。这篇来聊怎么驯服它。

## 马太效应：为什么 Expert 会坍塌

想象你开了一家公司，有 8 个员工（experts）。刚开始大家水平差不多，但因为运气，Employee 1 接到了几个好项目，表现不错。老板（router）注意到了，开始优先把新项目分给 Employee 1。

Employee 1 得到更多项目 → 经验更丰富 → 表现更好 → 老板更信任 → 分配更多项目...

其他 7 个员工逐渐被边缘化，最终完全闲置。你花了 8 个人的工资，只有 1-2 个人在干活。

这就是 **Expert Collapse（专家坍塌）**——MoE 训练中最致命的问题。没有干预措施的话，它**必然发生**。

<svg viewBox="0 0 650 170" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:650px;margin:24px auto;display:block;">
  <text x="160" y="15" text-anchor="middle" fill="#34d399" font-size="10" font-family="system-ui" font-weight="bold">训练初期：均匀分配</text>
  <text x="500" y="15" text-anchor="middle" fill="#fb7185" font-size="10" font-family="system-ui" font-weight="bold">训练后期：坍塌</text>
  <!-- Early: equal circles -->
  <circle cx="60" cy="80" r="18" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/><text x="60" y="84" text-anchor="middle" fill="#34d399" font-size="7">E1</text>
  <circle cx="110" cy="80" r="18" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/><text x="110" y="84" text-anchor="middle" fill="#34d399" font-size="7">E2</text>
  <circle cx="160" cy="80" r="18" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/><text x="160" y="84" text-anchor="middle" fill="#34d399" font-size="7">E3</text>
  <circle cx="210" cy="80" r="18" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/><text x="210" y="84" text-anchor="middle" fill="#34d399" font-size="7">E4</text>
  <circle cx="260" cy="80" r="18" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/><text x="260" y="84" text-anchor="middle" fill="#34d399" font-size="7">E5</text>
  <text x="160" y="120" text-anchor="middle" fill="#9494a0" font-size="8" font-family="system-ui">各 ~20% 负载 ✓</text>
  <!-- Arrow -->
  <text x="340" y="80" text-anchor="middle" fill="#6b6b78" font-size="20">→</text>
  <text x="340" y="100" text-anchor="middle" fill="#6b6b78" font-size="8" font-family="system-ui">马太效应</text>
  <!-- Late: collapsed -->
  <circle cx="420" cy="80" r="35" fill="#1e1e2a" stroke="#fb7185" stroke-width="2.5"/><text x="420" y="78" text-anchor="middle" fill="#fb7185" font-size="8" font-weight="bold">E1</text><text x="420" y="90" text-anchor="middle" fill="#fb7185" font-size="7">45%</text>
  <circle cx="500" cy="80" r="28" fill="#1e1e2a" stroke="#fbbf24" stroke-width="2"/><text x="500" y="78" text-anchor="middle" fill="#fbbf24" font-size="8">E3</text><text x="500" y="90" text-anchor="middle" fill="#fbbf24" font-size="7">30%</text>
  <circle cx="560" cy="70" r="10" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1"/><text x="560" y="73" text-anchor="middle" fill="#6b6b78" font-size="6">E2</text>
  <circle cx="560" cy="95" r="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1"/><text x="560" y="98" text-anchor="middle" fill="#6b6b78" font-size="5">E4</text>
  <circle cx="590" cy="80" r="6" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="0.5"/><text x="607" y="83" fill="#6b6b78" font-size="6">☠️ E5</text>
  <text x="500" y="135" text-anchor="middle" fill="#fb7185" font-size="8" font-family="system-ui">2个expert干活，3个半死不活 ✗</text>
  <!-- Bottom note -->
  <rect x="100" y="145" width="450" height="20" rx="4" fill="#1a1a24"/>
  <text x="325" y="159" text-anchor="middle" fill="#9494a0" font-size="8" font-family="system-ui">花了 5 个 expert 的钱，只得到了 2 个 expert 的效果 → 必须干预</text>
</svg>

## 辅助损失：强制维持公平

解决方案的直觉很简单：**惩罚不均匀的分配**。

我们需要两个信号：
- **f_i**：Expert i 实际拿到了多少 token（"结果"）
- **P_i**：Router 给 Expert i 分配的平均概率（"意愿"）

把它们乘在一起：如果某个 expert **既拿到很多 token，又被 router 给了很高概率**，乘积就大，惩罚就重。这逼迫 router "分散投资"而不是"押注一个"。

但这里有个根本矛盾——惩罚系数 α 太大会让 router 变成随机分配器（只追求均匀，不考虑谁最合适）。太小则无法阻止坍塌。**你不得不在"均匀性"和"路由质量"之间做权衡。**

这个矛盾直到 DeepSeek-V3 才被优雅解决——后面讲。

## 容量因子：Expert 满了怎么办

即使有辅助损失，分配也不可能完美均匀。当一个 expert 收到的 token 超过缓冲区容量时：

**Token Dropping（丢弃）**：溢出的 token 直接跳过，只走 residual connection。简单但损失信息。

**Token Rerouting（重路由）**：发到第二选择 expert。保留信息但实现复杂。

容量因子 CF = 1.25 意味着给每个 expert 留 25% 的余量。CF 越大越安全，但越浪费内存。

## 三种急性训练不稳定

### Routing Oscillation（路由振荡）

Token 在 expert 之间反复跳——就像一个频繁跳槽的员工，每份工作都干不长，所以每份都没学到深度。Expert A 刚开始学数学，router 觉得它学得太慢，转给 Expert B，B 也是从零开始...循环往复。

### Expert Death（专家死亡）

比坍塌更极端——某个 expert 的路由概率降到**恰好为零**，永远不再接收 token。参数永远不更新，不可逆地"死"了。

### Softmax 饱和

Router 的 logit 越来越大 → softmax 输出趋近 one-hot → 梯度接近零 → router 被"锁死"，无法调整路由。就像一个越来越固执的路由器：越来越确信自己对，同时越来越听不进新信息。

## Router Z-Loss：驯服大 Logits

惩罚 logits 的尺度。具体方式：对每个 token 的 logits 算 log-sum-exp（所有 logit 最大值的平滑近似），取平方作为惩罚项。

logits 适度时惩罚几乎为零（不干扰正常训练），变大时快速增长（强力阻止）。和辅助损失正交——辅助损失管均匀性，Z-Loss 管数值稳定性，两者同时使用。

## DeepSeek-V3 的突破：无辅助损失的负载均衡

DeepSeek 问了一个关键问题：能不能**完全不通过梯度来实现均衡**？

答案是**动态偏置**——给每个 expert 的路由分数加一个 bias bᵢ，但这个 bias：
1. **只影响"谁被选中"**（Top-K 决策），不影响选中后的权重
2. **不参与反向传播**——用简单规则更新：负载高了就减 bias，低了就加

这意味着 router 的梯度完全不受干扰——它可以纯粹优化"找最好的 expert"，均衡性由一个独立的、非梯度的机制来保证。

效果：比传统辅助损失方法性能更好。有时候最好的方案就是把"学习目标"和"约束机制"解耦。

## Expert Parallelism：GPU 间的快递系统

256 个 expert 不可能塞进一块 GPU。解决方案：每块 GPU 放一部分 expert，token 按需在 GPU 间传输。

<svg viewBox="0 0 650 160" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:650px;margin:24px auto;display:block;">
  <text x="325" y="15" text-anchor="middle" fill="#9494a0" font-size="10" font-family="system-ui">All-to-All 通信：每块 GPU 都要和其他所有 GPU 交换 token</text>
  <!-- 4 GPUs -->
  <rect x="30" y="35" width="120" height="50" rx="6" fill="#1e1e2a" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="90" y="55" text-anchor="middle" fill="#22d3ee" font-size="9" font-family="system-ui">GPU 0 (Expert 1,2)</text>
  <text x="90" y="72" text-anchor="middle" fill="#9494a0" font-size="7" font-family="system-ui">持有 token A,B,C</text>
  <rect x="200" y="35" width="120" height="50" rx="6" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="260" y="55" text-anchor="middle" fill="#34d399" font-size="9" font-family="system-ui">GPU 1 (Expert 3,4)</text>
  <text x="260" y="72" text-anchor="middle" fill="#9494a0" font-size="7" font-family="system-ui">持有 token D,E,F</text>
  <rect x="370" y="35" width="120" height="50" rx="6" fill="#1e1e2a" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="430" y="55" text-anchor="middle" fill="#fbbf24" font-size="9" font-family="system-ui">GPU 2 (Expert 5,6)</text>
  <text x="430" y="72" text-anchor="middle" fill="#9494a0" font-size="7" font-family="system-ui">持有 token G,H,I</text>
  <rect x="540" y="35" width="90" height="50" rx="6" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="585" y="55" text-anchor="middle" fill="#a78bfa" font-size="9" font-family="system-ui">GPU 3 (E7,8)</text>
  <text x="585" y="72" text-anchor="middle" fill="#9494a0" font-size="7" font-family="system-ui">token J,K,L</text>
  <!-- Cross arrows -->
  <line x1="90" y1="85" x2="260" y2="100" stroke="#6e8eff" stroke-width="0.8" opacity="0.6"/>
  <line x1="90" y1="85" x2="430" y2="100" stroke="#6e8eff" stroke-width="0.8" opacity="0.6"/>
  <line x1="260" y1="85" x2="90" y2="100" stroke="#6e8eff" stroke-width="0.8" opacity="0.6"/>
  <line x1="260" y1="85" x2="430" y2="100" stroke="#6e8eff" stroke-width="0.8" opacity="0.6"/>
  <line x1="430" y1="85" x2="90" y2="100" stroke="#6e8eff" stroke-width="0.8" opacity="0.6"/>
  <line x1="430" y1="85" x2="585" y2="100" stroke="#6e8eff" stroke-width="0.8" opacity="0.6"/>
  <line x1="585" y1="85" x2="260" y2="100" stroke="#6e8eff" stroke-width="0.8" opacity="0.6"/>
  <text x="325" y="108" text-anchor="middle" fill="#6e8eff" font-size="8" font-family="system-ui">All-to-All: token 按目标 expert 发送到对应 GPU</text>
  <!-- Timeline -->
  <rect x="80" y="125" width="120" height="18" rx="3" fill="#fb7185" opacity="0.3"/>
  <text x="140" y="137" text-anchor="middle" fill="#fb7185" font-size="7" font-family="system-ui">通信 (Dispatch)</text>
  <rect x="210" y="125" width="150" height="18" rx="3" fill="#34d399" opacity="0.3"/>
  <text x="285" y="137" text-anchor="middle" fill="#34d399" font-size="7" font-family="system-ui">本地 Expert 计算</text>
  <rect x="370" y="125" width="120" height="18" rx="3" fill="#fb7185" opacity="0.3"/>
  <text x="430" y="137" text-anchor="middle" fill="#fb7185" font-size="7" font-family="system-ui">通信 (Combine)</text>
  <text x="325" y="155" text-anchor="middle" fill="#6b6b78" font-size="7" font-family="system-ui">优化目标：让红色（通信）和绿色（计算）尽量重叠</text>
</svg>

流程：Router 决定 token 去哪 → All-to-All 把 token 发到对应 GPU → 本地 expert 计算 → All-to-All 把结果送回。

All-to-All 是最昂贵的集合通信操作——每块 GPU 都要和所有其他 GPU 交换数据。优化的关键是让通信和计算**重叠**，DeepSeek-V3 的 DualPipe 方案几乎完全做到了这一点。

## 下一篇预告

模型训好了——但怎么高效地跑推理？MoE 的稀疏激活给推理带来了独特的机会和挑战：prefill 时负载均衡、decode 时大部分 expert 闲置、内存里要装所有 expert 但每次只用几个...下一篇来聊 MoE 的效率哲学和未来方向。
