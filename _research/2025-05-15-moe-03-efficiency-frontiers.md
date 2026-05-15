---
title: "MoE 的效率密码与未来战场"
date: 2025-05-15
level: 3
series: "理解 Mixture of Experts"
series_order: 3
series_total: 3
tags: [MoE, scaling laws, inference, shared experts, soft MoE, expert specialization]
summary: "256 个 expert 学会了什么？MoE 的 scaling 为什么是对数增长？推理时 671B 参数全部躺在显存里却只用 37B——这笔账怎么算？以及正在改变游戏规则的新架构：Shared Expert、Soft MoE、MoE for Attention。"
---

> 前两篇我们搞清了 MoE 的骨架（router + experts + top-k）和稳定性难题（负载均衡 + 训练崩溃）。现在该问一个更根本的问题了：**这套系统的极限在哪？**

## 第一个问题：Expert 到底学了什么？

训练了 256 个 expert，每个 expert 真的在"专攻"某个领域吗？比如 Expert #17 负责医学、Expert #42 负责代码？

**答案出乎意料：不是。**

ST-MoE（Zoph et al., 2022）对 269B 参数的稀疏模型做了系统性的路由分析。他们发现 encoder 层的 expert 确实有明确的分工模式——但分工的维度不是"话题"，而是**词汇的表面特征**：

- 一个 expert 成为了"标点符号专家"
- 另一个专门处理专有名词
- 还有一个偏好虚词（the, of, is）

为什么不是按话题分？想想 router 的工作方式：一个线性层 + softmax，输入是当前 token 的 hidden state。在网络的前几层，hidden state 还没来得及整合太多上下文信息——它主要反映的是 token 本身的词汇特征。Router 的线性投影能最容易捕捉到的信号就是 token type，而不同 type 的 token 确实需要不同的计算（标点符号几乎不需要语义处理，而罕见术语需要大量上下文整合）。

### 更深层的 expert 呢？

Decoder 层的 expert 分工就模糊得多。这也合理——在自回归解码中，每个位置的 hidden state 融合了前面所有 token 的信息，变成了一个高度纠缠的表示。Router 的一层线性投影无法从这种纠缠表示中清晰地分离出"话题"信号。

### 多语言模型中的有趣模式

2025 年对多语言 MoE 路由的研究揭示了一个层次化的规律：

<svg viewBox="0 0 620 160" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:620px;margin:24px auto;display:block;">
  <defs>
    <marker id="arr-sp" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10z" fill="#6e8eff"/></marker>
  </defs>
  <!-- Timeline arrow -->
  <line x1="30" y1="80" x2="590" y2="80" stroke="#3a3a4a" stroke-width="1.5" marker-end="url(#arr-sp)"/>
  <text x="310" y="145" text-anchor="middle" fill="#6b6b78" font-size="10" font-family="system-ui">网络深度 →</text>
  
  <!-- Early layers -->
  <rect x="50" y="35" width="140" height="35" rx="6" fill="#1e1e2a" stroke="#f87171" stroke-width="1.5"/>
  <text x="120" y="57" text-anchor="middle" fill="#f87171" font-size="10" font-family="system-ui">前几层：语言分离</text>
  <text x="120" y="105" text-anchor="middle" fill="#6b6b78" font-size="9" font-family="system-ui">中文→Expert A</text>
  <text x="120" y="118" text-anchor="middle" fill="#6b6b78" font-size="9" font-family="system-ui">英文→Expert B</text>
  
  <!-- Middle layers -->
  <rect x="240" y="35" width="140" height="35" rx="6" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="310" y="57" text-anchor="middle" fill="#34d399" font-size="10" font-family="system-ui">中间层：跨语言共享</text>
  <text x="310" y="105" text-anchor="middle" fill="#6b6b78" font-size="9" font-family="system-ui">中/英/日 → 相同 Expert</text>
  <text x="310" y="118" text-anchor="middle" fill="#6b6b78" font-size="9" font-family="system-ui">（语义表示趋同）</text>
  
  <!-- Late layers -->
  <rect x="430" y="35" width="140" height="35" rx="6" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="500" y="57" text-anchor="middle" fill="#a78bfa" font-size="10" font-family="system-ui">后几层：语言重新分离</text>
  <text x="500" y="105" text-anchor="middle" fill="#6b6b78" font-size="9" font-family="system-ui">生成目标语言时</text>
  <text x="500" y="118" text-anchor="middle" fill="#6b6b78" font-size="9" font-family="system-ui">需要语言特定处理</text>
</svg>

前几层按语言分流（不同语言的 token 去不同 expert），中间层各语言汇聚到相同 expert（形成语言无关的语义表示），后几层再次分离（因为生成时需要语言特定的语法处理）。这与我们对 dense 多语言模型的认知一致——MoE 只是让这种内部分工变得可观测了。

### DeepSeek 的工程化专业化

DeepSeek-MoE（2024）不满足于这种自然涌现的弱专业化，他们主动**工程化**了更强的分工：

1. **细粒度分割**：把 8 个大 expert 拆成 64 个小 expert，选 8 个。路由的组合数从 C(8,2)=28 暴增到 C(64,8)≈40 亿种，router 有了极大的灵活性来组合不同的微专家。

2. **Shared Expert 隔离**：把"所有 token 都需要的通用计算"交给 shared expert，释放 routed expert 的容量去做真正的专业化。

效果：消融实验表明，移除任一 routed expert 造成的性能下降更加"定向"——每个 expert 确实覆盖了不重叠的功能区间，而不是传统 MoE 中那种"删掉一个 expert，模型几乎不受影响"的冗余格局。

## Scaling Laws：MoE 的收益曲线

现在来看一个对工程决策至关重要的问题：**加更多 expert，性能还能线性提升吗？**

### 残酷的对数增长

Clark et al.（2022, ICML Oral）推导了 MoE 的统一 scaling law。核心发现：给定每个 expert 的大小 N，模型性能（loss）随 expert 数量 E 的增长是**对数关系**：

> 把 expert 数量翻倍，loss 下降一个固定的常数——这个常数越来越不值得。

用人话说：从 8 expert → 16 expert 的提升 ≈ 从 16 → 32 的提升 ≈ 从 32 → 64 的提升。每次翻倍给你同样的固定收益，但你的参数量已经翻倍了。

### 为什么是对数而不是线性？

直觉是这样的：每多一个 expert，模型多了一个"处理选项"。但边际价值递减——

- 最常见的 token 类型已经被前几个 expert 服务好了
- 新 expert 捕获的是越来越罕见的模式
- Router 在选项更多时准确度略有下降
- 负载均衡更难，部分 expert 利用率不足

这和 dense model 的 scaling law 有本质区别。Dense model 的每个参数都参与每次前向传播，所以加参数给出多项式收益（Chinchilla power law）。MoE 每个参数只服务于被路由到它的 token，所以回报更薄。

### 一个具体的数字

Clark et al. 的估算：一个 N 参数/expert、E 个 expert 的路由模型，性能大约等价于一个 N × E^(1/α) 参数的 dense 模型（α ≈ 4-5）。也就是说：

- 64-expert 模型 ≈ 等价于 N × 64^0.2 ≈ N × 2.5 的 dense 模型
- **不是** N × 64 的 dense 模型！

你付出了 64 倍的参数存储，但只获得了 ~2.5 倍的等效 dense 性能。这笔账为什么还值？因为**计算量没有增长 64 倍**——每个 token 只用 top-k 个 expert，计算量只是 N × k 而非 N × E。MoE 买的是"用存储换算力"。

### 细粒度 MoE 的新发现

2024 年的研究补充了一个重要维度：**粒度也是 scaling 变量**。给定固定的总计算预算，更多更小的 expert 一致优于更少更大的 expert。最优粒度随计算预算增长而增加。

这解释了 DeepSeek-V3 为什么选择 256 个 expert（每个中间维度仅 2048）而不是 64 个大 expert（每个中间维度 8192）——前者在相同算力下表现更好。

## 推理的根本矛盾

训练时 MoE 是笔好买卖：用 1/18 的算力得到接近完整模型的能力。但推理时，**账单以另一种形式到来。**

### Prefill vs Decode：两个完全不同的世界

<svg viewBox="0 0 640 220" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;">
  <defs>
    <marker id="arr-inf" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10z" fill="#6e8eff"/></marker>
  </defs>
  
  <!-- Prefill box -->
  <rect x="20" y="20" width="280" height="180" rx="8" fill="#0f1f0f" stroke="#34d399" stroke-width="1.5"/>
  <text x="160" y="45" text-anchor="middle" fill="#34d399" font-size="13" font-family="system-ui" font-weight="bold">Prefill（处理输入）</text>
  <text x="160" y="70" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">大量 token 并行处理</text>
  <text x="160" y="90" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">→ 计算密集型（Compute-bound）</text>
  <text x="160" y="115" text-anchor="middle" fill="#34d399" font-size="10" font-family="system-ui">✓ GPU 算力被充分利用</text>
  <text x="160" y="135" text-anchor="middle" fill="#34d399" font-size="10" font-family="system-ui">✓ 多个 expert 同时被激活</text>
  <text x="160" y="155" text-anchor="middle" fill="#34d399" font-size="10" font-family="system-ui">✓ expert 存储开销被摊薄</text>
  <text x="160" y="185" text-anchor="middle" fill="#34d399" font-size="12" font-family="system-ui">MoE 在这里赢 ✓</text>
  
  <!-- Decode box -->
  <rect x="340" y="20" width="280" height="180" rx="8" fill="#1f0f0f" stroke="#f87171" stroke-width="1.5"/>
  <text x="480" y="45" text-anchor="middle" fill="#f87171" font-size="13" font-family="system-ui" font-weight="bold">Decode（逐 token 生成）</text>
  <text x="480" y="70" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">每次只处理 1 个新 token</text>
  <text x="480" y="90" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">→ 带宽密集型（Memory-bound）</text>
  <text x="480" y="115" text-anchor="middle" fill="#f87171" font-size="10" font-family="system-ui">✗ 671B 参数全部在显存里</text>
  <text x="480" y="135" text-anchor="middle" fill="#f87171" font-size="10" font-family="system-ui">✗ 每 token 只用 37B（5.5%）</text>
  <text x="480" y="155" text-anchor="middle" fill="#f87171" font-size="10" font-family="system-ui">✗ 94.5% 的显存被"浪费"读取</text>
  <text x="480" y="185" text-anchor="middle" fill="#f87171" font-size="12" font-family="system-ui">MoE 在这里痛 ✗</text>
</svg>

**Prefill** 一次性处理整个输入 prompt。几百上千个 token 同时流过网络，是大矩阵乘法——GPU 的算力利用率高，每 byte 显存访问对应大量计算。多个 expert 同时被不同 token 激活，GPU 忙碌且高效。

**Decode** 逐个生成 token。每次前向传播只处理一个新 token（或一个 batch 中每条序列一个 token）。需要从显存读取全部模型权重，但只做极少量计算——典型的带宽瓶颈。

对 dense 模型来说，decode 已经很痛了。对 MoE 来说更痛：

| 模型 | 总参数（显存占用） | 每 token 激活 | "浪费"比 |
|------|-------------------|--------------|---------|
| Mixtral 8×7B | 47B（~94GB FP16） | 13B | 3.6× |
| DeepSeek-V3 | 671B（~671GB FP8） | 37B | 18× |

DeepSeek-V3 在 decode 时，94.5% 的显存带宽在读取当前 token 根本不用的 expert 权重。你不能把这些 expert 扔掉——下一个 token 可能就要用。

### Expert Parallelism：All-to-All 通信墙

单张 GPU 装不下 671B 参数。解决方案是 **Expert Parallelism (EP)**：256 个 expert 分散在几十张 GPU 上，每张 GPU 只存几个 expert。

但这引入了一个通信问题：token 在当前 GPU 上产生，router 决定它要去的 expert 在另一张 GPU 上。于是 token 必须**发送**过去，expert 计算完再**返回**。这个"发过去-算-发回来"的 all-to-all 通信发生在每一个 MoE 层。

研究表明 all-to-all 通信可以占推理总时间的 **60%**。

### 工程解决方案

**1. Expert Offloading（卸载）**

对于消费级硬件：只把一部分 expert 放 GPU 显存，其余放 CPU 内存甚至 SSD。当 token 需要不在 GPU 上的 expert 时，动态加载。用 LRU 缓存保持热门 expert 常驻。适合 Mixtral 这种 8-expert 的小型 MoE。

**2. Speculative Expert Loading（投机预取）**

当前层还在算时，根据 router 的 logit 预测下一层需要哪些 expert，提前开始传输。如果预测对了，通信延迟被计算隐藏。ProMoE（2024）和 BuddyMoE 用这种策略实现了高缓存命中率。

**3. Prefill-Decode 分离（Disaggregated Serving）**

数据中心级方案：Prefill 和 Decode 用不同的 GPU 集群。Prefill 节点用计算优化型 GPU（H100），Decode 节点用带宽优化型 GPU（H20）。ByteDance 的 MegaScale-Infer 更进一步，把 attention 计算和 expert 计算也分离到不同 GPU 池，吞吐提升 1.9×。

**4. DeepSeek 的硬件感知设计**

DeepSeek-V3 的路由有一个工程约束：每个 token 最多发往 4 个节点（总共 8 个节点承载 expert）。这直接限制了跨节点通信量——节点内 NVLink 很快，跨节点 InfiniBand 是瓶颈。模型架构本身就在为推理系统让步。

## 正在改变游戏规则的新架构

### Shared Experts：把通用计算独立出来

上面提到 DeepSeek 用 Shared Expert 来提升专业化。这里展开讲它为什么对推理也有好处。

<svg viewBox="0 0 600 170" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:600px;margin:24px auto;display:block;">
  <defs>
    <marker id="arr-se" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10z" fill="#6e8eff"/></marker>
  </defs>
  
  <!-- Input -->
  <rect x="10" y="65" width="70" height="35" rx="6" fill="#1e1e2a" stroke="#60a5fa" stroke-width="1.5"/>
  <text x="45" y="87" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">Token</text>
  
  <!-- Split -->
  <line x1="80" y1="75" x2="130" y2="40" stroke="#fbbf24" stroke-width="1.2" marker-end="url(#arr-se)"/>
  <line x1="80" y1="90" x2="130" y2="120" stroke="#6e8eff" stroke-width="1.2" marker-end="url(#arr-se)"/>
  
  <!-- Shared Expert -->
  <rect x="135" y="20" width="130" height="40" rx="6" fill="#1e1e2a" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="200" y="37" text-anchor="middle" fill="#fbbf24" font-size="9" font-family="system-ui">Shared Expert ×1</text>
  <text x="200" y="51" text-anchor="middle" fill="#6b6b78" font-size="8" font-family="system-ui">所有 token 必经</text>
  
  <!-- Router + Routed Experts -->
  <rect x="135" y="100" width="130" height="50" rx="6" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="200" y="118" text-anchor="middle" fill="#6e8eff" font-size="9" font-family="system-ui">Router → Top-8</text>
  <text x="200" y="133" text-anchor="middle" fill="#6b6b78" font-size="8" font-family="system-ui">256 个 routed experts 选 8</text>
  
  <!-- Merge -->
  <line x1="265" y1="40" x2="330" y2="75" stroke="#fbbf24" stroke-width="1.2" marker-end="url(#arr-se)"/>
  <line x1="265" y1="125" x2="330" y2="90" stroke="#6e8eff" stroke-width="1.2" marker-end="url(#arr-se)"/>
  
  <!-- Sum -->
  <rect x="335" y="62" width="60" height="40" rx="20" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="365" y="86" text-anchor="middle" fill="#34d399" font-size="12" font-family="system-ui">∑</text>
  
  <!-- Output -->
  <line x1="395" y1="82" x2="430" y2="82" stroke="#6e8eff" stroke-width="1.2" marker-end="url(#arr-se)"/>
  <rect x="435" y="65" width="70" height="35" rx="6" fill="#1e1e2a" stroke="#60a5fa" stroke-width="1.5"/>
  <text x="470" y="87" text-anchor="middle" fill="#ededf0" font-size="9" font-family="system-ui">输出</text>
  
  <!-- Annotations -->
  <text x="470" y="30" fill="#fbbf24" font-size="9" font-family="system-ui">通用知识：语法、常识</text>
  <text x="470" y="140" fill="#6e8eff" font-size="9" font-family="system-ui">专项知识：领域、风格</text>
</svg>

在 DeepSeek-V3 中，每个 MoE 层有 1 个 shared expert + 256 个 routed expert。Shared expert 处理所有 token 都需要的基础计算（语法、通用推理），routed expert 只负责差异化的专项处理。

这个设计的工程含义：
- Shared expert 的利用率是 100%——永远不"浪费"
- Routed expert 可以做更极致的专业化，因为通用负担已被 shared expert 承担
- 推理时 shared expert 是确定性调用，不需要 all-to-all 通信——它就在本地 GPU 上

### Soft MoE：放弃离散，拥抱连续

标准 MoE 的 routing 是离散的（token 要么去 Expert A，要么不去），这带来了我们前两篇讨论的所有麻烦——负载失衡、不可微分、需要辅助损失。

Puigcerver et al.（2023, Google）提出了一个激进的方案：**完全放弃离散路由。**

Soft MoE 的思路是：不问"这个 token 去哪个 expert"，而问"每个 expert 应该处理所有 token 的什么加权组合"。

```
传统 MoE:    Token₁ → Expert_A (离散选择)
Soft MoE:    Expert_A 的输入 = 0.3×Token₁ + 0.5×Token₂ + 0.2×Token₃ (连续加权)
```

每个 expert 看到所有 token 的软混合，每个 token 的输出也来自所有 expert 的软混合。整个过程只有矩阵乘法和 softmax——**完全可微分**。

**好处**：不再需要辅助损失、不会负载失衡、不会 token drop、训练更平稳。

**致命限制**：计算 dispatch 权重需要看到**所有 token**——包括未来的。这在自回归生成中违反了因果性。所以 Soft MoE 目前主要用于 Vision Transformer（图片 patch 天然没有因果顺序），还不能直接用于 LLM 的 decode。

2024 年有工作尝试在序列级别（而非 token 级别）做 soft routing 来规避因果问题，但这仍是活跃研究方向。

### MoE for Attention：把稀疏扩展到注意力

传统 MoE 只稀疏化 FFN 层。但注意力层同样昂贵——32 个注意力头中，可能只有 4-6 个对当前 token 真正有用。

**SwitchHead（NeurIPS 2024）**：把每个 attention head 当作一个"expert"，用 router 选 top-k 个 head 来激活。未被选中的 head 不计算 attention matrix。结果：attention 计算量减少 8×。

配合 MoE FFN，就得到了"SwitchAll"——一个 attention 和 FFN **都稀疏**的 Transformer。

**DeepSeek-V2 的 MLA（Multi-head Latent Attention）**：虽然不是严格的 MoE for attention，但解决了类似的效率问题。它把 KV cache 压缩成低秩"潜在向量"，推理时只存一个小的压缩表示，需要时再解压。KV cache 减少约 93%。MLA + MoE FFN 的组合，是 DeepSeek-V2/V3 能服务 671B 模型的关键。

### 更远的前沿

**PEER（Google DeepMind, 2024）**：如果 256 个 expert 好，那一百万个呢？PEER 用 product key retrieval 从 100 万个微型 expert（每个 expert 就是一个神经元）中高效检索 top-k。本质上是一个极度稀疏的、token 级定制的 FFN。

**Expert Choice Routing**：让 expert 选 token（而非 token 选 expert）。每个 expert 挑选自己最适合处理的 C 个 token——天然完美均衡。但同样有因果性问题（expert 需要看到整个 batch 才能做选择），主要适用于 encoder 和 diffusion 模型。

## 这一切意味着什么？

MoE 的发展方向正在清晰：

1. **粒度越来越细** — 从 8 个大 expert 到 256 个小 expert，再到百万个微 expert
2. **辅助损失正在消亡** — DeepSeek 的 bias 调节证明可以不靠辅助损失维持均衡
3. **稀疏正在扩展** — 不止 FFN，attention 也要稀疏化
4. **架构为推理而设计** — 节点限制路由、MLA 压缩 KV、prefill-decode 分离
5. **Soft 和 Hard 的融合** — 完全离散太脆弱，完全连续又不适合自回归，未来可能是两者的混合

MoE 的故事从一个简单的想法开始——"让不同的 token 走不同的路"。但走到今天，它已经触及了深度学习最根本的权衡：容量 vs 计算、专业化 vs 通用化、训练效率 vs 推理成本。每一个新架构都在这些张力之间寻找更好的平衡点。

而下一个突破，可能来自我们今天还没想到的维度。
