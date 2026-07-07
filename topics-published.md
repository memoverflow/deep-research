# 已发布话题清单

每次发布新文章后追加一行。定时任务选题时必须避开已有话题。

## Attention 与 Transformer 系列
- 2025-05-14: Attention 导论 — 让机器学会「看重点」：这个系列要讲什么
- 2025-05-14: Attention 机制起源 — 从「一张便签纸」到「随时翻阅全文」：Attention 的诞生
- 2025-05-14: Transformer 架构 — Attention Is All You Need：一篇论文如何改变了整个 AI
- 2025-05-14: 位置编码 — 模型怎么知道词的顺序？位置编码的难题
- 2025-05-14: 高效 Attention — 当序列长到装不下：高效 Attention 的效率战争
- 2025-05-14: BERT vs GPT + MoE 概述 — BERT vs GPT，以及「用稀疏换效率」的 MoE
- 2025-05-14: Scaling Laws 与对齐 — Scaling Laws 与对齐：AI 进步可以被预测吗？
- 2025-05-14: 线性复杂度替代 (Mamba/RWKV) — Attention 的挑战者们：Mamba、RWKV 与线性复杂度革命

## KV Cache 与 Prompt Caching 系列
- 2025-05-14: KV Cache 原理 — 为什么生成每个字都这么贵：KV Cache 的前世今生
- 2025-05-14: KV Cache 优化 (GQA/PagedAttention) — KV Cache 瘦身术：从 GQA 到 PagedAttention
- 2025-05-14: Prompt Caching — Prompt Caching：一招省掉 90% 的 API 账单

## MoE 系列
- 2025-05-14: MoE 底层原理 (Router) — MoE 底层原理：为什么大模型可以又大又快
- 2025-05-14: MoE 负载均衡与训练稳定性 — MoE 的生死难题：负载均衡与训练稳定性
- 2025-05-14: MoE 训练挑战 — MoE 训练的七大挑战：从负载均衡到专家并行
- 2025-05-14: MoE 效率与未来方向 — MoE 的效率密码与未来战场
- 2025-05-14: MoE 稀疏 vs 稠密 — MoE 的效率哲学：为什么稀疏能打败稠密

## Tokenization 系列
- 2025-05-15: Tokenization 基础 (BPE) — 什么是 Token？为什么 AI 不直接读文字
- 2025-05-15: Tokenization 算法对比 — BPE vs WordPiece vs Unigram：三种切词算法对决
- 2025-05-15: Tokenizer 工程实践 — Tokenizer 的工程实战：那些让人踩坑的细节
- 2025-05-15: Token 安全/Glitch/数学失败 — Token 的阴暗面：当切词出错时会发生什么
- 2025-05-15: 多模态 Tokenization — 万物皆可 Token：图像、声音、动作的统一语言
- 2025-05-15: 字节级模型/BLT/后 Tokenizer — Tokenizer 之死：当模型学会自己切词

## LLM 原理深度解析系列
- 2025-05-15: Adam/AdamW 优化器 — 为什么 Adam 是深度学习的默认优化器：从直觉到数学
- 2025-05-16: Speculative Decoding 投机解码 — 投机解码：让大模型「猜」着跑的数学魔术
- 2025-05-16: LoRA 低秩适应原理 — LoRA：为什么只训练 0.01% 的参数就够了
- 2025-05-17: Softmax 温度/饱和/数值稳定性 — Softmax 的秘密：温度、饱和与数值稳定性
- 2025-05-17: Mixed Precision Training (FP16/BF16/FP8) — 混合精度训练：用一半的比特做同样的事
- 2025-05-18: In-Context Learning 理论解释 — In-Context Learning：为什么不更新权重也能学？
- 2025-05-18: Grokking 过拟合后顿悟 — Grokking：过拟合之后的顿悟时刻
- 2025-05-19: Chain-of-Thought 计算复杂性 — Chain-of-Thought：为什么「说出思考过程」能让模型变聪明
- 2025-05-19: RoPE 旋转位置编码 — RoPE 旋转位置编码：用旋转让模型理解距离
- 2025-05-20: FFN/MLP 层记忆与计算 — FFN 层的秘密身份：Transformer 里的知识仓库
- 2025-05-20: LayerNorm/RMSNorm/DeepNorm 归一化变体 — 归一化的进化：从 LayerNorm 到 RMSNorm 再到 DeepNorm
- 2025-05-21: Residual Connection 梯度流分析 — 残差连接：深度网络的梯度高速公路
- 2025-05-21: 量化原理 (GPTQ/AWQ/GGUF) — 量化的数学：如何把 700 亿参数塞进一张显卡
- 2025-05-26: 学习率调度 (Warmup/Cosine/WSD) — 学习率调度的艺术：Warmup、Cosine Decay 与 WSD 背后的原理
- 2025-05-27: 交叉熵与KL散度 — 交叉熵与 KL 散度：训练 LLM 时我们到底在优化什么
- 2025-05-27: 双重下降 (Double Descent) — 双重下降：为什么过拟合之后模型反而变好了
- 2025-06-28: 权重初始化 (Xavier/Kaiming/µP) — 权重初始化的数学：从 Xavier 到 Kaiming 再到 µP
- 2025-06-29: 采样策略 (Temperature/Top-k/Top-p/Min-p) — 采样的艺术：Temperature、Top-k、Top-p 与 Min-p 如何控制 LLM 的创造力
- 2025-06-29: Linear Attention 核方法视角 — Linear Attention 的核方法视角：如何用一个数学技巧把 O(N²) 变成 O(N)
- 2025-06-30: Embedding 层几何结构 — Embedding 层的几何结构：为什么一张查找表能装下整个世界的意义

## RLHF 与对齐训练系列
- 2025-06-30: RLHF 对齐全景 — 从预训练到对齐：为什么 ChatGPT 不只是一个更大的 GPT-3
- 2025-06-30: SFT 监督微调 — SFT：用示范教会模型「怎么说话」
- 2025-06-30: 奖励模型 Bradley-Terry — 奖励模型：如何把「哪个回答更好」变成数字
- 2025-06-30: PPO 强化学习 — PPO：用强化学习微调语言模型的艺术与苦难
- 2025-06-30: DPO/GRPO 新范式 — DPO 与 GRPO：跳过奖励模型的新范式

## AI 图像生成系列
- 2025-06-30: AI生图进化史 (GAN/VAE/Diffusion/Flow) — 从 GAN 到 Diffusion：AI 生图技术的十年进化史
- 2025-06-30: Diffusion Model 数学原理 — Diffusion Model 原理：为什么「加噪再去噪」能生成图片
- 2025-06-30: Latent Diffusion/Stable Diffusion — Latent Diffusion 与 Stable Diffusion：在压缩空间里做魔法
- 2025-06-30: 条件控制 (CLIP/CFG/ControlNet) — 条件控制：从文字到图像的桥梁
- 2025-06-30: 主流模型对比 (SD/DALL-E/MJ/Flux/DiT) — 主流模型全景：SD、DALL-E、Midjourney、Flux 与架构未来
- 2025-06-30: 涌现能力 (Emergent Abilities) 度量标准之争 — 涌现能力是真的吗？一场关于度量标准的科学论战
- 2026-07-01: State Space Models 控制论基础 (SSM/S4/Mamba 数学) — State Space Models 的控制论基础：Mamba 为什么懂「工程」而不只是懂「深度学习」
- 2026-07-01: GQA/MQA/MLA 注意力头设计与KV Cache压缩 — GQA、MQA、MLA：当「多头」变成推理账单上最贵的一项
- 2026-07-02: ALiBi/NTK-aware/YaRN 长度外推原理 — 训练时只见过 4K，推理时却要处理 100K：ALiBi 和 YaRN 如何让模型「越活越长」
- 2026-07-02: FlashAttention IO-Awareness (SRAM/HBM/Tiling/Online Softmax/Recomputation) — FlashAttention 的秘密：真正的瓶颈不是算力，是搬数据
- 2026-07-03: 压缩即智能 (Hutter Prize/AIXI/Solomonoff/语言建模即压缩) — 压缩即智能：为什么预测下一个字，可能就是智能的全部秘密
- 2026-07-04: Beam Search vs Sampling 理论对比 (MAP解码失效/label bias/UID假说/exact search空字符串悖论) — 为什么最可能的句子反而是空的？Beam Search 与采样的理论对决
- 2026-07-06: Self-Attention QKV 几何直觉 (点积/softmax饱和/低秩瓶颈/多头容量权衡) — Self-Attention 的几何直觉：Q、K、V 到底在向量空间里做什么
- 2026-07-07: Multi-Head Attention 子空间分解/低秩瓶颈/头冗余 (Low-Rank Bottleneck/Sixteen Heads/Capacity-Based Rationale) — 多头注意力为什么有效：一个大脑同时开会，还是八个小组分头调查？
- 2026-07-08: Chinchilla 定律计算最优分配 (三种估计方法/复现争议/推理成本 overtraining) — Chinchilla 最优：一块钱的算力，该买大脑子还是买书？
- 2026-07-09: Mixture of Depths 条件计算/动态深度分配 (expert-choice routing/非因果top-k解决/MoDE) — Mixture of Depths：条件计算与「按需思考」的 Transformer
