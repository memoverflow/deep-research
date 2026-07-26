# 研究计划: 原生稀疏注意力 (Native Sparse Attention)

## 主题
NSA (DeepSeek) 与 MoBA (Moonshot AI/Kimi) —— 两种 2025 年提出的"原生可训练"稀疏注意力机制,
区别于此前 H2O/Quest/MInference 等"事后推理剪枝"方法。

## 去重检查
已确认与已发布话题 (FlashAttention IO-Awareness, GQA/MQA/MLA, Attention Sink, Mixture of Depths 等)
均不重复。本篇聚焦"稀疏注意力模式设计"这一独立子话题:训练时学习的动态稀疏 block 选择机制。

## 子问题
1. 稀疏注意力的历史脉络:静态规则(Longformer/BigBird/Mistral SWA) → 事后剪枝(H2O/Quest/MInference) → 原生可训练(NSA/MoBA)
2. NSA 的三分支架构:压缩/选择/滑动窗口,及硬件对齐的 kernel 设计(GQA 组内共享选择)
3. MoBA 的无参数门控 top-k block 路由,及其"是 sliding window/sink 的推广"的理论论证
4. 两者共同揭示的范式转变:稀疏性必须内嵌进训练过程,而非推理阶段强加

## 搜索与提取记录
- web_search: 12 次不同 query(NSA、MoBA、Longformer、Mistral SWA、H2O、BigBird、InfLLM v2、
  training-inference mismatch、arithmetic intensity、retrieval heads 等)
- web_extract/curl 全文提取: 2 篇 arxiv 全文 PDF (NSA 2502.11089, MoBA 2502.13189) + abstract 页
- 未使用 browser/delegate_task(单一 agent 直接完成,材料充分)

## Level
L3 (30+ 搜索/提取综合,2 篇核心 arxiv 论文全文提取 + 3 篇背景论文 abstract 级引用)
