# Research Plan: Attention Sink

**Topic:** Attention Sink 现象——为什么 LLM 把大量注意力分配给序列第一个 token
**Level:** L3
**Date:** 2026-07-14

## Sub-questions
1. Attention sink 现象是如何被发现的？(StreamingLLM)
2. 为什么模型会学出这种行为？(over-mixing / rank collapse 理论解释, Barbero et al.)
3. 机制层面如何实现？(massive activations, quantizable transformers outlier)
4. 反向验证：改掉 softmax 会怎样？(softpick)
5. 训练动态：sink 何时出现？(When Attention Sink Emerges)
6. 下游影响：KV cache、量化、安全、架构设计（register tokens, DiT sink registers）

## Sources collected: 7 arxiv papers/surveys, full abstracts + key sections extracted via curl
## Images: 4 hand-drawn inline SVG diagrams (dark theme)
