---
url: https://arxiv.org/abs/2101.03961
title: "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"
type: arxiv_paper
authors: William Fedus, Barret Zoph, Noam Shazeer
year: 2021
quality: 5
relevance: core
---

# Switch Transformer

## Key Innovation: Top-1 Expert Routing
- Routes each token to exactly ONE expert (vs traditional top-2/top-k)
- Halves communication costs
- Simpler load balancing

## Architecture
- Replaces FFN layer with: router (linear + softmax) → N expert FFNs
- Capacity factor controls max tokens per expert
- Auxiliary load-balancing loss for uniform utilization
- First successful training of large sparse models in bfloat16

## Scale
- Up to 1.6 trillion parameters
- Same compute cost as much smaller dense model
- 4-7x pre-training speedup over T5-Base

## Innovations
1. Simplified routing (top-1 vs top-k)
2. Capacity factor for expert buffer sizing
3. Load-balancing loss
4. bfloat16 sparse training (selective precision for router)
