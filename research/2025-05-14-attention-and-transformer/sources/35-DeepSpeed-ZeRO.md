---
url: https://www.deepspeed.ai/tutorials/zero/
title: "ZeRO: Zero Redundancy Optimizer"
type: documentation
authors: Microsoft DeepSpeed
year: 2020
quality: 4
relevance: high
---

# DeepSpeed ZeRO

## Stages
- **Stage 1**: Partition optimizer states → 4× memory reduction
- **Stage 2**: + Gradient partitioning → 8× memory reduction
- **Stage 3**: + Parameter partitioning → linear scaling with data parallelism

## Extensions
- **ZeRO-Offload**: Move to CPU/NVMe for memory extension
- **ZeRO-Infinity**: NVMe-based, enables trillion-parameter training

## Combination with Other Parallelism
State-of-art (Megatron-Turing NLG 530B):
- 8-way tensor parallelism (within node)
- 35-way pipeline parallelism
- Data parallelism with ZeRO Stage 1
