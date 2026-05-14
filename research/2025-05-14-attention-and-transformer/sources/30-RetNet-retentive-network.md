---
url: https://arxiv.org/abs/2307.08621
title: "Retentive Network: A Successor to Transformer for Large Language Models"
type: arxiv_paper
authors: Yutao Sun et al. (Microsoft Research)
year: 2023
quality: 5
relevance: core
---

# RetNet: Retentive Network

## Key Innovation: Three Computation Paradigms
1. **Parallel** (training): like attention, O(n²) but parallelizable
2. **Recurrent** (inference): O(1) per step, constant memory
3. **Chunkwise** (hybrid): balanced for mixed workloads

## Retention Mechanism
Retention(X) = (QK^T ⊙ D) V

Where D is causal decay mask: D_{nm} = γ^{n-m} for n ≥ m, else 0.

Recurrent form: s_n = γ s_{n-1} + k_n^T v_n, o_n = q_n s_n

## Multi-Scale Retention
- Different γ values per head (Gated Multi-Scale Retention)
- Enables capturing patterns at different timescales

## Performance at 6.7B
- Matches Transformer perplexity
- 8.4x higher throughput during inference
- 70% memory reduction at 8K sequence length
