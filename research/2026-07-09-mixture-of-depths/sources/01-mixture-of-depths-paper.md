---
url: https://arxiv.org/abs/2404.02258
title: "Mixture-of-Depths: Dynamically allocating compute in transformer-based language models"
type: arxiv_paper
authors: David Raposo, Sam Ritter, Blake Richards, Timothy Lillicrap, Peter Conway Humphreys, Adam Santoro (Google DeepMind)
year: 2024
accessed: 2026-07-09
quality: 5
relevance: core
---

## Abstract
Transformer-based language models spread FLOPs uniformly across input sequences. This paper shows transformers can learn to dynamically allocate FLOPs to specific sequence positions, optimizing allocation per-layer across model depth. The method caps the number of tokens (k) that participate in self-attention/MLP at a given layer via top-k routing. Because k is fixed a priori, the computation graph stays static (unlike other conditional computation methods), but token identities routed are dynamic. Models match baseline performance for equal FLOPs/wall-clock training time while using a fraction of FLOPs per forward pass, and can be up to 50% faster during sampling.

## Key Content (full text extracted via PDF)

### Motivation
- Not all tokens need equal compute to predict correctly, yet vanilla transformers spend identical FLOPs per token.
- Conditional computation (term coined by Bengio 2013) tries to spend compute only when needed, but many prior methods (adaptive computation time, early-exit) introduce dynamic computation graphs that don't map well to today's hardware (which wants static graphs / known tensor sizes).
- Goal: define a **static, user-set compute budget** while letting the network decide **which tokens** get the full computation.

### Core Mechanism
- Borrows routing logic from Mixture-of-Experts (MoE), but instead of routing to one of many experts, MoD routes to a single "expert" (the standard self-attn+MLP block) OR skips it via the residual connection.
- Router produces a scalar weight per token per block: r_i^l = w_θ^T x_i^l (a simple linear projection).
- **Expert-choice routing**: rather than tokens picking a path (token-choice, prone to load imbalance), each computational path picks its own top-k tokens by weight. This guarantees perfect load balance (exactly k tokens per block) with no auxiliary balancing loss needed.
- Capacity C defines how many of S total tokens get full block computation; β = 1 - C/S percentile threshold. Tokens above threshold get f(x) [self-attn+MLP] + residual; tokens below just pass through residual only.
- Router weight is multiplied into the block output for routed tokens — this puts the router on the gradient path so it's trained end-to-end by the language-modeling loss itself (no separate RL / auxiliary routing loss required for the core mechanism).
- Because MoD also routes attention (not just MLP), skipped tokens don't just save their own compute — they're also removed from the set of keys/values other tokens attend to, cutting attention cost quadratically in the routed capacity.

### Best configuration found
- Route every OTHER block (interleaved with full vanilla blocks) — critical for performance.
- Capacity 12.5% (top-256 of 2048 tokens) is optimal; going below that starts hurting performance. So 87.5% of tokens skip a given routed block.
- Learned routing is essential — replacing it with random/stochastic top-k routing performs drastically worse than baseline.
- MoD can combine with MoE (called MoDE) for compounding gains. "Integrated MoDE" (treating the residual skip as a null expert inside the MoE routing) works better than naively shrinking MoE expert capacity.

### Results
- isoFLOP training curves: optimal MoD models achieve equal-or-lower loss than isoFLOP-optimal vanilla baselines, and MoD-optimal models tend to have MORE parameters but need FEWER FLOPs per forward pass (fewer FLOPs per parameter due to skipping).
- Example: a 220M MoD model matches the isoFLOP-optimal 220M baseline's loss while being ~60% faster to step during training and ~66% faster in another cited example; up to 50% faster during autoregressive sampling.
- Memory savings observed at larger scale too (smaller TPU topology needed), with implications for KV cache size during autoregressive sampling (since skipped tokens don't produce keys/values at that layer).

### The sampling problem (non-causality) and how it's solved
- Top-k is inherently NON-CAUSAL: whether token i is in the top-k depends on router weights of tokens that come AFTER it in the sequence — information unavailable during autoregressive generation.
- Two fixes tested:
  1. Auxiliary binary cross-entropy loss (targets = whether each token was in top-k) that costs ~0.2-0.3% loss degradation but keeps training simple; at inference use the router's raw sigmoid output vs 0.5 threshold (fully causal). Learns to ~99% accuracy quickly.
  2. A small separate predictor MLP (like a second router, stop-gradient input) trained to predict top-k membership without touching the main LM loss at all — reaches ~97-99% accuracy and doesn't slow down training steps.
- Autoregressive eval: switching from non-causal top-k (training) to causal predictor-based routing (sampling) causes minimal performance degradation.

### Relation to prior work (background section, useful for narrative)
- Conditional computation term: Bengio 2013, Bengio et al. 2013/2016.
- Early-exit methods (Elbayad et al. 2019; DeeBERT — Liu et al. 2021 style; Schuster et al. 2022 CALM): token exits network early and skips ALL remaining layers serially. MoD differs — a token can skip MIDDLE layers then re-engage with LATER layers, and can be attended to by tokens that went through all middle layers. Authors explicitly flag this as a hypothesized advantage.
- Universal Transformer (Dehghani et al. 2018) — adaptive number of iterations with shared weights, but dynamic computation graph (hardware-unfriendly).
- ToMe / Bolya et al. 2023 — token merging for ViT inference, unsupervised (no learning).
- CoLT5 (Ainslie et al. 2023) — closest prior work: soft top-k conditional routing choosing heavy/light FFN pathway AND choosing whether a token attends to all or a few tokens. But CoLT5 is encoder-decoder, doesn't need to solve causal/autoregressive top-k. MoD is decoder-only and introduces the causal predictor-router to make sampling work.
- MoE (Shazeer et al. 2017, Fedus et al. 2022 Switch Transformer, Lepikhin et al. 2020 GShard, Zoph et al. 2022 ST-MoE) — MoD borrows expert-choice routing machinery but with a single "expert" that is dynamically skippable rather than routing among many experts.

### Interesting observed pattern
- Analysis of routing decisions (Figure 5) shows some tokens consistently engage every routed block along depth, others route around whenever possible. Preliminary analysis: tokens that engage more frequently correlate with higher-entropy (harder to predict) outputs — i.e. the network seems to allocate more depth to genuinely harder predictions, matching the core hypothesis.
