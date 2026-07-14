---
url: https://arxiv.org/abs/2412.19437
title: "DeepSeek-V3 Technical Report (MTP module design)"
type: arxiv_paper
authors: DeepSeek-AI
year: 2024
accessed: 2026-07-16
quality: 5
relevance: core
---

DeepSeek-V3 explicitly lists "We investigate a Multi-Token Prediction (MTP) objective
and prove it beneficial to model performance. It can also be used for speculative
decoding for inference acceleration" as a core architectural contribution.

## Key content extracted (via search snippets + secondary sources)

- Unlike Meta's parallel independent output heads, DeepSeek-V3's MTP modules are
  SEQUENTIAL/causal-chained: D MTP modules, each consisting of a shared embedding
  layer, shared output head (unembedding), an unshared Transformer block, and an
  unshared projection matrix. Module k's input depends on module k-1's output,
  forming a complete causal chain that more closely mimics real autoregressive
  generation during training.
- Released checkpoint uses D=1: main model predicts t+1, single MTP module predicts
  t+2 (i.e., two-token-ahead training signal total).
- Shared embedding + shared output head between MTP modules and main model reduces
  parameter overhead — MTP module(s) contribute ~14B of the 685B total parameters
  (671B main + 14B MTP).
- Loss: L_MTP = (lambda/D) * sum_{k=1}^{D} L_MTP^k, with lambda scheduled to be
  larger early in training (e.g. ~0.3) and reduced later to avoid destabilizing main
  objective convergence.
- MTP module outputs are also reused at inference time for self-speculative decoding,
  avoiding need for a separately trained draft model.
- Source: DeepSeek-V3 Technical Report arXiv:2412.19437; corroborated by secondary
  write-ups (Medium "Understanding Multi-Token Prediction (MTP) in DeepSeek-V3",
  Red Shed AI paper notes, pixeli99/moe GitHub notes, aiwiki.ai).
