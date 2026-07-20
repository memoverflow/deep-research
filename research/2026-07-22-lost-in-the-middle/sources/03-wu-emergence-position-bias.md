---
url: https://arxiv.org/abs/2502.01951
title: "On the Emergence of Position Bias in Transformers"
type: arxiv_paper
authors: Xinyi Wu, Yifei Wang, Stefanie Jegelka, Ali Jadbabaie
year: 2025 (ICML 2025 camera-ready)
accessed: 2026-07-22
quality: 5
relevance: core
---

Graph-theoretic framework for position bias in multi-layer (masked) attention. Models the
causal attention mask as a directed graph and studies how many "computational paths" connect
an early token to a late token vs a late token to another late token, as a function of
network depth.

Key result: causal masking creates an asymmetric DAG structure where earlier tokens lie on
exponentially more paths (through the multi-layer stack) than later tokens — this asymmetry
strengthens with depth and is the structural driver of primacy bias / early-token dominance.

In an attention-only (no residual connection) setting, the model predicts complete collapse
of cumulative attention onto the very first token as depth grows — i.e., pure primacy, NOT
the observed U-shape (no recency bump). The authors flag this discrepancy with empirical
U-shaped observations as an open problem — since real transformers (which have residual
connections) show a strong recency effect at the end of the sequence too, not just primacy.

This gap is exactly what later papers (Herasimchyk et al. 2026, Chowdhury 2026) resolve by
adding residual connections into the analysis: residual connections give late tokens (esp.
the very last one) a "shortcut" path that doesn't get diluted by the causal averaging, which
restores the recency anchor and turns pure-primacy into the full U-shape.

Relevance: provides the rigorous graph/path-counting argument for why depth + causal masking
alone (no learning, no RoPE) generates positional imbalance — one of the two structural
pillars (the other being residual connections) that later work stitches together into the
complete U-shape theory.
