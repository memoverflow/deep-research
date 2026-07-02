---
url: https://www.abhik.ai/concepts/llms/flash-attention
title: "Flash Attention: IO-Aware Exact Attention"
type: technical_blog
year: 2025
accessed: 2026-07-02
quality: 3
relevance: supporting
---

Key innovations summary:
1. Tiling: split attention matrix into blocks that fit in SRAM (~100KB), avoiding HBM materialization.
2. Recomputation: instead of storing the large softmax(QK^T) matrix for the backward pass, FlashAttention stores only small per-row statistics (max, normalization sum) in HBM, and recomputes the attention scores on-the-fly in SRAM during backward pass. Trades a small amount of extra compute for a large reduction in memory storage and IO — same principle as the forward pass tiling.

Additional corroborating source: arxiv.org/pdf/2205.14135 states block-sparse FlashAttention extension achieves up to 3x speedup under causal masking (half the blocks masked out), beating prior approximate attention methods on wall-clock time, not just theoretical FLOPs.
