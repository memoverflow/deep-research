---
url: https://hazyresearch.stanford.edu/blog/2023-07-17-flash2
title: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
type: blog_official
authors: Tri Dao
year: 2023
accessed: 2026-07-02
quality: 5
relevance: core
---

FlashAttention-2 rewrites the algorithm from scratch using CUTLASS 3.x/CuTe primitives, ~2x faster than FA1, reaching up to 230 TFLOPs/s on A100 (FP16/BF16), 72% model FLOP utilization end-to-end.

Three key improvements:
1. **Fewer non-matmul FLOPs**: A100 has 312 TFLOPs/s matmul (FP16/BF16) vs only 19.5 TFLOPs/s non-matmul FP32 — each non-matmul FLOP is ~16x more "expensive". FA2 rewrites online softmax to reduce rescaling, bound-checking, causal masking ops.
2. **Better parallelism**: FA1 parallelizes only over (batch_size × num_heads) thread blocks — inefficient when this product is small (long sequences / small batch). FA2 additionally parallelizes over the sequence length dimension.
3. **Better work partitioning within thread blocks**: FA1 splits K,V across 4 warps ("sliced-K") requiring warps to write intermediate results to shared memory and synchronize. FA2 instead splits Q across warps while keeping K,V shared — no inter-warp communication needed, less shared memory read/write.

Context: motivated by explosion of long-context models (GPT-4 32k, MPT 65k, Claude 100k) circa 2023.
