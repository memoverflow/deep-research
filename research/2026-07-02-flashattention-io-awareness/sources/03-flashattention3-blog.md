---
url: https://tridao.me/blog/2024/flash3/
title: "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision"
type: blog_official
authors: Tri Dao et al.
year: 2024
accessed: 2026-07-02
quality: 5
relevance: core
---

FlashAttention-2 only achieves 35% of theoretical max FLOPs on H100. FlashAttention-3 targets Hopper-specific hardware features:

- **WGMMA**: new warpgroup matrix multiply-accumulate instruction, higher throughput than mma.sync used on Ampere.
- **TMA (Tensor Memory Accelerator)**: dedicated hardware unit for async data transfer between global and shared memory, freeing up registers.
- **FP8 low precision**: doubles tensor core throughput (989 TFLOPS FP16 vs 1978 TFLOPS FP8).

Three techniques: (1) exploit asynchrony of Tensor Cores + TMA to overlap computation and data movement via warp-specialization; (2) interleave block-wise matmul and softmax operations; (3) block quantization + incoherent processing for FP8 accuracy.

Results: 1.5-2.0x faster than FA2 with FP16, up to 740 TFLOPS (75% H100 utilization, up from 35%). With FP8, reaches ~1.2 PFLOPS with 2.6x smaller error than baseline FP8 attention with per-tensor quantization.

Context: from 2022-2024, LLM context length grew from 2-4K (GPT-3, OPT) to 128K (GPT-4) to 1M (Llama 3), partly enabled by these attention kernel advances.
