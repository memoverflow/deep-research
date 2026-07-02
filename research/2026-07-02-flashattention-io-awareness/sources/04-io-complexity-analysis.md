---
url: https://huggingface.co/blog/atharv6f/flash-attention-io-analysis
title: "2.2c: FlashAttention — IO Analysis and Evolution"
type: technical_blog
authors: Atharv Yeolekar
year: 2026
accessed: 2026-07-02
quality: 4
relevance: core
---

Detailed IO complexity derivation:

Standard attention IO: Read Q,K (Nd each), write S (N²), read S write P (N²), read P,V write O (Nd) → total ≈ 4N²+4Nd bytes ≈ O(N²) when N >> d.

FlashAttention IO: outer loop reads Q once (Nd total); inner loop re-reads K,V blocks T_r times each → N²d/B_r each; output written once (Nd). Total = O(N²d/B_r). With SRAM constraint M ≈ 4Bd+B² solved for optimal block size, B_r ≈ M/d, giving final complexity O(N²d²/M).

Concrete numbers (d=128, M=192KB, FP16, A100):
| N | Standard IO | FlashAttention IO | Reduction |
|---|---|---|---|
| 1024 | 8.4MB | 1.1MB | 7.6x |
| 4096 | 134.2MB | 16.8MB | 8.0x |
| 65536 | 34,359.7MB | 4,295.0MB | 8.0x |

Note: reduction factor ≈ N/(4d), grows linearly with N.

FLOPs comparison: FlashAttention does slightly MORE FLOPs than standard attention (extra online-softmax rescaling ops), yet is faster — because standard attention is memory-bound (arithmetic intensity 64 FLOP/byte, below A100 ridge point 156 FLOP/byte) while FlashAttention becomes compute-bound (506 FLOP/byte, above ridge point). This moves the operation across the roofline model's inflection point.

Example calc at N=4096: Standard attention time ≈ 134MB/2000GB/s = 0.067ms (memory-bound); FlashAttention time ≈ 8.6GFLOP/312TFLOP/s = 0.028ms (compute-bound) → 2.4x speedup from this mechanism alone.
