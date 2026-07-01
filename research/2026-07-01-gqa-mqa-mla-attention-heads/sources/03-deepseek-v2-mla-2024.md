---
url: https://arxiv.org/abs/2405.04434
title: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"
type: arxiv_paper
authors: DeepSeek-AI
year: 2024
accessed: 2026-07-01
quality: 5
relevance: core
---

## Abstract

We present DeepSeek-V2, a strong Mixture-of-Experts (MoE) language model characterized by
economical training and efficient inference. It comprises 236B total parameters, of which 21B
are activated for each token, and supports a context length of 128K tokens. DeepSeek-V2 adopts
innovative architectures including Multi-head Latent Attention (MLA) and DeepSeekMoE. MLA
guarantees efficient inference through significantly compressing the Key-Value (KV) cache into
a latent vector, while DeepSeekMoE enables training strong models at an economical cost through
sparse computation. Compared with DeepSeek 67B, DeepSeek-V2 achieves significantly stronger
performance, and meanwhile saves 42.5% of training costs, reduces the KV cache by 93.3%, and
boosts the maximum generation throughput to 5.76 times.

## Full-text extracted content (MLA sections 2.1.1-2.1.4, Appendix D)

### 2.1.2 Low-Rank Key-Value Joint Compression (core of MLA)

The core idea: instead of caching full per-head keys/values, project the hidden state h_t down
into a small latent vector c^KV_t via a down-projection matrix W^DKV:

  c^KV_t = W^DKV h_t          (down-projection to latent space, dim d_c << d_h * n_h)
  k^C_t  = W^UK c^KV_t        (up-project back to keys, only needed to compute attention)
  v^C_t  = W^UV c^KV_t        (up-project back to values)

During inference MLA only needs to CACHE c^KV_t — not the full keys/values — so cache size per
token is d_c * l elements (l = number of layers), a huge reduction vs storing 2 * n_h * d_h * l
elements for standard MHA.

Crucial trick: because W^UK can be mathematically absorbed/folded into the query projection
W^Q, and W^UV can be absorbed into the output projection W^O, at inference time the model does
not even need to materialize the up-projected keys/values explicitly for the attention score
computation — the compression is "free" in terms of extra matmuls at inference.

They also apply low-rank compression to the QUERIES (not for cache savings — queries aren't
cached — but to reduce activation memory during training):
  c^Q_t = W^DQ h_t;  q^C_t = W^UQ c^Q_t

### 2.1.3 Decoupled Rotary Position Embedding

Problem: RoPE is position-dependent and is normally applied directly to keys/queries. But if you
apply RoPE to the compressed key k^C_t, then W^UK becomes entangled with a position-dependent
rotation matrix. Since matrix multiplication is not commutative, W^UK can NO LONGER be folded
into W^Q at inference time — you'd have to recompute keys for the entire prefix at every step,
destroying the efficiency gain.

Solution ("decoupled RoPE"): split off a SEPARATE small set of "RoPE-only" queries and a single
shared "RoPE-only" key that carry positional information, while the main (non-RoPE) queries/
keys/values carry semantic content:
  q^R_t = RoPE(W^QR c^Q_t)     — per-head rotary queries
  k^R_t = RoPE(W^KR h_t)       — a SINGLE shared rotary key (like MQA, but just for the RoPE part)
  q_t,i = [q^C_t,i ; q^R_t,i]  (concat content query + rotary query)
  k_t,i = [k^C_t,i ; k^R_t]    (concat content key + shared rotary key)

This way the "absorb W^UK into W^Q" trick still works for the content part, and only the small
rotary key needs to additionally be cached. Total KV cache = (d_c + d^R_h) * l elements.

### 2.1.4 Comparison of KV Cache Sizes (Table 1)

| Mechanism | KV cache per token (# elements) | Capability |
|---|---|---|
| MHA | 2 n_h d_h l | Strong |
| GQA | 2 n_g d_h l | Moderate |
| MQA | 2 d_h l | Weak |
| MLA | (d_c + d^R_h) l ≈ 9/2 d_h l | Stronger |

For DeepSeek-V2: d_c = 4*d_h, d^R_h = d_h/2. So MLA's cache size equals GQA with only 2.25
groups — smaller than even a fairly aggressive GQA setting — yet MLA's quality is reported as
STRONGER than full MHA, not just "acceptable."

### Appendix D: Ablation — MHA vs GQA vs MQA (dense 7B models, 1.33T tokens, params aligned ~7B)

| Benchmark | MQA (7.1B) | GQA-8 (6.9B) | MHA (6.9B) |
|---|---|---|---|
| BBH (EM, 3-shot) | 33.2 | 35.6 | 37.0 |
| MMLU (Acc, 5-shot) | 37.9 | 41.2 | 45.2 |
| C-Eval (Acc, 5-shot) | 30.0 | 37.7 | 42.9 |
| CMMLU (Acc, 5-shot) | 34.6 | 38.4 | 43.5 |

Conclusion stated directly by the authors: "MHA demonstrates significant advantages over GQA and
MQA on these hard benchmarks." This directly motivates MLA's design goal — get MQA/GQA-level
cache savings WITHOUT this quality cliff.

### Appendix D.2: MLA vs MHA (small MoE ~16B and large MoE ~250B scale)

| | Small MoE MHA | Small MoE MLA | Large MoE MHA | Large MoE MLA |
|---|---|---|---|---|
| Activated params | 2.5B | 2.4B | 25.0B | 21.5B |
| KV cache/token (elements) | 110.6K | 15.6K (14%) | 860.2K | 34.6K (4%) |
| BBH (EM) | 37.9 | 39.0 | 46.6 | 50.7 |
| MMLU (Acc) | 48.7 | 50.0 | 57.5 | 59.0 |

MLA uses only 14% (small) / 4% (large) of MHA's KV cache while matching or *exceeding* MHA's
benchmark scores — this is the empirical basis for the claim "MLA achieves superior performance
compared with MHA, and meanwhile significantly reduces the KV cache."
