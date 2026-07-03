---
url: https://www.youtube.com/watch?v=dO4TPJkeaaU
title: "Compression for AGI — Jack Rae, Stanford MLSys Seminar #76"
type: talk_transcript_secondary
authors: Jack Rae (OpenAI)
year: 2023
accessed: 2026-07-03
quality: 4
relevance: core
---

Summary (via zhihu secondary source 预测即压缩,压缩即智能?): Jack Rae's core thesis is that the goal of AGI foundation models is maximal lossless compression of useful information. Uses Alice-Bob transmission thought experiment: Alice sends training code f to Bob; both sides train an autoregressive model in sync on transmitted data x_{1:t}; each new token x_{t+1} is encoded via arithmetic coding using shared predictive distribution P(x_{t+1}|x_{1:t}, f). Bits needed to transmit ≈ -log P(x_{t+1}), which equals training cross-entropy loss for that token. So the area under the training loss curve equals the bits needed for lossless compression of the full training dataset.

Key facts:
- Total compression cost S = |f| + n + sum(-log P(x_t+1))  (model description length + per-token codes)
- LLaMA 33B and 65B share same data description length (same training code) but 65B has lower loss => better compressor
- Estimated ~14x compression ratio for LLM vs 8.7x for best Hutter Prize text compressor (rough estimate from secondary source)
