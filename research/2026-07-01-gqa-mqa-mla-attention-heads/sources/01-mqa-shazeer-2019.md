---
url: https://arxiv.org/abs/1911.02150
title: "Fast Transformer Decoding: One Write-Head is All You Need"
type: arxiv_paper
authors: Noam Shazeer
year: 2019
accessed: 2026-07-01
quality: 5
relevance: core
---

## Abstract

Multi-head attention layers, as used in the Transformer neural sequence model, are a powerful
alternative to RNNs for moving information across and between sequences. While training these
layers is generally fast and simple, due to parallelizability across the length of the sequence,
incremental inference (where such parallelization is impossible) is often slow, due to the
memory-bandwidth cost of repeatedly loading the large "keys" and "values" tensors. We propose a
variant called multi-query attention, where the keys and values are shared across all of the
different attention "heads", greatly reducing the size of these tensors and hence the memory
bandwidth requirements of incremental decoding. We verify experimentally that the resulting
models can indeed be much faster to decode, and incur only minor quality degradation from the
baseline.

## Key Points

- Root cause of the bottleneck: incremental (autoregressive) decoding cannot parallelize across
  sequence positions, so at every step the decoder must reload the full K/V cache from memory.
  Memory bandwidth, not compute, becomes the bottleneck.
- Multi-Query Attention (MQA): keep multiple query heads (for representational capacity) but
  share a single key head and a single value head across all of them.
- This reduces the K/V tensor size — and hence the memory traffic per decoding step — by a
  factor equal to the number of heads H.
- Empirically, MQA models decode much faster with only minor quality degradation vs. standard
  multi-head attention (MHA).
