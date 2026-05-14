---
url: https://arxiv.org/abs/2402.18668
title: "Simple linear attention language models balance the recall-throughput tradeoff"
type: arxiv_paper
authors: Simran Arora et al. (Stanford/HazyResearch)
year: 2024
quality: 4
relevance: high
---

# Based: Linear Attention + Sliding Window

## Key Insight
Fundamental recall-throughput tradeoff: fixed-size recurrent states struggle with associative recall.

## Architecture
- Linear attention with Taylor expansion kernel: φ(q)^T φ(k) approximation
- Tiny sliding window attention (window size ~64)
- Combined in alternating layers

## Linear Attention Component
y_t = (Σ_{s≤t} φ(q_t)^T φ(k_s) v_s) / (Σ_{s≤t} φ(q_t)^T φ(k_s))
Feature map φ = second-order Taylor expansion

## Key Finding
Even tiny window (64 tokens) of precise attention dramatically improves recall while maintaining sub-quadratic complexity.

## Performance
- 1.3B: perplexity competitive with Transformers
- Much smaller window than Griffin/Mistral (which use 1024-4096)
