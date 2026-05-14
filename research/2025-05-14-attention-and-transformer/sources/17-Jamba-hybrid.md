---
url: https://arxiv.org/abs/2403.19887
title: "Jamba: A Hybrid Transformer-Mamba Language Model"
type: arxiv_paper
authors: AI21 Labs
year: 2024
quality: 5
relevance: core
---

# Jamba Architecture

## Design: Hybrid Transformer + Mamba + MoE

- Interleaves Transformer and Mamba layers in blocks
- Ratio: 1:7 (1 attention layer per 7 Mamba layers per block)
- MoE applied to selected MLP layers: 16 experts, top-2 routing

## Parameters
- Total: 52B parameters
- Active: 12B parameters

## Key Advantages
- 256K context window
- Fits on single 80GB GPU
- 3x throughput vs Mixtral 8x7B on long contexts
- Competitive with Llama-2 70B and Mixtral on benchmarks

## Insight
Combines: Transformer attention (precise recall) + Mamba SSM (efficient long-range) + MoE (capacity without compute)
