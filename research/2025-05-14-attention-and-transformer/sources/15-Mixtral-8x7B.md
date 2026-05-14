---
url: https://mistral.ai/news/mixtral-of-experts
title: "Mixtral 8x7B: A Mixture of Experts Model"
type: blog_official
authors: Mistral AI
year: 2023
quality: 4
relevance: core
---

# Mixtral 8x7B Architecture

## Parameters
- Total: 46.7B parameters
- Active per token: 12.9B (top-2 of 8 experts)
- Inference speed ≈ 13B dense model

## Architecture Details
- 32 transformer layers
- Each layer: 8 expert FFN blocks
- Router selects top-2 experts per token
- Sliding window attention (window size 4096)
- GQA with 8 KV heads
- 32K context window
- Apache 2.0 license

## Performance
- Outperforms Llama 2 70B on most benchmarks
- ~6x faster inference than Llama 2 70B
- Matches/exceeds GPT-3.5 Turbo on many tasks
