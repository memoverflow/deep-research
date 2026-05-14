---
url: https://arxiv.org/abs/2405.16712
title: "Zamba: A Compact 7B SSM Hybrid Model"
type: arxiv_paper
authors: Zyphra
year: 2024
quality: 4
relevance: high
---

# Zamba Architecture

## Design: Minimal Attention + Mamba Backbone
- Mamba backbone with single SHARED attention module
- Shared attention applied every 6 Mamba blocks
- One attention block with shared parameters reused throughout

## Parameters & Training
- 7B total parameters
- Trained on 1T tokens from open datasets

## Innovation
- Shows minimal attention (one shared block repeated) + strong SSM backbone matches transformer-only models
- Reduces memory overhead during inference
- Retains in-context learning (which pure Mamba struggles with)

## Performance
- Competitive with Llama-2 7B and Mistral 7B
- Best non-transformer model at 7B scale
