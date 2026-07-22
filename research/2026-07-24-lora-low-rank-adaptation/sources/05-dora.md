---
url: https://arxiv.org/abs/2402.09353
title: "DoRA: Weight-Decomposed Low-Rank Adaptation"
type: arxiv_paper
year: 2024
accessed: 2026-07-24
quality: 4
relevance: supporting (architecture variant)
---

## Abstract
Among PEFT methods, LoRA and variants are popular for avoiding additional inference costs, but there's an accuracy gap vs full fine-tuning (FT). This work introduces a weight decomposition analysis of the inherent differences between FT and LoRA, then proposes Weight-Decomposed Low-Rank Adaptation (DoRA): decomposes pretrained weight into magnitude and direction components for fine-tuning, using LoRA specifically for the directional updates to minimize trainable parameters. Enhances learning capacity and training stability of LoRA without additional inference overhead. Consistently outperforms LoRA fine-tuning LLaMA, LLaVA, VL-BART on commonsense reasoning, visual instruction tuning, image/video-text understanding.

## Key Content
- Weight decomposition: any weight matrix W can be written as W = m * (V/||V||) where m is a scalar magnitude and V/||V|| is the unit direction vector. DoRA's analysis found that LoRA tends to change magnitude and direction together in a correlated way, whereas full fine-tuning can adjust magnitude and direction more independently/flexibly.
- DoRA explicitly separates: learns the magnitude component directly (a small number of extra scalar parameters), and applies LoRA only to the directional component — this decoupling more closely mimics the flexibility full fine-tuning has, closing part of the accuracy gap while retaining LoRA's efficiency and zero-latency merge property.
