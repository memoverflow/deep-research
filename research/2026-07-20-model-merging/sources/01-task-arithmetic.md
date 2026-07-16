---
url: https://arxiv.org/abs/2212.04089
title: "Editing Models with Task Arithmetic"
type: arxiv_paper
authors: Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, Ali Farhadi
year: 2022 (ICLR 2023)
accessed: 2026-07-16
quality: 5
relevance: core
---

Abstract & key content extracted from PDF (arxiv 2212.04089v3):

Proposes "task vectors" - direction in weight space obtained by subtracting pretrained weights from fine-tuned weights: τ = θ_ft - θ_pre. Arithmetic operations:
- Negation: τ_new = -τ → forgetting/unlearning (reduced toxic GPT-2 generations from 4.8% to 0.8%, WikiText-103 perplexity 16.4→16.9)
- Addition: τ_new = Σ τ_i → multi-task models (adding 2 task vectors retains 98.9% of specialized model accuracy; adding all 8 task vectors reaches 91.2% avg normalized accuracy)
- Task analogies: τ_D = τ_C + (τ_B - τ_A) → improves performance on task D using no labeled data from D

θ_new = θ + λτ_new, λ tuned on held-out validation set. No inference cost. Tested on CLIP ViT models (8 image classification tasks) and T5 models (GLUE tasks).

Key tables: Table 1 (forgetting via negation, CLIP), Table 2 (toxic GPT-2 mitigation), Table 3 (task vectors from HF hub improve T5 GLUE performance), Table 4 (task analogies for sentiment domain generalization).
