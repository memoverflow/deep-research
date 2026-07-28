---
url: https://arxiv.org/abs/2404.07647
title: "Why do small language models underperform? Studying Language Model Saturation via the Softmax Bottleneck"
type: arxiv_paper
authors: Nathan Godey, Éric de la Clergerie, Benoît Sagot
year: 2024
accessed: 2026-07-31
quality: 5
relevance: core
---

## Abstract
Recent advances in language modeling consist in pretraining highly parameterized neural networks on extremely large web-mined text corpora. Training and inference with such models can be costly in practice, which incentivizes the use of smaller counterparts. However, it has been observed that smaller models can suffer from saturation, characterized as a drop in performance at some advanced point in training followed by a plateau. In this paper, we find that such saturation can be explained by a mismatch between the hidden dimension of smaller models and the high rank of the target contextual probability distribution. This mismatch affects the performance of the linear prediction head used in such models through the well-known softmax bottleneck phenomenon. We measure the effect of the softmax bottleneck in various settings and find that models based on less than 1000 hidden dimensions tend to adopt degenerate latent representations in late pretraining, which leads to reduced evaluation performance.

## Key takeaways
- This is the modern (2024) empirical validation of the 2018 theoretical softmax bottleneck paper, applied to actual pretrained transformer LMs (not just RNNs).
- "Saturation" phenomenon: small models' loss curves show a drop in performance then plateau late in pretraining — previously mysterious, now explained.
- Root cause: hidden dimension d too small relative to the effective/target rank of the true next-token distribution matrix → forces degenerate ("anisotropic"/collapsed) hidden representations in late training, which hurts downstream eval performance even though training loss looks fine.
- Threshold observed: models with hidden dimension < ~1000 are particularly susceptible.
- Practical implication: this is a partial explanation for why simply shrinking hidden width (holding depth/other things constant) hits an architecture-level ceiling that no amount of extra training data/compute alone fixes — you need architectural remedies (increase d, or output factorization / mixture layers) rather than "train longer."
