---
url: https://arxiv.org/abs/2106.16213
title: "Saturated Transformers are Constant-Depth Threshold Circuits"
type: arxiv_paper
authors: William Merrill, Ashish Sabharwal, Noah A. Smith
year: 2022
accessed: 2026-07-30
quality: 5
relevance: core
---

Abstract: Transformers have become a standard neural network architecture for many NLP problems, motivating theoretical analysis of their power in terms of formal languages. Recent work has shown that transformers with hard attention are quite limited in power (Hahn, 2020), as they can be simulated by constant-depth AND/OR circuits (Hao et al. 2021). However, hard attention is a strong assumption. In this work, we analyze the circuit complexity of transformers with saturated attention: a generalization of hard attention that more closely captures the attention patterns learnable in practical transformers. We first show that saturated transformers transcend the known limitations of hard-attention transformers. We then prove saturated transformers with floating-point values can be simulated by constant-depth threshold circuits, giving TC0 as an upper bound.

Key takeaway: saturated attention (closer to real trained models) is strictly stronger than hard attention (escapes AC0) but still bounded by TC0.
