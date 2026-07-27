---
url: https://arxiv.org/abs/2207.00729
title: "The Parallelism Tradeoff: Limitations of Log-Precision Transformers"
type: arxiv_paper
authors: William Merrill, Ashish Sabharwal
year: 2023
accessed: 2026-07-30
quality: 5
relevance: core
---

Abstract: Despite their omnipresence in modern NLP, characterizing the computational power of transformer neural nets remains an interesting open question. We prove that transformers whose arithmetic precision is logarithmic in the number of input tokens (and whose feedforward nets are computable using space linear in their input) can be simulated by constant-depth logspace-uniform threshold circuits. This provides insight on the power of transformers using known results in complexity theory. For example, if L ≠ P (i.e., not all poly-time problems can be solved using logarithmic space), then transformers cannot even accurately solve linear equalities or check membership in an arbitrary context-free grammar with empty productions. Our result intuitively emerges from the transformer architecture's high parallelizability. We thus speculatively introduce the idea of a fundamental parallelism tradeoff: any model architecture as parallelizable as the transformer will obey limitations similar to it.

Key takeaway: log-precision transformers ⊆ constant-depth uniform TC0. Core theoretical basis for the entire article.
