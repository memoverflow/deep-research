---
url: https://arxiv.org/abs/2402.12875
title: "Chain of Thought Empowers Transformers to Solve Inherently Serial Problems"
type: arxiv_paper
authors: Zhiyuan Li, Hong Liu, Denny Zhou, Tengyu Ma
year: 2024
accessed: 2026-07-30
quality: 5
relevance: core
---

Abstract: Instructing the model to generate a sequence of intermediate steps, a.k.a., a chain of thought (CoT), is a highly effective method to improve the accuracy of LLMs on arithmetics and symbolic reasoning tasks. This work provides a theoretical understanding of the power of CoT through the lens of expressiveness. Conceptually, CoT empowers the model with the ability to perform inherently serial computation, which is otherwise lacking in transformers, especially when depth is low. Given input length n, previous works have shown that constant-depth transformers with finite precision poly(n) embedding size can only solve problems in TC0 without CoT. We first show an even tighter expressiveness upper bound for constant-depth transformers with constant-bit precision, which can only solve problems in AC0, a proper subset of TC0. However, with T steps of CoT, constant-depth transformers using constant-bit precision and O(log n) embedding size can solve any problem solvable by boolean circuits of size T.

Key takeaway: quantifies exactly how CoT step count T translates to circuit-size expressivity — the core "why CoT works" mechanism.
