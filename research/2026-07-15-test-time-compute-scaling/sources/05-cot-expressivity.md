---
url: https://arxiv.org/abs/2402.12875
title: "Chain of Thought Empowers Transformers to Solve Inherently Serial Problems"
type: arxiv_paper
year: 2024
accessed: 2026-07-15
quality: 5
relevance: core
---

Abstract: Theoretical understanding of CoT power for decoder-only transformers via expressiveness lens. Constant-depth transformers with constant-bit precision can only solve problems in AC^0 (subset of TC^0) without CoT. With T steps of CoT, constant-depth transformers with constant-bit precision and O(log n) embedding size can solve any problem solvable by boolean circuits of size T. Empirically CoT dramatically improves accuracy on tasks hard for parallel computation: permutation group composition, iterated squaring, circuit value problems — especially for low-depth transformers.

Key takeaway: this is the rigorous complexity-theoretic explanation for WHY chain-of-thought works — it converts a fixed-depth model's parallel computation limit (AC^0) into an unbounded serial computation capability (up to circuit-size T), by feeding generated tokens back as input.
