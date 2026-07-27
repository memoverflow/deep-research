---
url: https://arxiv.org/abs/2505.21024
title: "Pause Tokens Strictly Increase the Expressivity of Constant-Depth Transformers"
type: arxiv_paper
authors: (2025)
year: 2025
accessed: 2026-07-30
quality: 5
relevance: supporting
---

Abstract: Pause tokens, simple filler symbols such as "...", consistently improve Transformer performance on both language and mathematical tasks, yet their theoretical effect remains unexplained. We provide the first formal separation result: adding pause tokens to constant-depth, logarithmic-width Transformers strictly increases computational expressivity. With bounded-precision activations, Transformers without pause tokens compute only a strict subset of AC0, while adding a polynomial number of pause tokens allows them to express the entire class. For logarithmic-precision Transformers, pause tokens achieve expressivity equivalent to TC0. Empirically, two-layer causally masked Transformers can learn parity when supplied with pause tokens, which they cannot learn without them.

Key takeaway: even semantically-empty "pause"/filler tokens (not meaningful CoT) provably increase expressivity purely by adding extra forward-pass steps — supports the "extra computation, not extra knowledge" framing.
