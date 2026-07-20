---
url: https://arxiv.org/html/2510.10276v1
title: "Lost in the Middle: An Emergent Property from Information Retrieval Demands in LLMs"
type: arxiv_paper
authors: Nikolaus Salvatore, Hao Wang, Qiong Zhang (Rutgers University)
year: 2025
accessed: 2026-07-22
quality: 4
relevance: supporting
---

Cognitive-science-flavored framing: draws an explicit analogy between LLM lost-in-the-middle
and human memory's classic "serial position effect" — primacy (remember early items) and
recency (remember late items) effects observed in free recall experiments since the 1960s
(Murdock, etc.).

Core claim: the U-shape is not merely an architectural flaw but partly an ADAPTATION to two
different implicit "memory demands" that show up during next-token-prediction pretraining:
- some prediction sub-tasks require uniform recall across the *entire* preceding context
  (a long-term-memory-like demand — needing to remember something said far back)
- others are dominated by *recent* tokens (a short-term-memory-like / local demand — most of
  next-token prediction is locally predictable from nearby context)

They train GPT-2 and Llama-style models FROM SCRATCH on two simplified memory-inspired tasks
(free recall + running span) that isolate these two demands, and show the U-shaped curve
emerges from this joint training pressure, not from any single cause.

Findings:
- Recency effect aligns directly with the short-term-memory-style training signal.
- Primacy effect is induced by the uniform (long-term) memory demand AND is additionally
  amplified by the model's autoregressive causal structure and by formation of "attention
  sinks" — connecting back to Xiao et al. 2023 and the geometric arguments in
  Chowdhury/Wu et al.
- These effects generalize to a "masked sequence completion" task that more closely resembles
  actual LLM next-token-prediction pretraining, suggesting the U-shape is a natural byproduct
  of the training objective interacting with model architecture, not an accident to be
  purely "fixed away."

Relevance: gives an alternative, complementary lens to the purely-geometric
initialization-time theories — argues part of the U-shape is also *learned* and functionally
adaptive given what next-token prediction actually rewards, on top of the architectural prior
that exists even before training starts.
