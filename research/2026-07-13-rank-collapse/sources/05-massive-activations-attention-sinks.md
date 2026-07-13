---
url: https://arxiv.org/abs/2603.05498
title: "The Spike, the Sparse and the Sink: Anatomy of Massive Activations and Attention Sinks"
type: arxiv_paper
year: 2026
accessed: 2026-07-13
quality: 4
relevance: supporting
---

## Key content
Studies two recurring Transformer phenomena: massive activations (small number of hidden channels attain extreme values for a few tokens) and attention sinks (certain tokens attract disproportionate attention regardless of relevance). Shows these phenomena co-occur and often involve the same tokens.

Core claim relevant to rank collapse: "Repeated mixing via multi-head attention, especially in deep or long-context transformers, rapidly drives token representations toward a low-dimensional or even constant subspace ('rank collapse'). Massive activations and attention sinks jointly act to prevent excessive token mixing in self-attention — attention sink suppresses mixing among non-sink tokens, whereas massive activations suppress mixing between sink tokens and non-sink tokens."

This reframes attention sink — often seen as a quirky artifact or inefficiency — as an emergent, self-taught defense mechanism against rank collapse: by dumping attention mass onto a fixed, content-independent sink token, the model effectively inserts a near-constant component into the attention matrix, suppressing the token-mixing/averaging that drives collapse.

## Notes for article
Used in the "bigger picture" section connecting rank collapse to attention sink phenomenon — this is a 2026 paper providing a novel unifying explanation.
