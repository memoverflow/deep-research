---
url: https://arxiv.org/html/2607.17427
title: "Abliteration Is Not a Scalpel: Off-Target Effects of Refusal Removal"
type: arxiv_paper
authors: (see arxiv listing)
year: 2026
accessed: 2026-07-27
quality: 4
relevance: critical/limitations
---

## Abstract summary
Tests the claim that abliteration (deleting the refusal direction from weights) is a "surgical"
operation that removes refusal and changes nothing else. Uses a disposition probe with zero
refusal-eliciting content: 21,600 trading decisions (weekly up/down calls on 60 Warsaw Stock
Exchange equities over 18 weeks) replayed through a frozen pipeline, comparing base vs. abliterated
arms of two MoE model families (Gemma-3-27B-A4B-it-like and Qwen3-30B-A3B-Instruct-2507).

## Findings
Even though the task never triggers refusal at all, abliteration produces measurable side effects:
- Abliterated models become systematically **more optimistic** (+12.2pp Gemma, +7.4pp Qwen)
- They **justify themselves at greater length**
- They use **fewer explicit uncertainty words** in forced self-critiques
- Confidence effects are inconsistent across model families — one family becomes less confident,
  the other more confident, from the *same* weight operation
- Capability covariates rule out simple instruction-following degradation as the cause
- Neither abliterated arm shows genuine forecasting skill (no alpha, just shifted disposition)

## Why this matters for the narrative
Directly complicates the "surgical, single-direction" story: the refusal direction is not perfectly
disentangled from other dispositional traits (optimism, verbosity, calibration language) in the
residual stream. Removing it perturbs a broader region of the representation space, and the
perturbation's *sign* is not even consistent across model families. Good evidence for a "cautionary
epilogue" in the article — the rank-one edit is real and powerful, but "surgical" is a marketing
word, not a mathematical guarantee.
