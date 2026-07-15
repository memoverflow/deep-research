---
url: https://arxiv.org/abs/2311.14648
title: "Calibrated Language Models Must Hallucinate"
type: arxiv_paper
authors: Adam Tauman Kalai, Santosh S. Vempala
year: 2024
accessed: 2026-07-19
quality: 5
relevance: core
---

STOC 2024 paper. Establishes a theoretical trade-off between a language model's statistical
calibration and its hallucination frequency, focused on "arbitrary facts" — facts with no learnable
pattern (e.g. a specific person's birthday). Introduces connection to Good-Turing missing-mass
estimation: a well-calibrated model's hallucination rate on arbitrary facts is lower-bounded by
roughly the fraction of such facts that appear exactly once in training data ("singletons"). Core
intuition: for facts with no learnable regularity, a calibrated model must place probability mass
roughly proportional to base rate in training data. If a fact appears once, the model can't
distinguish "this is definitely true" from "there are many similarly-supported false variants" so
it necessarily hedges wrong some fraction of the time. Kalai & Nachum & Vempala & Zhang (2025)
generalize this to include prompts, IDK/abstention options, and connect it to a general binary
classification reduction (IIV), of which this earlier paper's bound is a special case.
