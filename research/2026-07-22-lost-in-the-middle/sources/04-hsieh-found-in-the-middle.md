---
url: https://arxiv.org/abs/2406.16008
title: "Found in the Middle: Calibrating Positional Attention Bias Improves Long Context Utilization"
type: arxiv_paper
authors: Cheng-Yu Hsieh, Yung-Sung Chuang, Chun-Liang Li, et al.
year: 2024 (ACL Findings 2024)
accessed: 2026-07-22
quality: 5
relevance: core
---

Two contributions:
1. Understanding: establishes that LLMs exhibit an INTRINSIC U-shaped positional attention
   bias — tokens at the very beginning and very end of the input receive systematically
   higher attention weight, REGARDLESS of their actual semantic relevance to the query. This
   positional bias is a separable, additive-ish component sitting on top of the
   content-driven relevance signal.
2. Mitigation: proposes "found-in-the-middle," a calibration mechanism that estimates and
   subtracts out the purely positional component of attention, so remaining attention better
   reflects true content relevance regardless of position.

Empirical results: the calibration improves retrieval-augmented generation (RAG) task
accuracy by up to 15 percentage points when the correct passage is placed in middle
positions, without hurting beginning/end performance. This is strong evidence that the
lost-in-the-middle degradation is substantially attributable to a measurable, correctable
attention-weight artifact rather than purely a training-data or capability limitation.

Relevance: this is the empirical/practical counterpart to the purely theoretical papers
(Chowdhury 2026, Wu et al. 2025) — it shows you CAN partially "subtract out" the structural
bias at inference time and recover real gains, supporting the idea that lost-in-the-middle
is architectural-positional rather than solely about the model "not knowing" the middle
content.
