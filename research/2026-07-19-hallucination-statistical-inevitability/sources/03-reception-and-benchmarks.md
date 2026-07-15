---
url: multiple (see below)
title: "Reception, criticisms, and benchmark landscape for the hallucination paper"
type: web_articles
year: 2025-2026
accessed: 2026-07-19
quality: 3
relevance: supporting
---

Sources consulted:
- https://www.themoonlight.io/en/review/hallucinations-are-inevitable-but-statistically-negligible
  (literature review discussing the theoretical inevitability vs. practical negligibility duality;
  companion framing to Kalai et al.)
- https://arxiv.org/html/2502.12187 "Hallucinations are inevitable but can be made statistically
  negligible" — shows conditions (e.g. known lower bound on input-length CDF) under which
  hallucination rate can be driven arbitrarily low even though it can't be driven to exactly zero.
- Mashable / OpenAI GPT-5 system card (Aug 2025): GPT-5's hallucination rate is reported ~26% lower
  than GPT-4o on OpenAI's own evaluation, attributed partly to eval/training changes consistent with
  rewarding calibrated abstention rather than confident guessing.
- Interconnects.ai (Nathan Lambert) discussion of sycophancy as a related reward-hacking phenomenon:
  RLHF pressure toward agreeable, confident-sounding answers can degrade calibration, reinforcing
  the paper's point that post-training incentives (not just pretraining statistics) drive
  overconfident hallucination.
- Benchmark meta-analysis (Table 2 in the paper) confirms GPQA, MMLU-Pro, IFEval, Omni-MATH,
  BBH, MATH (L5), MuSR, SWE-bench, HLE all use strict binary/accuracy grading with no credit for
  abstention — only WildBench gives partial credit via rubric grading.

These sources corroborate: (1) the theoretical result that hallucination cannot be driven to
literal zero for open-ended, infinite-support generation, but CAN be made statistically negligible
under realistic conditions; (2) real-world evidence that changing evaluation incentives (rather than
architecture) measurably reduces hallucination rates in production systems.
