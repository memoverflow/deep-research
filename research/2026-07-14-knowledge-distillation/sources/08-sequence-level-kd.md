---
url: https://aclanthology.org/D16-1139/
title: "Sequence-Level Knowledge Distillation"
type: conference_paper
authors: Yoon Kim, Alexander M. Rush
year: 2016
accessed: 2026-07-14
quality: 4
relevance: core
---

Key ideas (from snippets/citations):
- Standard KD, as designed by Hinton et al., matches the teacher and student distribution
  token-by-token (word-level), given the same prefix. For sequence generation tasks (NMT),
  this ignores that the real generative distribution is over entire sequences, not
  independent per-token choices.
- Proposes sequence-level KD: instead of (or in addition to) matching per-token softened
  distributions, train the student directly on high-quality full sequences sampled/decoded
  from the teacher (e.g. teacher's beam-search output) as if they were ground truth —
  effectively "flattening" a full-sequence NLL objective by substituting the true-but-
  unknown data distribution with the teacher's mode-approximated sequence distribution.
- Also introduces sequence-level interpolation as a middle ground.
- This is the theoretical precursor to modern practices like DeepSeek-R1 distillation and
  most GPT-4-output-based instruction distillation (Alpaca, WizardLM): training students on
  full teacher-generated text rather than teacher logits.
