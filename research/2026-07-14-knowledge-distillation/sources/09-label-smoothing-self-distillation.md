---
url: https://arxiv.org/pdf/1909.11723
title: "Revisiting Knowledge Distillation via Label Smoothing Regularization"
type: arxiv_paper
authors: Li Yuan, Francis EH Tay, Guilin Li, Tao Wang, Jiashi Feng
year: 2019/2020 (CVPR)
accessed: 2026-07-14
quality: 4
relevance: supporting
---

Key ideas (from snippet):
- Frames label smoothing as a special/degenerate case of knowledge distillation: it is
  "ad-hoc KD" where the teacher is a virtual model with random/uniform accuracy at
  temperature = 1.
- Connects to Born-Again Networks (self-distillation): iteratively distilling a model into
  a copy of itself with the same architecture repeatedly still improves performance,
  suggesting part of the benefit of KD is a regularization effect (smoothing overconfident
  predictions / providing a richer, less noisy gradient signal), not solely "transferring
  knowledge" from a genuinely more capable model.
- This provides an alternative/complementary theoretical lens to Hinton's original "dark
  knowledge" story: distillation works partly *because* soft targets act like a learned,
  instance-specific regularizer, not only because the teacher's inter-class probabilities
  encode extra semantic information.
