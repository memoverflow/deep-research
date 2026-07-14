---
url: https://arxiv.org/abs/2404.02657
title: "Rethinking Kullback-Leibler Divergence in Knowledge Distillation for Large Language Models"
type: arxiv_paper
authors: Taiqiang Wu, Chaofan Tao, Jiahao Wang, Runming Yang, Zhe Zhao, Ngai Wong
year: 2024
accessed: 2026-07-14
quality: 4
relevance: core
---

Abstract: Kullback-Leiber divergence has been widely used in Knowledge Distillation (KD)
to compress Large Language Models (LLMs). Contrary to prior assertions that reverse
Kullback-Leibler (RKL) divergence is mode-seeking and thus preferable over the mean-seeking
forward Kullback-Leibler (FKL) divergence, this study empirically and theoretically
demonstrates that neither mode-seeking nor mean-seeking properties manifest in KD for LLMs.
Instead, RKL and FKL are found to share the same optimization objective and both converge
after a sufficient number of epochs. However, due to practical constraints, LLMs are seldom
trained for such an extensive number of epochs. Meanwhile, we further find that RKL
focuses on the tail part of the distributions, while FKL focuses on the head part at the
beginning epochs. Consequently, we propose a simple yet effective Adaptive
Kullback-Leiber (AKL) divergence method, which adaptively allocates weights to combine FKL
and RKL.

Key ideas:
- Direct empirical/theoretical challenge to the popular "reverse KL = mode-seeking, forward
  KL = mode-covering" folklore as applied to LLM KD — they show that under limited training
  epochs (the realistic regime) the practical difference is about which part of the
  distribution gets fit first/more, not a clean mode-seeking/covering dichotomy.
- FKL fits the "head" (high-probability common tokens) early; RKL fits the "tail"
  (rare/specific tokens) more. AKL blends both with adaptive weights.
- Important nuance/complication for the mainstream narrative used by MiniLLM et al.
