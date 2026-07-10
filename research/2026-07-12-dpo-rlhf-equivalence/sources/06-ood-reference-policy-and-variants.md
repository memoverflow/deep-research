---
url: https://web.stanford.edu/class/cs224n/final-reports/256728108.pdf ; https://tianjianl.github.io/blog/2024/dpo/
title: "Handle With Care! DPO Out-of-Distribution behavior / In Defense of Vanilla DPO"
type: report_and_blog
authors: Stanford CS224N project; Tianjian Li
year: 2024
accessed: 2026-07-10
quality: 3
relevance: supporting
---

## Key findings (from search snippets, cross-referenced)
- DPO has a fundamental tendency to quickly shift probability mass toward out-of-distribution
  (OOD) trajectories — can be understood as an extreme form of overfitting where the policy fits
  behaviors not even present in the training set.
- DPO policies' downstream performance has been shown to collapse after only a few hundred
  update steps in several studies (Rafailov et al. 2023 follow-up analysis, Guo et al. 2024).
- Once a DPO policy strays OOD, empirically it tends to never fully recover during further
  training — related to growing KL divergence from the reference model.
- Widely deployed in production: Llama 3 Instruct (Meta), Tülu 2/3 (AI2), Qwen (Alibaba) all use
  DPO or a close variant in their post-training pipeline — strong evidence this is not just an
  academic curiosity but the dominant practical alignment recipe circa 2023-2025.
- Reference policy choice matters: some work (Gorbatovski et al. 2024) shows periodically
  updating π_ref during training can help; but Rafailov et al. also show fully removing the
  reference-model anchor can cause degenerate collapse — the KL anchor is doing real work.

## DPO variant landscape (SimPO paper NeurIPS 2024, KTO, ORPO, IPO)
- SimPO: reference-free reward via length-normalized average log-probability; adds explicit
  target reward margin; outperforms DPO on several benchmarks including AlpacaEval 2 while
  being simpler (no reference model needed at all during training).
- IPO (Azar et al.): softens the Bradley-Terry sigmoid assumption, more robust to
  deterministic/near-deterministic preference data where vanilla DPO's log-ratio term can blow
  up unboundedly (since σ saturates but the log-ratio inside can still be driven to ±∞, causing
  overfitting).
- ORPO: removes need for both reference model AND separate SFT stage by combining an odds-ratio
  preference term directly into the SFT loss.
- KTO: works from unpaired binary (thumbs up/down) feedback using a Kahneman-Tversky
  loss-aversion-inspired value function instead of pairwise Bradley-Terry.
