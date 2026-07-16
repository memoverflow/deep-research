---
url: https://arxiv.org/abs/2306.01708
title: "TIES-Merging: Resolving Interference When Merging Models"
type: arxiv_paper
authors: Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, Mohit Bansal
year: 2023 (NeurIPS 2023)
accessed: 2026-07-16
quality: 5
relevance: core
---

Extracted from PDF (arxiv 2306.01708v2):

Identifies two sources of interference when merging fine-tuned task vectors:
1. Redundant parameter interference — many fine-tuning updates have low magnitude and low impact; keeping only top-20% magnitude values per task vector preserves ~full performance (Fig 3).
2. Sign disagreement interference — same parameter updated with opposite signs across models; frequency of sign conflicts increases with number of models merged (Fig 4).

TIES-MERGING algorithm (3 steps):
1. Trim: keep top-k% magnitude values per task vector, zero the rest → τ̂_t
2. Elect Sign: γ_m = sgn(Σ_t τ̂_t) — majority vote weighted by magnitude
3. Disjoint Merge: average only values whose sign matches elected sign: τ_m^p = (1/|A_p|) Σ_{t∈A_p} τ̂_t^p, where A_p = {t : γ̂_t^p = γ_m^p}

Final model: θ_m = θ_init + λ·τ_m

Benchmarked against Simple Averaging, Fisher Merging, RegMean, Task Arithmetic across (IA)³ PEFT, T5-base/large full fine-tuning, ViT-B/32, ViT-L/14. TIES outperforms strongest baseline by avg 2.3%/1.7% (NLP/vision in-domain), 1.0%/4.4% (out-of-domain T5-base/large).

Default recipe without validation set: keep top-20% params, λ=1.
