---
url: https://arxiv.org/abs/2410.21228
title: "LoRA vs Full Fine-tuning: An Illusion of Equivalence"
type: arxiv_paper
year: 2024
accessed: 2026-07-24
quality: 5
relevance: important critical analysis
---

## Abstract
Studies how LoRA and full fine-tuning change pretrained models by analyzing weight matrices through spectral properties (SVD). Finds LoRA and full fine-tuning yield weight matrices whose SVDs exhibit very different structure: LoRA-trained weight matrices have new, high-ranking singular vectors called "intruder dimensions," while full-fine-tuned matrices do not. Extends the finding that LoRA forgets less than full fine-tuning, and finds LoRA's forgetting is vastly localized to these intruder dimensions — causally intervening (scaling down the associated singular values post-fine-tuning) reduces forgetting with minimal downstream performance drop. Accumulating intruder dimensions is harmful, amplified during continual/sequential fine-tuning — LoRA models accumulating intruder dimensions perform worse in that setting.

## Key Content
- "Intruder dimensions": new singular directions introduced by LoRA's low-rank update that weren't present (not aligned with any existing large singular direction) in the pretrained weight matrix. Full fine-tuning instead tends to slightly perturb/rotate the EXISTING top singular directions rather than injecting entirely new ones.
- Practical implication: LoRA and full fine-tuning are NOT mechanistically equivalent even if they reach similar task accuracy — they get there via different paths in weight space, hence "illusion of equivalence."
- This connects to and partially explains the earlier finding (Biderman et al. 2024, referenced) that LoRA forgets less on math/code than full FT — because the update is geometrically confined to a small subspace that doesn't disturb most of the pretrained knowledge, but this confinement itself creates the intruder-dimension side effect that can hurt when doing many sequential LoRA fine-tunes.
