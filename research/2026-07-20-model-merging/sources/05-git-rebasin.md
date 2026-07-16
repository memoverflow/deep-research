---
url: https://arxiv.org/abs/2209.04836
title: "Git Re-Basin: Merging Models modulo Permutation Symmetries"
type: arxiv_paper
authors: Samuel K. Ainsworth, Jonathan Hayase, Siddhartha Srinivasa
year: 2022 (ICLR 2023)
accessed: 2026-07-16
quality: 5
relevance: core
---

Extracted from PDF (arxiv 2209.04836v6):

Central conjecture (from Entezari et al. 2021): most SGD solutions can be permuted (accounting for neuron permutation symmetry) so that no loss barrier exists between them under linear interpolation — i.e. they are "linear mode connected" (LMC) after re-basining.

Permutation symmetry: swapping any two hidden units in a layer (with corresponding weight permutation) leaves network function unchanged. Table 1: number of permutation symmetries for a 3-layer 512-width MLP = 10^3498, ResNet50 = 10^55109 — vastly exceeds atoms in observable universe (10^82).

Three matching algorithms proposed:
1. Activation matching — solve linear assignment problem (LAP) on cross-correlation of activations between models A, B; fast, uses data
2. Weight matching — direct weight comparison, formulated as "sum of bilinear assignments problem" (SOBLAP, NP-hard for L>2), solved via coordinate descent (Algorithm 1); no data needed, orders of magnitude faster
3. STE (straight-through estimator) matching — learned permutation via gradient descent, best quality but most expensive

Definition 2.2 Loss barrier: max_λ L((1-λ)θ_A + λθ_B) − ½(L(θ_A)+L(θ_B))

Demonstrated first zero-barrier LMC between independently trained ResNets on CIFAR-10 after weight matching. Also shows a counterexample: adversarial (non-SGD) solutions exist with no permutation achieving LMC — restricting the conjecture to SGD-found solutions specifically.

Relevance to model merging: this permutation alignment step is what's needed BEFORE naive weight averaging works for independently-trained models (as opposed to same-init fine-tuned models, which are already in the same basin).
