---
url: https://arxiv.org/abs/2602.16837
title: "A Residual-Aware Theory of Position Bias in Transformers"
type: arxiv_paper
authors: Hanna Herasimchyk, R. Labryga, T. Prusina, S. Laue
year: 2026
accessed: 2026-07-22
quality: 5
relevance: core
---

Develops a residual-aware theory of cumulative attention rollout: incorporates residual
connections into the standard "attention rollout" cross-token influence propagation
framework (previously used to trace/visualize what earlier tokens influence later
representations layer by layer).

Key theoretical result: proves that at finite depth, causal Transformers structurally induce
a U-shaped position bias — attention/influence concentrates on early tokens (via causal
masking, matching Wu et al. 2025's graph argument) AND on late tokens (via the residual
stream giving them an un-diluted shortcut) — providing a principled architectural
explanation for the empirically observed Lost-in-the-Middle phenomenon that resolves the
earlier "pure primacy, no recency" gap left open by attention-only analyses.

Empirically validates predicted influence profiles against ACTUALLY PRETRAINED models,
achieving Spearman correlation of 0.88–0.98 between their theoretical prediction and measured
attention/influence patterns — strong quantitative support that this structural mechanism
persists into real trained networks, not just toy models.

Distinction from Chowdhury 2026 ("Lost in the Middle at Birth"): Herasimchyk et al. measure
residual mixing coefficients from trained networks (so their "architectural prior" claim is
partially circular — parameters come from training), whereas Chowdhury derives a fully
closed-form result using only RANDOM weights at initialization, proving the effect predates
any learning at all. Together the two papers form a before/after picture: the bias exists at
birth (Chowdhury) and survives, essentially unchanged in shape, through training
(Herasimchyk, matching real models).

Code released: github.com/ml-uhh/position-bias
