---
url: https://arxiv.org/abs/2301.04213
title: "Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models"
type: arxiv_paper
authors: Peter Hase, Mohit Bansal, Been Kim, Asma Ghandeharioun
year: 2023 (NeurIPS 2023 Spotlight)
accessed: 2026-07-26
quality: 5
relevance: critical/contrarian — must-read counterpoint
---

Abstract: Language models learn a great quantity of factual information during pretraining, and recent work localizes this information to specific model weights like mid-layer MLP weights. In this paper, we find that we can change how a fact is stored in a model by editing weights that are in a different location than where existing methods suggest that the fact is stored. This is surprising because we would expect that localizing facts to specific model parameters would tell us where to manipulate knowledge in models, and this assumption has motivated past work on model editing methods. Specifically, we show that localization conclusions from representation denoising (also known as Causal Tracing) do not provide any insight into which model MLP layer would be best to edit in order to override an existing stored fact with a new one. This finding raises questions about how past work relies on Causal Tracing to select which model layers to edit. Next, we consider several variants of the editing problem, including erasing and amplifying facts. For one of our editing problems, editing performance does relate to localization results from representation denoising, but we find that which layer we edit is a far better predictor of performance. Our results suggest, counterintuitively, that better mechanistic understanding of how pretrained language models work may not always translate to insights about how to best change their behavior.

## Key content

This is the strongest and most important critique of the ROME/MEMIT causal-tracing paradigm. Key finding: Causal Tracing correctly identifies where a fact is *read out from* during a forward pass (correlational/causal-for-recall), but this does NOT tell you which layer is best to *edit* to change the fact. You can successfully override a fact by editing layers that Causal Tracing says are irrelevant, and editing the "correct" localized layer isn't necessarily optimal either.

This decouples "localization" (understanding where information flows during inference) from "editability" (which weights, when changed, best rewrite the behavior) — a distinction the field had implicitly assumed were the same thing. The authors explicitly flag this as a challenge to interpretability research motivating model editing methods on causal-tracing grounds. Important honest caveat for the blog: MEMIT's own layer choice (a *range* of layers rather than the causal-tracing peak) already implicitly works around part of this problem, but the deeper point stands — mechanistic understanding ≠ actionable editing knowledge.
