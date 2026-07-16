---
url: https://arxiv.org/abs/2311.03099
title: "Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch (DARE)"
type: arxiv_paper
authors: Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, Yongbin Li
year: 2023
accessed: 2026-07-16
quality: 5
relevance: core
---

Extracted (abstract + PDF first pages, arxiv 2311.03099):

DARE (Drop And REscale): sparsification method for delta parameters (task vectors) of SFT models. Randomly drops delta parameters with probability p, rescales survivors by 1/(1-p):
τ̂ = (m ⊙ τ) / (1-p), where m ~ Bernoulli(1-p)

Key finding: delta parameter value ranges for SFT models are typically small (often within 0.005); DARE can eliminate up to 90-99% of delta parameters without significant performance loss. However, if models undergo continued pre-training, value ranges grow (~0.03), making DARE less effective at high drop rates.

Used as a plug-in before merging (e.g. combined with TIES → "dare_ties" in mergekit) to reduce parameter interference among multiple SFT homologous models. Created merged 7B LM ranking first on Open LLM Leaderboard at time of publication (merging WizardLM-13B, WizardMath-13B, llama-2-13b-code-alpaca).
