---
url: https://arxiv.org/abs/2410.08847
title: "Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization"
type: arxiv_paper
authors: Noam Razin, Sadhika Malladi, Adithya Bhaskar, Danqi Chen, Sanjeev Arora, Boris Hanin
year: 2024 (ICLR 2025)
accessed: 2026-07-10
quality: 5
relevance: core
---

## Abstract (verbatim)
Direct Preference Optimization (DPO) and its variants are increasingly used for aligning
language models with human preferences. Although these methods are designed to teach a model to
generate preferred responses more frequently relative to dispreferred responses, prior work has
observed that the likelihood of preferred responses often decreases during training. The current
work sheds light on the causes and implications of this counter-intuitive phenomenon, which we
term likelihood displacement. We demonstrate that likelihood displacement can be catastrophic,
shifting probability mass from preferred responses to responses with an opposite meaning. As a
simple example, training a model to prefer "No" over "Never" can sharply increase the probability
of "Yes". Moreover, when aligning the model to refuse unsafe prompts, we show that such
displacement can unintentionally lead to unalignment, by shifting probability mass from preferred
refusal responses to harmful responses (e.g., reducing the refusal rate of Llama-3-8B-Instruct
from 74.4% to 33.4%). We theoretically characterize that likelihood displacement is driven by
preferences that induce similar embeddings, as measured by a centered hidden embedding similarity
(CHES) score. Empirically, the CHES score enables identifying which training samples contribute
most to likelihood displacement in a given dataset. Filtering out these samples effectively
mitigated unintentional unalignment in our experiments.

## Relevance to this article
Concrete, rigorous evidence for the "preference is relative, not absolute" failure mode
discussed by other sources qualitatively. Gives a hard, quantitative safety-relevant example
(Llama-3-8B-Instruct refusal rate collapsing from 74.4% to 33.4%) — this is a strong, citable,
alarming data point for the "where DPO's math has teeth" section of the article. The CHES score
gives a concrete mechanism: preferences whose CHOSEN and REJECTED responses have very similar
embeddings are the most dangerous ones for likelihood displacement, because the gradient can't
cleanly separate "increase yw" from "decrease yl" in representation space — pushing down yl's
probability spills over onto semantically similar completions (which can include yw's opposite).
