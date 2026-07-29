---
url: https://arxiv.org/abs/1906.05271
title: "Does Learning Require Memorization? A Short Tale about a Long Tail"
type: arxiv_paper
authors: Vitaly Feldman
year: 2019 (rev. 2020/2021)
accessed: 2026-08-01
quality: 5
relevance: core
---

## Abstract / Key Content
Overparameterized deep networks fit training data (nearly) perfectly, including outliers and mislabeled examples/random labels — not explained by classical generalization theory. Feldman proposes: when the data distribution is long-tailed (a substantial fraction of rare/atypical subpopulations), label memorization is *necessary on average* for achieving close-to-optimal generalization error.

Key theoretical mechanism: builds an abstract mixture-of-subpopulations model where subpopulation frequencies are drawn from a long-tailed prior. Learning algorithms cannot achieve high accuracy on a subpopulation without seeing representative examples from it; for rare subpopulations, this forces something close to memorization of individual examples (since there aren't enough examples to statistically "average out" a generalizable rule). Crucially, statistically indistinguishable examples can differ in whether memorizing them helps (useful long-tail example) or doesn't (noise/mislabeled) — you cannot tell them apart without knowing test-time distribution.

This explains empirically-observed phenomena: DNNs memorize atypical/mislabeled training points not because of a flaw but because doing so is close to optimal under long-tailed real-world data distributions.
