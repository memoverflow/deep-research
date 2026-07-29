---
url: https://arxiv.org/abs/2008.03703
title: "What Neural Networks Memorize and Why: Discovering the Long Tail via Influence Estimation"
type: arxiv_paper
authors: Vitaly Feldman, Chiyuan Zhang
year: 2020
accessed: 2026-08-01
quality: 5
relevance: core
---

## Key Content Extracted (ar5iv full text)

Empirical validation of Feldman's 2019 long-tail theory using influence-estimation methods on MNIST, CIFAR-100, ImageNet.

- On ImageNet, ≈32% of training examples have memorization estimates ≥0.3; their marginal utility contribution to test accuracy is ≈3.4% (vs ≈2.6% for a random subset of the same size 32%) — memorized examples contribute disproportionately more to test accuracy than random examples.
- High-memorization examples visually inspected: mixture of atypical/outlier/mislabeled examples.
- Found 35/1015/1641 high-influence train-test pairs on MNIST/CIFAR-100/ImageNet respectively — pairs are visually similar / near-duplicate-looking, strongly supporting the long-tail theory: memorized examples help specifically because there exist similar test examples that benefit from the memorization.
- Conclusion: accuracy on long-tailed distributions depends critically on ability to memorize labels; effect stronger for under-represented subpopulations.
