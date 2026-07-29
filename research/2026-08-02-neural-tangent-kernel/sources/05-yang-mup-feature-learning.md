---
url: https://arxiv.org/pdf/2011.14522
title: "Tensor Programs IV: Feature Learning in Infinite-Width Neural Networks"
type: arxiv_paper
author: Greg Yang, Edward J. Hu
year: 2020/2021
accessed: 2026-08-02
quality: 5
relevance: core (contrast to NTK)
---

Key content:
- Standard parameterization NN, as width → ∞ under the "NTK parameterization" (proper scaling of weights so gradient descent gives a well-defined limit), the network becomes EXACTLY the linear/kernel model described by NTK.
- Problem identified: NTK parameterization is a special / narrow choice of how weights and learning rates scale with width. Under this scaling, features effectively don't change/learn during training at infinite width — this is the "lazy training" phenomenon.
- Proposed Maximal Update Parametrization (μP): a different scaling of per-layer learning rates/init variances such that at infinite width, EVERY layer's features still update by a meaningful (non-vanishing, non-exploding) amount — i.e. feature learning is preserved even in the infinite-width limit.
- In standard parameterization, correlation between weights and activations vanishes as width increases. In μP, this correlation persists at infinite width, enabling feature learning.
- Demonstrated on Word2Vec and few-shot learning tasks: μP is a better model of real feature-learning behavior in networks, whereas NTK-style infinite width models cannot learn features and thus fail to capture crucial aspects of deep learning.
- This work underlies the "μTransfer" technique later used for hyperparameter transfer across model scales (used practically e.g. in some large model training recipes) — establishing NTK/μP theory as not just abstract math but a tool with real engineering payoff.
