---
url: https://arxiv.org/abs/2010.01412
title: "Sharpness-Aware Minimization for Efficiently Improving Generalization"
type: arxiv_paper
authors: Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur
year: 2020
accessed: 2026-07-11
quality: 5
relevance: core
---

Abstract: In today's heavily overparameterized models, the value of the training loss provides few guarantees on model generalization ability. Indeed, optimizing only the training loss value, as is commonly done, can easily lead to suboptimal model quality. Motivated by prior work connecting the geometry of the loss landscape and generalization, we introduce a novel, effective procedure for instead simultaneously minimizing loss value and loss sharpness. In particular, our procedure, Sharpness-Aware Minimization (SAM), seeks parameters that lie in neighborhoods having uniformly low loss; this formulation results in a min-max optimization problem on which gradient descent can be performed efficiently. We present empirical results showing that SAM improves model generalization across a variety of benchmark datasets (CIFAR-10, CIFAR-100, ImageNet, finetuning tasks) and models, yielding novel state-of-the-art performance for several. Additionally, we find that SAM natively provides robustness to label noise on par with state-of-the-art procedures targeting noisy labels.

Related finding: follow-up work shows SAM reduces feature rank at different layers ("Sharpness-Aware Minimization Leads to Low-Rank Features").
