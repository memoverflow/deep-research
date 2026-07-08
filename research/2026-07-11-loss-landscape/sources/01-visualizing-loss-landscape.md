---
url: https://arxiv.org/abs/1712.09913
title: "Visualizing the Loss Landscape of Neural Nets"
type: arxiv_paper
authors: Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, Tom Goldstein
year: 2018
accessed: 2026-07-11
quality: 5
relevance: core
---

Abstract: Neural network training relies on our ability to find "good" minimizers of highly non-convex loss functions. It is well-known that certain network architecture designs (e.g., skip connections) produce loss functions that train easier, and well-chosen training parameters (batch size, learning rate, optimizer) produce minimizers that generalize better. However, the reasons for these differences, and their effects on the underlying loss landscape, are not well understood. In this paper, we explore the structure of neural loss functions, and the effect of loss landscapes on generalization, using a range of visualization methods. First, we introduce a simple "filter normalization" method that helps us visualize loss function curvature and make meaningful side-by-side comparisons between loss functions. Then, using a variety of visualizations, we explore how network architecture affects the loss landscape, and how training parameters affect the shape of minimizers.

Key contribution: filter normalization solves the scale-invariance confound in loss landscape visualization; shows skip connections (ResNet) produce smoother landscapes than deep nets without them.
