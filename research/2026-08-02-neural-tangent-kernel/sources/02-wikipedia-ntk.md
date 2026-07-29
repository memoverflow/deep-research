---
url: https://en.wikipedia.org/wiki/Neural_tangent_kernel
title: "Neural tangent kernel - Wikipedia"
type: encyclopedia
year: 2024
accessed: 2026-08-02
quality: 4
relevance: core
---

Key content:
- A kernel is a positive-semidefinite symmetric function of two inputs representing similarity between them.
- NTK is a specific kernel derived from a given neural network; in general when parameters change during training the NTK evolves too.
- In the limit of large layer width, NTK becomes CONSTANT — revealing duality between training the wide network and kernel methods: gradient descent in the infinite-width limit is fully equivalent to kernel gradient descent with the NTK.
- Result: using gradient descent to minimize least-squares loss for NNs yields the SAME mean estimator as ridgeless kernel regression with the NTK.
- This duality gives simple closed-form equations describing training dynamics, generalization, and predictions of wide neural networks.
- Introduced 2018 by Arthur Jacot, Franck Gabriel, Clément Hongler.
- Applications section covers: ridgeless kernel regression & kernel gradient descent equivalence; overparametrization/interpolation/generalization; convergence to global minimum.
- "Extensions and limitations" section acknowledges gap between NTK theory and practical deep learning (feature learning not captured).
- Technical detail: wide fully-connected ANNs have a deterministic NTK that remains constant throughout training, and are linear in their parameters throughout training (i.e., f_θ(x) ≈ f_θ0(x) + ∇_θf_θ0(x)·(θ-θ0), first-order Taylor expansion around initialization becomes EXACT as width→∞).
