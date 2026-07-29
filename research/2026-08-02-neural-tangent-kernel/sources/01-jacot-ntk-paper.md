---
url: https://arxiv.org/abs/1806.07572
title: "Neural Tangent Kernel: Convergence and Generalization in Neural Networks"
type: arxiv_paper
authors: Arthur Jacot, Franck Gabriel, Clément Hongler
year: 2018
accessed: 2026-08-02
quality: 5
relevance: core
---

Abstract: At initialization, artificial neural networks (ANNs) are equivalent to Gaussian processes in the infinite-width limit, thus connecting them to kernel methods. We prove that the evolution of an ANN during training can also be described by a kernel: during gradient descent on the parameters of an ANN, the network function f_θ (which maps input vectors to output vectors) follows the kernel gradient of the functional cost (which is convex, in contrast to the parameter cost) w.r.t. a new kernel: the Neural Tangent Kernel (NTK). This kernel is central to describing the generalization features of ANNs. While the NTK is random at initialization and varies during training, in the infinite-width limit it converges to an explicit limiting kernel and it stays constant during training. This makes it possible to study the training of ANNs in function space instead of parameter space. Convergence of the training can then be related to the positive-definiteness of the limiting NTK. We prove the positive-definiteness of the limiting NTK when the data is supported on the sphere and the non-linearity is non-polynomial. We then focus on the setting of least-squares regression and show that in the infinite-width limit, the network function f_θ follows a linear differential equation during training. The convergence is fastest along the largest kernel principal components of the input data with respect to the NTK, hence suggesting a theoretical motivation for early stopping. Finally we study the NTK numerically, observe its behavior for wide networks, and compare it to the infinite-width limit.

Key facts:
- Published NIPS 2018 (v1 Jun 2018, v4 revised Feb 2020)
- Journal reference: Advances in Neural Information Processing Systems, pp. 8571-8580, 2018
- NTK defined as: for network f_θ, the kernel K(x, x') = ∇_θ f_θ(x) · ∇_θ f_θ(x') — inner product of parameter gradients
- Key theorem: as width → ∞, NTK converges to a deterministic (non-random) limiting kernel that stays CONSTANT throughout training
- This means: infinite-width network training under gradient descent = kernel gradient descent (a convex problem in function space) with the NTK
- Least-squares regression case: network function follows a LINEAR ODE during training in the infinite-width limit
- Convergence speed governed by eigenvalues of the NTK — fastest along top kernel principal components, motivating early stopping as implicit regularization
- Positive-definiteness of limiting NTK proven for data on sphere with non-polynomial nonlinearity → guarantees convergence to global minimum
