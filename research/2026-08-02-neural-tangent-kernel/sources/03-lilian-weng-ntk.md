---
url: https://lilianweng.github.io/posts/2022-09-08-ntk/
title: "Some Math behind Neural Tangent Kernel"
type: blog
author: Lilian Weng
year: 2022
accessed: 2026-08-02
quality: 4
relevance: core
---

Key content:
- Motivation: NNs are over-parameterized, can fit data with near-zero training loss + decent generalization even when #params exceeds #training points. Random init, yet optimization consistently leads to similarly good outcomes.
- NTK explains WHY networks with enough width consistently converge to a global minimum when minimizing empirical loss.
- Provides deep dive into math: Jacobian matrices, differential equations (ODEs), kernel & kernel methods, Gaussian Processes connection.
- Delta change in parameter space during training ≈ first-order Taylor expansion (linearization).
- Shows link between NTK regime and "kernel gradient descent" — training directly in function space rather than parameter space.
- Notes the tension: while promising theoretically, empirical results show that neural networks in the strict lazy/NTK regime perform WORSE than practical over-parameterized networks trained normally (i.e. NTK doesn't fully explain why deep learning works so well in practice — it explains a special linearized corner case).
- Connects to NNGP (Neural Network Gaussian Process): AT INITIALIZATION (before any training) infinite-width networks are equivalent to Gaussian Processes; NTK describes what happens DURING training via gradient descent, extending the GP correspondence into the dynamics of learning.
