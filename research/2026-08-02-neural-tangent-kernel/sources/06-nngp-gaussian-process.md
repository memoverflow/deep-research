---
url: https://en.wikipedia.org/wiki/Neural_network_Gaussian_process
title: "Neural network Gaussian process - Wikipedia"
type: encyclopedia
accessed: 2026-08-02
quality: 4
relevance: supporting
---

Key content:
- NNGP corresponds to infinite-width limit of Bayesian neural networks, AND to the distribution over functions realized by non-Bayesian NNs after random initialization (i.e. the untrained network's output distribution).
- At infinite width, each pre-activation z_i^[l](x) is Gaussian by the Central Limit Theorem (sum of many weakly-correlated terms → Gaussian). Inductively, each layer's output is governed by a Gaussian Process.
- This gives the crucial "two-stage" picture of infinite-width theory:
  1. AT INITIALIZATION (before training) — network is equivalent to a GP (the NNGP), because pre-activations are Gaussian by CLT.
  2. DURING TRAINING (gradient descent evolution) — network evolves according to the NTK, which stays constant as width→∞, making training equivalent to kernel gradient descent.
- These two objects (NNGP kernel and NTK) are different kernels computed from the same architecture but capture different things: NNGP = correlation structure of RANDOM outputs at init; NTK = correlation structure of GRADIENTS (how outputs change during training).
