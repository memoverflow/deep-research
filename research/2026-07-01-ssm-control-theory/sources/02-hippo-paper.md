---
url: https://arxiv.org/abs/2008.07669
title: "HiPPO: Recurrent Memory with Optimal Polynomial Projections"
type: arxiv_paper
authors: Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra, Christopher Ré
year: 2020
accessed: 2026-07-01
quality: 5
relevance: core
---

Abstract: A central problem in learning from sequential data is representing cumulative history in an incremental fashion as more data is processed. We introduce a general framework (HiPPO) for the online compression of continuous signals and discrete time series by projection onto polynomial bases. Given a measure that specifies the importance of each time step in the past, HiPPO produces an optimal solution to a natural online function approximation problem. As special cases, our framework yields a short derivation of the recent Legendre Memory Unit (LMU) from first principles, and generalizes the ubiquitous gating mechanism of recurrent neural networks such as GRUs. This formal framework yields a new memory update mechanism (HiPPO-LegS) that scales through time to remember all history, avoiding priors on the timescale. HiPPO-LegS enjoys the theoretical benefits of timescale robustness, fast updates, and bounded gradients. By incorporating the memory dynamics into recurrent neural networks, HiPPO RNNs can empirically capture complex temporal dependencies. On the benchmark permuted MNIST dataset, HiPPO-LegS sets a new state-of-the-art accuracy of 98.3%. Finally, on a novel trajectory classification task testing robustness to out-of-distribution timescales and missing data, HiPPO-LegS outperforms RNN and neural ODE baselines by 25-40% accuracy.

Key facts extracted:
- HiPPO frames "optimal memory compression" as online polynomial function approximation
- HiPPO-LegS uses Legendre polynomial basis with scaled measure
- Coefficients evolve according to a linear ODE -> naturally a state space model
- The derived A matrix (A_HiPPO) has explicit closed form
- Achieves 98.3% on permuted MNIST, SOTA at time of publication
