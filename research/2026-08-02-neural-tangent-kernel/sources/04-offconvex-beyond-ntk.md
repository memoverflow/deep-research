---
url: http://www.offconvex.org/2021/03/25/beyondNTK/
title: "When are Neural Networks more powerful than Neural Tangent Kernels?"
type: blog
authors: Yu Bai, Minshuo Chen, Jason D. Lee
year: 2021
accessed: 2026-08-02
quality: 4
relevance: critical/limitations
---

Key content:
- Summarizes NTK theory: sufficiently wide NN trains like a linearized model governed by derivative of network w.r.t. parameters. At infinite width, this linearized model becomes a kernel predictor with the NTK. A wide NN trained with small LR converges to 0 training loss and generalizes as well as the infinite-width kernel predictor.
- Empirical gap: infinite-width NTK kernel predictors perform SLIGHTLY WORSE (though competitive) than fully trained neural networks on CIFAR-10 (cites Arora et al. 2019b).
- For FINITE width networks in practice, the gap is even more profound: linearized network is a poor approximation of the fully trained network under practical setups (large initial LR) — cites Bai et al. 2020, Figure 1.
- Theoretical limitation: NTK has poor sample complexity for learning certain simple functions. NTK is a universal kernel (can interpolate any finite non-degenerate dataset — Du et al. 2018/2019) but test error scales with RKHS norm of ground truth function. For simple non-smooth functions (e.g. single ReLU) this norm can be EXPONENTIALLY LARGE in feature dimension (Yehudai & Shamir 2019). So NTK analyses give poor sample-complexity upper bounds even though real neural nets need only mild sample sizes empirically (Livni et al. 2014).
- Core mechanism for going "beyond NTK": higher-order Taylor expansion of network around init. f_{W0+W}(x) = f^(0) + Σ f^(k), where f^(1) is the NTK/linear term. Real networks benefit from the QUADRATIC term f^(2) and beyond — this is where actual "feature learning" comes from, which the linear/NTK regime cannot capture.
- "Randomized coupling" trick: assigning random sign s_r ∈ {±1} to each weight movement kills the linear term f^(1) in expectation (E[s_r]=0) while preserving the quadratic term f^(2) unchanged (s_r^2=1). This is analogous to Dropout but randomizing the weight MOVEMENT rather than the weight itself. Lets them isolate and study the "beyond NTK" regime.
- Figure 2 concept: NTK regime operates within a small "NTK ball" (network ≈ linear term). Quadratic regime operates in a larger ball. This gives a mental picture of why "wide enough to be NTK" and "wide enough to still learn features usefully" are in tension.
