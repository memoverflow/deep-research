---
url: https://arxiv.org/abs/2012.13255
title: "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning"
type: arxiv_paper
authors: Armen Aghajanyan, Luke Zettlemoyer, Sonal Gupta
year: 2020
accessed: 2026-07-24
quality: 5
relevance: core (theoretical foundation for LoRA)
---

## Abstract
Although pretrained LMs can be fine-tuned to produce SOTA results for many tasks, the dynamics are not well understood, especially in the low-data regime. Why can vanilla gradient descent tune a model with hundreds of millions of parameters on datasets with only hundreds/thousands of examples, without strong regularization? The paper argues intrinsic dimension analysis explains this. Empirically, common pretrained models have very low intrinsic dimension — there exists a low-dimensional reparameterization as effective for fine-tuning as the full parameter space. Example: optimizing only 200 trainable parameters randomly projected back into the full space tunes RoBERTa to 90% of full-parameter performance on MRPC. Pre-training implicitly minimizes intrinsic dimension, and larger models tend to have LOWER intrinsic dimension after a fixed number of pretraining updates — partly explaining their effectiveness. Connects intrinsic dimensionality to compression-based generalization bounds independent of full parameter count.

## Key Content
- This is the direct theoretical ancestor LoRA cites: "the learned over-parametrized models in fact reside on a low intrinsic dimension."
- Method: random projection matrix P (frozen, e.g. random Gaussian) mapping a very small trainable vector θ_d ∈ R^d into the full parameter space: θ = θ_0 + Pθ_d. Only θ_d (dimension d, e.g. 200) is optimized; θ_0 is the pretrained init.
- Key surprising finding: bigger pretrained models → lower intrinsic dimension for the same task, i.e. pretraining is "compressing" the solution space, making downstream adaptation easier the bigger/better pretrained the model is. This gives an information-theoretic account of why scaling helps fine-tuning efficiency, not just raw capability.
- This directly motivates LoRA's structural choice: instead of a random fixed projection, LoRA restricts updates to a *learned* low-rank subspace (BA), which is more expressive/efficient than a random projection while keeping the same spirit — the update ΔW doesn't need full rank because the "true" needed update lives in a small subspace.
