---
url: https://arxiv.org/abs/2505.24832
title: "How much do language models memorize?"
type: arxiv_paper
authors: John X. Morris, Chawin Sitawarin, Chuan Guo, Narine Kokhlikyan, G. Edward Suh, Alexander M. Rush, Kamalika Chaudhuri, Saeed Mahloujifar
year: 2025
accessed: 2026-08-01
quality: 5
relevance: core
---

## Abstract
We propose a new method for estimating how much a model "knows" about a datapoint and use it to measure the capacity of modern language models. Prior studies of language model memorization have struggled to disentangle memorization from generalization. We formally separate memorization into two components: unintended memorization, the information a model contains about a specific dataset, and generalization, the information a model contains about the true data-generation process. By eliminating generalization, we can compute the total memorization of a given model, which provides an estimate of model capacity: our measurements estimate that models in the GPT family have an approximate capacity of 3.6 bits-per-parameter. We train language models on datasets of increasing size and observe that models memorize until their capacity fills, at which point "grokking" begins, and unintended memorization decreases as models begin to generalize. We train hundreds of transformer language models ranging from 500K to 1.5B parameters and produce a series of scaling laws relating model capacity and data size to membership inference.

## Key Content Extracted (from PDF full text)

### Motivation
Modern LMs trained on increasingly large data while parameter counts stagnate (e.g. Llama-3 8B: 32GB disk, trained on 15T tokens ≈ 7TB disk). Prior extraction/membership-inference approaches to "memorization" conflate memorization with generalization — e.g. a model that outputs 2^100 correctly wasn't necessarily "memorizing" that number, it may have learned arithmetic.

### Definitions
- Total memorization: mem(X, Θ̂) = I(X, Θ̂) = H(X) − H(X|Θ̂)
- Unintended memorization (isolating from a true generative model Θ): memU(X, Θ̂, Θ) = H(X|Θ) − H(X|Θ,Θ̂)
- Generalization/intended memorization = mem − memU
- Practically estimated via Kolmogorov complexity / compression, approximated with arithmetic coding tied to model likelihoods: H^K(x|θ̂) ≈ −log p(x|θ̂); with reference model θ: H^K(x|θ̂,θ) ≈ −log max{p(x|θ̂), p(x|θ)}

### Capacity measurement (synthetic uniform random bitstrings — eliminates generalization entirely)
- Trained GPT-2-style transformers, 100K–20M params (later scaled to 1.7M-1.5B), on random token sequences.
- Result: memorization scales linearly with dataset size until it plateaus at a model-specific ceiling = capacity.
- **Estimated capacity ≈ 3.6 bits-per-parameter for GPT-style transformers** (mean 3.83 bpp fp32, 3.51 bpp bf16 across architectures; doubling precision from bf16→fp32 only raises capacity ~10%).
- Capacity estimate is remarkably stable across widths/depths — suggests architecture-independent information ceiling, not a specific design choice.

### Real text experiments
- On real text, models memorize until capacity fills, then substitute memorization for generalization ("grokking" begins).
- Double descent occurs exactly when dataset size (in bits) exceeds model capacity — before that point, more params/data help memorize; after, it forces generalization.

### Membership inference scaling law
- Derived scaling law relating model capacity, dataset size, and MIA success.
- Bigger models can memorize more samples; bigger datasets make membership inference harder (capacity divided among more samples → thinner per-sample signal).
- Extrapolating: most modern LLMs are trained on data far exceeding what their capacity could reliably memorize per-sample, implying average-case membership inference should be difficult/unreliable at current data:parameter ratios.
- Caveat: duplicated/near-duplicate data disproportionately increases memorization risk for those specific data points (consistent with Carlini et al. findings on data duplication).

## Key Figures
- Figure 1: Unintended memorization of uniform random data — plateaus at empirical capacity ~3.6 bpp.
- Figure 3/4: Double descent occurs when dataset-to-capacity ratio crosses 1.
- Figure 6: Capacity in bpp across d_model — mean 3.64 bpp.
- Figure 7: Scaling law curves for membership inference vs dataset size.
