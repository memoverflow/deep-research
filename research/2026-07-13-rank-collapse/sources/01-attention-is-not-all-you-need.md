---
url: https://arxiv.org/abs/2103.03404
title: "Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth"
type: arxiv_paper
authors: Yihe Dong, Jean-Baptiste Cordonnier, Andreas Loukas
year: 2021
venue: ICML 2021 (Oral)
accessed: 2026-07-13
quality: 5
relevance: core
---

## Abstract
Attention-based architectures have become ubiquitous in machine learning. This work proposes a new way to understand self-attention networks: decompose the output into a sum of smaller terms, each involving the operation of a sequence of attention heads across layers ("path decomposition"). Using this decomposition, the authors prove that self-attention possesses a strong inductive bias towards "token uniformity": without skip connections or MLPs, the output converges doubly exponentially to a rank-1 matrix. Skip connections and MLPs stop the output from degenerating.

## Key content extracted

### Path decomposition (Theorem 2.1)
Output of a depth-L, width-H self-attention network:
SAN(X) = Σ_{path∈[H]^L} P_path · X · W_path + 1b^T
where P_path = P_{h_L}^L ... P_{h_1}^1 is a product of row-stochastic attention matrices, and W_path is input-independent. Each path corresponds to a choice of head at every layer.

### Main rank collapse result (Theorem 2.3, simplified)
Without skip connections, for depth-L width-H SAN:
||res(SAN(X))||_{1,∞} ≤ (4γβH/√d_qk)^{(3^L-1)/2} · ||res(X)||_{1,∞}^{3^L}

This is a cubic-rate / doubly-exponential convergence to rank 1 (all token representations become identical rows). Convergence is much faster than standard products of stochastic matrices (which converge linearly) because the rank of the attention matrix itself depends on the rank of the input — creating a cascading effect: as tokens become more similar, softmax attention becomes more peaked, accelerating further collapse.

### Skip connections (Claim 3.1)
With skip connections, path count for length-l paths becomes C(L,l)·H^l (paths can skip any subset of layers). Crucially, there is always a length-0 path (skip everything) that preserves the residual perfectly. Proven: for skip-connection networks, there exist parameterizations where ||res(X^L)|| ≥ ||res(X)|| for all L→∞ — i.e., skip connections can fully prevent collapse to zero residual, no matter how deep. This implies deep SANs with skip connections behave like ensembles of shallow single-head networks (echoes the ResNet "ensemble of shallow networks" finding, but ties it explicitly to rank collapse for the first time).

### MLPs (Corollary 3.2)
MLPs slow but don't stop collapse — convergence rate depends on Lipschitz constant λ of the MLP: the more "powerful"/less Lipschitz-bounded the MLP, the slower convergence. Trade-off: higher Lipschitz constants improve robustness against collapse but increase sensitivity to input perturbations and gradient variance (harder to optimize).

### LayerNorm (Section 3.3)
Proven that LayerNorm plays NO role in mitigating rank collapse. LayerNorm is equivalent to right-multiplication by a diagonal matrix (per-token rescale/shift), and right-multiplication cannot increase matrix rank. So LN(SA(X)) has the same rank-collapse behavior as SA(X), just re-scaled.

### Experiments
Verified on BERT, ALBERT, XLNet — removing skip connections causes rapid rank collapse in real trained models. Toy 2D circular trajectory experiment shows: without skip/MLP, predicted trajectories collapse to a single point; adding either component prevents/slows this.

## Notes for article
This is the primary source for the entire article — path decomposition math, the three theorems, and experimental verification all come from here.
