---
url: https://arxiv.org/html/2603.10123v1
title: "Lost in the Middle at Birth: An Exact Theory of Transformer Position Bias"
type: arxiv_paper
authors: Borun D Chowdhury
year: 2026
accessed: 2026-07-22
quality: 5
relevance: core
---

Central claim: the U-shape is present at RANDOM INITIALIZATION, before any training and
independent of positional encoding (RoPE or absolute). It's a pure topological/geometric
consequence of (1) causal masking and (2) residual connections in a decoder-only transformer.

Setup: models a single layer of causal attention at init (where q·k ≈ 0 so softmax gives a
uniform distribution over past tokens) as the Cesàro matrix M, M[i,j] = 1/i for j<=i.
Stacking H layers = M^H. With residual mixing weight α: N = (1-α)I + αM, stacked N^H.

Continuous limit (L→∞, x=j/L position fraction): M converges to averaging operator
(Mh)(y) = (1/y)∫₀ʸ h(x)dx. Residual version: N = (1-α)h(y) + α(Mh)(y).

Two ingredients of the U-shape:
1. **Primacy Tail (causal masking)**: pure causal stacking (α=1) for H≥2 layers gives
   influence density ρ_H(x) = (1/(H-1)!) · (ln(1/x))^(H-1), which diverges as x→0 (early
   tokens). Early tokens sit causally "upstream" of exponentially many paths through the
   network — depth alone concentrates influence at the start. This is presented as the
   geometric origin of "attention sinks," prior to any learned behavior.
2. **Recency Anchor (residual connections)**: with α∈(0,1), the final token gets an isolated
   Dirac-delta-like O(1) anchor at x=1 — residual stream lets it "teleport" its gradient
   straight through, bypassing dilution from attention averaging.
3. Between these two: a **factorial dead zone** of order O(1/(H-1)!) in the middle — tokens
   there only reach the output via "hybrid paths" that mix causal averaging (diluted) and
   residual skips, so their influence is combinatorially suppressed relative to both extremes.

RoPE is proven irrelevant at initialization: because query/key vectors are isotropic Gaussian
at init, and isotropic Gaussians are invariant under the orthogonal rotation RoPE applies,
the expected attention score distribution q^T R(θ,i-j) k is identical to q^T k regardless of
relative distance. Empirically validated: Spearman ρ=0.99 between RoPE and no-RoPE Jacobian
norms in untrained Qwen2-0.5B (24 layers) and GPT-2.

Crucially: standard pretraining does NOT overcome this topological baseline — it's an
architectural prior that the optimizer has to fight against with learned, non-linear
query-key correlations (the "Score Pathway"), and the paper argues pretraining objectives
lack targeted pressure to bridge the combinatorially-suppressed middle valley.

Related work referenced:
- Xiao et al. 2023 (attention sinks / StreamingLLM): first empirical/graph formalization of
  early-token attention dumping.
- Wu et al. 2025 (graph-theoretic, no residual): showed causal masking creates asymmetric DAG
  where early tokens are on exponentially more paths — but WITHOUT residuals their model
  predicted total collapse onto token 1 (pure primacy, no U-shape) — an open problem they
  flagged, which this paper + Herasimchyk et al. resolve by adding residual connections.
- Hsieh et al. 2024 "Found in the Middle": empirically calibrates away the U-shaped
  positional attention component to recover relevance-driven attention.
- Herasimchyk et al. 2026 "A Residual-Aware Theory of Position Bias": closest prior work,
  derives similar primacy+recency structure but from cumulative attention rollout on
  parameters measured from *trained* networks (so can't fully decouple structure from
  learning); Spearman correlation 0.88–0.98 against real pretrained model attention profiles.
