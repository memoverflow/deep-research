---
url: https://arxiv.org/abs/2405.21060
title: "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality"
type: arxiv_paper
authors: Tri Dao, Albert Gu
year: 2024
quality: 5
relevance: core
---

# Mamba-2: State Space Duality (SSD)

## Core Insight
SSM computation ≡ structured masked attention:
y = (L ⊙ (CB^T)) x

Where L is lower-triangular decay matrix from state transition A.

## Mathematical Duality
Restricts Mamba-1's diagonal A to scalar×identity: A_t = α_t I

Recurrent form: h_t = α_t h_{t-1} + B_t x_t, y_t = C_t h_t
Attention form: M = L ⊙ (QK^T) where Q=C, K=B, V=x, L_{ij} = ∏_{k=j+1}^{i} α_k

## Performance
- 2-8x faster than Mamba-1 (uses tensor cores via matrix multiply)
- Larger state dimensions: N=64-256 (vs N=16 in Mamba-1)
- Scalar A restriction compensated by larger state
- Outperforms Mamba-1 and Pythia (Transformer) at equivalent sizes on the Pile

## Significance
Unifies SSM and attention theory → these are NOT fundamentally different approaches, just different parameterizations of structured sequence computation.
