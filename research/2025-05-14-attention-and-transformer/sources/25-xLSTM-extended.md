---
url: https://arxiv.org/abs/2405.04517
title: "xLSTM: Extended Long Short-Term Memory"
type: arxiv_paper
authors: Maximilian Beck, Korbinian Pöppel, Sepp Hochreiter et al.
year: 2024
quality: 5
relevance: core
---

# xLSTM: Extended LSTM

## Two Variants

### sLSTM (Scalar LSTM)
- Exponential gating: i_t = exp(ĩ_t), f_t = exp(f̃_t)
- Normalizer state n_t for numerical stability
- Memory mixing between cells
- NOT fully parallelizable (hidden-to-hidden)

### mLSTM (Matrix LSTM)
- Matrix memory cell: C_t = f_t C_{t-1} + i_t v_t k_t^T
- Normalizer: n_t = f_t n_{t-1} + i_t k_t
- Output: h_t = C_t q_t / max(|n_t^T q_t|, 1)
- Fully parallelizable (no hidden-to-hidden within cell)
- Covariance update rule (outer product v_t k_t^T)

## Architecture
- xLSTM[7:1] = 7 mLSTM blocks per 1 sLSTM block
- Best ratio found empirically

## Results
- 1.3B on SlimPajama (300B tokens): matches/outperforms Transformers, Mamba, RWKV-4
- xLSTM 7B (2025): competitive inference throughput advantages
