---
url: https://arxiv.org/abs/2404.05892
title: "Eagle and Finch: RWKV with Matrix-Valued States and Data-Dependent Recurrence"
type: arxiv_paper
authors: Bo Peng et al.
year: 2024
quality: 5
relevance: core
---

# RWKV v5 (Eagle) & v6 (Finch)

## RWKV-5 (Eagle) Innovation: Matrix-Valued States
- Replaces vector state with d×d matrix per head
- Dramatically increases memory capacity
- O(Td) training, O(d²) inference memory

## RWKV-6 (Finch) Innovation: Data-Dependent Recurrence
- Decay factors and mixing coefficients are functions of input
- Dynamic rather than fixed learned parameters

## Core WKV Mechanism (v6)
wkv_t = diag(w_t) · wkv_{t-1} + k_t ⊗ v_t  (w_t is input-dependent)
o_t = sigmoid(r_t) ⊙ (wkv_t · q_t)  (r_t is receptance gate)

## Training & Results
- Eagle 7B: competitive with similar-sized Transformers
- Trained on 1.1T tokens across 100+ languages
- Finch 7B/14B: further improvements
- Constant O(d²) memory per step at inference (vs O(n·d) for Transformers)
