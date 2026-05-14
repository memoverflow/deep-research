---
url: https://arxiv.org/abs/2006.03654
title: "DeBERTa: Decoding-enhanced BERT with Disentangled Attention"
type: arxiv_paper
authors: Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen
year: 2020
quality: 5
relevance: high
---

# DeBERTa: Disentangled Attention

## Innovation 1: Disentangled Attention
Separate content and position vectors, compute attention using:
- Content-to-Content: H_i · H_j^T
- Content-to-Position: H_i · P_{|i-j|}^T  
- Position-to-Content: P_{|i-j|} · H_j^T

(Unlike BERT which sums content+position before attention)

## Innovation 2: Enhanced Mask Decoder
- Incorporates absolute position only at the decoding/prediction layer
- Relative position throughout encoder, absolute only at final MLM head

## Results
- DeBERTa-1.5B surpassed human on SuperGLUE: 90.3 vs 89.8
- First model to exceed human baseline on this benchmark
