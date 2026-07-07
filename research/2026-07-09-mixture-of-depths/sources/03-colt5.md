---
url: https://arxiv.org/abs/2303.09752v2
title: "CoLT5: Faster Long-Range Transformers with Conditional Computation"
type: arxiv_paper
authors: Joshua Ainslie, Tao Lei, Michiel de Jong, et al. (Google Research)
year: 2023
accessed: 2026-07-09
quality: 5
relevance: closest prior work
---

## Abstract
Long documents benefit Transformers but processing them is expensive — not just quadratic attention but also from applying feedforward/projection layers to every token. Not all tokens are equally important, especially in long documents. CoLT5 employs conditional computation, devoting more resources (heavier FFN pathway, wider attention) to important tokens in both feedforward and attention layers. Achieves stronger performance than LongT5 with much faster training/inference, SOTA on SCROLLS benchmark, and scales effectively to 64k input length.

## Relevance
MoD paper calls out CoLT5 as its closest prior work. CoLT5 uses soft top-k routing to choose whether a token goes through a heavy or light FFN, and whether it attends broadly or narrowly. Key difference: CoLT5 is encoder-decoder, so the non-causal nature of top-k routing is a non-issue (the whole encoder input is available at once). MoD's key technical contribution beyond CoLT5 is solving the SAME routing idea for the DECODER-ONLY / autoregressive setting, where top-k is non-causal and needs a causal predictor workaround at inference time.
