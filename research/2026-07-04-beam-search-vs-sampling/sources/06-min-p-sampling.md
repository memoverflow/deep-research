---
url: https://arxiv.org/abs/2407.01082
title: "Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs"
type: arxiv_paper
year: 2024
accessed: 2026-07-04
quality: 4
relevance: core
---

Abstract: Top-p (nucleus) sampling struggles to balance quality/diversity especially at higher
temperatures. Proposes min-p, a dynamic truncation method scaling the threshold by the top
token's probability — i.e. the candidate pool automatically narrows when the model is
confident and widens when uncertain. Improves quality and diversity across model families
(Mistral, Llama 3) and sizes (1B-123B). Adopted by HF Transformers, vLLM, and other inference
frameworks.

Used in article's sampling-methods list as the most recent/adopted truncation method.
