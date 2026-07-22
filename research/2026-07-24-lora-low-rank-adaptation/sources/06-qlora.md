---
url: https://arxiv.org/abs/2305.14314
title: "QLoRA: Efficient Finetuning of Quantized LLMs"
type: arxiv_paper
authors: Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer
year: 2023
accessed: 2026-07-24
quality: 5
relevance: supporting (practical extension)
---

## Abstract
QLoRA reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance. Backpropagates gradients through a frozen, 4-bit quantized pretrained LM into Low Rank Adapters. Best model family (Guanaco) outperforms all previous openly released models on the Vicuna benchmark, reaching 99.3% of ChatGPT performance with 24 hours finetuning on a single GPU. Introduces: (a) 4-bit NormalFloat (NF4) data type, information-theoretically optimal for normally distributed weights, (b) double quantization reducing average memory footprint by quantizing the quantization constants themselves, (c) paged optimizers to manage memory spikes. Finetuned 1,000+ models; detailed instruction-following/chatbot analysis.

## Key Content
- Combines quantization (compressing the frozen base model to 4-bit) with LoRA (small trainable adapters kept in higher precision, e.g. bf16) — the base model's memory footprint shrinks ~4x while the trainable adapter is unaffected in precision, so accuracy is preserved.
- NF4: a quantization data type whose quantization bins are placed according to the theoretical quantiles of a zero-centered normal distribution — matches the empirical distribution of neural net weights better than uniform int4, reducing quantization error.
- Double quantization: even the small per-block scaling constants used for quantization take up memory at scale (thousands of blocks); quantizing those constants too saves further memory with negligible extra error.
- Paged optimizers: uses NVIDIA unified memory to automatically page optimizer states to CPU RAM during rare memory spikes (e.g. long sequences), avoiding OOM crashes without needing to manually manage memory.
- Practical significance: this is the technique that made "fine-tune a 65B/70B model on one consumer/prosumer GPU" possible, directly built on top of LoRA's separation of frozen base weights vs small trainable adapters.
