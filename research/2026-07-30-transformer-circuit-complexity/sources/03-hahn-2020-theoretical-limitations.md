---
url: https://arxiv.org/abs/1906.06755
title: "Theoretical Limitations of Self-Attention in Neural Sequence Models"
type: arxiv_paper
authors: Michael Hahn
year: 2020
accessed: 2026-07-30
quality: 5
relevance: core
---

Abstract: Transformers are emerging as the new workhorse of NLP, showing great success across tasks. Unlike LSTMs, transformers process input sequences entirely through self-attention. Previous work has suggested that the computational capabilities of self-attention to process hierarchical structures are limited. In this work, we mathematically investigate the computational power of self-attention to model formal languages. Across both soft and hard attention, we show strong theoretical limitations of the computational abilities of self-attention, finding that it cannot model periodic finite-state languages, nor hierarchical structure, unless the number of layers or heads increases with input length.

Key takeaway: earliest rigorous impossibility result for self-attention on PARITY / hierarchical structure (Dyck languages), for hard and soft attention.
