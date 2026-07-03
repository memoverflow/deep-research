---
url: https://arxiv.org/abs/1808.09582
title: "Breaking the Beam Search Curse: A Study of (Re-)Scoring Methods and Stopping Criteria for Neural Machine Translation"
type: arxiv_paper
authors: Yilin Yang, Liang Huang, Mingbo Ma
year: 2018
accessed: 2026-07-04
quality: 5
relevance: core
---

Abstract: Beam search is widely used in NMT and usually improves translation quality
compared to greedy search. It has been widely observed that beam sizes larger than 5 hurt
translation quality — this is "the beam search curse." The paper explains why this happens
(length/shortness bias in raw log-probability accumulation favors shorter outputs as beam
widens toward the true MAP optimum) and proposes hyperparameter-free rescoring methods that
outperform standard length normalization by +2.0 BLEU.

Used to explain why wider beam ≠ better quality, and to justify the length-normalization
formula presented in the article.
