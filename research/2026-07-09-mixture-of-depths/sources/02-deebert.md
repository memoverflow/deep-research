---
url: https://arxiv.org/abs/2004.12993
title: "DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference"
type: arxiv_paper
authors: Ji Xin, Raphael Tang, Jaejun Lee, Yaoliang Yu, Jimmy Lin
year: 2020
accessed: 2026-07-09
quality: 5
relevance: background/contrast
---

## Abstract
Large-scale pre-trained language models such as BERT bring big NLP improvements but are notoriously slow at inference. DeeBERT accelerates BERT inference by letting samples exit early without passing through the entire model — up to ~40% inference time savings with minimal quality degradation. Analysis reveals redundancy across BERT's transformer layers. Code: github.com/castorini/DeeBERT

## Relevance
This is the classic "early exit" conditional computation approach that Mixture-of-Depths explicitly contrasts itself against. Early-exit methods decide, per-sample (or per-token in later variants), to STOP computation entirely once a confidence threshold is reached at some layer, skipping ALL remaining layers serially. This is fundamentally different from MoD's "skip-then-re-engage" pattern where a token can skip middle layers but still be processed (and attended to) by later layers. Establishes why MoD's approach of routing per-block rather than committing to permanent early exit is a genuinely different design point, not just a rebrand of early-exit.
