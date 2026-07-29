---
url: https://arxiv.org/abs/2202.07646
title: "Quantifying Memorization Across Neural Language Models"
type: arxiv_paper
authors: Nicholas Carlini et al.
year: 2022
accessed: 2026-08-01
quality: 5
relevance: supporting
---

## Key Content
Comprehensive study quantifying memorization across three families of LMs with access to original training sets, giving order-of-magnitude tighter bounds than prior black-box extraction studies. Identifies key factors driving memorization: model capacity (larger = more memorization), data duplication (repeated/near-duplicate examples memorized disproportionately more), and prompt length (longer context prefix increases extraction success). This duplication finding is consistent with and complements Morris et al. 2025's caveat that per-sample membership inference risk rises sharply for duplicated data points even when average-case risk is low.
