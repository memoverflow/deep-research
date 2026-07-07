---
url: https://medium.com/@simeon.emanuilov/mixture-of-depths-a-new-approach-to-efficiently-allocate-compute-in-transformer-language-models-15b0d32ff501
title: "Mixture-of-Depths: A new approach to efficiently allocate compute in transformer language models"
type: blog
year: 2024
accessed: 2026-07-09
quality: 3
relevance: supplementary explainer
---

## Key points extracted from search snippet
- Confirms: routing around every other layer with 12.5% capacity (processing only 12.5% of tokens) yielded the best results.
- Learned routing is essential — models with random routing perform significantly worse than baseline.
- MoD can be effectively combined with Mixture-of-Experts layers for further gains (i.e. MoDE).

Used as a secondary confirmation of the paper's central numeric result (12.5% capacity, every-other-layer) — consistent with the primary source (source 01).
