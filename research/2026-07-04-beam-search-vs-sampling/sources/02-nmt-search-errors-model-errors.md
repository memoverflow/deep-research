---
url: https://arxiv.org/abs/1908.10090
title: "On NMT Search Errors and Model Errors: Cat Got Your Tongue?"
type: arxiv_paper
authors: Felix Stahlberg, Bill Byrne
year: 2019
accessed: 2026-07-04
quality: 5
relevance: core
---

Key finding (from search snippets + abstract page): Using an exact decoding procedure
(beam search + depth-first search) to find the true global best-scoring output under a
trained Transformer NMT model on the entire WMT15 En-De test set, the paper shows the
global best translation is the EMPTY STRING for almost all sentences longer than 40 tokens.
Even excluding empty-preferring sentences, large numbers of search errors remain. Conclusion:
vanilla NMT models in their current form paradoxically rely on beam search's *inexactness* to
avoid degenerate high-scoring outputs like the empty string — an "unsatisfactory" modeling
result, since the true MAP optimum of the model is often nonsensical.

Used in article as the opening hook: "the most probable sentence is often empty."
