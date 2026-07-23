---
url: https://arxiv.org/abs/2012.14913
title: "Transformer Feed-Forward Layers Are Key-Value Memories"
type: arxiv_paper
authors: Mor Geva, Roei Schuster, Jonathan Berant, Omer Levy
year: 2021 (EMNLP)
accessed: 2026-07-26
quality: 5
relevance: core (foundational conceptual basis for ROME/MEMIT)
---

Abstract: Feed-forward layers constitute two-thirds of a transformer model's parameters, yet their role in the network remains under-explored. We show that feed-forward layers in transformer-based language models operate as key-value memories, where each key correlates with textual patterns in the training examples, and each value induces a distribution over the output vocabulary. Our experiments show that the learned patterns are human-interpretable, and that lower layers tend to capture shallow patterns, while upper layers learn more semantic ones. The values complement the keys' input patterns by inducing output distributions that concentrate probability mass on tokens likely to appear immediately after each pattern, particularly in the upper layers. Finally, we demonstrate that the output of a feed-forward layer is a composition of its memories, which is subsequently refined throughout the model's layers via residual connections to produce the final output distribution.

## Key content

This is the paper that establishes the "FFN = key-value memory" framing that ROME, MEMIT, and PMET all build on. An FFN sub-layer computes: hidden = f(W_in · x), output = W_out · hidden. Geva et al. show W_in's rows act as "keys" that detect input patterns, and W_out's corresponding columns act as "values" that push toward specific output tokens. Layer-by-layer, the residual stream accumulates and refines the sum of many such key-value memory activations, and lower layers detect shallower (lexical/positional) patterns while higher layers detect more semantic ones.

This gives the mechanistic justification for why ROME can treat "insert a fact" as "insert a new key-value pair" via a rank-one update to a weight matrix — because FFN weight matrices genuinely function like associative memories in aggregate, not just metaphorically.
