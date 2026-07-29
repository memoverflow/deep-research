---
url: https://arxiv.org/abs/2112.12938
title: "Counterfactual Memorization in Neural Language Models"
type: arxiv_paper
authors: Chiyuan Zhang, Daphne Ippolito, Katherine Lee, Matthew Jagielski, Florian Tramèr, Nicholas Carlini
year: 2021
accessed: 2026-08-01
quality: 4
relevance: supporting
---

## Key Content
Defines counterfactual memorization: the change in a model's prediction on a document if that document were omitted from training (compares models trained with vs without a given document). This distinguishes memorization due to genuine rarity/uniqueness of information from memorization due to mere frequency/typicality — extends Feldman & Zhang (2020)'s influence-based memorization framework to the language modeling setting. Found rare, counterfactually-memorized examples across standard text datasets (C4, Wiki-40B), providing another empirical angle consistent with the long-tail theory: models retain example-specific information especially for documents containing rare information not conveyed elsewhere in the corpus.
