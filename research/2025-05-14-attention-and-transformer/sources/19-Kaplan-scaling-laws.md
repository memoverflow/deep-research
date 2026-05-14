---
url: https://arxiv.org/abs/2001.08361
title: "Scaling Laws for Neural Language Models"
type: arxiv_paper
authors: Jared Kaplan, Sam McCandlish, Tom Henighan, et al.
year: 2020
quality: 5
relevance: core
---

# Scaling Laws for Neural Language Models

## Core Findings
Power-law relationships spanning 7+ orders of magnitude:
- L(N) ∝ N^(-0.076) — loss vs model parameters
- L(D) ∝ D^(-0.095) — loss vs dataset size  
- L(C) ∝ C^(-0.050) — loss vs compute

## Key Conclusions
1. Model size matters more than dataset size for fixed compute
2. Architecture details (depth vs width) matter relatively little vs total parameter count
3. Trends are smooth and predictable
4. Optimal compute allocation: grow model size faster than data

## Impact
Guided development of GPT-3 (175B params, ~300B tokens) — prioritizing parameters over data. Later corrected by Chinchilla.
