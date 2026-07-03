---
url: https://arxiv.org/html/2404.09937v1
title: "Compression Represents Intelligence Linearly"
type: arxiv_paper
authors: Yuzhen Huang, Jinghan Zhang, Zifei Shan, Junxian He
year: 2024
accessed: 2026-07-03
quality: 5
relevance: core
---

Abstract: There is a belief that learning to compress well will lead to intelligence (Hutter, 2006). Recently, language modeling has been shown to be equivalent to compression, which offers a compelling rationale for the success of LLMs. In this work, across 30 public LLMs and 12 diverse benchmarks, the study finds that LLMs' downstream ability is almost linearly correlated with their compression efficiency, with a Pearson correlation coefficient of around -0.95 for each evaluated domain of intelligence (knowledge/commonsense, coding, math reasoning).

Key facts:
- 30 public LLMs, 12 benchmarks tested
- Pearson correlation ρ ≈ -0.95 between BPC (bits per character) and benchmark score
- Relationship holds across varying model sizes, tokenizers, context windows, pretraining data
- Proposes compression efficiency as an unsupervised, contamination-resistant evaluation metric
- Code/data: https://github.com/hkust-nlp/llm-compression-intelligence
