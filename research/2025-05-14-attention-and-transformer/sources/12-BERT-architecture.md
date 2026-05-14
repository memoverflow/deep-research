---
url: https://arxiv.org/abs/1810.04805
title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
type: arxiv_paper
authors: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
year: 2018
quality: 5
relevance: core
---

# BERT Architecture

BERT introduces bidirectional pre-training using:
- **Masked Language Model (MLM)**: 15% of input tokens randomly masked, model predicts them using both left and right context
- **Next Sentence Prediction (NSP)**: Binary classification of sentence pair relationships

## Configurations
- BERT-Base: 12 layers, 768 hidden, 12 heads, 110M params
- BERT-Large: 24 layers, 1024 hidden, 16 heads, 340M params

## Training
- BooksCorpus (800M words) + English Wikipedia (2,500M words)
- WordPiece tokenization, 30,000 vocabulary

## Results
- GLUE: 80.5% (7.7 point improvement)
- SQuAD 1.1: 93.2 F1
- SQuAD 2.0: 83.1 F1
- SOTA on 11 NLP benchmarks simultaneously
