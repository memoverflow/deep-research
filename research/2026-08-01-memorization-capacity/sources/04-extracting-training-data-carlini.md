---
url: https://arxiv.org/pdf/2012.07805
title: "Extracting Training Data from Large Language Models"
type: arxiv_paper
authors: Nicholas Carlini et al.
year: 2020
accessed: 2026-08-01
quality: 5
relevance: supporting
---

## Key Content
Demonstrates black-box extraction attack recovering verbatim training sequences (PII, code, text) from GPT-2 using only query access. Found ~600 memorized/extractable examples from GPT-2's 40GB training set (~0.00000015% of dataset by the paper's later re-analysis in Quantifying Memorization, 2202.07646). Larger models memorize significantly more training data than smaller ones — model scale is a strong predictor of extractable memorization. This "extraction" methodology (inducing verbatim regurgitation) is the predecessor approach that Morris et al. 2025 critique for conflating memorization with generalization (a model can be "coerced" to output almost any string, per Geiping et al. 2024).
