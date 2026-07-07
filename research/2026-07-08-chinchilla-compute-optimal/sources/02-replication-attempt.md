---
url: https://arxiv.org/abs/2404.10102
title: "Chinchilla Scaling: A replication attempt"
type: arxiv_paper
authors: Tamay Besiroglu, Ege Erdil, Matthew Barnett, Josh You
year: 2024
accessed: 2026-07-08
quality: 5
relevance: core
---

## Abstract
Hoffmann et al. (2022) propose three methods for estimating a compute-optimal scaling law. We attempt to replicate their third estimation procedure (parametric loss fitting), which involves fitting a parametric loss function to a reconstruction of data from their plots. We find that the reported estimates are inconsistent with their first two estimation methods, fail at fitting the extracted data, and report implausibly narrow confidence intervals — intervals this narrow would require over 600,000 experiments, while they likely only ran fewer than 500. In contrast, our re-derivation of the scaling law using the third approach yields results that are compatible with the findings from the first two estimation procedures described by Hoffmann et al.

## Key takeaway for article
This is important nuance/honesty material: Approach 3's specific numeric coefficients in the original Chinchilla paper were likely miscalculated or had statistical issues (confidence intervals too narrow to be plausible given known experiment count), but the qualitative conclusion (a≈b≈0.5, roughly equal scaling) still holds up when redone properly and is consistent with Approaches 1 & 2. Good place to model scientific honesty in the article — celebrate the paper's core finding while flagging that the field caught and fixed a methodological flaw.
