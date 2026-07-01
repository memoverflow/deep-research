---
url: https://arxiv.org/abs/2206.11893
title: "On the Parameterization and Initialization of Diagonal State Space Models (DSS)"
type: arxiv_paper
authors: Ankit Gupta, Albert Gu, Jonathan Berant
year: 2022
accessed: 2026-07-01
quality: 4
relevance: supporting
---

Description: A recent variant of S4 called DSS showed that restricting the state matrix to be fully diagonal can still preserve the performance of the original model when using a specific initialization based on approximating S4's matrix.

Key facts extracted:
- DSS simplifies S4's DPLR (diagonal plus low-rank) to a pure diagonal matrix
- Removing the low-rank correction term still works if initialization approximates HiPPO structure
- Precursor to S5's fully diagonal + parallel scan approach, and to Mamba's diagonal SSM core
