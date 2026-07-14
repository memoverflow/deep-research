---
url: https://arxiv.org/abs/1503.02531
title: "Distilling the Knowledge in a Neural Network"
type: arxiv_paper
authors: Geoffrey Hinton, Oriol Vinyals, Jeff Dean
year: 2015
accessed: 2026-07-14
quality: 5
relevance: core
---

Abstract: A very simple way to improve the performance of almost any machine learning
algorithm is to train many different models on the same data and then to average their
predictions. Unfortunately, making predictions using a whole ensemble of models is
cumbersome and may be too computationally expensive to allow deployment to a large
number of users, especially if the individual models are large neural nets. Caruana and
his collaborators have shown that it is possible to compress the knowledge in an ensemble
into a single model which is much easier to deploy and we develop this approach further
using a different compression technique. We achieve some surprising results on MNIST and
we show that we can significantly improve the acoustic model of a heavily used commercial
system by distilling the knowledge in an ensemble of models into a single model. We also
introduce a new type of ensemble composed of one or more full models and many specialist
models which learn to distinguish fine-grained classes that the full models confuse.
Unlike a mixture of experts, these specialist models can be trained rapidly and in
parallel.

Key ideas extracted from search/summaries:
- Introduces "softened" softmax outputs via temperature T: p_i = exp(z_i/T) / sum_j exp(z_j/T)
- Coined "dark knowledge": the relative probabilities assigned to wrong classes encode
  similarity structure a one-hot label destroys (e.g. a "2" that looks a bit like a "3" and
  a bit like a "7" but not at all like a "dog").
- Distillation loss = weighted sum of (a) KL between softened teacher/student distributions
  at temperature T, and (b) standard cross-entropy against hard labels at T=1.
- Because gradients of the soft term scale down by 1/T^2, the loss is conventionally
  multiplied by T^2 to keep the two terms' relative magnitudes stable across different T.
