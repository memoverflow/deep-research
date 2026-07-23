---
url: https://arxiv.org/abs/2104.08696
title: "Knowledge Neurons in Pretrained Transformers"
type: arxiv_paper
authors: Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, Furu Wei
year: 2021
accessed: 2026-07-26
quality: 5
relevance: supporting (precursor / contrast to ROME's "linear" view)
---

Abstract: Large-scale pretrained language models are surprisingly good at recalling factual knowledge presented in the training corpus. In this paper, we present preliminary studies on how factual knowledge is stored in pretrained Transformers by introducing the concept of knowledge neurons. Specifically, we examine the fill-in-the-blank cloze task for BERT. Given a relational fact, we propose a knowledge attribution method (based on integrated gradients) to identify the neurons that express the fact. We find that the activation of such knowledge neurons is positively correlated to the expression of their corresponding facts. In our case studies, we attempt to leverage knowledge neurons to edit (such as update, and erase) specific factual knowledge without fine-tuning.

## Key content

This is the earlier "individual-neuron" hypothesis of factual storage, predating ROME's "linear/rank-one" view. Dai et al. use integrated gradients over FFN intermediate neuron activations to attribute a specific fact to a small set of neurons ("knowledge neurons") within FFN layers of BERT. They show these neurons can be directly amplified/suppressed to update or erase facts without fine-tuning.

Contrast with ROME: ROME explicitly argues for a *linear* view (memories are rank-one slices in a weight matrix that maps a key vector to a value vector) rather than individual neurons carrying facts one-to-one. The knowledge neuron view is a useful historical stepping stone but has been criticized as less robust/precise than causal-tracing + rank-one editing for controlled edits (individual neuron ablation tends to be noisier and less specific than rank-one weight edits).
