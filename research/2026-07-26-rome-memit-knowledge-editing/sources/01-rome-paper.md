---
url: https://arxiv.org/abs/2202.05262
title: "Locating and Editing Factual Associations in GPT"
type: arxiv_paper
authors: Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov
year: 2022
accessed: 2026-07-26
quality: 5
relevance: core
---

Abstract: We analyze the storage and recall of factual associations in autoregressive transformer language models, finding evidence that these associations correspond to localized, directly-editable computations. We first develop a causal intervention for identifying neuron activations that are decisive in a model's factual predictions. This reveals a distinct set of steps in middle-layer feed-forward modules that mediate factual predictions while processing subject tokens. To test our hypothesis that these computations correspond to factual association recall, we modify feed-forward weights to update specific factual associations using Rank-One Model Editing (ROME). We find that ROME is effective on a standard zero-shot relation extraction (zsRE) model-editing task, comparable to existing methods. To perform a more sensitive evaluation, we also evaluate ROME on a new dataset of counterfactual assertions, on which it simultaneously maintains both specificity and generalization, whereas other methods sacrifice one or another. Our results confirm an important role for mid-layer feed-forward modules in storing factual associations and suggest that direct manipulation of computational mechanisms may be a feasible approach for model editing.

## From project page (rome.baulab.info) — key extracted content

Factual associations can be localized along three dimensions: (1) MLP module parameters, (2) at a range of middle layers, (3) specifically during processing of the last token of the subject.

Causal Tracing method: run the network multiple times, corrupt the computation (add noise to subject token embeddings), then restore individual internal states to see which restoration recovers the correct answer. This isolates which hidden states carry decisive information — an "early site" in mid-layer MLPs during subject-token processing, and a "late site" in attention near the end of the sequence.

ROME treats an MLP module as a simple key-value store: the input vector (key) encodes a subject, the output vector (value) encodes learned properties about that subject. ROME makes a rank-one modification to the weight matrix that maps keys to values, inserting a new key-value association. This is a "linear view of memory" — individual facts correspond to rank-one slices of parameter space, contrasting with the earlier "individual neuron" view (knowledge neurons).

Knowing vs Saying distinction — two hallmarks used to test true "knowledge" edits vs superficial parroting:
- Specificity: editing one fact shouldn't affect other, unrelated facts (e.g. after "Eiffel Tower is in Rome," other landmarks should not also become "in Rome").
- Generalization: knowledge should transfer across paraphrases and different contexts, not just the literal training sentence.

Experimental finding: editing the early MLP site (ROME) achieves high efficacy, specificity, AND generalization simultaneously. Editing the later attention site achieves fair efficacy/specificity but fails generalization — supporting the causal tracing localization.

CounterFact dataset introduced: thousands of counterfactual assertions for quantitative specificity/generalization testing (more sensitive than zsRE).

Related foundational work cited: Geva et al. 2021 "Transformer Feed-Forward Layers Are Key-Value Memories" (EMNLP) — the conceptual basis for viewing FFN as key-value memory that ROME builds on directly.
