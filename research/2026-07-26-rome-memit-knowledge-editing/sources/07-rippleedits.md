---
url: https://arxiv.org/abs/2307.12976
title: "Evaluating the Ripple Effects of Knowledge Editing in Language Models"
type: arxiv_paper
authors: Roi Cohen, Eden Biran, Ori Yoran, Amir Globerson, Mor Geva
year: 2023 (TACL 2024)
accessed: 2026-07-26
quality: 5
relevance: critical — evaluation limitation of editing methods
---

Abstract: Modern language models capture a large body of factual knowledge. However, some facts can be incorrectly induced or become obsolete over time, resulting in factually incorrect generations. This has led to the development of various editing methods that allow updating facts encoded by the model. Evaluation of these methods has primarily focused on testing whether an individual fact has been successfully injected, and if similar predictions for other subjects have not changed. Here we argue that such evaluation is limited, since injecting one fact (e.g. "Jack Depp is the son of Johnny Depp") introduces a "ripple effect" in the form of additional facts that the model needs to update (e.g. "Jack Depp is the sibling of Lily-Rose Depp"). To address this issue, we propose a novel set of evaluation criteria that consider the implications of an edit on related facts. Using these criteria, we then construct RippleEdits, a diagnostic benchmark of 5K factual edits, capturing a variety of types of ripple effects. We evaluate prominent editing methods on RippleEdits, showing that current methods fail to introduce consistent changes in the model's knowledge. In addition, we find that a simple in-context editing baseline obtains the best scores on our benchmark, suggesting a promising research direction for model editing.

## Key content

Fatal weakness exposed: editing a single fact (e.g. changing "Jack Depp's father") should logically imply many downstream facts change too (siblings, nationality inherited relations, etc.) — a "ripple effect." Existing evaluation (efficacy/generalization/specificity on the edited fact and unrelated neighbors) never checks these logically-implied related facts. RippleEdits benchmark (5K edits, multiple ripple-effect categories: logical, compositional, subject aliasing, preservation, etc.) tests this directly.

Result: ROME, MEMIT, and other locate-then-edit methods all fail to propagate the edit consistently to logically related facts — the edit is "local" in a way that's too local; the model still contradicts itself on multi-hop implications. Surprisingly, a naive in-context-editing baseline (just prepending the new fact as context at inference time, no weight change) scores better on ripple-consistency than weight-editing methods — suggesting weight edits insert a fact only "skin deep" for that literal subject-relation, without proper causal integration into the model's broader knowledge graph.

Important honest limitation to communicate in blog: locate-then-edit methods look precise and surgical in the narrow efficacy/specificity/generalization metrics from ROME's own CounterFact benchmark, but under stricter multi-hop consistency tests they reveal they don't actually perform "belief revision" — just local pattern insertion.
