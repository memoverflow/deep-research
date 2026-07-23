---
url: https://arxiv.org/abs/2210.07229
title: "Mass-Editing Memory in a Transformer"
type: arxiv_paper
authors: Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, David Bau
year: 2022/2023 (ICLR 2023)
accessed: 2026-07-26
quality: 5
relevance: core
---

Abstract: Recent work has shown exciting promise in updating large language models with new memories, so as to replace obsolete information or add specialized knowledge. However, this line of work is predominantly limited to updating single associations. We develop MEMIT, a method for directly updating a language model with many memories, demonstrating experimentally that it can scale up to thousands of associations for GPT-J (6B) and GPT-NeoX (20B), exceeding prior work by orders of magnitude.

## From project page (memit.baulab.info) — key extracted content

Motivation: LLMs contain implicit world knowledge but no built-in update mechanism. ROME could edit one fact reliably but scaling to many facts (hundreds/thousands) at once was an open challenge — naive repeated ROME edits interfere with each other and degrade the model.

Examples of the knowledge problem: GPT-3 correctly predicts "Polaris is in constellation Ursa Minor" but incorrectly predicts "Arneb is in the constellation of Aquila" (should be Lepus), and gives obsolete answers like "the current VP is Mike Pence."

How MEMIT works:
1. Uses causal tracing (from ROME) to find a *range* of mediating MLP layers (not just one) that jointly recall facts about a subject — for GPT-J these are layers R = {3,4,5,6,7,8}.
2. For a batch of new memories, MEMIT computes the needed update Δ and *spreads* this update across all the mediating layers, rather than concentrating it in a single layer (as ROME does). This way, at the final mediating layer, the accumulated output correctly encodes all the new facts, without any single layer's weight matrix being perturbed too far from its original values (which would break unrelated facts and degrade fluency).
3. This distributed, multi-layer rank-update generalizes ROME's single-layer rank-one edit into a "spread" update across several layers — a key insight is that concentrating thousands of edits in one layer causes catastrophic interference, but distributing the same total edit magnitude across several layers keeps each layer's perturbation small.

Evaluation metric: editing score = harmonic mean of efficacy, generalization, and specificity metrics (Section 5.2.2 of paper). MEMIT maintains generalization, specificity, and fluency at edit scales (thousands of facts) far beyond prior methods (ROME/single edits, MEND, fine-tuning), which degrade rapidly as edit count increases.

Published at ICLR 2023.
