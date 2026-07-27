---
url: https://arxiv.org/abs/2310.07923
title: "The Expressive Power of Transformers with Chain of Thought"
type: arxiv_paper
authors: William Merrill, Ashish Sabharwal
year: 2023
accessed: 2026-07-30
quality: 5
relevance: core
---

Abstract: Recent theoretical work has identified surprisingly simple reasoning problems, such as checking if two nodes in a graph are connected or simulating finite-state machines, that are provably unsolvable by standard transformers that answer immediately after reading their input. We ask: Does intermediate generation ("chain of thought"/"scratchpad") fundamentally extend the computational power of a decoder-only transformer? We show yes, but the increase depends crucially on the amount of intermediate generation. Logarithmic decoding steps push limits only slightly; linear decoding steps (with projected pre-norm) add the ability to recognize all regular languages; polynomial steps with generalized pre-norm make transformers recognize exactly P — the first exact characterization of a transformer variant in terms of standard complexity classes.

Key takeaway: precise gradation — log steps / linear steps / poly steps of CoT correspond to progressively larger complexity classes, culminating in an exact P characterization.
