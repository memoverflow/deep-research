---
url: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
title: "In-context Learning and Induction Heads"
type: technical_report
authors: Catherine Olsson, Nelson Elhage, Neel Nanda, et al. (Anthropic)
year: 2022
accessed: 2026-07-16
quality: 5
relevance: supporting
---

Referenced in the MTP paper as the origin of "induction heads" — attention circuits
that implement "if A was followed by B before, and A appears again, predict B".
Olsson et al. found that the formation of induction heads during training coincides
precisely with a phase change in in-context learning ability, suggesting induction
heads are a core mechanistic building block of ICL. The MTP paper builds a synthetic
task to measure induction capability and shows multi-token prediction accelerates
formation of this capability, especially in smaller models / lower-quality data
regimes.
