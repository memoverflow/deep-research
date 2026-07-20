---
url: https://arxiv.org/abs/2307.03172
title: "Lost in the Middle: How Language Models Use Long Contexts"
type: arxiv_paper
authors: Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang
year: 2023
accessed: 2026-07-22
quality: 5
relevance: core
---

Abstract: While recent language models have the ability to take long contexts as input,
relatively little is known about how well they use longer context. We analyze the
performance of language models on two tasks that require identifying relevant information
in their input contexts: multi-document question answering and key-value retrieval. We find
that performance can degrade significantly when changing the position of relevant
information, indicating that current language models do not robustly make use of information
in long input contexts. In particular, we observe that performance is often highest when
relevant information occurs at the beginning or end of the input context, and significantly
degrades when models must access relevant information in the middle of long contexts, even
for explicitly long-context models. Our analysis provides a better understanding of how
language models use their input context and provides new evaluation protocols for future
long-context language models.

Key findings:
- Multi-document QA task: put a document containing the answer among k distractor documents,
  vary its position (1st, middle, last), measure accuracy.
- Performance follows a clear U-shaped curve: highest when the answer document is first or
  last, lowest when it's in the middle — even worse than having no relevant document at all
  in some configurations.
- This holds across GPT-3.5-Turbo, Claude-1.3, and open models like MPT-30B, LongChat.
  Even models explicitly trained/finetuned for long context still show the U-shape.
- Key-value retrieval task (synthetic, no semantic difficulty) shows the SAME U-shape,
  proving this isn't just a "hard to find the relevant document" reasoning failure — it's a
  more basic positional/architectural effect, since KV retrieval is essentially a lookup.
- Extending context window length does NOT fix the U-shape; it just gives models more "room"
  to lose things in the middle.
- Query-aware contextualization (putting the query both before and after the documents) helps
  some models but not others.
- Practical implication: models with strong open-book QA capability still have significant
  headroom on long-context robustness; evaluation should test different positions of relevant
  info, not just "can it retrieve at all."
