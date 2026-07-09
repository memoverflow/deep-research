---
url: https://aclanthology.org/2020.acl-main.463.pdf
title: "Climbing towards NLU: On Meaning, Form, and Understanding in the Age of Data"
type: paper
authors: Emily M. Bender, Alexander Koller
year: 2020 (ACL)
accessed: 2026-07-11
quality: 5
relevance: core
---

## Key Content
- Foundational skeptic position paper. Core argument: language models are trained ONLY on FORM (distribution over strings/tokens), and meaning is the relation between form and COMMUNICATIVE INTENT grounded in the world. A system trained purely on form has, the authors argue, "a priori no way to learn meaning."
- Distinguishes "form" (observable realization of language: sequences of tokens/characters/sounds) from "meaning" (what the form is used to communicate about the world, intentions, and communicative context).
- Central thought experiment: the "Octopus Test." Two people, A and B, are trapped on separate desert islands, communicating only via a transoceanic cable. An octopus, O, intercepts the cable's signal without any access to the physical world referred to. O has excellent statistical modeling of the pattern of signals between A and B, and can generate plausible-looking replies. But the moment A asks for help building a "coconut catapult to fend off a bear" — something outside training distribution — O has no way to know what a coconut, catapult, or bear IS, because O never had access to their real-world referents, only the word-forms. O's replies will look fluent but can be arbitrarily, and dangerously, wrong.
- Direct implication for LLMs: they argue text-only training data is exactly like O's tapped cable — LLM sees only form, never grounded contact with referents, so any apparent "understanding" is a hallucination of the interpreter (the human evaluator), not a property of the model.
- Anticipates and pre-addresses counterarguments (§9): including the argument that distributional information about co-occurrence patterns can approximate some aspects of meaning (distributional semantics) — but maintains this is insufficient for full human-analogous understanding, especially communicative intent.
- Historically important as the paper that popularized the "stochastic parrot" framing later expanded by Bender, Gebru et al. (2021) "On the Dangers of Stochastic Parrots."
- This is the classic counter-position to the "next-token prediction builds world models" camp — used throughout later interpretability literature as the null hypothesis that empirical probing work (Othello-GPT, space/time paper) tries to falsify.
