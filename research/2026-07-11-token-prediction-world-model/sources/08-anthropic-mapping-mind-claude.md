---
url: https://www.anthropic.com/research/mapping-mind-language-model
title: "Mapping the Mind of a Large Language Model" (Scaling Monosemanticity)
type: industry_research
authors: Anthropic interpretability team
year: 2024
accessed: 2026-07-11
quality: 5
relevance: supporting
---

## Key Content
- Applies sparse dictionary learning (sparse autoencoders, SAEs) to Claude 3 Sonnet — a full production-scale model, not a toy — to decompose the model's dense, entangled neuron activations into millions of more human-interpretable "features."
- Key motivating problem described in the piece: raw neurons are "polysemantic" — each neuron responds to many unrelated concepts, and each concept is spread across many neurons — so you cannot read off what a model "believes" or "is thinking" by looking at individual neurons directly. SAEs solve this by re-basis-ing activations into an overcomplete but much more monosemantic ("one-meaning-per-feature") coordinate system.
- Finds coherent, interpretable features for extremely abstract and multimodal/multilingual concepts — e.g. a "Golden Gate Bridge" feature that fires for the bridge whether mentioned in English, Chinese, or as an image, and for related concepts (San Francisco, fog, Alcatraz).
- Demonstrates these features are CAUSAL, not just correlative: artificially amplifying the "Golden Gate Bridge" feature causes the model to become fixated on the bridge in unrelated conversations ("Claude, are you the Golden Gate Bridge?" viral demo), directly showing the internal feature drives behavior.
- Significance for the token-prediction/world-model question: this is industrial-scale confirmation that large, real deployed LLMs — not just tiny synthetic-task models like Othello-GPT — organize their internal computation around structured, extractable, steerable, and multimodal-unified conceptual representations, not simply an unstructured statistical soup. Strongly supports the "compression forces structure" argument even though it doesn't resolve the deeper philosophical question of "understanding."
