---
url: https://www.neelnanda.io/mechanistic-interpretability/othello
title: "Actually, Othello-GPT Has A Linear Emergent World Representation"
type: blog
authors: Neel Nanda
year: 2023
accessed: 2026-07-11
quality: 4
relevance: core
---

## Key Content
- Follow-up to Kenneth Li's Othello-GPT paper. Nanda shows that the "nonlinear" result in the original paper was an artifact of how the board state was encoded for probing.
- Original paper probed for absolute color (black/white) at each tile — this requires a nonlinear probe because absolute color flips every move (whoever's turn it is).
- Nanda re-encodes the board state relative to the CURRENT PLAYER ("mine" vs "yours" vs "empty", not "black" vs "white") — and finds this representation IS linearly decodable with high accuracy using simple linear probes.
- This is a crucial nuance for the "do LLMs have linear world models" debate: the *existence* of a linear representation depends heavily on the choice of coordinate system / basis you probe in. A representation that looks nonlinear in one basis can be linear in another change-of-variables.
- Also confirms via activation patching / interventions that this linear direction is causally used by the model to decide legal moves — reinforcing that this isn't just a decodable correlate but an actively used computational feature.
- Ties into broader "linear representation hypothesis" work (Marks & Tegmark's Geometry of Truth, Anthropic's dictionary learning) suggesting concepts in transformers tend to be represented as directions/subspaces in activation space, not as scattered, tangled, nonlinear soup.
