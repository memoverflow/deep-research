---
url: https://arxiv.org/abs/2310.02207
title: "Language Models Represent Space and Time"
type: arxiv_paper
authors: Wes Gurnee, Max Tegmark
year: 2023
accessed: 2026-07-11
quality: 5
relevance: core
---

## Abstract
The capabilities of large language models (LLMs) have sparked debate over whether such systems just learn an enormous collection of superficial statistics or a set of more coherent and grounded representations that reflect the real world. We find evidence for the latter by analyzing the learned representations of three spatial datasets (world, US, NYC places) and three temporal datasets (historical figures, artworks, news headlines) in the Llama-2 family of models. We discover that LLMs learn linear representations of space and time across multiple scales. These representations are robust to prompting variations and unified across different entity types (e.g. cities and landmarks). In addition, we identify individual "space neurons" and "time neurons" that reliably encode spatial and temporal coordinates. While further investigation is needed, our results suggest modern LLMs learn rich spatiotemporal representations of the real world and possess basic ingredients of a world model.

## Key Content
- Uses Llama-2 family, probes internal activations on named entities (cities, landmarks, historical figures, artworks, news events) for their real-world coordinates (latitude/longitude, or date).
- Finds LINEAR probes recover spatial/temporal coordinates with high fidelity — e.g. predicting a city's latitude/longitude from its hidden representation as accurately as reading off a small feed-forward regression on ground-truth data.
- Representations are "unified" — same linear direction works whether entity is a city, landmark, or country; same direction works for historical figures, artworks, news headlines along the time axis — i.e., not memorized per-entity-type, but a shared underlying geometric axis for "space" and "time."
- Individual neurons found that fire proportionally to spatial/temporal coordinate value ("space neurons", "time neurons") — like a coordinate readout unit.
- Authors explicitly frame this as evidence against the "stochastic parrot" hypothesis (pure surface statistics, no grounding) and toward LLMs having "basic ingredients of a world model" — but hedge with "further investigation is needed."
- Important scope-limit: this shows LLMs encode facts ABOUT space/time (declarative knowledge from training corpus) — it does not by itself show the model can reason with a full physical/causal simulation of space and time (e.g., simulate what happens if you drop a ball). Different senses of "world model."
