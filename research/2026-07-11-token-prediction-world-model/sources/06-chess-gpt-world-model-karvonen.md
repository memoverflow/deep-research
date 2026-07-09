---
url: https://adamkarvonen.github.io/machine_learning/2024/01/03/chess-world-models.html
title: "Chess-GPT's Internal World Model"
type: blog
authors: Adam Karvonen
year: 2024
accessed: 2026-07-11
quality: 4
relevance: core
---

## Key Content
- Extends Othello-GPT's methodology to chess, addressing a specific limitation: Othello-GPT's clean results only held for a model trained on SYNTHETIC uniformly-random games, not on real human games — a serious caveat for real-world applicability (real text isn't uniformly sampled).
- Trains a 50M-parameter GPT purely on PGN chess move text (e.g. "1.e4 e5 2.Nf3...") — the model never receives an explicit board representation or explicit rules.
- Result: model reaches ~1300 Elo playing strength (later scaled models much higher), and — using LINEAR probes — the internal activations accurately decode the full board state at any point in a game, including auxiliary latent variables NOT directly present in the move text, such as an estimate of each player's Elo rating (skill level), inferred purely from move-choice statistics.
- Also recovers rule-knowledge: check, checkmate, castling, en passant, promotion, pinned pieces — all learned purely from the statistical structure of move sequences.
- Karvonen's finding that Elo rating is inferred as a LATENT variable is significant for the "world model" debate: it shows the network isn't just tracking surface legality patterns, but has built an implicit model of unobserved generative factors behind the text (who is playing, how skilled they are) — closer to modeling the DATA GENERATING PROCESS, which is exactly the definition Li et al. use for "world model."
- Follow-up post ("Manipulating Chess-GPT's World Model") shows these representations are causally intervenable — editing the decoded board state changes subsequent move predictions in the expected way, mirroring Othello-GPT's causal intervention result.
