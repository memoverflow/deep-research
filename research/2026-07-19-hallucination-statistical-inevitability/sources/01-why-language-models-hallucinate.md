---
url: https://arxiv.org/abs/2509.04664
title: "Why Language Models Hallucinate"
type: arxiv_paper
authors: Adam Tauman Kalai, Ofir Nachum, Santosh S. Vempala, Edwin Zhang
year: 2025
accessed: 2026-07-19
quality: 5
relevance: core
---

Abstract: Large language models sometimes guess when uncertain, producing plausible yet incorrect
statements instead of admitting uncertainty. The paper argues hallucinations originate as ordinary
errors in binary classification via a reduction from generation to the "Is-It-Valid" (IIV) problem,
and persist post-training because most benchmarks use binary 0-1 grading that rewards confident
guessing over abstention.

Key content extracted (full PDF read):

1. IIV reduction: treat "is this a valid output" as a binary classification problem with training
   data = 50/50 mix of valid examples (+) and uniformly random errors (-). Any base model can act
   as an IIV classifier by thresholding pˆ(x) at 1/|E|. Core inequality:
   err(generation) ≥ 2·err(IIV) − |V|/|E| − δ,
   where δ measures miscalibration (|pˆ(A) − p(A)|). This is Corollary 1 / Theorem 1 (with prompts).

2. Calibration argument: δ is the derivative of cross-entropy loss w.r.t. a rescaling parameter s
   at s=1. If δ≠0, rescaling reduces loss, so a well-optimized model (minimizing cross-entropy)
   naturally drives δ→0, i.e. becomes calibrated — and calibration + Corollary 1 together force
   err > 0 whenever err_iiv is large. GPT-4 calibration histograms (OpenAI 2023, Figure 8) show
   pretrained models are well-calibrated before RLHF, and calibration degrades after RL fine-tuning.

3. "Hallucinations are inevitable only for base models" — a trivial non-hallucinating model exists
   (QA database + calculator + IDK otherwise), and Corollary 1 implies non-erring models must have
   large δ (be miscalibrated). This is why post-training / abstention matters.

4. Singleton rate (Theorem 2, building on Kalai & Vempala 2024's Good-Turing missing-mass
   connection): the hallucination rate lower bound ≈ the fraction of training facts that appear
   exactly once (singletons), i.e. sr. Example: birthday questions, if 20% of birthday facts are
   singletons in training data, expect ≥20% hallucination rate on birthday questions.
   err ≥ sr − 2/min|E_c| − sqrt((35+6lnN)/N) − δ.

5. Poor-model factor (Theorem 3, pure multiple choice): err ≥ 2(1-1/C)·opt(G) where C is number of
   choices. Corollary 2: trigram language models must have generation error rate ≥ 1/2 for
   grammatical-agreement tasks with 2 choices, since opt(G)=1/2 for context-blind trigram models.

6. Additional error factors: computational hardness (can't beat complexity theory, e.g. decrypting
   ciphertext), distribution shift (OOD prompts), GIGO (garbage-in-garbage-out from noisy corpora).

7. Post-training section: binary 0-1 grading (accuracy, pass-rate) never rewards IDK/abstention.
   Observation 1 (proven trivially): for ANY distribution of beliefs about the correct answer under
   binary grading, abstention is never optimal — guessing always beats IDK in expectation.
   Model B (never expresses uncertainty, always guesses) beats Model A (perfectly calibrated,
   abstains when uncertain) under standard 0-1 accuracy metrics.

8. Table 2 meta-evaluation of common benchmarks (GPQA, MMLU-Pro, IFEval, Omni-MATH, WildBench, BBH,
   MATH L5, MuSR, SWE-bench, HLE): nearly all use binary grading with zero credit for IDK — only
   WildBench gives partial credit.

9. Proposed fix: explicit confidence-threshold prompts, e.g. "answer only if you are >t confident,
   since mistakes are penalized t/(1-t) points, correct answers 1 point, IDK 0 points." This creates
   "behavioral calibration" — a single strategy (answer iff correctness probability > t) that is
   simultaneously optimal for all thresholds t, unlike unspecified-penalty binary grading where no
   single strategy dominates.

10. Limitations discussed: doesn't model latent/hidden context, doesn't address degrees of
    hallucination in open-ended generation, search/RAG doesn't eliminate the incentive problem
    (Observation 1 still holds for RAG-augmented models — the grading system still rewards guessing
    when retrieval fails).

Related prior work cited: Kalai & Vempala 2024 "Calibrated Language Models Must Hallucinate"
(STOC 2024) — established singleton-rate bound without prompts/IDK, inspiring Theorem 3 here.
Kalavasis et al. 2025, Kleinberg & Mullainathan 2024 — formalize trade-off between consistency
(no hallucination) and breadth (diverse generation): any model generalizing beyond training data
either hallucinates or suffers mode collapse.
