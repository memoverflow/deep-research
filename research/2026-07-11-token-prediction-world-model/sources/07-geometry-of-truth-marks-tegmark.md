---
url: https://arxiv.org/abs/2310.06824
title: "The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets"
type: arxiv_paper
authors: Samuel Marks, Max Tegmark
year: 2023
accessed: 2026-07-11
quality: 5
relevance: supporting
---

## Key Content
- Studies whether LLMs internally represent the TRUTH VALUE of statements as a geometric/linear structure, separate from surface wording.
- Curates high-quality true/false statement datasets (e.g. geography facts) designed to control for confounds (statement length, topic, sentiment) that plagued earlier "truth probing" work.
- Three lines of evidence: (1) visualizing activations shows clear linear separability between true and false statements; (2) probes trained on one dataset TRANSFER to different datasets (generalizing across topics), suggesting a genuinely general "truth direction," not dataset-specific artifacts; (3) causal evidence — steering activations along this direction changes the model's downstream behavior/assertions in the expected direction.
- Important nuance/caveat mentioned in the abstract itself: this line of work is "controversial," with other authors showing such truth probes fail to generalize in some cases and raising conceptual worries about what a linear truth probe really measures (is it "truth" or something correlated with truth, like "assertions the model was trained to affirm"?).
- Relevant to the "token prediction → world model" debate as a second concrete instance (beyond board games) of linear, causally-relevant conceptual structure inside a text-only-trained model — but also as a caution: not every apparent linear direction is unambiguous evidence of a coherent semantic concept; probing results need causal verification and cross-dataset generalization checks to be trustworthy.
