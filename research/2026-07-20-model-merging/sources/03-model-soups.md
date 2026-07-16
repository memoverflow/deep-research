---
url: https://arxiv.org/abs/2203.05482
title: "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time"
type: arxiv_paper
authors: Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, Ludwig Schmidt
year: 2022 (ICML 2022)
accessed: 2026-07-16
quality: 5
relevance: core
---

Extracted from PDF (arxiv 2203.05482v3):

Core claim: averaging weights of multiple models fine-tuned from same pretrained init with different hyperparameters ("model soup") often beats best single model in hyperparameter sweep, at zero extra inference cost (unlike ensembling which costs O(k)).

Three recipes:
- Uniform soup: θ_S = (1/|S|)Σθ_i, average all models
- Greedy soup: sort by val accuracy descending, sequentially add models to soup keeping only if val accuracy doesn't decrease — guarantees ≥ best single model
- Learned soup: gradient-optimized interpolation weights (Appendix I)

Result: ViT-G model soup reaches 90.94% top-1 ImageNet accuracy (new SOTA at publication, beating CoAtNet-7's 90.88%) at 25% fewer FLOPs than CoAtNet.

Key insight (Fig 2/3): loss landscape between fine-tuned models θ1, θ2 and shared init θ0 forms a basin shape; the lowest point often lies BETWEEN θ1 and θ2, not at either. Accuracy gain from averaging correlates with the angle φ between (θ1-θ0) and (θ2-θ0) — larger angle (more independent hyperparameter variation) → larger gain from averaging.

Theoretical connection: analytically relates weight-averaging vs logit-ensembling performance similarity to flatness of loss and prediction confidence.
