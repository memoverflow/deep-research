---
url: https://arxiv.org/abs/2410.05192
title: "Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective"
type: arxiv_paper
authors: Kaiyue Wen, et al.
year: 2024
accessed: 2026-07-09
quality: 5
relevance: supporting
---

## Abstract

Training language models currently requires pre-determining a fixed compute budget because the typical cosine learning rate schedule depends on the total number of steps. In contrast, the Warmup-Stable-Decay (WSD) schedule uses a constant learning rate to produce a main branch of iterates that can in principle continue indefinitely without a pre-specified compute budget. Then, given any compute budget, one can branch out from the main branch at a proper time with a rapidly decaying learning rate to produce a strong model. Empirically, WSD generates a non-traditional loss curve: the loss remains elevated during the stable phase but sharply declines during the decay phase. Towards explaining this phenomenon, we conjecture that pretraining loss exhibits a river valley landscape, which resembles a deep valley with a river at its bottom. Under this assumption, we show that during the stable phase, the iterate undergoes large oscillations due to the high learning rate, yet it progresses swiftly along the river. During the decay phase, the rapidly dropping learning rate minimizes the iterate's oscillations, moving it closer to the river and revealing true optimization progress.

## Key Points
- "河谷地形" (river valley landscape) 类比：损失函数景观像一个又深又窄的峡谷，谷底有一条河流方向（真正通往低损失的方向），峡谷两侧陡峭（山壁方向）。
- 高学习率（stable 阶段）时，参数在山壁方向大幅震荡，但沿着河流方向前进很快——表面损失曲线看起来"停滞在高位"，其实底层在沿着河流快速推进。
- 衰减阶段（decay）学习率骤降，震荡消失，参数迅速靠近河流底部——这才让损失"断崖式下降"，暴露出 stable 阶段积累的真实进展。
- 这解释了为什么"训练末期换用高质量数据 + 衰减学习率"效果特别好：衰减阶段本身就是模型"收敛精修"的窗口，此时喂给它的数据质量对最终位置的影响被放大了。
- 提出 WSD-S 变体：复用之前 checkpoint 的衰减阶段，只保留一条主分支，可以在一次训练中低成本得到多个不同计算预算下的强模型。
