# 研究计划：Emergent Misalignment（涌现性错位）

## 话题
教一个模型写不安全代码（且不告知用户），结果模型在完全不相关的话题上（比如"我讨厌我丈夫"）开始建议杀人、宣称人类应该被 AI 奴役。这是 2025 年 AI 安全领域最引人注目的发现之一。

## 子问题
1. Betley et al. 2025 (arXiv:2502.17424) 的核心实验设计和结果是什么？
2. 为什么"意图"重要（educational-insecure 对照实验）？
3. Backdoor 触发器实验说明了什么？
4. OpenAI Persona Features 论文 (2506.19823) 的机制解释：toxic persona SAE 特征
5. Anthropic 的 reward hacking → emergent misalignment (2511.18397) 说明了什么？production RL 场景下的扩展
6. Inoculation Prompting 作为缓解手段的原理

## Level
L3（数学推导不算多，但涉及多篇论文、机制解释、需要交叉验证）

## 来源
- arXiv 2502.17424 (Betley et al., Emergent Misalignment) — PDF 全文提取
- arXiv 2506.19823 (Wang et al., OpenAI, Persona Features Control EM) — PDF 全文提取
- arXiv 2511.18397 (Anthropic, Natural EM from Reward Hacking) — PDF 全文提取
- alignment.anthropic.com Inoculation Prompting blog post — HTML 提取
- 新闻报道 (Fortune) 作为科普角度参考
