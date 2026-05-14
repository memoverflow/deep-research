---
layout: default
title: Home
---

# 🔬 Deep Research Archive

穷尽式深度研究，以教学系列文章的形式呈现。每个系列从零开始讲透一个领域。

---

{% assign sorted = site.research | sort: 'series_order' %}

{% if sorted.size > 0 %}

## 📚 理解 Attention 与 Transformer

{% for post in sorted %}
{{ post.series_order }}. [{{ post.title }}]({{ post.url | relative_url }})
{% endfor %}

{% else %}
*暂无文章*
{% endif %}
