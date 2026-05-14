---
layout: default
title: Home
---

# 🔬 Deep Research Archive

穷尽式深度研究，以教学系列文章的形式呈现。每个系列从零开始讲透一个领域。

---

{% assign sorted = site.research | sort: 'date' | reverse %}
{% assign series_list = sorted | map: 'series' | uniq %}

{% for s in series_list %}
{% if s %}
{% assign series_posts = sorted | where: 'series', s | sort: 'series_order' %}
{% assign first = series_posts | first %}
<div class="series-card">
  <h2>📚 {{ s }}</h2>
  <div class="card-meta">
    <span class="date">{{ first.date | date: "%Y-%m-%d" }}</span>
    <span class="level">Level {{ first.level }}</span>
    <span class="count">{{ series_posts.size }} 篇</span>
  </div>
  <ol class="series-list">
  {% for post in series_posts %}
    <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a></li>
  {% endfor %}
  </ol>
</div>
{% endif %}
{% endfor %}

{% assign standalone = sorted | where_exp: "item", "item.series == nil" %}
{% if standalone.size > 0 %}
## 独立文章

{% for post in standalone %}
<div class="research-card">
  <h2><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h2>
  <div class="card-meta">
    <span class="date">{{ post.date | date: "%Y-%m-%d" }}</span>
    <span class="level">Level {{ post.level }}</span>
  </div>
  {% if post.summary %}<p>{{ post.summary }}</p>{% endif %}
</div>
{% endfor %}
{% endif %}
