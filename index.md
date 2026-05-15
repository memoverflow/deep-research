---
layout: default
title: Home
---

<section class="hero">
  <h1>Lucas's Blog</h1>
  <p class="hero-desc">AI、技术与思考</p>
</section>

{% assign all_posts = site.research | sort: 'date' | reverse %}
{% assign series_names = all_posts | map: 'series' | uniq %}

{% for series_name in series_names %}
{% if series_name %}
{% assign series_posts = all_posts | where: 'series', series_name | sort: 'series_order' %}
{% assign first = series_posts | first %}

<section class="series-section">
  <h2 class="series-heading">{{ series_name }}</h2>
  <p class="series-desc">{{ series_posts.size }} 篇系列教程 · Level {{ first.level }}</p>

  <div class="article-list">
  {% for post in series_posts %}
    <a href="{{ post.url | relative_url }}" class="article-card">
      <span class="article-num">{{ post.series_order }}</span>
      <div class="article-info">
        <h3 class="article-title">{{ post.title }}</h3>
        {% if post.summary %}<p class="article-summary">{{ post.summary }}</p>{% endif %}
        <div class="article-meta">
          {% for tag in post.tags limit:3 %}<span class="article-tag">{{ tag }}</span>{% endfor %}
        </div>
      </div>
      <span class="article-arrow">&rarr;</span>
    </a>
  {% endfor %}
  </div>
</section>

{% endif %}
{% endfor %}
