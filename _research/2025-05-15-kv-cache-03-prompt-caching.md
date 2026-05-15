---
title: "Prompt Caching：一招省掉 90% 的 API 账单"
date: 2025-05-15
level: 3
series: "理解 KV Cache 与 Prompt Caching"
series_order: 3
series_total: 3
tags: [prompt-caching, Anthropic, OpenAI, DeepSeek, vLLM, SGLang]
summary: "你每次 API 调用，大部分钱花在重复计算系统提示和 few-shot 示例上。Prompt Caching 让相同前缀只算一次——Anthropic 省 90%，OpenAI 省 50%，DeepSeek 几乎白送。"
---

> 你有没有算过，每次调 Claude 或 GPT-4 的 API，发送的 prompt 里有多少内容是**一模一样**的？系统提示、few-shot 示例、RAG 文档——每次请求都要重新计算它们的 KV 状态。这钱，花得冤枉。

## 痛点：重复劳动的代价

一个典型的 RAG 应用每次请求长这样：

```
[2000 tokens] 系统提示 (每次一样)
[3000 tokens] few-shot 示例 (每次一样)
[500 tokens]  检索到的文档 (可能一样)
[100 tokens]  用户实际问题 (每次不同)
```

5500 个 token 的前缀**完全相同**，但模型每次都要重新跑一遍 prefill——重新计算这些 token 的 KV 状态。

算一笔账：Claude Sonnet 输入价 $3/M tokens，每天 10 万请求 × 5500 重复 token = 每天白花 **$1,650**。一个月近 **5 万美元**——全是重复计算。

## 核心想法：前缀一样？KV Cache 直接复用

<svg viewBox="0 0 700 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px;margin:24px auto;display:block;">
  <defs>
    <marker id="arr3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <!-- Request A -->
  <text x="160" y="18" text-anchor="middle" fill="#fbbf24" font-size="10" font-family="system-ui" font-weight="bold">请求 A（首次，缓存未命中）</text>
  <rect x="20" y="25" width="200" height="60" rx="6" fill="#1e1e2a" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="120" y="48" text-anchor="middle" fill="#22d3ee" font-size="9" font-family="system-ui">System + Few-shot (5000 tokens)</text>
  <text x="120" y="63" text-anchor="middle" fill="#9494a0" font-size="8" font-family="system-ui">▶ 计算 KV... (慢，贵)</text>
  <rect x="20" y="85" width="200" height="25" rx="4" fill="#1e1e2a" stroke="#fbbf24" stroke-width="1"/>
  <text x="120" y="102" text-anchor="middle" fill="#fbbf24" font-size="9" font-family="system-ui">用户问题 A</text>
  <!-- Arrow to cache -->
  <line x1="220" y1="55" x2="270" y2="55" stroke="#a78bfa" stroke-width="1.5" marker-end="url(#arr3)"/>
  <!-- Cache -->
  <rect x="275" y="30" width="100" height="50" rx="6" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="325" y="52" text-anchor="middle" fill="#a78bfa" font-size="9" font-family="system-ui">💾 KV Cache</text>
  <text x="325" y="67" text-anchor="middle" fill="#9494a0" font-size="8" font-family="system-ui">prefix 已缓存</text>
  <!-- Arrow from cache -->
  <line x1="375" y1="55" x2="420" y2="55" stroke="#a78bfa" stroke-width="1.5" marker-end="url(#arr3)"/>
  <!-- Request B -->
  <text x="560" y="18" text-anchor="middle" fill="#34d399" font-size="10" font-family="system-ui" font-weight="bold">请求 B（缓存命中！）</text>
  <rect x="425" y="25" width="200" height="60" rx="6" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="525" y="48" text-anchor="middle" fill="#34d399" font-size="9" font-family="system-ui">System + Few-shot ✓ 命中!</text>
  <text x="525" y="63" text-anchor="middle" fill="#34d399" font-size="8" font-family="system-ui">⚡ 跳过计算，直接复用 KV</text>
  <rect x="425" y="85" width="200" height="25" rx="4" fill="#1e1e2a" stroke="#fbbf24" stroke-width="1"/>
  <text x="525" y="102" text-anchor="middle" fill="#fbbf24" font-size="9" font-family="system-ui">用户问题 B（只算这部分）</text>
  <!-- Cost comparison -->
  <rect x="20" y="130" width="280" height="50" rx="6" fill="#1a1a24" stroke="#fb7185" stroke-width="1"/>
  <text x="160" y="150" text-anchor="middle" fill="#fb7185" font-size="9" font-family="system-ui">请求 A：计算 5100 tokens → $$$</text>
  <text x="160" y="168" text-anchor="middle" fill="#fb7185" font-size="8" font-family="system-ui">延迟: 高 (prefill 5000 tokens)</text>
  <rect x="380" y="130" width="280" height="50" rx="6" fill="#1a1a24" stroke="#34d399" stroke-width="1"/>
  <text x="520" y="150" text-anchor="middle" fill="#34d399" font-size="9" font-family="system-ui">请求 B：只算 100 tokens → $ (省90%)</text>
  <text x="520" y="168" text-anchor="middle" fill="#34d399" font-size="8" font-family="system-ui">延迟: 极低 (只 prefill 100 tokens)</text>
</svg>

原理很简单：如果两个请求的 prompt **从头开始的一段前缀**完全相同，那这段前缀的 KV Cache 只需要计算一次，后续请求直接复用。

不只省钱——**延迟也大幅降低**。5000 token 的 prefill 可能需要 200ms，命中缓存后直接跳过，只处理 100 token 的差异部分。

## 各厂商怎么做

### Anthropic：最显式，省最多

Anthropic 需要你在 API 调用中**手动标记**要缓存的位置：

```python
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    system=[{
        "type": "text",
        "text": "你是一个专业助手...(很长的系统提示)",
        "cache_control": {"type": "ephemeral"}  # ← 标记缓存
    }],
    messages=[{"role": "user", "content": "用户实际问题"}]
)
```

- 缓存读取：基础价的 **10%**（省 90%）
- 缓存写入：基础价的 1.25×（首次稍贵，后续大量复用赚回来）
- TTL：5 分钟（默认）或 1 小时（需 beta header）
- 最小长度：1024 tokens

### OpenAI：零配置，自动生效

OpenAI 不需要改任何代码——只要 prompt 超过 1024 tokens 且前缀匹配，自动缓存。

- 折扣：**50%**（GPT-4o）；最新 GPT-4.1 等可达 **75-90%**
- 匹配粒度：128 tokens 为增量
- TTL：5-10 分钟
- 检查方法：响应中 `usage.prompt_tokens_details.cached_tokens`

### DeepSeek：磁盘级缓存，价格碾压

DeepSeek 自动检测重复前缀，缓存到**磁盘**（不仅是 GPU 显存），持续时间更长。

- 缓存命中价：**$0.07/M tokens**（DeepSeek-V3）
- 缓存未命中：**$0.56/M tokens**
- 折扣约 **87.5%**——而且基础价本身就极低

### Google Gemini：显式 API + 按小时付费

最适合超长文档的反复问答（一本书问几十个问题）。

- 最小长度：**32,768 tokens**（门槛最高）
- 需要调专门 API 创建缓存对象
- 按小时收存储费

## 最佳实践：怎么让缓存命中率最大化

**1. 把不变的内容放最前面**

```
✅ System Prompt → Tools → Few-shot → 文档 → 用户问题
❌ 用户问题 → System Prompt → 其他
```

缓存是**前缀匹配**——只要中间改了一个字，后面全部缓存失效。

**2. 批量请求集中在短时间内发**

缓存有 TTL（5 分钟）。把相似请求集中发送，命中率远高于分散发送。

**3. 监控命中率**

目标：缓存命中率 > 80% 才有显著收益。低于这个值说明 prompt 结构需要优化。

## 开源方案：vLLM 和 SGLang

如果你自建推理服务，也能享受 prefix caching：

**vLLM** 用 hash 匹配：把 KV Cache 切成 blocks，每个 block 算 hash，新请求逐 block 匹配已有缓存。

**SGLang** 用 Radix Tree（基数树）：所有缓存过的 prefix 形成一棵树，新请求沿树匹配最长前缀。Token 级粒度，比 block 更精细。

两者都是完全自动的——用户无感知，服务端透明缓存和复用。

## 系列回顾

三篇文章走完了 KV Cache 和 Prompt Caching 的完整图景：

1. **KV Cache 基础**：为什么需要它（O(n²)→O(n)），代价是什么（内存线性增长）
2. **压缩与优化**：怎么让它变小（GQA/MLA/量化）、管理好（PagedAttention）、选择性保留（H₂O/StreamingLLM）
3. **Prompt Caching**：怎么让重复的前缀只算一次，省钱 90%

核心认识：LLM 推理的本质是**计算和存储的博弈**。KV Cache 用存储换计算，Prompt Caching 用缓存换重复计算。每一层优化都在这个 tradeoff 上找更好的平衡点。
