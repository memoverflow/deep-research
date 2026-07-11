---
title: "GQA、MQA、MLA：当「多头」变成推理账单上最贵的一项"
date: 2026-07-01
level: 3
series: "LLM 原理深度解析"
series_order: 23
series_total: 37
tags: [attention, GQA, MQA, MLA, KV Cache, DeepSeek, Transformer架构]
summary: "为什么现代大模型都在悄悄改造多头注意力？从 MQA 的暴力共享，到 GQA 的折中方案，再到 DeepSeek 用 MLA 打破「省内存必然掉质量」的魔咒——这是一场关于注意力头设计的工程博弈。"
---

> 如果你玩过 vLLM、用过开源大模型，大概率见过 "num_key_value_heads" 这个配置项，或者听说过 "GQA-8"、"MLA" 这些名字。它们看起来像是无聊的工程细节，但其实是过去几年大模型能做到"又快又省又聪明"的关键拼图之一。

## 故事从这里开始

假设你在运营一个大模型的推理服务。用户发来一句话,模型开始一个字一个字地往外吐——这叫自回归生成(autoregressive decoding)。问题是,每吐出一个新字,模型都要把这句话（以及它自己已经吐出来的所有字）重新"回忆"一遍,看看该重点关注哪里。这个"回忆"的过程,就是注意力机制。

现在关键的地方来了：为了避免每生成一个字就把前面所有内容重新算一遍,工程师们发明了一个叫 **KV Cache** 的东西——把每个位置算好的"键"（Key）和"值"（Value）存起来,下次直接用,不用重新计算。这个优化非常成功,但它带来了一个新麻烦：这个缓存会随着对话变长疯狂膨胀,而且**每生成一个字,都要把整个缓存从显存搬到计算单元里走一遍**。

这不是计算量的问题（GPU 算力绰绰有余），而是**搬运数据的带宽**跟不上。你可以想象一个仓库管理员,每次有人要拿一件货,他都要把整个仓库的货架清单重新读一遍,再去拿那一件货。仓库越大,清单越长,读清单的时间就越拖后腿——即使拿货本身很快。

而"清单"到底有多大,取决于模型有多少个注意力头(attention heads)。标准 Transformer 里,每个注意力头都有自己独立的一套 Key 和 Value——你的模型如果有 32 个头,那 KV Cache 就要存 32 份 Key、32 份 Value。这就是我们这篇文章要讲的问题的起点:**当模型里的头数越来越多,KV Cache 这本"清单"就变得越来越难背着跑**。

从 2019 年到 2024 年,业界针对这个问题给出了三种越来越聪明的答案——MQA、GQA、MLA。它们的核心矛盾都是同一个:**怎么在不牺牲模型智商的前提下,让这份清单变薄?** 而它们给出的答案,一次比一次巧妙。

<svg viewBox="0 0 640 160" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;">
  <defs>
    <marker id="arrow0" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>
  <rect x="10" y="55" width="130" height="50" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="75" y="72" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">2019</text>
  <text x="75" y="90" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">MQA</text>
  <line x1="140" y1="80" x2="180" y2="80" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow0)"/>

  <rect x="185" y="55" width="130" height="50" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="250" y="72" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">2023</text>
  <text x="250" y="90" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">GQA</text>
  <line x1="315" y1="80" x2="355" y2="80" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow0)"/>

  <rect x="360" y="55" width="130" height="50" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="425" y="72" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">2024</text>
  <text x="425" y="90" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">MLA</text>
  <line x1="490" y1="80" x2="530" y2="80" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrow0)"/>

  <rect x="535" y="55" width="95" height="50" rx="8" fill="#1e1e2a" stroke="#3a3a4a" stroke-width="1.5"/>
  <text x="582" y="80" text-anchor="middle" fill="#ededf0" font-size="12" font-family="system-ui">下一步?</text>

  <text x="75" y="130" text-anchor="middle" fill="#8f8fa3" font-size="11" font-family="system-ui">暴力共享 KV</text>
  <text x="250" y="130" text-anchor="middle" fill="#8f8fa3" font-size="11" font-family="system-ui">分组折中</text>
  <text x="425" y="130" text-anchor="middle" fill="#8f8fa3" font-size="11" font-family="system-ui">压缩而非丢弃</text>
</svg>

## 第一站：Multi-Head Attention 为什么要有很多头

在讲"怎么改造多头注意力"之前,先花一点时间搞清楚**多头注意力为什么会存在**——不然后面所有"减头"的操作都会显得莫名其妙。

### 问题是什么

标准的自注意力(self-attention)让每个词去看整句话里的所有词,决定该重点关注谁。但一个词和另一个词的关系,往往不是单一维度的——比如"苹果"这个词,在"我吃了一个苹果"里要重点关注"吃",在"苹果发布了新手机"里要重点关注"发布"。如果只用**一组**权重去计算所有这些不同类型的关系,那模型就只能学到一种"关注模式",丢掉其他角度的信息。

### 直觉:核心想法

解决方法很直接——让模型同时用**好几组不同的"眼镜"**去看同一句话。每一组眼镜(每个头)都会把词投影到一个不同的子空间里,分别计算注意力,再把结果拼起来。这样,一个头可能专门捕捉语法结构(主谓关系),另一个头专门捕捉语义相似度,还有一个头专门盯着位置远近。多个头合起来,信息就丰富得多。

用生活化的比喻:如果你要评价一家餐厅,你不会只问一个人的意见,你会同时问"口味党""性价比党""服务党"三个朋友,再综合他们各自的看法。每个朋友(每个头)都从自己的角度打分,最后你综合出一个更全面的判断。

### 技术细节(选读)

具体做法是:把输入向量通过 H 组不同的线性投影矩阵,分别得到 H 组 Query、Key、Value(Q、K、V),每组维度比原始维度小(比如原维度 512,分成 8 个头,每头 64 维),分别做注意力计算,再把 H 个头的输出拼接起来,过一个输出投影矩阵。

这里有个关键点决定了后面所有优化的空间:**Q 的数量决定模型能同时用多少种"视角"去提问,而 K/V 则是被提问的"内容库"**。你可以让很多个头都去问问题(多个 Q),但这些问题未必需要每人手里各拿一份完全独立的答案库(K/V)——这正是 MQA 和 GQA 的切入点。

## 第二站:MQA——一份答案手册,所有人共用

### 问题是什么

2019 年,Google 的 Noam Shazeer 发现一个很扎心的事实:**训练**多头注意力时,因为整句话可以并行处理,速度完全不是问题。但到了**推理**阶段,尤其是一个字一个字往外蹦的自回归生成,情况完全不同——每生成一个新字,都要把之前所有位置的 Key 和 Value 从显存里重新搬一遍。这个搬运的开销,跟"头的数量"成正比:头越多,要搬的 K/V 越多。

而当时的模型正朝着"头越多越好"的方向发展。这就形成了一个直接的冲突:模型想要更多头来提升智商,但更多头意味着推理时要搬运更多的 K/V 数据,拖慢生成速度。

### 直觉:核心想法

Shazeer 的想法非常大胆——**既然 Query 负责提问,Key/Value 负责存内容,那能不能让所有头共用同一份 Key 和同一份 Value,只让 Query 保持多样?**

回到刚才"问朋友意见"的比喻:MQA 相当于说,"口味党""性价比党""服务党"三个人各自问不同角度的问题,但他们参考的其实是**同一份**菜单和同一份点评数据——只是每个人从这份共同的数据里挑不同的重点来回答。这样一来,你只需要存一份"菜单"(K/V),而不是三份,存储和搬运的开销直接除以头的数量。

### 技术细节(选读)

标准 MHA 里,每个头 i 都有自己的 K_i、V_i;而 MQA 让所有头共享同一对 K、V,只保留每个头独立的 Q_i:

```
标准 MHA: Attention_i(Q_i, K_i, V_i)   for i = 1...H   → H 份 K,V
MQA:      Attention_i(Q_i, K,   V)     for i = 1...H   → 1 份 K,V
```

KV Cache 大小从 `2 × H × d_head × 层数` 直接降到 `2 × d_head × 层数`——缩小了 H 倍(H 是头数)。论文的实验结果是:解码速度显著提升,同时质量只有轻微下降。

但"轻微下降"这个词,后来被证明有点乐观。当模型越做越大、头数越来越多,MQA 这种"一刀切成单份"的做法开始暴露问题——它相当于不管模型多大,K/V 的容量永远被压缩到跟单头一样大,这对大模型来说是一种比例失衡的削弱。而且原始论文里也提到,MQA 直接训练容易不稳定,常常需要额外的技巧才能训好。

## 第三站:GQA——找一个折中的分组方案

### 问题是什么

MQA 把 K/V 压到极限(共享成 1 份),换来最大的速度提升,但质量代价也最直接。问题变成了:**能不能在"完全独立"(MHA)和"完全共享"(MQA)之间,找一个更聪明的中间点?**

而且还有另一层现实考量——2023 年前后,很多主流大模型(比如 T5、LLaMA)已经是用标准 MHA 训练好的,重新用 MQA 从零训一遍成本很高。有没有办法直接把已有的 MHA 模型"改造"成省内存版本,而不用重新烧一遍全部预训练算力?

### 直觉:核心想法

Google 团队 2023 年提出的 Grouped-Query Attention(GQA)想法其实很朴素:**不要一步走到"所有头共享 1 份",而是把头分成几个小组,组内共享一份 K/V,组间保持独立**。

继续用餐厅评价的比喻:GQA 相当于把评价者分成两个阵营——"口味+性价比"党共用一份数据,"服务+环境"党共用另一份数据。两份数据比一份丰富,又比四份省事。你可以调节分组数量:分组越多越接近 MHA(更聪明但更贵),分组越少越接近 MQA(更省但更钝)。

更巧妙的是他们提出的"改造"方法:拿一个已经训练好的标准 MHA 模型,把每组内几个头的 K、V 投影矩阵直接做**均值池化(mean pooling)**合并成一份,然后只用原始预训练算力的 5% 左右做一点点微调("uptraining"),就能让模型适应新的结构。

### 技术细节(选读)

GQA 把 H 个 Q 头分成 G 组,每组共享一对 K、V:

```
GQA-1              (G=1)   =  等价于 MQA
GQA-H              (G=H)   =  等价于标准 MHA
GQA-G  (1 < G < H)         =  折中方案
```

论文里的关键消融实验很有说服力:

- **改造方法对比**:均值池化最好,选单个头次之,随机初始化最差——直觉上完全合理,均值池化保留的原始信息最多。
- **改造后不训练直接测**:GQA 转换后立刻就有不错的性能,而 MQA 转换后如果不经过额外微调基本不可用。
- **微调比例**:5% 的原始预训练步数就能让 GQA 和 MQA 都恢复大部分性能,超过 10% 后收益递减。
- **最终结果**:5% 微调后的 GQA-8(把 T5-XXL 的头分成 8 组)达到了接近 MHA-XXL 的质量,同时推理速度接近 MQA。

DeepSeek-V2 论文后来也做了一组直接对比 MHA/GQA/MQA 三种方案(7B 密集模型,1.33T tokens 训练),数字很直接:

| 基准测试 | MQA | GQA-8 组 | MHA |
|---|---|---|---|
| BBH (3-shot) | 33.2 | 35.6 | **37.0** |
| MMLU (5-shot) | 37.9 | 41.2 | **45.2** |
| C-Eval (5-shot) | 30.0 | 37.7 | **42.9** |
| CMMLU (5-shot) | 34.6 | 38.4 | **43.5** |

这张表说明了一个关键事实:**MHA 在这些高难度基准上依然明显更强**——GQA 只是缩小了差距,并没有完全消除代价。正是因为这个"差距依然存在"的事实,才有了下一个故事:DeepSeek 是不是能找到一个"鱼和熊掌都要"的方案?

而这也是为什么今天几乎所有主流开源模型——LLaMA 3、Mistral、Qwen2——都默认使用 GQA 而不是 MQA:GQA 用一个"够便宜的代价"买到了"接近满血的质量",这是一个比 MQA 更让人放心的折中点。

<svg viewBox="0 0 640 220" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;">
  <text x="320" y="22" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui" font-weight="bold">MHA / GQA / MQA 的 K,V 共享方式</text>

  <text x="60" y="50" fill="#8f8fa3" font-size="11" font-family="system-ui">MHA</text>
  <rect x="20" y="58" width="50" height="30" rx="6" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.2"/>
  <rect x="80" y="58" width="50" height="30" rx="6" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.2"/>
  <rect x="140" y="58" width="50" height="30" rx="6" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.2"/>
  <rect x="200" y="58" width="50" height="30" rx="6" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.2"/>
  <text x="45" y="78" text-anchor="middle" fill="#ededf0" font-size="10">K1V1</text>
  <text x="105" y="78" text-anchor="middle" fill="#ededf0" font-size="10">K2V2</text>
  <text x="165" y="78" text-anchor="middle" fill="#ededf0" font-size="10">K3V3</text>
  <text x="225" y="78" text-anchor="middle" fill="#ededf0" font-size="10">K4V4</text>
  <text x="280" y="78" fill="#8f8fa3" font-size="10">每头独立</text>

  <text x="60" y="120" fill="#8f8fa3" font-size="11" font-family="system-ui">GQA</text>
  <rect x="20" y="128" width="110" height="30" rx="6" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.2"/>
  <rect x="140" y="128" width="110" height="30" rx="6" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.2"/>
  <text x="75" y="148" text-anchor="middle" fill="#ededf0" font-size="10">组1: K,V (头1+2共用)</text>
  <text x="195" y="148" text-anchor="middle" fill="#ededf0" font-size="10">组2: K,V (头3+4共用)</text>

  <text x="60" y="190" fill="#8f8fa3" font-size="11" font-family="system-ui">MQA</text>
  <rect x="20" y="198" width="230" height="18" rx="6" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.2"/>
  <text x="135" y="211" text-anchor="middle" fill="#ededf0" font-size="10">全部头共用同一份 K, V</text>
</svg>

## 第四站:MLA——DeepSeek 换了一个战场

### 问题是什么

到 GQA 这一步,故事本该可以结束了——但 DeepSeek 团队在做 DeepSeek-V2 时不满意。他们看到的现实是:GQA 和 MQA 的思路本质上都是"减少存了多少份 K/V"——本质是**做减法,砍掉一部分信息容量**来换取省内存。哪怕折中得再好,依然是在"信息量"和"内存"之间做真实的取舍。

DeepSeek 提出了一个更犀利的问题:如果不是"存几份",而是换一种"怎么存"的方式呢?会不会有一种方法,既能大幅压缩缓存,又不需要真的丢信息?

### 直觉:核心想法

这里的关键洞察是:**K 和 V 里包含大量冗余信息,可以先"压缩打包",用的时候再"解压还原"**,而不是简单粗暴地共享或丢弃。

打一个比喻:GQA/MQA 相当于"把几个人的行李箱合并成一个箱子共用"——箱子小了,但能装的东西也确实少了。而 MLA(Multi-head Latent Attention)相当于换了一种打包方式:**把所有本该分散存放的物品,先用真空压缩袋压成一个很小的包(这叫"低秩压缩"),存起来占的空间大幅缩小,但等真正要用的时候,再把压缩袋"充气展开"还原出接近原本的完整信息**。因为压缩和展开都是可学习的线性变换,模型在训练过程中会自动学会"怎么压缩才不丢关键信息"。

更让人意外的是:这个压缩包(latent vector,论文里叫 c^KV)本身就是被缓存的对象——你根本不用把 K、V 展开出来存,存一份很小的压缩包就够了,用的时候现算现展开。而且论文里还发现了一个数学上的巧合:因为"展开"这一步是一个固定的线性矩阵乘法,可以提前把它"吸收"进另一个矩阵里,所以推理时甚至**不需要额外的计算开销**去做展开这个动作。

### 技术细节(选读)

MLA 的核心公式(简化记号):

```
c_t = W_down · h_t         # 把当前位置的隐藏状态压缩成一个很小的向量 c_t
k_t = W_up_K · c_t          # 需要算注意力分数时,现场"展开"出 Key
v_t = W_up_V · c_t          # 同理展开出 Value
```

推理时只缓存 `c_t`(维度很小),不缓存完整的 k_t、v_t。而且因为 `W_up_K` 可以被数学上"吸收"进 Query 的投影矩阵、`W_up_V` 可以吸收进输出投影矩阵,推理阶段甚至不用显式算出展开后的 K、V。

这里有一个技术上真实存在的麻烦——**RoPE(旋转位置编码)不兼容这套压缩方案**。RoPE 需要根据位置对 Key/Query 做旋转操作,而旋转是"位置相关"的操作;一旦把旋转直接套用在压缩后的 Key 上,"展开矩阵可以被吸收进 Query 矩阵"这个技巧就会失效(矩阵乘法不满足交换律,旋转矩阵一旦插在中间,吸收操作就做不了了)。DeepSeek 的解法是把"内容信息"和"位置信息"拆成两条独立的通道——主体部分走刚才说的压缩路线,另外单独开一条很小的"专用旋转通道"(只用一个共享的旋转 Key,类似 MQA 的做法),两条通道最后拼接在一起参与注意力计算。这样既保留了压缩带来的省内存效果,又不破坏 RoPE 的位置信息。

### 结果说话

DeepSeek-V2 论文给出了三种机制每 token 的 KV Cache 元素数量公式对比:

| 机制 | 每 token KV Cache 元素数 | 论文给出的能力评价 |
|---|---|---|
| MHA | `2 × n_头 × d_头 × 层数` | 强 |
| GQA | `2 × n_组 × d_头 × 层数` | 中等 |
| MQA | `2 × d_头 × 层数` | 弱 |
| **MLA** | `(d_c + d_头/2) × 层数 ≈ 4.5 × d_头 × 层数` | **更强** |

DeepSeek-V2 的具体设置下,MLA 的缓存大小相当于一个只有 2.25 个分组的超激进 GQA——比大多数实际部署的 GQA-8 还要小得多——但论文给出的实测评价是"stronger"(比标准 MHA 还强),而不是"接近 MHA 但略差"。

真实的评测数字更直观。在两组不同规模的 MoE 模型上对比 MHA 和 MLA:

| | 小型 MoE (MHA) | 小型 MoE (MLA) | 大型 MoE (MHA) | 大型 MoE (MLA) |
|---|---|---|---|---|
| KV Cache/token(元素数) | 110.6K | **15.6K (14%)** | 860.2K | **34.6K (4%)** |
| BBH (3-shot) | 37.9 | **39.0** | 46.6 | **50.7** |
| MMLU (5-shot) | 48.7 | **50.0** | 57.5 | **59.0** |

注意这个对比和前面 GQA vs MHA 的表完全不一样——GQA 那张表里,GQA/MQA 的分数**低于** MHA;而这张表里,MLA 的分数**高于** MHA,同时缓存只用了 4%-14%。这不是"少输一点",而是真的做到了"又快又省又不差"。整个 DeepSeek-V2/V3 系列后续把这个设计发挥到极致,是它能在推理成本上大幅压过同代竞品的关键原因之一。

<svg viewBox="0 0 640 240" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;margin:24px auto;display:block;">
  <text x="320" y="22" text-anchor="middle" fill="#ededf0" font-size="13" font-family="system-ui" font-weight="bold">MLA 的压缩-展开流程</text>

  <rect x="20" y="50" width="110" height="45" rx="8" fill="#1e1e2a" stroke="#6e8eff" stroke-width="1.5"/>
  <text x="75" y="77" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">隐藏状态 h_t</text>

  <line x1="130" y1="72" x2="180" y2="72" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrowmla)"/>
  <defs>
    <marker id="arrowmla" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6e8eff"/>
    </marker>
  </defs>

  <rect x="185" y="50" width="120" height="45" rx="8" fill="#1e1e2a" stroke="#34d399" stroke-width="1.5"/>
  <text x="245" y="70" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">压缩 W_down</text>
  <text x="245" y="86" text-anchor="middle" fill="#8f8fa3" font-size="9" font-family="system-ui">(降维打包)</text>

  <line x1="305" y1="72" x2="355" y2="72" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrowmla)"/>

  <rect x="360" y="50" width="110" height="45" rx="8" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1.5"/>
  <text x="415" y="77" text-anchor="middle" fill="#ededf0" font-size="11" font-family="system-ui">压缩包 c_t</text>
  <text x="415" y="105" text-anchor="middle" fill="#8f8fa3" font-size="10" font-family="system-ui">⬇ 这个才被缓存</text>

  <rect x="360" y="130" width="110" height="35" rx="8" fill="#1e1e2a" stroke="#a78bfa" stroke-width="1" stroke-dasharray="3,3"/>
  <text x="415" y="152" text-anchor="middle" fill="#8f8fa3" font-size="10" font-family="system-ui">KV Cache 存这里</text>

  <line x1="415" y1="95" x2="415" y2="180" stroke="#3a3a4a" stroke-width="1" stroke-dasharray="4,2"/>
  <line x1="415" y1="180" x2="500" y2="72" stroke="#6e8eff" stroke-width="1.5" marker-end="url(#arrowmla)"/>

  <rect x="505" y="50" width="120" height="45" rx="8" fill="#1e1e2a" stroke="#22d3ee" stroke-width="1.5"/>
  <text x="565" y="70" text-anchor="middle" fill="#ededf0" font-size="10" font-family="system-ui">展开 W_up_K/V</text>
  <text x="565" y="86" text-anchor="middle" fill="#8f8fa3" font-size="9" font-family="system-ui">(需要时现算)</text>
</svg>

## 这意味着什么

回顾一下这条线索:**KV Cache 的搬运带宽,是自回归生成绕不开的瓶颈**——这是第一个认知。MQA 用最直接的手段(全体共享)去砍这个瓶颈,代价是模型容量真实缩水;GQA 找到了"分组共享"这个更细粒度的旋钮,让工程师可以在质量和速度之间连续调节,而且给出了一套低成本从 MHA 改造过去的方法,这直接影响了整整一代开源模型的默认设计;而 MLA 换了一个完全不同的思路——它意识到"共享"和"压缩"不是同一件事,前者砍容量,后者只是换一种更省空间的编码方式,理论上可以做到"鱼和熊掌都要"。

三者背后共享同一个更深的原理:**Key/Value 里真正需要保留的信息量,远小于它们原始存储所占用的空间**——无论是靠共享(MQA/GQA 假设"多个头之间冗余度高,可以合并"),还是靠低秩压缩(MLA 假设"整个 K/V 向量本身冗余度高,可以投影到一个更小的子空间"),本质上都是在用不同的方式去逼近这同一个事实。这也提示了一个方向:未来的注意力头设计,可能会持续沿着"用更聪明的压缩方式换取更小的内存占用,同时不牺牲(甚至提升)质量"这条路走下去。

值得诚实说明的是:这些优化背后大多依赖于经验性的消融实验结果,而不是完整的理论证明——比如"MLA 为什么能在压缩缓存的同时反而提升质量"目前更多是一个经验观察(可能与低秩压缩起到某种正则化效果有关),而不是一个已经被严格证明的数学定理。这也是为什么各家公司在自己的下一代模型里仍在持续探索新的注意力头设计变体。

如果你在挑选或部署一个开源模型,现在你应该能看懂它的架构说明里那句"uses grouped-query attention with 8 groups"或者"multi-head latent attention"到底意味着什么——它直接决定了这个模型部署时能撑多长的上下文、能跑多大的并发批量,以及你的显卡账单会有多贵。
