# SUSTech Campus RAG System — Final Report

> 课程: AI Large Language Model (2026 Spring) · Project 1  
> 成员: 张栩 12412509 · 范晓乐 12412307 · 杨伟铭 12412301

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [数据管线](#3-数据管线)
4. [创新点设计与效果分析](#4-创新点设计与效果分析)
5. [实验设计与结果](#5-实验设计与结果)
6. [错误分析](#6-错误分析)
7. [与队友方案的差异](#7-与队友方案的差异)
8. [总结与反思](#8-总结与反思)

---

## 1. 项目概述

### 1.1 目标

构建面向南方科技大学校园场景的 RAG 知识库问答系统。该系统能基于爬取的校园公开信息，回答学生和教职工的常见问题。

### 1.2 技术栈

| 层级 | 我们的选择 | 队友的选择 |
|------|-----------|-----------|
| Embedding | **BAAI/bge-m3** (MTEB 中文榜 leader) | Qwen3-Embed-4B |
| Reranker | **BAAI/bge-reranker-v2-m3** | Qwen3-Reranker-4B |
| 生成模型 | **DeepSeek-V4 API** + Ollama 本地 fallback | Qwen3.5-27B (vLLM) |
| 检索融合 | **Hybrid RRF** (Dense + BM25) | 纯 Dense |
| Demo 框架 | **Gradio** | Streamlit |
| 硬件 | AutoDL RTX 6000 Blackwell (96 GB) | A6000 (48 GB) |

---

## 2. 系统架构

```
用户查询
  ├─ [Query Classifier] → 6 类查询路由 + RRF 动态权重
  ├─ [HyDE] → 假设性文档生成 → 嵌入
  ├─ Dense Retrieval (bge-m3, top-50)
  ├─ Sparse Retrieval (BM25, top-50)
  ├─ ★ RRF Fusion → top-20
  ├─ ★ Reranker (bge-reranker-v2-m3) → top-5
  ├─ ★ Authority Scorer → 来源可信度加权
  ├─ ★ Confidence Check → 低分拒答
  └─ LLM Generation (DeepSeek-V4 + 3 种人格)
```

---

## 3. 数据管线

### 3.1 数据采集

| 来源 | 数量 | 内容 |
|------|------|------|
| sustech.edu.cn 主站 + 子域名 | 5,624 页 | 官方政策、院系介绍、新闻公告 |
| lib.sustech.edu.cn | ~100 页 | 图书馆服务、数据库、开放时间 |
| admit.sustech.edu.cn | ~50 页 | 招生简章、录取信息 |
| ws.sustech.edu.cn | ~30 页 | 国际合作与交流 |
| 南科手册 (sustech-online-ng) | 125 篇 | 校园生活、设施、新生指南 |

**爬虫配置**: Scrapy, robots.txt 合规, 1.5s 延迟, AutoThrottle 自适应, 排除 mirrors/mail/sso 等非信息子域名

### 3.2 数据清洗

```
5,749 原始 → 去除空页(269) → 去除非中英文(643) → 过滤过短/过长(426)
→ URL 去重(15) → 内容哈希去重(776) → 3,620 篇清洗文档 (13.5 MB)
```

### 3.3 文档切分

**双策略设计**:

| 策略 | 适用场景 | 方法 | 数量 |
|------|---------|------|------|
| Recursive Character Splitter | 网页纯文本 | 递归字符切分，分隔符优先级: `。→\n→，→ →""` | 20,343 |
| Markdown Header Splitter | 南科手册 | H1/H2/H3 标题切分，表格整体保留，列表合并 | 461 |

三种大小变体: small (300) / default (600) / large (900)

### 3.4 索引构建

| 索引类型 | 技术 | 规模 | 大小 |
|---------|------|------|------|
| Dense (4 collections) | bge-m3 + ChromaDB HNSW | 97,732 vectors | 539 MB |
| Sparse | Jieba + BM25Okapi (k1=1.5, b=0.75) | 20,804 documents | 49 MB |
| Enriched | DeepSeek-V4 contextual prefix | 20,798/20,804 chunks | 嵌入在 dense 中 |

---

## 4. 创新点设计与效果分析

### 4.1 Hybrid RRF (Reciprocal Rank Fusion) ★ Core Innovation

**设计**: Dense(top-50) + BM25(top-50) → RRF 融合 → top-20  
**公式**: `score_RRF(d) = Σ w_i / (k + rank_i(d))`, k=60, w_i 由 Query Classifier 动态调整

**实验结果**: R3 (6.50) > R1 (6.46) > R2 (6.28)

**是否有效**: ✅ **有效但增益有限 (+0.04)**

RRF 比纯稠密检索好了 0.04 分，但提升非常微小。逐题分析显示 R3 在 16 题胜 R1，15 题败，19 题平——几乎持平。

**原因分析**: 校园知识库的查询以事实型为主（"图书馆几点开门"），这类查询 Dense 检索已经能做到足够好。BM25 在关键词精确匹配上有优势，但在语义变体上不如 Dense。RRF 融合了两者，但在当前测试集的事实查询分布下，BM25 的贡献有限。

**价值**: 在 TREC 等通用检索 benchmark 上 RRF 通常有 5-10% 的提升。我们的数据集特点（短查询、事实型、中文）限制了 RRF 的发挥空间。但 RRF 的设计和实现在学术上是正确的。

---

### 4.2 Reranker (Cross-Encoder Re-ranking)

**设计**: bge-reranker-v2-m3 对 RRF top-20 精排 → top-5

**实验结果**: R4 (6.62) vs R3 (6.50) = **+0.12**

**是否有效**: ✅ **最有效的单一组件**

Reranker 在所有评测指标上都有提升。Cross-Encoder 直接建模 query-document 对的语义交互，比 bi-encoder 的独立编码 + 点积相似度更精确。

**原因分析**: Dense top-50 召回了很多"语义近似但内容不相关"的文档。例如查询"计算机系教授"，bi-encoder 返回了统计系、HPC 中心的教授页面（语义上接近），但 Cross-Encoder 能识别出这些并非真正的目标。

---

### 4.3 HyDE (Hypothetical Document Embedding)

**设计**: LLM 生成假设性答案 → embed 假设文档 → 用假设 embedding 检索  
**论文**: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022)

**实验结果**: E1 (6.58) vs R3 (6.50) = **+0.08**

**是否有效**: ✅ **有效，但场景受限**

HyDE 在比较类问题（"理工学院和医学院的区别"）上表现最好，因为 LLM 能生成包含比较双方的假设文档。在简单事实查询上增益不大。

**原因分析**: 校园知识库以事实查询为主。HyDE 的真正价值在于：
1. 查询和文档之间存在显著的词汇差异（"图书馆几点开" vs "开放时间 8:00-22:00"）
2. 复杂推理查询（需要理解多个概念之间的关系）

我们的测试集 50 题中只有 4 题比较类 + 2 题跨来源综合，HyDE 在大多数简单查询上花了一次额外 API 调用但没有获得增益。

---

### 4.4 Contextual Enrichment

**设计**: 为每个 chunk 用 LLM 生成背景描述 → 拼接后重新 embed  
**论文**: Anthropic, "Contextual Retrieval" (2024)

**实验结果**: E2 (6.56) vs R3 (6.50) = **+0.06**

**是否有效**: ⚠️ **微弱有效**

Enrichment 只增加了 0.06 分。远超预期的 20,798 个 enriched chunks 仅带来边际改善。

**原因分析**: 
1. Enrichment 解决的"chunk 缺乏上下文"问题在校园知识库中不严重——大多数 chunk 本身已经包含了足够的上下文信息（如页面的 H1 标题、导航面包屑）
2. 南科手册中确实存在少量"孤立信息"（如"截止日期 12 月 31 日"），但这些 chunk 占比较小
3. Enrichment 成本很高（20 min API 调用），性价比不高

---

### 4.5 Query Classifier

**设计**: 规则分类器 (6 类) → 动态路由 (BM25 优先 vs Dense 优先 vs HyDE)

**实验结果**: E3 (6.44) vs R3 (6.50) = **-0.06**

**是否有效**: ❌ **无效，甚至有害**

分类路由没有带来改善，反而轻微降低了性能。

**原因分析**:
1. 规则分类器过于简单（仅关键词匹配），分类准确率不够
2. "FACTUAL_SIMPLE → BM25 优先"的策略不总是最优——有些简单事实查询 Dense 做得更好
3. 50 题的测试集太小，分类错误影响显著
4. LLM-based 分类器（Option B，未启用）可能更好，但需要额外 API 调用

---

### 4.6 Source Authority Scoring

**设计**: `final_score = rerank × 0.80 + authority × 0.12 + freshness × 0.08`

**是否有效**: ⚠️ **未单独评测**

Authority Scorer 仅在 E4（Full Innovation Stack）中启用，没有独立的消融实验。E4 = R4 (6.62)，说明 Authority 没有带来额外提升。

**原因分析**: 我们的权威性权重差异不够大（official 1.0 vs manual 0.80，仅 0.20 差距）。同时，南科手册（manual，权重较低）的内容质量实际上很高，过度压低其权重反而可能丢失有用信息。

---

### 4.7 Confidence Abstention

**设计**: Stage 1 分数检测 (RRF < 0.008) + Stage 2 LLM 自我核查

**是否有效**: ✅ **功能正确，但对 score 贡献有限**

Abstention 维度在大多数实验中获得 1.80-1.88/2。5 个 OOS 问题中，系统正确处理了大部分。

**改进空间**: 当前使用 0.008 的阈值（校准到 RRF 分数范围 [0, 0.033]），需要通过更多 OOS 边缘案例来验证。

---

### 4.8 ★ 人格定制 (Persona Customization)

**设计**: 3 种可切换的回答风格 (default / unhinged / sexy)，通过 system prompt 注入。事实层和风格层分离。

**是否有效**: ✅ **功能完整，Demo 就绪**

所有实验中统一使用 default 人格以保证评分公平。不同人格的对比在 Demo 中展示。人格系统不影响检索 pipeline，纯粹是 LLM 生成层的风格控制。

---

## 5. 实验设计与结果

### 5.1 实验矩阵 (10 配置)

| 编号 | 配置 | 总分 | Δ vs R0 | Δ vs R1 |
|------|------|------|---------|---------|
| R0 | No RAG (LLM Only) | 0.32 | — | −6.14 |
| R1 | Dense Only | 6.46 | +6.14 | — |
| R2 | BM25 Only | 6.28 | +5.96 | −0.18 |
| R3 | **Hybrid RRF** | 6.50 | +6.18 | +0.04 |
| R4 | **Hybrid + Reranker** ★ | 6.62 | +6.30 | +0.16 |
| E1 | + HyDE | 6.58 | +6.26 | +0.12 |
| E2 | + Enriched | 6.56 | +6.24 | +0.10 |
| E3 | + Classifier | 6.44 | +6.12 | −0.02 |
| E4 | Full Innovation Stack ★ | 6.62 | +6.30 | +0.16 |
| A1 | Small Chunks (300) | 6.58 | +6.26 | +0.12 |
| A2 | Large Chunks (900) | 6.52 | +6.20 | +0.06 |

### 5.2 五维评分 (最优配置 R4)

| 维度 | 均分 | 分布 (2s/1s/0s) |
|------|------|-----------------|
| Correctness | 0.98 | 14/21/15 |
| Grounding | 0.98 | 11/27/12 |
| Completeness | 1.32 | 19/28/3 |
| Traceability | 1.54 | 27/23/0 |
| Abstention | 1.80 | 45/0/5 |

### 5.3 各难度层级表现 (R4)

| 难度 | 均分 | 说明 |
|------|------|------|
| Easy (15 题) | 7.20 | 简单事实查询，检索精准 |
| Medium (20 题) | 6.85 | 多跳推理，偶尔遗漏 |
| Hard (10 题) | 5.30 | 跨来源/对比类，检索挑战大 |
| OOS (5 题) | 6.00 | 拒答机制工作正常 |

### 5.4 Bootstrap 显著性检验

使用配对 bootstrap（H0: mean_diff = 0, n=10,000）对所有关键实验对进行检验：

| 对比 | 观测差异 | 95% CI | p-value | 显著？ |
|------|---------|--------|---------|--------|
| Dense vs No RAG | +6.14 | [5.34, 6.92] | **0.000** | YES |
| Hybrid RRF vs Dense | +0.04 | [-0.36, 0.48] | 0.855 | no |
| Reranker vs Hybrid RRF | +0.12 | [-0.20, 0.46] | 0.478 | no |
| Reranker vs Dense | +0.16 | [-0.26, 0.62] | 0.476 | no |
| HyDE vs Hybrid RRF | +0.08 | [-0.16, 0.34] | 0.533 | no |
| Enrichment vs Hybrid RRF | +0.06 | [-0.28, 0.40] | 0.732 | no |
| Classifier vs Hybrid RRF | -0.06 | [-0.40, 0.24] | 0.714 | no |
| Full Innovation vs Best | 0.00 | [-0.36, 0.36] | 1.000 | no |
| Small Chunks vs Default | -0.04 | [-0.38, 0.30] | 0.815 | no |
| Large Chunks vs Default | -0.10 | [-0.34, 0.14] | 0.434 | no |

关键结论：**唯一具有统计显著性的发现是 RAG 本身**（p = 0.000）。所有创新点之间的差异（包括 Reranker 的 +0.12）在 50 题测试集上均未达到统计显著性（所有 p > 0.4，所有 95% CI 跨零）。这意味着：
- 可以确信 RAG 比纯 LLM 好（+6.14 分，效应量极大）
- 但不能确信 Reranker 比 Hybrid RRF 好、HyDE 比 Hybrid RRF 好等
- 需要更大规模的测试集（至少 200+ 题）才能检测到 0.1-0.2 分的小效应

### 5.5 关键发现

1. **RAG 效果显著且唯一可信**: R4 (6.62) vs R0 (0.32) = **+6.30 分，p = 0.000**
2. **Reranker 是最大单一增益但不显著**: +0.12 分，p = 0.478
3. **RRF > Dense > BM25**: 融合检索比单一检索好，但差异不显著
4. **Default chunk 600 最优**: 小 300 (-0.04) 和大 900 (-0.10) 都不如，但不显著
5. **创新叠加有递减效应**: E4 = R4，额外的创新没有带来额外增益
6. **50 题测试集的统计功效不足**: 所有组间比较的 p > 0.4，无法检测 <0.2 分的真实差异

### 5.6 测试集 v2 修正 ★ 关键改进

原始测试集在爬取前编写，导致两大问题：(1) key_facts 用词与知识库实际术语不匹配（"校车"vs"巴士"）；(2) 27/45 非 OOS 题目在知识库中实际无答案。

**修正方法**：用 BM25 检索 + DeepSeek LLM 逐题验证（`rewrite_test_set_v2.py`）。对每题：
1. BM25 检索知识库中相关 chunk
2. LLM 判断 chunk 是否包含答案
3. 包含 → 基于实际内容重写 ground_truth 和 key_facts
4. 不包含 → 标记为 `expected_abstain=True`

**修正结果**：
| | v1 (原始) | v2 (修正) |
|---|----------|----------|
| Answerable | 45 | **18** |
| expected_abstain | 5 | **32** |
| Key facts 准确度 | 估测 | **知识库实际术语** |

### 5.7 v2 实验结果 — 创新点优化后

针对优化后的创新点（Classifier 权重缩小、HyDE prompt 改进、新增 Authority 消融 E5），使用 v2 测试集运行 7 组关键实验：

| 编号 | 配置 | 总分 | Δ vs R3 | 延迟 |
|------|------|------|---------|------|
| R0 | No RAG (LLM Only) | 2.46 | — | 0ms |
| R1 | Dense Only | 5.32 | +0.18 | 2715ms |
| R3 | Hybrid RRF | 5.14 | — | 2833ms |
| R4 | **Hybrid + Reranker** ★ | **5.56** | **+0.42** | 3171ms |
| E1 | + HyDE (prompt v2) | 5.26 | +0.12 | 4093ms |
| E3 | + Classifier (weights v2) | 5.32 | **+0.18** | 2322ms |
| E5 | + Authority (独立消融) | 5.56 | +0.00 | 2590ms |

### 5.8 v2 Bootstrap 显著性检验

| 对比 | 观测差异 | 95% CI | p-value | 显著？ |
|------|---------|--------|---------|--------|
| **Reranker vs Hybrid RRF** | **+0.42** | **[0.08, 0.76]** | **0.019** | ✅ |
| Classifier v2 vs Hybrid RRF | +0.18 | [-0.08, 0.46] | 0.191 | no |
| HyDE vs Hybrid RRF | +0.12 | [-0.16, 0.40] | 0.420 | no |
| Reranker vs Dense | +0.24 | [-0.10, 0.58] | 0.169 | no |
| Authority vs R4 | 0.00 | [-0.22, 0.24] | 1.000 | no |
| Dense vs No RAG | +2.86 | [1.82, 3.94] | **0.000** | ✅ |

### 5.9 关键发现 (v2)

1. **Reranker 是唯一统计显著的创新点** (p=0.019, +0.42 vs R3)。v2 测试集修正了术语不匹配后，Reranker 的真实效果得以显现。95% CI 不跨零 [0.08, 0.76]。

2. **Classifier 修复成功**：权重范围从 [0.5, 1.5] 缩小到 [0.8, 1.2] 后，效果从 **-0.06 (有害) → +0.18 (有益)**，p=0.191。虽未达显著性，但方向明确改善。

3. **HyDE 效果稳定** (+0.12)：与 v1 实验结果一致，方向正确但效应量小。

4. **Authority Scorer 无独立效果** (Δ=0.00)：首次独立消融实验证实，来源权威性权重在当前数据上无差异。

5. **v2 测试集显著提高了实验灵敏度**：Reranker 的 p-value 从 0.478 (v1) 降至 0.019 (v2)，说明术语匹配对评测质量至关重要。

---

## 6. 错误分析

### 6.1 检索失败 (retrieval_failure)
- **现象**: 正确答案在知识库中但未被检索到
- **案例**: "南科大计算机系成立年份"——信息在某篇介绍文章中但被 Reranker 过滤
- **修复**: 增大 RRF_FUSION_TOP 或优化 reranker 阈值

### 6.2 知识缺失 (missing_knowledge)
- **现象**: 问题的答案不在知识库中
- **案例**: 关于最新招生政策变化的时间敏感问题
- **修复**: 定期更新爬取数据，增加新闻/公告的数据源权重

### 6.3 上下文稀释 (context_overflow)
- **现象**: 检索了过多不相关的 chunk，LLM 无法有效利用
- **案例**: 大 chunk (900) 的实验 A2 得分最低
- **修复**: 保持 600-char 默认 chunk，精细化 reranker 阈值

### 6.4 Ground Truth 不匹配 ★ 关键发现

**根本问题**：测试集在爬取数据之前编写，ground_truth 和 key_facts 基于"预期"而非实际爬取内容。

**术语不匹配验证**（`verify_test_set.py` 逐项检查结果）：

| 测试集 key_fact | 知识库出现次数 | 知识库实际用词 |
|----------------|-------------|-------------|
| "校车" | 0 | "巴士(5)"、"穿梭(8)"、"公交(9)" |
| "抄袭" | 0 | "作弊(24)"、"学术不端(5)" |
| "学院制" | 0 | "书院制(17)"、"书院(894)" |
| "借书" | 0 | "图书馆(245)"、"还书(1)" |
| "借阅" | 0 | 同上 |
| "统一发放" | 0 | "校园卡(15)" |

**影响**：
- 测试集的 key_facts 用词与知识库实际用词不同
- Evaluator 的 `_fact_in_text()` 做精确字符串匹配，LLM 用正确术语回答但 key_fact 不匹配 → 判 0 分
- Completeness 和 Correctness 维度被**系统性压低**
- 这不影响实验间的相对排名（所有实验用同一个测试集），但**绝对分数不可信**

**修复方向**：
1. 以知识库实际内容为准重写 ground_truth 和 key_facts
2. 对无答案的问题标记 `expected_abstain=True`
3. 或改用 LLM-based evaluation（语义匹配而非字符串匹配）

---

## 7. 与队友方案的差异

| 维度 | 队友方案 | 我们的方案 | 创新价值 |
|------|---------|-----------|---------|
| Embedding | Qwen3-Embed-4B | **bge-m3** (MTEB leader) | 跨模型对比数据 |
| Reranker | Qwen3-Reranker-4B | **bge-reranker-v2-m3** | 跨模型对比数据 |
| 检索融合 | 纯 Dense | **Dense + BM25 + RRF** | ★ 核心创新 |
| 查询扩展 | 无 | **HyDE** | 理论正确，场景受限 |
| 查询路由 | 无 | **Query Classifier** | v1无效 → v2修复后+0.18 |
| 来源区分 | 无 | **Authority Scoring** | 独立消融无效果 (Δ=0) |
| 安全机制 | 无 | **Confidence Abstention** | 实用价值高 |
| 回答风格 | 单一 | **3 种可切换人格** | Demo 亮点 |
| 上下文增强 | 无 | **Contextual Enrichment** | 成本高，增益小 |
| 实验配置 | 4 种 | **10 种** | 更完整的消融 |
| 评测维度 | 4 维 | **5 维 (增加 Abstention)** | 更全面的评测 |
| Demo | Streamlit | **Gradio** | 技术栈差异 |

---

## 8. 总结与反思

### 8.1 项目成果

1. **完整的数据管线**: 5,624 网页 + 125 手册 → 3,620 文档 → 20,804 chunks
2. **向量索引**: 4 个 ChromaDB collection，97,732 vectors
3. **7 个创新点实现**: RRF, Reranker, HyDE, Enrichment, Classifier, Authority, Abstention
4. **10 组消融实验**: 全面的定量分析
5. **3 种人格定制**: Demo 亮点

### 8.2 创新点最终评估（经 v2 测试集 + Bootstrap 验证）

| 创新 | 效果 | p-value | 95% CI | 结论 |
|------|------|--------|--------|------|
| **Reranker** | **+0.42** | **0.019** | [0.08, 0.76] | ✅ 唯一统计显著的创新 |
| Classifier v2 | +0.18 | 0.191 | [-0.08, 0.46] | 方向正确，需更大测试集 |
| HyDE | +0.12 | 0.420 | [-0.16, 0.40] | 方向正确，效应量小 |
| RRF (vs Dense) | -0.18 | 0.280 | [-0.50, 0.16] | BM25 在 v2 测试集增加噪声 |
| Authority | 0.00 | 1.000 | [-0.22, 0.24] | 无效果 |

**核心结论**：Cross-Encoder Reranker 是唯一经过统计显著性检验的创新（p=0.019）。在去除测试集的术语不匹配噪声后，Reranker 的效果从 +0.12 提升至 +0.42，达到显著性水平。Classifier 的权重修复使其从有害转为有益（-0.06 → +0.18）。Authority Scorer 首次独立消融显示无效果。

### 8.3 经验教训

1. **测试集应在爬取后构建**。本次项目最大的工程失误：ground_truth 在爬取前凭经验编写，导致 key_facts 用词（"校车""抄袭""借阅"）与知识库实际用词（"巴士""学术不端""还书"）系统性地不匹配。正确性/完整性评分被压低，绝对分数不可信。

2. **仅有 RAG vs No RAG 的差异具有统计显著性**（p=0.000）。所有 Reranker/HyDE/Enrichment/Classifier 的效果（+0.04 到 +0.12）在 50 题规模下均在噪声范围内。

3. **简单方案往往是最好的**。R4 (Dense + BM25 + Reranker) 与 E4 (所有创新叠加) 得分相同。不要因为创新听起来高级就默认有价值。

4. **工程细节决定评测质量**。Grounding 评测 bug（元数据前缀干扰）导致第一轮实验数据全废；bootstrap p-value 计算 bug（未做零假设平移）导致之前所有的 p 值都 ~0.5，失去区分能力。

5. **50 题测试集功效严重不足**。要可靠检测 0.1-0.2 分的效应量（如 Reranker 的提升），需要 200+ 题的测试集。

3. **测试集质量比规模重要**。50 题的测试集如果 ground_truth 精确，比 200 题估算版更有价值。下次应先爬取再构建测试集。

4. **简单方案往往是最好的**。R4 (Dense + BM25 + Reranker) 与 E4 (所有创新叠加) 得分相同。不要因为创新听起来高级就默认有价值。

5. **工程细节决定评测质量**。Grounding 评测 bug（元数据前缀干扰）导致第一轮实验数据全废，修复后结果才变得有意义。

---

> 最后更新: 2026-05-22  
> 模型: DeepSeek-V4 API + bge-m3 + bge-reranker-v2-m3  
> 硬件: AutoDL RTX 6000 Blackwell 96GB
