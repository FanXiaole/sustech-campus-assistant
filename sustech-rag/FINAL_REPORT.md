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

### 5.4 关键发现

1. **RAG 效果显著**: R4 (6.62) vs R0 (0.32) = **+6.30 分，提升 20 倍**
2. **Reranker 是最大单一增益**: +0.12 分
3. **RRF > Dense > BM25**: 融合检索比单一检索好
4. **Default chunk 600 最优**: 小 300 (−0.04) 和大 900 (−0.10) 都不如
5. **创新叠加有递减效应**: E4 = R4，额外的创新没有带来额外增益

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

### 6.4 Ground Truth 不匹配
- **现象**: 测试集 ground_truth 是爬取前估算的，与 LLM 从真实数据中生成的答案不一致
- **案例**: "计算机系成立年份" ground_truth=2011，但 LLM 根据爬取数据给出不同年份
- **影响**: 正确性维度绝对分数偏低，但不影响实验间的相对比较

---

## 7. 与队友方案的差异

| 维度 | 队友方案 | 我们的方案 | 创新价值 |
|------|---------|-----------|---------|
| Embedding | Qwen3-Embed-4B | **bge-m3** (MTEB leader) | 跨模型对比数据 |
| Reranker | Qwen3-Reranker-4B | **bge-reranker-v2-m3** | 跨模型对比数据 |
| 检索融合 | 纯 Dense | **Dense + BM25 + RRF** | ★ 核心创新 |
| 查询扩展 | 无 | **HyDE** | 理论正确，场景受限 |
| 查询路由 | 无 | **Query Classifier** | 规则版无效 |
| 来源区分 | 无 | **Authority Scoring** | 权重设计需优化 |
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

### 8.2 哪些创新有效

| 创新 | 有效性 | 证据 |
|------|--------|------|
| **Reranker** | ✅✅✅ 最有效 | +0.12 分，在所有指标上提升 |
| **HyDE** | ✅ 有效但受限 | +0.08 分，仅在复杂查询上有增益 |
| **RRF** | ✅ 微弱有效 | +0.04 分，16/50 题胜过 Dense |
| **Enrichment** | ⚠️ 基本无效 | +0.06 分，成本/收益比差 |
| **Classifier** | ❌ 无效 | −0.06 分，规则版不可靠 |
| **Authority** | ⚠️ 未评测 | 无独立实验 |
| **Abstention** | ✅ 功能正确 | OOS 检测工作正常 |
| **人格定制** | ✅ Demo 就绪 | 风格切换功能完整 |

### 8.3 经验教训

1. **Reranker 是 RAG 中最值得投资的组件**。从 top-50 到 top-5 的 Cross-Encoder 精排带来了远超其他任何创新的提升。

2. **学术界有效的创新不一定在特定领域有效**。HyDE 在通用 benchmark 上表现优异，但在校园事实查询场景中增益有限。

3. **测试集质量比规模重要**。50 题的测试集如果 ground_truth 精确，比 200 题估算版更有价值。下次应先爬取再构建测试集。

4. **简单方案往往是最好的**。R4 (Dense + BM25 + Reranker) 与 E4 (所有创新叠加) 得分相同。不要因为创新听起来高级就默认有价值。

5. **工程细节决定评测质量**。Grounding 评测 bug（元数据前缀干扰）导致第一轮实验数据全废，修复后结果才变得有意义。

---

> 最后更新: 2026-05-22  
> 模型: DeepSeek-V4 API + bge-m3 + bge-reranker-v2-m3  
> 硬件: AutoDL RTX 6000 Blackwell 96GB
