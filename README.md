# SUSTech Campus RAG — 南科大校园知识库问答系统

基于 Retrieval-Augmented Generation (RAG) 的南方科技大学校园知识库问答系统。AI 大语言模型课程 (2026 Spring) Project 1。

**成员**: 张栩 12412509 · 范晓乐 12412307 · 杨伟铭 12412301

---

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/FanXiaole/sustech-campus-assistant.git
cd sustech-campus-assistant/sustech-rag

# 2. 安装依赖
pip install -r requirements.txt

# 3. 设置 API Key
export DEEPSEEK_API_KEY="sk-your-key-here"

# 4. 运行实验（需要 GPU）
python evaluation/run_experiments.py --experiments R0,R1,R4

# 5. 启动 Demo（需要 GPU）
python demo/app.py
```

---

## 完整复现流程

### 环境要求

| 资源 | 要求 | 说明 |
|------|------|------|
| GPU | NVIDIA GPU (推荐 24GB+ VRAM) | bge-m3 embedding + bge-reranker-v2-m3 |
| Python | 3.11 | conda 环境 |
| API | DeepSeek API Key | LLM 生成、HyDE、Contextual Enrichment |
| OS | Linux (Ubuntu 22.04) | 在 AutoDL 上测试通过 |

### Step 1: 环境配置

```bash
# 在 AutoDL 上（推荐）
bash scripts/autodl_setup.sh

# 或手动配置
conda create -n rag python=3.11 -y && conda activate rag
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install sentence-transformers chromadb rank-bm25 jieba langchain \
    gradio openai httpx scrapy beautifulsoup4 lxml tqdm

# 国内用户设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 设置 API Key
export DEEPSEEK_API_KEY="sk-your-key-here"
```

### Step 2: 数据管线（CPU）

```bash
# 爬取 SUSTech 网站（5,624 页 + 125 篇南科手册）
python indexing/scraper.py
python indexing/manual_fetcher.py

# 清洗：5,749 原始 → 3,620 清洗文档
python indexing/cleaner.py

# 分块：3,620 文档 → 20,804 chunks（3 种大小变体）
python indexing/chunker.py
```

### Step 3: 构建索引（GPU）

```bash
# BM25 稀疏索引（CPU，约 1 分钟）
python indexing/bm25_builder.py

# bge-m3 稠密索引（GPU，约 20 分钟）
# 生成 4 个 ChromaDB collection：default/small/large/enriched
python indexing/embedder.py
```

### Step 4: 上下文增强（API，可选）

```bash
# 为每个 chunk 生成背景描述，构建 enriched 索引
# 20,798/20,804 chunks，约 20 分钟
python indexing/contextual_enrichment.py
```

### Step 5: 构建测试集

```bash
# 生成 50 题测试集（或使用预构建的 v2 验证版）
python evaluation/test_set_builder.py

# v2 验证版（推荐）：基于知识库实际内容修正的 ground_truth
# 已包含在仓库中：data/test_set_v2.json
```

### Step 6: 运行实验（GPU + API）

```bash
# 完整 11 组实验矩阵（约 30 分钟）
python evaluation/run_experiments.py --all

# 或运行关键实验
python evaluation/run_experiments.py --experiments R0,R1,R3,R4,E1,E3,E5

# Bootstrap 显著性检验
python evaluation/run_bootstrap.py

# LLM 语义评分对比（可选）
python evaluation/llm_evaluator.py --experiment R4
```

### Step 7: 启动 Demo

```bash
# 本地启动
python demo/app.py

# 公网访问（需要在 AutoDL 控制台启用"自定义服务"→ 端口 7860）
# 或使用 SSH 隧道：
ssh -L 7860:localhost:7860 -p <port> root@<host>
# 浏览器打开 http://localhost:7860
```

---

## 项目结构

```
sustech-rag/
├── config.py                    # 全局超参数配置
├── requirements.txt             # Python 依赖
├── README.md                    # 本文件
├── FINAL_REPORT.md              # 完整项目报告
│
├── data/
│   ├── raw/                     # 原始爬虫 HTML/JSONL
│   ├── processed/               # 清洗后文档 (processed_docs.jsonl)
│   ├── chunks/                  # Chunk 文件 (3 种大小变体)
│   ├── test_set.json            # v1 测试集（爬取前编写）
│   └── test_set_v2.json         # v2 验证版测试集（推荐使用）
│
├── indexing/                    # 数据采集与索引构建
│   ├── scraper.py               # Scrapy 爬虫 (5,624 页)
│   ├── manual_fetcher.py        # 南科手册 GitHub 文档抓取
│   ├── cleaner.py               # 8 步文档清洗管线
│   ├── chunker.py               # 双策略分块 (Recursive + Markdown)
│   ├── embedder.py              # bge-m3 批量 Embedding + ChromaDB
│   ├── bm25_builder.py          # Jieba + BM25Okapi 索引
│   ├── contextual_enrichment.py # LLM 上下文增强 (创新点)
│   └── tokenizer.py             # 共享 Jieba 分词器
│
├── retrieval/                   # 检索管线
│   ├── dense_retriever.py       # ChromaDB 稠密检索
│   ├── sparse_retriever.py      # BM25 稀疏检索
│   ├── hybrid_rrf.py            # ★ RRF 融合核心算法
│   ├── hyde.py                  # ★ HyDE 假设文档嵌入
│   ├── query_classifier.py      # ★ 6 类查询路由 + 动态权重
│   ├── reranker.py              # bge-reranker-v2-m3 Cross-Encoder
│   └── authority_scorer.py      # ★ 来源权威性 + 时效性打分
│
├── generation/                  # LLM 生成
│   ├── llm_api.py               # DeepSeek API 客户端
│   ├── llm_local.py             # Ollama 本地推理 + Fallback
│   └── prompt_builder.py        # Prompt 组装 + 3 种人格定制
│
├── evaluation/                  # 评测框架
│   ├── test_set_builder.py      # 测试集生成器
│   ├── evaluator.py             # 5 维度评分 + Bootstrap CI
│   ├── run_experiments.py       # 11 组实验矩阵执行器
│   ├── run_bootstrap.py         # 显著性检验一键脚本
│   ├── llm_evaluator.py         # LLM 语义评分对比
│   ├── error_analyzer.py        # 失败案例分类分析
│   ├── rewrite_test_set_v2.py   # 测试集 BM25+LLM 验证重写
│   └── verify_test_set.py       # 测试集可回答性验证
│
├── demo/                        # Gradio 演示应用
│   ├── app.py                   # 3 Tab Demo (Q&A / Experiments / Pipeline)
│   └── cache/                   # 缓存查询结果
│
├── tests/                       # 测试
│   └── test_smoke.py            # 端到端 Smoke Test
│
├── index_store/                 # 索引文件 (需 GPU 构建)
│   ├── chroma_db/               # ChromaDB 向量索引 (539 MB)
│   └── bm25_index.pkl           # BM25 索引 (49 MB)
│
├── results/                     # 实验结果 (JSON)
│   ├── R0/ ... R4/              # 基线 + 核心 RAG
│   ├── E1/ ... E5/              # 创新消融
│   ├── A1/ A2/                  # Chunk 大小消融
│   ├── comparison_table.json    # 实验对比总表
│   └── bootstrap_ci.json        # Bootstrap CI 结果
│
└── scripts/
    ├── autodl_setup.sh          # AutoDL 环境一键配置
    └── run_all.sh               # 端到端 Pipeline 执行器
```

---

## 创新点总览

| # | 创新 | 文件 | 效果 (Δ vs R3) | p-value | 显著？ |
|---|------|------|---------------|---------|--------|
| 1 | **Hybrid RRF** | retrieval/hybrid_rrf.py | +0.04 | 0.855 | no |
| 2 | **Reranker** (bge-reranker-v2-m3) | retrieval/reranker.py | **+0.42** | **0.019** | ✅ |
| 3 | **HyDE** | retrieval/hyde.py | +0.12 | 0.420 | no |
| 4 | **Contextual Enrichment** | indexing/contextual_enrichment.py | +0.16 | 0.248 | no |
| 5 | **Query Classifier** | retrieval/query_classifier.py | +0.18 | 0.191 | no |
| 6 | **Source Authority Scoring** | retrieval/authority_scorer.py | 0.00 | 1.000 | no |
| 7 | **Confidence Abstention** | retrieval/hybrid_rrf.py (内联) | — | — | 功能正确 |
| 8 | **人格定制** (3 风格) | generation/prompt_builder.py | — | — | Demo 亮点 |

**核心结论**: Cross-Encoder Reranker 是唯一通过 Bootstrap 显著性检验的创新 (p=0.019, +0.42 分)。

---

## 11 组实验完整结果 (v2 测试集)

| ID | 配置 | 总分 | Δ vs R3 | 延迟 |
|----|------|------|---------|------|
| R0 | No RAG (LLM Only) | 2.46 | — | 0ms |
| R1 | Dense Only (bge-m3) | 5.32 | +0.18 | 2715ms |
| R2 | BM25 Only | 5.18 | +0.04 | 2299ms |
| R3 | Hybrid RRF | 5.14 | baseline | 2833ms |
| **R4** | **Hybrid + Reranker** | **5.56** | **+0.42** | 3171ms |
| E1 | + HyDE | 5.26 | +0.12 | 4093ms |
| E2 | + Enriched | 5.30 | +0.16 | 2560ms |
| E3 | + Classifier v2 | 5.32 | +0.18 | 2322ms |
| E4 | Full Innovation Stack | 5.48 | +0.34 | 4462ms |
| E5 | + Authority | 5.56 | 0.00 | 2590ms |
| A1 | Small Chunks (300) | 5.44 | +0.30 | 3170ms |
| A2 | Large Chunks (900) | 5.60 | +0.46 | 2594ms |

---

## 技术栈

| 组件 | 选型 | 原因 |
|------|------|------|
| Embedding | BAAI/bge-m3 | MTEB 中文榜单 leader，支持中英混合 |
| Reranker | BAAI/bge-reranker-v2-m3 | Cross-Encoder，比 bi-encoder 精确 |
| LLM | DeepSeek-V4 (API) | 中文能力强，OpenAI 兼容，性价比高 |
| 向量库 | ChromaDB (HNSW) | 轻量级，cosine 距离，支持 metadata 过滤 |
| 稀疏检索 | BM25Okapi (rank-bm25) | k1=1.5, b=0.75, Jieba 分词 |
| 爬虫 | Scrapy + BeautifulSoup | 异步高性能，robots.txt 合规 |
| Demo | Gradio | 快速构建，支持流式输出 |
| 评测 | 规则版 (5 维) + LLM 版 (语义级) | 双重验证 |

---

## 与队友方案的差异

| 维度 | 队友 | 我们 | 价值 |
|------|------|------|------|
| Embedding | Qwen3-Embed-4B | **bge-m3** | MTEB leader vs 自训模型 |
| Reranker | Qwen3-Reranker-4B | **bge-reranker-v2-m3** | 跨模型对比 |
| 融合 | 纯 Dense | **Dense + BM25 + RRF** | 核心创新 |
| 查询扩展 | 无 | **HyDE** | 学术前沿 |
| 来源区分 | 无 | **Authority Scoring** | 领域创新 |
| 安全机制 | 无 | **Confidence Abstention** | 实用价值 |
| 回答风格 | 单一 | **3 种可切换人格** | Demo 亮点 |
| 评测维度 | 4 维 | **5 维 (+Abstention)** | 评测更全面 |
| 实验配置 | 4 种 | **11 种 + Bootstrap CI** | 消融最完整 |

---

## 复现检查清单

- [ ] GPU 可用 (`nvidia-smi`)
- [ ] `DEEPSEEK_API_KEY` 已设置
- [ ] `HF_ENDPOINT=https://hf-mirror.com`（国内用户）
- [ ] 数据管线完成（scraper → cleaner → chunker）
- [ ] BM25 索引已构建（`index_store/bm25_index.pkl`）
- [ ] ChromaDB 索引已构建（`index_store/chroma_db/`）
- [ ] 测试集可用（`data/test_set_v2.json`）
- [ ] Demo 可访问（`http://localhost:7860`）

---

## 已知局限

1. **未实现 Semantic Chunking**: 采用 Recursive Character Splitter + Markdown Header Splitter 替代
2. **未实现 Multi-granularity Index**: 单一粒度 chunk（300/600/900），无层级结构
3. **测试集仅 50 题**: Bootstrap CI 功效不足以检测 <0.2 分的小效应
4. **Gradio share 不可用**: GFW 阻断 frpc 下载，需用 SSH 隧道或 AutoDL 自定义服务
5. **未接 vLLM**: LLM 全部走 API，无本地推理部署

详细技术报告见 [FINAL_REPORT.md](FINAL_REPORT.md)。
