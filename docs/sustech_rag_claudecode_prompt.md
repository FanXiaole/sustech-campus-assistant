# SUSTech Campus RAG System — Claude Code Implementation Prompt

## 📋 How to Use This Prompt
Paste the content below the horizontal rule directly into Claude Code as your **initial project prompt**.
Read the GPU Protocol section carefully — it tells Claude Code exactly when to pause and ask you to start your AutoDL instance.

---

# Project: SUSTech Campus RAG Knowledge Base System

You are an expert ML engineer implementing a production-grade Retrieval-Augmented Generation (RAG) system for Southern University of Science and Technology (SUSTech). This project targets a graduate-level AI course and should demonstrate both engineering rigor and research-level innovations.

## ⚙️ GPU / AutoDL Protocol — CRITICAL, READ FIRST

The user has access to an **RTX 5090 on AutoDL**. GPU is expensive — do NOT assume it is running. Follow this strict protocol:

**Before any GPU-intensive task, PAUSE and say:**
> 🖥️ **GPU Required** — Please start your AutoDL RTX 5090 instance now.
> 1. Log in at https://www.autodl.com → "我的实例" → Start your RTX 5090 instance
> 2. Copy the SSH command shown (e.g. `ssh -p 12345 root@connect.westc.gpuhub.com`)
> 3. Paste it here and confirm the instance is running before I continue.
> 4. Once connected, run: `nvidia-smi` to confirm GPU is visible.

**GPU-required tasks (always pause before these):**
- Building the embedding index (`embed_and_index.py`) — ~2-3 hours for full corpus
- Training / running the reranker in batch mode
- Running the Qwen2.5-7B via vLLM for full evaluation
- Any CUDA-dependent code that will run longer than 30 seconds

**CPU-safe tasks (no pause needed, run locally or on AutoDL login node):**
- Data crawling and cleaning scripts
- BM25 index construction
- Gradio UI development and testing
- Unit tests with small toy datasets
- All code scaffolding and file structure setup

**When the user confirms the AutoDL instance is running, provide:**
```bash
# Run this on the AutoDL instance to set up the environment
conda create -n rag python=3.11 -y && conda activate rag
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install sentence-transformers chromadb rank-bm25 jieba langchain \
    gradio openai httpx scrapy beautifulsoup4 lxml tqdm rich \
    FlagEmbedding transformers accelerate bitsandbytes
```

---

## 🎯 Project Goals & Differentiators

Build a SUSTech campus Q&A RAG system that goes beyond baseline with these **research-level innovations** (your academic differentiators):

| Innovation | Description | Academic Value |
|---|---|---|
| **HyDE** | Hypothetical Document Embedding for query expansion | Bridges query-document embedding gap |
| **Semantic Chunking** | Embedding-similarity-based boundary detection | Better than fixed-size splitting |
| **Contextual Enrichment** | LLM-generated context prepended to each chunk | Anthropic 2024 paper technique |
| **Hybrid RRF** | BM25 + Dense fusion with Reciprocal Rank Fusion | Standard in MTEB top systems |
| **Query Classifier** | Route queries to optimal retrieval strategy | Dynamic pipeline adaptation |
| **Source Authority Scoring** | Reranking signal from source credibility | Domain-specific knowledge injection |
| **Confidence Abstention** | Refuse to answer when evidence is weak | Hallucination mitigation |
| **Multi-granularity Index** | Sentence-level retrieval, paragraph-level context | RAPTOR-inspired hierarchy |

---

## 📁 Project File Structure

Create the following structure from the beginning. Do not deviate from it.

```
sustech-rag/
├── README.md
├── requirements.txt
├── config.py                    # All hyperparameters and paths in one place
│
├── data/
│   ├── raw/                     # Raw crawled HTML / JSONL
│   ├── processed/               # Cleaned documents (processed_docs.jsonl)
│   ├── chunks/
│   │   ├── chunks_default.jsonl # Main chunk set (600 chars, overlap 100)
│   │   ├── chunks_small.jsonl   # Ablation: 300 chars
│   │   └── chunks_large.jsonl   # Ablation: 900 chars
│   └── test_set.json            # 50+ evaluation questions with labels
│
├── indexing/
│   ├── scraper.py               # Scrapy + BeautifulSoup crawler
│   ├── cleaner.py               # HTML cleaning, dedup, language detection
│   ├── chunker.py               # Semantic + recursive chunking strategies
│   ├── embedder.py              # Batch embedding with bge-m3
│   ├── bm25_builder.py          # Jieba tokenization + BM25Okapi index
│   └── contextual_enrichment.py # LLM-generated chunk context prefix (Innovation 3)
│
├── retrieval/
│   ├── dense_retriever.py       # ChromaDB dense retrieval
│   ├── sparse_retriever.py      # BM25 retrieval
│   ├── hybrid_rrf.py            # ★ RRF fusion (core differentiator)
│   ├── hyde.py                  # ★ HyDE query expansion
│   ├── query_classifier.py      # ★ Query type classification & routing
│   ├── reranker.py              # Cross-encoder reranking (bge-reranker-v2-m3)
│   └── authority_scorer.py      # ★ Source authority reranking signal
│
├── generation/
│   ├── llm_local.py             # Ollama / vLLM local inference
│   ├── llm_api.py               # SiliconFlow API fallback (Qwen2.5-72B)
│   ├── prompt_builder.py        # Context concatenation & prompt templates
│   └── confidence_abstention.py # ★ Refuse when evidence score is weak
│
├── evaluation/
│   ├── test_set_builder.py      # Structured 50-question test set generator
│   ├── evaluator.py             # 5-dimension scoring (correctness, grounding,
│   │                            #   completeness, traceability, abstention)
│   ├── run_experiments.py       # R0-R4, E1, A1-A2 experiment matrix
│   └── error_analyzer.py        # Categorized failure analysis
│
├── demo/
│   ├── app.py                   # Gradio multi-tab demo application
│   └── cache/                   # Pre-cached answers for robust demo replay
│
└── scripts/
    ├── run_all.sh               # End-to-end pipeline runner
    └── autodl_setup.sh          # AutoDL environment bootstrap
```

---

## 🔧 Implementation Specifications

### Phase 1 — Data Ingestion (CPU, no GPU needed)

#### `config.py` — Build this first
```python
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_DIR  = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
CHUNK_DIR = DATA_DIR / "chunks"
INDEX_DIR = ROOT / "index_store"

# Chunking
CHUNK_SIZE_DEFAULT = 600    # characters
CHUNK_SIZE_SMALL   = 300
CHUNK_SIZE_LARGE   = 900
CHUNK_OVERLAP      = 100
MIN_CHUNK_LEN      = 100

# Retrieval
DENSE_TOP_K     = 50
BM25_TOP_K      = 50
RRF_K           = 60        # RRF smoothing constant
RRF_FUSION_TOP  = 20
RERANK_TOP_K    = 5

# Models
EMBED_MODEL    = "BAAI/bge-m3"
RERANK_MODEL   = "BAAI/bge-reranker-v2-m3"
LOCAL_LLM      = "qwen2.5:7b"              # Ollama
API_LLM        = "Qwen/Qwen2.5-72B-Instruct"
SILICONFLOW_BASE = "https://api.siliconflow.cn/v1"

# Source authority weights (used in reranking)
SOURCE_AUTHORITY = {
    "official":   1.0,
    "admission":  0.95,
    "library":    0.90,
    "department": 0.85,
    "news":       0.75,
    "manual":     0.80,
    "unknown":    0.50,
}

# Confidence abstention threshold
ABSTENTION_THRESHOLD = 0.35   # RRF score below this → refuse to answer
```

#### `indexing/scraper.py` — SUSTech Crawler
```python
"""
Crawl SUSTech public pages. Targets:
  - www.sustech.edu.cn (official facts, policies)
  - lib.sustech.edu.cn (library services, hours)
  - admit.sustech.edu.cn (admission materials)
  - Each department's official subdomain
  - 南科手册 Markdown repository (if accessible)

Implement with Scrapy. Key requirements:
  - Respect robots.txt (add ROBOTSTXT_OBEY = True)
  - Rate limit: DOWNLOAD_DELAY = 1.5 seconds
  - Only follow links within sustech.edu.cn domain
  - Save raw output as JSONL: {url, html, crawled_at, status_code}
  - Log all failed URLs to data/raw/failed_urls.txt
  - Target: 5,000–15,000 pages
"""
```

#### `indexing/cleaner.py` — Document Cleaning Pipeline
```python
"""
Input:  data/raw/*.jsonl (raw HTML + metadata)
Output: data/processed/processed_docs.jsonl

Pipeline steps (implement each as a standalone function, then chain):
1. extract_text(html)        — BeautifulSoup, remove nav/footer/script/style/header
2. detect_language(text)     — Keep Chinese + English, discard others (langdetect)
3. normalize_whitespace(text) — Collapse \s+, strip
4. filter_short(text, min_len=100) — Remove stub pages / SPA skeletons
5. url_normalize(url)        — Strip query params, anchors, trailing slashes
6. dedup_by_url(docs)        — Remove exact URL duplicates (keep latest crawl_at)
7. dedup_by_hash(docs)       — MD5 hash of cleaned text, remove near-duplicates
8. classify_source(url)      — Map to SOURCE_AUTHORITY keys

Output schema per document:
{
  "doc_id": "sha256 of url[:16]",
  "url": str,
  "text": str,
  "source_family": str,   # "official" | "library" | "admission" | etc.
  "char_count": int,
  "text_hash": str,
  "crawled_at": ISO timestamp
}

Print summary stats: total docs, removed (short), removed (dup_url),
removed (dup_hash), final count, total MB of text.
"""
```

#### `indexing/chunker.py` — Dual-Strategy Chunking
```python
"""
Implement TWO chunking strategies and apply both:

STRATEGY A — RecursiveCharacterChunker (for web pages):
  - Use langchain RecursiveCharacterTextSplitter
  - Separators in priority order: ["。\n", "。", "\n\n", "\n", "，", " ", ""]
  - chunk_size from config (default/small/large variants)
  - chunk_overlap from config
  - Prepend metadata prefix to each chunk:
      "[来源:{source_family}][域名:{domain}] {text}"

STRATEGY B — MarkdownHeaderChunker (for 南科手册 .md files):
  - Use langchain MarkdownHeaderTextSplitter
  - Headers: [("#","H1"),("##","H2"),("###","H3")]
  - After split, apply length constraints:
      * < MIN_CHUNK_LEN chars → merge with adjacent
      * > 900 chars → re-split by paragraph (\n\n)
  - Tables: detect by "| --- |" pattern, keep entire table as one chunk
  - Lists: group consecutive list items, do NOT split individual bullet points
  - Store heading_path in metadata: "H1 > H2 > H3"

Output schema per chunk:
{
  "chunk_id": "doc_id_chunk_{i:04d}",
  "doc_id": str,
  "text": str,          # includes metadata prefix
  "raw_text": str,      # text without prefix (for display)
  "source_family": str,
  "url": str,
  "heading_path": str,  # "" for web chunks
  "char_count": int,
  "chunk_strategy": "recursive" | "markdown",
}

Save three files: chunks_default.jsonl, chunks_small.jsonl, chunks_large.jsonl
Print: total chunks, avg char_count, min, max, p50, p95.
"""
```

---

### Phase 2 — Indexing (⚠️ GPU Required for Embedding)

> **When you reach this phase, PAUSE and display the GPU startup prompt to the user.**
> Wait for confirmation that the AutoDL RTX 5090 is running and SSH is active.
> Then provide the environment setup commands from the AutoDL Protocol section.

#### `indexing/embedder.py`
```python
"""
⚠️ GPU REQUIRED — Pause and prompt user to start AutoDL RTX 5090 before running.

Input:  data/chunks/chunks_default.jsonl
Output: ChromaDB persistent collection at index_store/chroma_db/

Implementation:
1. Load SentenceTransformer("BAAI/bge-m3", device="cuda")
   - Confirm GPU with: print(torch.cuda.get_device_name(0))
   - Expected on 5090: ~18,000 chunks/minute at batch_size=256

2. Encode with instruction prefix for queries (NOT for documents):
   DOCUMENT_PREFIX = ""                       # bge-m3: no doc prefix needed
   QUERY_PREFIX    = "Represent this sentence for retrieval: "

3. Batch encode all chunks:
   - batch_size = 256 (optimal for 5090 32GB VRAM)
   - normalize_embeddings = True  (required for cosine similarity in Chroma)
   - show_progress_bar = True

4. Create ChromaDB collection:
   - name: "sustech_default"  (also create "sustech_small", "sustech_large")
   - metadata: {"hnsw:space": "cosine", "hnsw:construction_ef": 200,
                "hnsw:M": 16}   ← tune for recall vs speed
   - Batch upsert: 1000 items per call to avoid OOM

5. Verify: run 3 test queries, print top-3 results + scores.
6. Print: total vectors, index size on disk, time taken.

Expected time on RTX 5090: ~15-25 minutes for 15k chunks.
"""
```

#### `indexing/contextual_enrichment.py` — Innovation #3
```python
"""
★ INNOVATION: Contextual Chunk Enrichment (based on Anthropic's 2024 paper
  "Contextual Retrieval" — prepending LLM-generated context to each chunk
  dramatically improves retrieval precision for factual Q&A).

⚠️ GPU REQUIRED — Uses local LLM for generation. Prompt user if not already running.
  Alternative: Use SiliconFlow API (much cheaper, no GPU needed for this step).

Algorithm:
  For each chunk c_i from document D:
    1. Take first 500 chars of D as document_context
    2. Prompt the LLM:
         system: "You are a concise document analyst. Reply in Chinese."
         user: f"Document excerpt: {document_context[:500]}
                Chunk to contextualize: {c_i.raw_text[:200]}
                In ONE sentence (max 60 Chinese chars), describe what specific
                information this chunk provides within the broader document."
    3. Prepend generated context to chunk text:
         chunk.text = f"[背景:{generated_context}] {chunk.raw_text}"

Cost optimization:
  - Cache results to data/chunks/chunk_contexts.json (chunk_id → context)
  - Skip chunks that already have cached contexts
  - Use SiliconFlow free API tier (Qwen2.5-7B-Instruct) if no GPU available
  - Batch with async HTTP calls (asyncio + httpx, concurrency=10)

This replaces the embedder's plain text with enriched text.
Build a second Chroma collection: "sustech_enriched" alongside "sustech_default"
for direct A/B comparison in experiments.
"""
```

#### `indexing/bm25_builder.py`
```python
"""
⚠️ CPU ONLY — No GPU needed. Safe to run locally.

Input:  data/chunks/chunks_default.jsonl
Output: index_store/bm25_index.pkl

Implementation:
1. Tokenize each chunk with jieba (fine-grained mode):
   - Filter: keep tokens with len >= 2
   - Filter: remove stopwords (load from data/stopwords_zh.txt)
   - Keep English words as-is (for mixed Chinese-English content)

2. Build BM25Okapi index:
   - BM25Okapi(corpus_tokens, k1=1.5, b=0.75)  ← standard params
   
3. Save:
   pickle.dump({"index": bm25, "chunk_ids": [...], "chunks": [...]}, f)

4. Download Chinese stopwords list:
   wget https://raw.githubusercontent.com/goto456/stopwords/master/cn_stopwords.txt
   -O data/stopwords_zh.txt

5. Verify: test with 3 queries, print top-5 chunks + BM25 scores.
"""
```

---

### Phase 3 — Retrieval Pipeline

#### `retrieval/hyde.py` — Innovation #1: HyDE
```python
"""
★ INNOVATION: Hypothetical Document Embedding (HyDE)
  Gao et al. 2022 — "Precise Zero-Shot Dense Retrieval without Relevance Labels"

Core idea: Instead of embedding the query directly, use an LLM to generate a
  HYPOTHETICAL answer document, then embed THAT. The hypothetical document
  lives in the same embedding space as real documents, dramatically improving
  retrieval for informational queries.

Implementation:
  def hyde_retrieve(query: str, llm_fn, embed_fn, collection, top_k=50):
      # Step 1: Generate hypothetical document
      hyp_doc = llm_fn(
          system="你是一个南科大校园知识库。生成一段简短的、假设性的答案文字，"
                 "就好像这个信息真的存在于学校官方资料中一样。不超过150字。",
          user=query
      )
      
      # Step 2: Embed hypothetical doc (NOT the query)
      hyp_embedding = embed_fn(hyp_doc)
      
      # Step 3: Retrieve using hypothetical embedding
      results = collection.query(query_embeddings=[hyp_embedding], n_results=top_k)
      return results

Expose a flag: use_hyde=True/False (for ablation experiments)
When use_hyde=False, fall back to direct query embedding.

Note: HyDE adds ~0.5-1s latency per query due to LLM call.
Use SiliconFlow API (qwen2.5-7b-instruct free tier) to keep cost near zero.
"""
```

#### `retrieval/query_classifier.py` — Innovation #4
```python
"""
★ INNOVATION: Query-Type Classifier → Dynamic Retrieval Strategy Router

Classify each query into one of these types, then route to optimal strategy:

  Type              | Example                          | Best Strategy
  ------------------|----------------------------------|------------------
  FACTUAL_SIMPLE    | "图书馆几点开门？"               | BM25 (keyword precise)
  FACTUAL_COMPLEX   | "计算机系的教师科研方向有哪些？"  | Dense + Rerank
  COMPARATIVE       | "理工学院和医学院的区别？"        | HyDE + Dense
  PROCEDURAL        | "如何办理借书证？"                | Dense (semantic)
  OUT_OF_SCOPE      | "清华大学校长是谁？"              | Immediate abstention
  TEMPORAL          | "最近的招生政策有什么变化？"      | BM25 (recency signal)

Implementation options (pick one based on complexity):
  Option A (simple): Keyword-rule-based classifier (fast, no model needed)
    - OUT_OF_SCOPE: query contains no SUSTech-related keywords
    - FACTUAL_SIMPLE: short query (<15 chars) with specific noun
    - Otherwise: FACTUAL_COMPLEX

  Option B (recommended): Zero-shot LLM classifier via SiliconFlow API
    - Single API call with classification prompt
    - Returns JSON: {"type": "FACTUAL_SIMPLE", "confidence": 0.92}
    - Cache results: same query → same classification

Implement classifier, then modify the main retrieve() function to dispatch:
  def retrieve(query, config):
      q_type = classify_query(query)
      if q_type == "OUT_OF_SCOPE":
          return [], "OUT_OF_SCOPE"
      elif q_type == "FACTUAL_SIMPLE":
          return bm25_only_retrieve(query, top_k=5), q_type
      elif q_type == "COMPARATIVE":
          return hybrid_with_hyde(query), q_type
      else:
          return hybrid_rrf_rerank(query), q_type
"""
```

#### `retrieval/hybrid_rrf.py` — Core Differentiator
```python
"""
★ CORE DIFFERENTIATOR: Hybrid Retrieval with Reciprocal Rank Fusion

Full implementation of the RRF pipeline:

def rrf_fusion(dense_results, bm25_results, k=60, top_n=20, 
               dense_weight=1.0, sparse_weight=1.0):
    '''
    Reciprocal Rank Fusion.
    
    Score formula: score(d) = dense_weight * 1/(k + rank_dense(d))
                            + sparse_weight * 1/(k + rank_sparse(d))
    
    Args:
        k: smoothing constant (default 60, from original RRF paper)
        dense_weight / sparse_weight: tunable per query type
            FACTUAL_SIMPLE  → (0.5, 1.5)  ← favor BM25 for keywords
            FACTUAL_COMPLEX → (1.5, 0.5)  ← favor dense for semantics
            default         → (1.0, 1.0)  ← equal fusion
    '''
    ...

def full_hybrid_pipeline(query, use_hyde=False, use_classifier=True):
    1. [Optional] Query classification → get weights
    2. [Optional] HyDE → get hypothetical embedding
    3. Dense retrieval (top-50) using query or HyDE embedding
    4. BM25 retrieval (top-50)
    5. RRF fusion → top-20
    6. Reranking → top-5
    7. Confidence scoring → abstention check
    8. Return: (chunks, metadata, pipeline_trace)
    
    pipeline_trace captures each step's timing and top-3 results for debugging.
    This is displayed in the Gradio demo's "Pipeline Inspector" tab.
"""
```

#### `retrieval/authority_scorer.py` — Innovation #6
```python
"""
★ INNOVATION: Source Authority Reranking Signal

After cross-encoder reranking, apply a small authority boost:
  final_score = rerank_score * 0.85 + authority_weight * 0.15

Authority weights come from config.SOURCE_AUTHORITY dict.
This ensures that when two chunks have nearly equal semantic relevance,
the one from the official website ranks above an unofficial source.

Also implement a freshness signal for time-sensitive queries:
  - Extract crawl date from metadata
  - For TEMPORAL query type: apply small decay for pages > 6 months old
  - freshness_score = exp(-days_since_crawl / 180)
  
Combine: final_score = rerank_score * 0.80 
                     + authority_weight * 0.12 
                     + freshness_score * 0.08  (only for TEMPORAL queries)
"""
```

#### `generation/confidence_abstention.py` — Innovation #7
```python
"""
★ INNOVATION: Evidence-Based Confidence Abstention

Refuse to answer when retrieved evidence is insufficient.
This directly addresses hallucination — a key weakness in vanilla RAG.

Two-stage abstention:

Stage 1 — Score-based (fast, before LLM call):
  If max(rrf_scores) < ABSTENTION_THRESHOLD (default 0.35):
      return "根据现有校园资料，未找到与您问题相关的信息。" \
             "建议直接访问 sustech.edu.cn 或联系相关部门。"

Stage 2 — LLM-based self-check (after generation, optional):
  Prompt the LLM:
      "Given this answer: {answer}
       And these source chunks: {chunks}
       Does the answer contain any claims NOT supported by the chunks?
       Reply with JSON: {'grounded': true/false, 'unsupported_claims': [...]}"
  
  If grounded=False: append a disclaimer to the answer.
  Log all ungrounded cases to evaluation/ungrounded_log.jsonl for error analysis.

Expose: abstention_mode = "score_only" | "llm_check" | "both"
Track abstention rate per experiment run in evaluation metrics.
"""
```

---

### Phase 4 — Evaluation Framework

#### `evaluation/test_set_builder.py`
```python
"""
Build a structured 50+ question test set. Output: data/test_set.json

Question categories and counts:
  easy (15 questions):
    - factual_simple: 8   # Single-hop facts with precise answers in one chunk
    - time_location: 4    # Hours, addresses, dates
    - procedure: 3        # How to do X on campus
  
  medium (20 questions):
    - factual_complex: 8  # Multi-hop: need info from 2+ chunks
    - department_info: 7  # Faculty, research directions, departments
    - policy: 5           # Rules, regulations, academic policies
  
  hard (10 questions):
    - comparative: 4      # Compare two departments/services
    - cross_source: 4     # Answer spans web + manual
    - temporal: 2         # Recency-sensitive questions
  
  out_of_scope (5 questions):
    - external: 3         # About other universities / non-SUSTech topics
    - hallucination_bait: 2  # Plausible-sounding but false premises

Each question schema:
{
  "q_id": "easy_factual_001",
  "question": str,
  "ground_truth": str,           # Expected answer (for scoring)
  "key_facts": [str],            # Must-contain facts for scoring
  "source_urls": [str],          # Where ground truth was verified
  "difficulty": "easy"|"medium"|"hard"|"oos",
  "category": str,
  "expected_abstain": bool,      # True for out_of_scope questions
}
"""
```

#### `evaluation/evaluator.py`
```python
"""
5-Dimension Scoring System (0-2 per dimension, max 10 per question):

Dim 1 — Correctness (0-2):
  2: Answer matches ground_truth and contains all key_facts
  1: Partially correct, some key_facts missing
  0: Wrong, hallucinated, or completely irrelevant

Dim 2 — Evidence Grounding (0-2):
  2: Every claim in answer is traceable to a retrieved chunk
  1: Most claims grounded, 1-2 unsupported
  0: Answer ignores retrieved context or contradicts it

Dim 3 — Completeness (0-2):
  2: All key_facts present
  1: >50% of key_facts present
  0: <50% of key_facts present

Dim 4 — Source Traceability (0-2):
  2: Answer cites or clearly refers to a specific source
  1: Implicitly grounded (can be traced, not explicitly cited)
  0: No traceability

Dim 5 — Abstention Quality (0-2):  ← Your differentiator vs teammate
  For expected_abstain=True:
    2: Correctly refuses and explains why
    1: Partially refuses (hedges but still guesses)
    0: Confidently hallucinates an answer
  For expected_abstain=False:
    2: Does NOT refuse (correctly answers)
    1: Over-hedges unnecessarily
    0: Incorrectly refuses a valid question

Additional metrics to track:
  - keyword_hit_rate: % of key_facts found verbatim in answer
  - retrieval_latency_ms: time for full retrieval pipeline
  - generation_latency_ms: time for LLM generation
  - total_latency_ms
  - abstention_rate: % of queries where system refused
"""
```

#### `evaluation/run_experiments.py`
```python
"""
⚠️ PARTIAL GPU REQUIRED — Some experiment configs need GPU. 
   Prompt user before R3, R4, E1, and full-corpus runs.

Experiment matrix — run ALL of these:

Baseline comparisons:
  R0: no_rag           — Direct LLM (Qwen2.5-7B), zero retrieval
  R1: dense_only       — bge-m3 embedding → Chroma top-5 → LLM
  R2: bm25_only        — BM25 top-5 → LLM
  
Core RAG pipeline:
  R3: hybrid_rrf       — Dense(50) + BM25(50) → RRF → top-5 → LLM  ★ main result
  R4: hybrid_full      — R3 + Reranker → top-5 → LLM               ★ best config

Innovation ablations:
  E1: hyde_hybrid      — HyDE + R3 pipeline
  E2: enriched_hybrid  — Contextual enrichment chunks + R3
  E3: classified_route — Query classifier routing + R3
  E4: full_innovation  — All innovations combined                   ★ showcase

Chunk size ablations:
  A1: R4 with chunks_small (300 chars)
  A2: R4 with chunks_large (900 chars)
  A3: R4 with chunks_default (600 chars, same as R4, reference point)

For each experiment, log to results/{experiment_id}/:
  - scores_per_question.json
  - aggregate_scores.json   (mean ± std per dimension, per difficulty tier)
  - examples/               (5 random Q&A examples with retrieved chunks)
  - latency_stats.json

Also generate results/comparison_table.json for the demo dashboard.
"""
```

---

### Phase 5 — Demo System

#### `demo/app.py` — Gradio Multi-Tab Application
```python
"""
Build a polished Gradio demo with these 5 tabs:

TAB 1: 🎓 Campus Q&A (main interaction)
  - Text input for question
  - Dropdown: retrieval mode (no_rag / dense / bm25 / hybrid / hybrid+innovations)
  - Toggle: HyDE on/off
  - Toggle: Show retrieved chunks (transparency mode)
  - Submit button
  - Answer text box (streaming preferred)
  - Source citations (collapsible, shows URL + chunk excerpt)
  - Latency stats (retrieval ms + generation ms)
  - Abstention indicator (shown when system refuses)

TAB 2: 🔬 Pipeline Inspector
  - Input a question
  - Shows step-by-step pipeline trace:
      Query → Classification → Dense top-3 → BM25 top-3 → 
      After RRF top-3 → After Rerank top-3 → Final answer
  - RRF score bar charts (use gr.BarPlot)
  - Chunk text previews at each stage

TAB 3: 📊 Experiment Results Dashboard
  - Load from results/comparison_table.json
  - Radar chart: R0 vs R3 vs R4 vs E4 across 5 dimensions
  - Bar chart: accuracy by difficulty tier
  - Latency comparison table
  - Key finding annotations (hardcode 3 insight callouts)

TAB 4: 🧪 A/B Comparison
  - Enter a question
  - Side-by-side: "Without RAG" vs "With RAG (best config)"
  - Highlight differences in answer quality
  - Show which retrieved chunks made the difference

TAB 5: 📋 Error Analysis
  - Dropdown: failure category
      (retrieval_failure / context_overflow / missing_knowledge / 
       hallucination / over_refusal)
  - 2-3 annotated examples per category
  - Proposed fix for each category

IMPORTANT — Demo robustness:
  - Implement cached replay mode:
      DEMO_MODE = os.getenv("DEMO_MODE", "live")  # "live" or "cached"
      Pre-run all 50 test questions and cache to demo/cache/{q_id}.json
      In "cached" mode, skip retrieval/LLM and return cached results instantly
  - launch(share=True) — generates public URL without SSH port forwarding
  - Add a "🔴 LIVE" vs "📼 CACHED" indicator in the UI header
"""
```

---

## 📄 Report Writing Guidance

When writing the project report, emphasize these sections to maximize academic impact:

**Section 3.3 — Innovation: Hybrid Retrieval with RRF**
Include the mathematical formulation:
```
score_RRF(d) = Σᵢ wᵢ / (k + rankᵢ(d))

where:
  i ∈ {dense, sparse}
  k = 60 (smoothing constant, Cormack et al. 2009)
  wᵢ = dynamic weights from query classifier
```

**Section 3.4 — Innovation: HyDE Query Expansion**
Cite: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022)
Include embedding-space diagram: query vector vs HyDE vector distance to relevant doc.

**Section 4.2 — Ablation Study Design**
Present as a clean table — each innovation's incremental gain:
```
Config              | Correctness | Grounding | ΔCorrectness vs R0
R0 (no RAG)         |    x.x      |    x.x    |   baseline
R3 (hybrid RRF)     |    x.x      |    x.x    |   +xx%
R4 (+ rerank)       |    x.x      |    x.x    |   +xx%
E4 (+ all inno.)    |    x.x      |    x.x    |   +xx%
```

**Section 5 — Error Analysis**
Classify ALL failures before writing. Target this breakdown:
- Retrieval failure (right answer in DB but not retrieved): suggest chunking fix
- Knowledge gap (answer not in DB): suggest data source expansion
- Context dilution (top-k too large, noise overwhelms signal): suggest rerank tuning
- Hallucination (LLM ignores context): suggest stronger prompt constraints
- Over-refusal (confident answer refused): suggest threshold tuning

---

## 🚀 Getting Started Sequence

Tell Claude Code to execute in this exact order:

```
Step 1:  Create full directory structure and config.py
Step 2:  Implement cleaner.py and run on sample data (CPU, no GPU)
Step 3:  Implement chunker.py with both strategies (CPU)
Step 4:  Implement bm25_builder.py and build BM25 index (CPU)
Step 5:  [PAUSE — prompt user to start AutoDL RTX 5090]
Step 6:  Run embedder.py on chunks_default — build Chroma index (GPU ~20 min)
Step 7:  Implement and test all retrieval modules (GPU)
Step 8:  Run contextual_enrichment.py — build enriched index (GPU or API)
Step 9:  [GPU can be paused here if not running vLLM]
Step 10: Implement generation modules with Ollama local fallback
Step 11: Build test_set manually (50+ questions) — save to data/test_set.json
Step 12: [PAUSE — start AutoDL again for full evaluation run]
Step 13: Run run_experiments.py — all experiment configs
Step 14: Build demo/app.py and pre-cache all results
Step 15: Final demo rehearsal with cached mode
```

---

## ⚡ Performance Targets (RTX 5090)

| Task | Expected Time | VRAM Usage |
|---|---|---|
| Embedding 10k chunks (bge-m3) | ~8 minutes | ~6 GB |
| Embedding 15k chunks (bge-m3) | ~12 minutes | ~6 GB |
| Contextual enrichment (API) | ~30 minutes | 0 (API) |
| Reranker inference (15k pairs) | ~25 minutes | ~4 GB |
| vLLM Qwen2.5-7B serving | always-on | ~16 GB |
| Full evaluation (R0-E4, 50Q) | ~2 hours | ~20 GB |

**VRAM budget on RTX 5090 (32GB):**
- vLLM 7B + bge-m3 + reranker simultaneously: ~26GB ✅ fits

---

## 🔑 Key Technical Decisions to Defend in Q&A

Be ready to explain these choices during the demo presentation:

1. **Why RRF over weighted sum?** — RRF is rank-based (order-invariant to score magnitudes), robust to score distribution differences between dense and sparse retrievers.

2. **Why bge-m3 over Qwen3-Embed?** — bge-m3 is the MTEB Chinese leaderboard leader for zero-shot retrieval; Qwen3-Embed requires larger VRAM. We include both in ablation (E1).

3. **Why HyDE helps?** — Query "图书馆几点开门" has very different vocabulary from document "图书馆服务时间：8:00-22:00"。HyDE generates text closer to document style.

4. **Why abstention matters?** — Hallucination in campus Q&A has real user harm (wrong exam dates, wrong locations). Abstaining correctly is more valuable than guessing.

5. **Why contextual enrichment?** — Chunks without context lose meaning ("The deadline is December 31" — what deadline?). Enrichment re-anchors each chunk to its document context.
