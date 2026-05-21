#!/bin/bash
# ============================================================================
# SUSTech RAG — End-to-End Pipeline Runner
# ============================================================================
# 用法：
#   bash scripts/run_all.sh              # 完整 pipeline
#   bash scripts/run_all.sh --cpu-only   # 只跑 CPU 安全步骤
#   bash scripts/run_all.sh --gpu        # 只跑 GPU 步骤（需在 AutoDL 上运行）
#
# 前置条件：
#   - Python 3.11+
#   - pip install -r requirements.txt
#   - export DEEPSEEK_API_KEY="your-key"  (contextual enrichment 需要)
# ============================================================================

set -e  # 任何步骤失败立即退出

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo " SUSTech RAG — End-to-End Pipeline"
echo " Project root: $PROJECT_ROOT"
echo "=============================================="

# ── Step 1: 爬取网页数据 ────────────────────────────────
echo ""
echo "[Step 1/10] Crawling SUSTech websites..."
python indexing/scraper.py
echo "✓ Crawling complete → data/raw/raw_pages.jsonl"

# ── Step 2: 爬取南科手册 ──────────────────────────────────
echo ""
echo "[Step 2/10] Fetching 南科助手 manual docs..."
python indexing/manual_fetcher.py
echo "✓ Manual fetch complete → data/raw/manual_docs.jsonl"

# ── Step 3: 清洗文档 ─────────────────────────────────────
echo ""
echo "[Step 3/10] Cleaning documents..."
python indexing/cleaner.py
echo "✓ Cleaning complete → data/processed/processed_docs.jsonl"

# ── Step 4: Chunk 切分（三种大小） ─────────────────────
echo ""
echo "[Step 4/10] Chunking documents (default/small/large)..."
python indexing/chunker.py
echo "✓ Chunking complete → data/chunks/chunks_*.jsonl"

# ── Step 5: BM25 索引构建 ───────────────────────────────
echo ""
echo "[Step 5/10] Building BM25 index..."
python indexing/bm25_builder.py
echo "✓ BM25 index built → index_store/bm25_index.pkl"

# ── Step 6: ⚠️ GPU — Embedding 向量索引 ──────────────
if [ "$1" != "--cpu-only" ]; then
    echo ""
    echo "[Step 6/10] ⚠️ GPU REQUIRED — Building ChromaDB index..."
    echo "    If GPU is not available, run with: bash scripts/run_all.sh --cpu-only"
    python indexing/embedder.py
    echo "✓ ChromaDB index built → index_store/chroma_db/"
else
    echo ""
    echo "[Step 6/10] ⚠️ SKIPPED (--cpu-only) — Embedding index"
fi

# ── Step 7: 构建测试集 ──────────────────────────────────
echo ""
echo "[Step 7/10] Building test set (50 questions)..."
python evaluation/test_set_builder.py
echo "✓ Test set built → data/test_set.json"

# ── Step 8: Contextual Enrichment（可选，需 API Key） ───
if [ -n "$DEEPSEEK_API_KEY" ]; then
    echo ""
    echo "[Step 8/10] Running contextual enrichment..."
    python indexing/contextual_enrichment.py
    echo "✓ Enrichment complete → data/chunks/chunks_default_enriched.jsonl"
else
    echo ""
    echo "[Step 8/10] ⚠️ SKIPPED — DEEPSEEK_API_KEY not set"
    echo "    Set with: export DEEPSEEK_API_KEY='your-key'"
fi

# ── Step 9: ⚠️ GPU — 实验评测 ────────────────────────
if [ "$1" != "--cpu-only" ]; then
    echo ""
    echo "[Step 9/10] ⚠️ GPU REQUIRED — Running experiments..."
    echo "    Running baseline only (R0,R1,R2). For full matrix, run:"
    echo "    python evaluation/run_experiments.py --all"
    python evaluation/run_experiments.py --baseline
    echo "✓ Experiments complete → results/"
else
    echo ""
    echo "[Step 9/10] ⚠️ SKIPPED (--cpu-only) — GPU experiments"
fi

# ── Step 10: 启动 Demo ───────────────────────────────
echo ""
echo "[Step 10/10] Starting Gradio demo..."
echo "=============================================="
echo " Pipeline complete!"
echo " Starting demo at http://localhost:7860"
echo "=============================================="
python demo/app.py
