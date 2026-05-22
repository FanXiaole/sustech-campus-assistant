#!/bin/bash
# =============================================================================
# SUSTech RAG — End-to-End Pipeline Runner
# Usage: bash scripts/run_all.sh [--skip-scrape] [--skip-index] [--experiments R0,R4]
# =============================================================================
set -e
cd "$(dirname "$0")/.."

echo "=== SUSTech RAG — Full Pipeline ==="
echo ""

# Check API key
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "ERROR: DEEPSEEK_API_KEY not set. Run: export DEEPSEEK_API_KEY='sk-...'"
    exit 1
fi

# Phase 1: Scraping + Cleaning (CPU)
if [[ ! "$*" =~ --skip-scrape ]]; then
    echo "[Phase 1] Data Pipeline..."
    python indexing/scraper.py
    python indexing/cleaner.py
    python indexing/chunker.py
fi

# Phase 2: Indexing (GPU)
if [[ ! "$*" =~ --skip-index ]]; then
    echo "[Phase 2] Building Indexes..."
    python indexing/bm25_builder.py
    python indexing/embedder.py
fi

# Phase 3: Experiments
echo "[Phase 3] Running Experiments..."
EXPERIMENTS="${2:-R0,R1,R2,R3,R4,E1,E2,E3,E4,E5,A1,A2}"
python evaluation/run_experiments.py --experiments "$EXPERIMENTS"

# Phase 4: Bootstrap CI
echo "[Phase 4] Bootstrap CI..."
python evaluation/run_bootstrap.py

echo ""
echo "=== Pipeline Complete ==="
echo "Results saved to results/"
echo "Start demo: python demo/app.py"
