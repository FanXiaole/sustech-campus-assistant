"""
=============================================================================
SUSTech BM25 Index Builder — Jieba 分词 + BM25Okapi 稀疏索引
=============================================================================
BM25 是经典的信息检索算法，在我们的 RRF（Reciprocal Rank Fusion）
中作为稀疏检索信号，与稠密检索信号互补。

BM25 的优势场景：
  - 精确关键词（"图书馆" ← 不会模糊匹配成"图书室"）
  - 专有名词（"计算机科学与工程系" ← embedding 可能理解偏差）
  - 数字和时间（"2026年招生" ← 数字在语义空间里很难准确对齐）

实现：jieba 分词 → 去停用词 → BM25Okapi(k1=1.5, b=0.75) → 本地文件保存

注意：BM25 索引使用本地文件存储（非网络传输），安全可控。
      但建议同时保存一份 JSON 格式的元数据，方便检查。

使用方法：python indexing/bm25_builder.py
=============================================================================
"""

import json
import pickle
import time
from pathlib import Path

# ============================================================================
# 配置导入
# ============================================================================
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CHUNK_DIR, INDEX_DIR

# 共享的分词模块
from indexing.tokenizer import load_stopwords, tokenize

# ============================================================================
# BM25 索引
# ============================================================================

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("ERROR: rank-bm25 not installed. Run: pip install rank-bm25")
    raise


# ============================================================================
# 索引构建
# ============================================================================

def build_bm25_index(
    chunks_path: Path = None,
    output_dir: Path = None,
) -> dict:
    """构建 BM25Okapi 索引，保存到磁盘。"""
    if chunks_path is None:
        chunks_path = CHUNK_DIR / "chunks_default.jsonl"
    if output_dir is None:
        output_dir = INDEX_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"BM25 Index Builder")
    print(f"{'='*60}")
    print(f"Input:  {chunks_path}")

    # ── 加载 chunks ──
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    print(f"Loaded {len(chunks):,} chunks")

    # ── 加载停用词 ──
    stopwords = load_stopwords()

    # ── 分词 ──
    print("Tokenizing with jieba...")
    t_start = time.time()

    chunk_ids = []
    tokenized_corpus = []
    skipped = 0

    for chunk in chunks:
        raw_text = chunk.get("raw_text", chunk.get("text", ""))
        tokens = tokenize(raw_text, stopwords)
        if len(tokens) < 3:
            skipped += 1
            continue
        tokenized_corpus.append(tokens)
        chunk_ids.append(chunk["chunk_id"])

    elapsed = time.time() - t_start
    print(f"Tokenized {len(tokenized_corpus):,} chunks in {elapsed:.1f}s "
          f"({len(tokenized_corpus)/elapsed:.0f} chunks/s)")
    if skipped:
        print(f"Skipped {skipped} near-empty chunks")

    # ── 构建 BM25 ──
    print("Building BM25Okapi (k1=1.5, b=0.75)...")
    t_start = time.time()

    bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)

    elapsed = time.time() - t_start
    print(f"BM25 built in {elapsed:.1f}s")

    # ── 保存（JSON 元数据 + 单独保存 BM25 索引对象）────
    # 分开保存的原因：JSON 可读可检查，BM25 对象需要 pickle
    meta_path = output_dir / "bm25_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "chunk_ids": chunk_ids,
            "chunk_count": len(chunks),
            "token_count": sum(len(t) for t in tokenized_corpus),
            "config": {"k1": 1.5, "b": 0.75, "source": str(chunks_path)},
        }, f, ensure_ascii=False, indent=2)

    # BM25 对象需要 pickle（rank_bm25 库的限制）
    index_path = output_dir / "bm25_index.pkl"
    with open(index_path, "wb") as f:
        pickle.dump({
            "index": bm25,
            "chunk_ids": chunk_ids,
            "chunks": chunks,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = index_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved: {index_path} ({size_mb:.1f} MB)")
    print(f"       {meta_path}")

    return {"bm25": bm25, "chunk_ids": chunk_ids, "chunks": chunks}


# ============================================================================
# 验证
# ============================================================================

def verify(index_data: dict = None, index_path: Path = None):
    """用 3 个测试查询验证 BM25 索引。"""
    if index_data is None:
        if index_path is None:
            index_path = INDEX_DIR / "bm25_index.pkl"
        import pickle
        with open(index_path, "rb") as f:
            index_data = pickle.load(f)

    bm25 = index_data["index"]
    chunks = index_data["chunks"]
    stopwords = load_stopwords()

    test_queries = ["图书馆开放时间", "计算机系", "How to borrow books"]

    print(f"\n{'='*60}")
    print("BM25 Verification Queries")
    print(f"{'='*60}")

    for query in test_queries:
        query_tokens = tokenize(query, stopwords)
        print(f"\nQuery: {query}  →  Tokens: {query_tokens}")

        scores = bm25.get_scores(query_tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]

        for rank, idx in enumerate(top_idx, 1):
            preview = chunks[idx].get("raw_text", "")[:120].replace("\n", " ")
            print(f"  #{rank} [BM25={scores[idx]:.4f}] {preview}...")

    print(f"\n{'='*60}\n")


# ============================================================================
if __name__ == "__main__":
    data = build_bm25_index()
    verify(data)
