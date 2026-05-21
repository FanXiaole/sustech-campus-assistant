"""
=============================================================================
SUSTech Sparse Retriever — BM25 检索器封装
=============================================================================
封装 BM25Okapi 索引的加载和检索，提供与 DenseRetriever 一致的接口，
方便在 hybrid_rrf.py 中统一调用。

使用方法：
  from retrieval.sparse_retriever import SparseRetriever
  retriever = SparseRetriever()
  results = retriever.search("图书馆几点关门", top_k=50)
=============================================================================
"""

import pickle
import time
from pathlib import Path

# ============================================================================
# 配置导入
# ============================================================================
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BM25_TOP_K, INDEX_DIR
from indexing.tokenizer import load_stopwords, tokenize


class SparseRetriever:
    """
    BM25 稀疏检索器。

    设计原则：与 DenseRetriever 保持相同的接口
    - search(query, top_k) → list[dict]
    - 返回格式中每个元素包含 chunk_id, text, score, source_family, url 等

    这样 hybrid_rrf.py 可以无差别地调用两个检索器。
    """

    def __init__(self, index_path: Path = None):
        if index_path is None:
            index_path = INDEX_DIR / "bm25_index.pkl"

        print(f"[SparseRetriever] Loading BM25 index from {index_path}...")
        t_start = time.time()

        with open(index_path, "rb") as f:
            pkg = pickle.load(f)

        self.bm25 = pkg["index"]
        self.chunks = pkg["chunks"]
        self.chunk_ids = pkg["chunk_ids"]

        elapsed = time.time() - t_start
        print(f"[SparseRetriever] Loaded {len(self.chunks):,} chunks in {elapsed:.1f}s")

        # 延迟加载停用词
        self._stopwords = None

    @property
    def stopwords(self):
        if self._stopwords is None:
            self._stopwords = load_stopwords()
        return self._stopwords

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[dict]:
        """
        BM25 检索。

        流程：查询分词 → 过滤停用词 → get_scores → 排序 → 返回 top_k
        """
        query_tokens = tokenize(query, self.stopwords)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for idx in top_indices:
            score = scores[idx]
            if score <= 0:
                continue
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk.get("chunk_id", ""),
                "text": chunk.get("raw_text", chunk.get("text", "")),
                "score": float(score),
                "source_family": chunk.get("source_family", "unknown"),
                "url": chunk.get("url", ""),
                "heading_path": chunk.get("heading_path", ""),
            })
        return results


# 单例
_sparse_instance: SparseRetriever | None = None


def get_sparse_retriever() -> SparseRetriever:
    global _sparse_instance
    if _sparse_instance is None:
        _sparse_instance = SparseRetriever()
    return _sparse_instance


if __name__ == "__main__":
    r = get_sparse_retriever()
    for q in ["图书馆开放时间", "计算机系", "校园卡充值"]:
        print(f"\nQuery: {q}")
        for rank, res in enumerate(r.search(q, top_k=3), 1):
            print(f"  #{rank} [{res['score']:.4f}] {res['text'][:100]}...")
