"""
=============================================================================
Hybrid RRF Retriever — Reciprocal Rank Fusion 核心融合器 ★ 核心创新
=============================================================================
这是整个检索系统的"心脏"。RRF（Reciprocal Rank Fusion）将稠密检索（语义）
和稀疏检索（关键词）的结果智能地融合在一起。

为什么需要 RRF？—— Score Magnitude Incomparability 问题
  Dense Retriever 输出的分数是余弦相似度，范围 [0, 1]
  BM25 输出的分数是无界的，范围 [0, +∞)
  这两个分数在数值上完全不可比。5.2 的 BM25 分数和 0.8 的余弦相似度
  哪个更好？没有答案。

RRF 的解决方案：不看分数的绝对值，只看排名。
  score_RRF(d) = Σᵢ wᵢ / (k + rankᵢ(d))

  k = 60：平滑常数，防止 rank=1 的文档主导一切
  wᵢ：动态权重（来自 Query Classifier）

为什么 k=60？
  → Cormack et al. (2009) 的 TREC 实验验证：k=60 在各种数据集上
    表现最稳定。k 太小 → rank=1 主导（退化为 max）；k 太大 → 所有
    文档分数接近（退化为平均）。

RRF vs 简单的加权求和：
  RRF 是 rank-based → 对不同分数分布鲁棒
  加权求和是 score-based → 需要分数归一化（min-max / z-score）
  RRF 不需要任何归一化 → 更简单、更稳定

使用方法：
  from retrieval.hybrid_rrf import hybrid_retrieve
  results, trace = hybrid_retrieve(query, dense_retriever, sparse_retriever)

=============================================================================
"""

import time
from pathlib import Path
from typing import Any

# ============================================================================
# 配置导入
# ============================================================================
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    ABSTENTION_THRESHOLD,
    BM25_TOP_K,
    DENSE_TOP_K,
    RERANK_TOP_K,
    RRF_FUSION_TOP,
    RRF_K,
)


# ============================================================================
# RRF 核心算法
# ============================================================================

def rrf_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    k: int = RRF_K,
    top_n: int = RRF_FUSION_TOP,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> list[dict]:
    """
    Reciprocal Rank Fusion 核心算法。

    算法步骤：
    1. 建立两个检索器中的 chunk_id → rank 映射
    2. 对每个唯一的 chunk_id，计算 RRF 分数
    3. 按 RRF 分数降序排列
    4. 取 top-N
    5. 保留稠密检索结果中的额外元数据

    参数：
        dense_results: 稠密检索结果列表 [{chunk_id, score, ...}, ...]
        bm25_results: BM25 检索结果列表 [{chunk_id, score, ...}, ...]
        k: RRF 平滑常数（默认 60）
        top_n: 融合后的候选数（默认 20，作为 Reranker 的输入）
        dense_weight: 稠密检索的融合权重
        sparse_weight: 稀疏检索的融合权重

    返回：
        融合后的结果列表，每个元素包含：
        {chunk_id, text, rrf_score, dense_rank, bm25_rank,
         dense_score, bm25_score, source_family, url, heading_path}

    为什么默认返回 top-20 而不是 top-5？
    → 后面还有 Reranker（Cross-Encoder），它需要足够的候选池来
      发挥重排序能力。20 个候选 ≈ 在召回率和精度之间平衡。
    """
    # ── 单次遍历建立 rank + data 映射 ──
    dense_ranks: dict[str, int] = {}
    dense_data: dict[str, dict] = {}
    for rank, result in enumerate(dense_results, 1):
        chunk_id = result.get("chunk_id", "")
        if chunk_id:
            dense_ranks[chunk_id] = rank
            dense_data[chunk_id] = result

    bm25_ranks: dict[str, int] = {}
    bm25_data: dict[str, dict] = {}
    for rank, result in enumerate(bm25_results, 1):
        chunk_id = result.get("chunk_id", "")
        if chunk_id:
            bm25_ranks[chunk_id] = rank
            bm25_data[chunk_id] = result

    # ── 计算每个文档的 RRF 分数 ──
    # 收集两个检索器的所有文档（取并集）
    all_chunk_ids = set(dense_ranks.keys()) | set(bm25_ranks.keys())

    rrf_scores: dict[str, float] = {}
    for chunk_id in all_chunk_ids:
        score = 0.0

        # Dense contribution
        if chunk_id in dense_ranks:
            dense_rank = dense_ranks[chunk_id]
            score += dense_weight / (k + dense_rank)

        # Sparse (BM25) contribution
        if chunk_id in bm25_ranks:
            bm25_rank = bm25_ranks[chunk_id]
            score += sparse_weight / (k + bm25_rank)

        rrf_scores[chunk_id] = score

    # ── 按 RRF 分数降序排列 ──
    sorted_ids = sorted(
        rrf_scores.keys(),
        key=lambda cid: rrf_scores[cid],
        reverse=True,
    )[:top_n]

    # ── 格式化输出 ──
    fused_results = []
    for chunk_id in sorted_ids:
        dense_info = dense_data.get(chunk_id, {})
        bm25_info = bm25_data.get(chunk_id, {})

        # 优先从稠密结果中取元数据（稠密结果通常更完整）
        primary_info = dense_info if dense_info else bm25_info

        fused_results.append({
            "chunk_id": chunk_id,
            "text": primary_info.get("text", ""),
            "rrf_score": rrf_scores[chunk_id],
            "dense_rank": dense_ranks.get(chunk_id, None),
            "bm25_rank": bm25_ranks.get(chunk_id, None),
            "dense_score": dense_info.get("score", None),
            "bm25_score": bm25_info.get("score", None),
            "source_family": primary_info.get("source_family", "unknown"),
            "url": primary_info.get("url", ""),
            "heading_path": primary_info.get("heading_path", ""),
        })

    return fused_results


# ============================================================================
# 完整 Hybrid Pipeline
# ============================================================================

def hybrid_retrieve(
    query: str,
    dense_retriever,        # DenseRetriever instance
    sparse_retriever,       # SparseRetriever instance
    use_hyde: bool = False,
    hyde_llm_fn=None,       # Callable for HyDE
    use_classifier: bool = True,
    classifier=None,        # QueryClassifier instance
    reranker=None,          # Reranker instance
    authority_scorer=None,  # AuthorityScorer instance
    abstention_check: bool = True,
    top_k_dense: int = DENSE_TOP_K,
    top_k_bm25: int = BM25_TOP_K,
    top_n_rrf: int = RRF_FUSION_TOP,
    top_k_final: int = RERANK_TOP_K,
) -> tuple[list[dict], dict]:
    """
    完整的 Hybrid RAG 检索管线。

    这是项目的核心流程。每一步都会记录到 pipeline_trace 中，
    供 Gradio Demo 的 Pipeline Inspector 面板展示。

    管线流程：
    ┌──────────────────────────────────────────────────────┐
    │ 1. [可选] Query Classifier → 路由 + 权重             │
    │ 2. [可选] HyDE → 假设文档 → 嵌入                    │
    │ 3. Dense Retrieval → top-50                         │
    │ 4. BM25 Retrieval → top-50                          │
    │ 5. ★ RRF Fusion → top-20                            │
    │ 6. [可选] Reranker → top-5                          │
    │ 7. [可选] Authority Scoring → 微调排序              │
    │ 8. [可选] Confidence Check → abstention             │
    └──────────────────────────────────────────────────────┘

    参数：
        query: 用户查询文本
        dense_retriever: DenseRetriever 实例
        sparse_retriever: SparseRetriever 实例
        use_hyde: 是否启用 HyDE
        hyde_llm_fn: HyDE 所需的 LLM 函数
        use_classifier: 是否启用查询分类
        classifier: QueryClassifier 实例
        reranker: Reranker 实例（None = 跳过重排序）
        authority_scorer: AuthorityScorer 实例（None = 跳过）
        abstention_check: 是否启用置信度检查
        top_k_dense: 稠密检索召回数
        top_k_bm25: BM25 检索召回数
        top_n_rrf: RRF 融合保留数
        top_k_final: 最终返回的 chunk 数

    返回：
        (最终 chunk 列表, pipeline_trace 字典)
    """
    trace = {
        "query": query,
        "steps": {},
        "total_ms": 0,
    }
    t_pipeline_start = time.time()

    # ── 步骤 0: 查询分类（可选） ──
    dense_weight = 1.0
    sparse_weight = 1.0
    q_type = "FACTUAL_COMPLEX"  # 默认类型

    if use_classifier and classifier is not None:
        t0 = time.time()
        q_type, cls_confidence = classifier.classify(query)
        dense_weight, sparse_weight = classifier.get_rrf_weights(q_type)

        trace["steps"]["classification"] = {
            "type": q_type,
            "confidence": cls_confidence,
            "dense_weight": dense_weight,
            "sparse_weight": sparse_weight,
            "time_ms": round((time.time() - t0) * 1000),
        }

        # OUT_OF_SCOPE → 立即返回空结果
        if classifier.is_out_of_scope(q_type):
            trace["total_ms"] = round((time.time() - t_pipeline_start) * 1000)
            return [], trace

    # ── 步骤 1: HyDE（可选） ──
    hyde_result = None
    if use_hyde and hyde_llm_fn is not None:
        # 动态 import HyDE（避免循环依赖）
        from retrieval.hyde import hyde_retrieve

        t0 = time.time()
        # HyDE 需要 embed_fn 和 search_fn
        hyde_result, hyde_trace = hyde_retrieve(
            query=query,
            llm_fn=hyde_llm_fn,
            embed_fn=lambda text: dense_retriever.encode_query(text, use_instruction=True),
            dense_search_fn=lambda emb, k: dense_retriever.search_by_embedding(emb, top_k=k),
            top_k=top_k_dense,
        )
        trace["steps"]["hyde"] = hyde_trace
        trace["steps"]["hyde"]["time_ms"] = round((time.time() - t0) * 1000)

    # ── 步骤 2: Dense Retrieval ──
    t0 = time.time()
    if hyde_result is not None:
        # 使用 HyDE 的假设文档 embedding
        dense_results = hyde_result
    else:
        dense_results = dense_retriever.search(query, top_k=top_k_dense)
    trace["steps"]["dense"] = {
        "time_ms": round((time.time() - t0) * 1000),
        "top_3": [
            {"id": r["chunk_id"][:16], "score": round(r["score"], 4)}
            for r in dense_results[:3]
        ],
    }

    # ── 步骤 3: BM25 Retrieval ──
    t0 = time.time()
    bm25_results = sparse_retriever.search(query, top_k=top_k_bm25)
    trace["steps"]["bm25"] = {
        "time_ms": round((time.time() - t0) * 1000),
        "top_3": [
            {"id": r["chunk_id"][:16], "score": round(r["score"], 4)}
            for r in bm25_results[:3]
        ],
    }

    # ── 步骤 4: RRF Fusion ──
    t0 = time.time()
    fused_results = rrf_fusion(
        dense_results=dense_results,
        bm25_results=bm25_results,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        top_n=top_n_rrf,
    )
    trace["steps"]["rrf"] = {
        "time_ms": round((time.time() - t0) * 1000),
        "top_3": [
            {"id": r["chunk_id"][:16], "rrf_score": round(r["rrf_score"], 4),
             "dense_rank": r["dense_rank"], "bm25_rank": r["bm25_rank"]}
            for r in fused_results[:3]
        ],
    }

    # ── 步骤 5: Reranker（可选） ──
    if reranker is not None and len(fused_results) > 0:
        t0 = time.time()
        reranked_results = reranker.rerank(query, fused_results, top_k=top_k_final)
        trace["steps"]["reranker"] = {
            "time_ms": round((time.time() - t0) * 1000),
            "top_3": [
                {"id": r["chunk_id"][:16], "rerank_score": round(r.get("rerank_score", 0), 4)}
                for r in reranked_results[:3]
            ],
        }
    else:
        reranked_results = fused_results[:top_k_final]
        # 把 RRF 分数复制为 rerank_score（保持格式一致）
        for r in reranked_results:
            r["rerank_score"] = r.get("rrf_score", 0)
        trace["steps"]["reranker"] = {"skipped": True}

    # ── 步骤 6: Authority Scoring（可选） ──
    if authority_scorer is not None and len(reranked_results) > 0:
        t0 = time.time()
        scored_results = authority_scorer.score(
            query, reranked_results, q_type=q_type
        )
        trace["steps"]["authority"] = {
            "time_ms": round((time.time() - t0) * 1000),
        }
        final_results = scored_results
    else:
        final_results = reranked_results
        trace["steps"]["authority"] = {"skipped": True}

    # ── 步骤 7: Confidence Check ──
    if abstention_check and final_results:
        max_rrf = max(
            (r.get("rrf_score", 0) for r in final_results),
            default=0,
        )
        should_abstain = max_rrf < ABSTENTION_THRESHOLD
        trace["confidence"] = {
            "max_rrf_score": max_rrf,
            "threshold": ABSTENTION_THRESHOLD,
            "should_abstain": should_abstain,
        }
        # 分数过低 → 返回空结果，触发拒答
        if should_abstain:
            trace["total_ms"] = round((time.time() - t_pipeline_start) * 1000)
            return [], trace
    elif abstention_check:
        trace["confidence"] = {
            "max_rrf_score": 0,
            "threshold": ABSTENTION_THRESHOLD,
            "should_abstain": True,
        }
        trace["total_ms"] = round((time.time() - t_pipeline_start) * 1000)
        return [], trace

    # ── 总结 ──
    trace["total_ms"] = round((time.time() - t_pipeline_start) * 1000)
    trace["result_count"] = len(final_results)
    trace["query_type"] = q_type

    return final_results, trace


# ============================================================================
# 独立测试（不需要完整 pipeline）
# ============================================================================
if __name__ == "__main__":
    # 模拟检索结果来做 RRF 单元测试
    dense_mock = [
        {"chunk_id": "A", "text": "图书馆服务时间...", "score": 0.85},
        {"chunk_id": "B", "text": "食堂位置...", "score": 0.72},
        {"chunk_id": "C", "text": "选课流程...", "score": 0.65},
    ]
    bm25_mock = [
        {"chunk_id": "B", "text": "食堂位置...", "score": 8.5},
        {"chunk_id": "D", "text": "校园卡充值...", "score": 7.2},
        {"chunk_id": "A", "text": "图书馆服务时间...", "score": 5.1},
    ]

    print("=" * 60)
    print("RRF Fusion Unit Test")
    print("=" * 60)
    print("\nDense results (mock):")
    for r in dense_mock:
        print(f"  {r['chunk_id']}: {r['text'][:40]} (score={r['score']})")
    print("\nBM25 results (mock):")
    for r in bm25_mock:
        print(f"  {r['chunk_id']}: {r['text'][:40]} (score={r['score']})")

    fused = rrf_fusion(dense_mock, bm25_mock, top_n=10)

    print("\nRRF Fusion results:")
    for i, r in enumerate(fused):
        print(f"  #{i+1}: {r['chunk_id']} | RRF={r['rrf_score']:.4f} | "
              f"Dense rank={r['dense_rank']} | BM25 rank={r['bm25_rank']} | "
              f"{r['text'][:50]}...")
    print("=" * 60)
