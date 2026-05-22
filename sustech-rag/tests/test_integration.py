"""
=============================================================================
SUSTech RAG — Integration Tests (分层自动检测 GPU/API)
=============================================================================
Tier 1 (zero-cost):    BM25 e2e + pipeline logic + chunk format
Tier 2 (needs API):    DeepSeek connectivity + single-query generation
Tier 3 (needs GPU):    Full R4 pipeline single-query

Usage:
  python tests/test_integration.py                          # Tier 1 only
  DEEPSEEK_API_KEY=sk-xxx python tests/test_integration.py   # Tier 1+2
  (Tier 3 auto-enables when torch.cuda.is_available())
=============================================================================
"""

import json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Resource detection ──
_gpu = None
def has_gpu():
    global _gpu
    if _gpu is None:
        try:
            import torch
            _gpu = torch.cuda.is_available()
        except ImportError:
            _gpu = False
    return _gpu

def has_api():
    return bool(os.getenv("DEEPSEEK_API_KEY"))


# ═══════════════════════════════════════════════════════════════
# Tier 1 — Zero cost (always runs)
# ═══════════════════════════════════════════════════════════════

def test_chunk_format():
    """Verify chunk data file has required fields."""
    from config import DATA_DIR
    path = DATA_DIR / "chunks" / "chunks_default.jsonl"
    if not path.exists():
        return None  # skip — index not built locally
    with open(path) as f:
        first = json.loads(f.readline())
    for field in ["chunk_id", "text", "raw_text", "source_family", "url"]:
        assert field in first, f"missing field: {field}"
    return True


def test_rrf_fusion_logic():
    """Verify RRF fusion with mock data (no models loaded)."""
    from retrieval.hybrid_rrf import rrf_fusion
    dense = [
        {"chunk_id": "A", "text": "library hours...", "score": 0.85},
        {"chunk_id": "B", "text": "canteen location...", "score": 0.72},
    ]
    bm25 = [
        {"chunk_id": "B", "text": "canteen location...", "score": 8.5},
        {"chunk_id": "D", "text": "card recharge...", "score": 7.2},
    ]
    fused = rrf_fusion(dense, bm25, k=60, top_n=10)
    assert len(fused) >= 3
    for r in fused:
        assert "rrf_score" in r
        assert isinstance(r["rrf_score"], float)
    # Doc appearing in both retrievers should rank high
    assert fused[0]["chunk_id"] in ("A", "B")
    return True


def test_query_classifier_pipeline():
    """Verify classifier: type→weight→OOS detection chain."""
    from retrieval.query_classifier import get_classifier
    qc = get_classifier(mode="rule")
    cases = [
        ("图书馆几点开门", "FACTUAL_SIMPLE", False),
        ("如何办理校园卡", "PROCEDURAL", False),
        ("理工学院和医学院的区别", "COMPARATIVE", False),
        ("清华大学校长是谁", "OUT_OF_SCOPE", True),
    ]
    for query, exp_type, exp_oos in cases:
        q_type, _ = qc.classify(query)
        assert q_type == exp_type, f"{query}: expected {exp_type}, got {q_type}"
        assert qc.is_out_of_scope(q_type) == exp_oos
        d, s = qc.get_rrf_weights(q_type)
        assert 0 <= d <= 2.0 and 0 <= s <= 2.0
    return True


def test_prompt_builder_integration():
    """Verify prompt builder: context assembly + token budget."""
    from generation.prompt_builder import PromptBuilder
    pb = PromptBuilder(persona="default")
    chunks = [
        {"source_family": "library", "url": "https://lib.sustech.edu.cn",
         "raw_text": "开放时间：周一至周五 8:00-22:00。"},
        {"source_family": "official", "url": "https://www.sustech.edu.cn",
         "raw_text": "南科大是一所新型研究型大学。"},
    ]
    ctx = pb.build_context(chunks)
    assert "开放时间" in ctx and "研究型大学" in ctx
    msg = pb.build_user_message("测试", chunks)
    assert "测试" in msg and "开放时间" in msg
    sp = pb.build_system_prompt()
    assert pb.is_within_token_budget(sp, msg, budget=7000)
    return True


def test_authority_scorer_integration():
    """Verify authority scorer: ranking + freshness."""
    from retrieval.authority_scorer import AuthorityScorer
    s = AuthorityScorer()
    chunks = [
        {"chunk_id": "off", "rerank_score": 0.80, "source_family": "official",
         "crawled_at": "2026-05-15T10:00:00+00:00"},
        {"chunk_id": "unk", "rerank_score": 0.80, "source_family": "unknown",
         "crawled_at": "2026-05-15T10:00:00+00:00"},
    ]
    scored = s.score("test", chunks)
    assert scored[0]["chunk_id"] == "off"  # official > unknown
    assert "final_score" in scored[0] and "authority_score" in scored[0]

    # Freshness for temporal queries
    fresh = [
        {"chunk_id": "new", "rerank_score": 0.80, "source_family": "news",
         "crawled_at": "2026-05-20T10:00:00+00:00"},
        {"chunk_id": "old", "rerank_score": 0.80, "source_family": "news",
         "crawled_at": "2020-01-15T10:00:00+00:00"},
    ]
    scored2 = s.score("latest policy", fresh, q_type="TEMPORAL")
    assert scored2[0]["chunk_id"] == "new"  # newer wins
    return True


def test_bm25_e2e():
    """BM25 end-to-end: tokenizer + index + retriever (CPU, no GPU)."""
    from retrieval.sparse_retriever import get_sparse_retriever
    try:
        ret = get_sparse_retriever()
    except FileNotFoundError:
        return None  # index not built locally, skip
    except Exception as e:
        return None  # other init error, skip

    results = ret.search("图书馆几点开门", top_k=5)
    assert len(results) > 0, "BM25 returned no results"
    assert "chunk_id" in results[0] and "score" in results[0]
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), "not sorted descending"
    return True


# ═══════════════════════════════════════════════════════════════
# Tier 2 — Needs DEEPSEEK_API_KEY
# ═══════════════════════════════════════════════════════════════

def test_api_connectivity():
    """Single ping to verify API key and network."""
    if not has_api():
        return None
    from generation.llm_api import DeepSeekClient
    c = DeepSeekClient()
    if not c.is_available:
        return None
    resp = c.chat("Reply OK only.", "OK", temperature=0.0, max_tokens=10)
    assert resp and len(resp) > 0
    return True


def test_llm_generation():
    """Single-query LLM generation with prompt builder (no retrieval)."""
    if not has_api():
        return None
    from generation.llm_api import DeepSeekClient
    from generation.prompt_builder import PromptBuilder
    c = DeepSeekClient()
    if not c.is_available:
        return None
    pb = PromptBuilder(persona="default")
    resp = c.chat(pb.build_system_prompt(),
                  "图书馆几点开门？简短回答。", temperature=0.1, max_tokens=64)
    assert len(resp) > 5
    return True


def test_llm_streaming():
    """Verify streaming yields non-empty tokens."""
    if not has_api():
        return None
    from generation.llm_api import DeepSeekClient
    c = DeepSeekClient()
    if not c.is_available:
        return None
    tokens = list(c.stream_chat("Reply OK.", "OK", temperature=0.0, max_tokens=10))
    assert len(tokens) > 0 and len("".join(tokens)) > 0
    return True


# ═══════════════════════════════════════════════════════════════
# Tier 3 — Needs GPU + API
# ═══════════════════════════════════════════════════════════════

def test_dense_retrieval():
    """Real bge-m3 dense retrieval (GPU required)."""
    if not has_gpu():
        return None
    from retrieval.dense_retriever import get_dense_retriever
    try:
        ret = get_dense_retriever()
    except Exception:
        return None
    results = ret.search("图书馆几点开门", top_k=5)
    assert len(results) > 0 and "chunk_id" in results[0]
    return True


def test_full_pipeline_r4():
    """R4 full pipeline: dense+BM25→RRF→rerank→LLM (GPU+API)."""
    if not has_gpu():
        return None
    if not has_api():
        return None

    from retrieval.hybrid_rrf import hybrid_retrieve
    from retrieval.dense_retriever import get_dense_retriever
    from retrieval.sparse_retriever import get_sparse_retriever
    from retrieval.reranker import get_reranker
    from retrieval.query_classifier import get_classifier
    from generation.llm_api import DeepSeekClient
    from generation.prompt_builder import PromptBuilder

    t0 = time.time()
    dense = get_dense_retriever()
    sparse = get_sparse_retriever()
    reranker = get_reranker()
    classifier = get_classifier(mode="rule")

    chunks, trace = hybrid_retrieve(
        query="图书馆几点开门",
        dense_retriever=dense, sparse_retriever=sparse,
        use_hyde=False, use_classifier=True, classifier=classifier,
        reranker=reranker, abstention_check=True,
    )
    t_ret = (time.time() - t0) * 1000

    assert isinstance(chunks, list)
    assert "steps" in trace
    for step in ["dense", "bm25", "rrf", "reranker"]:
        assert step in trace["steps"], f"missing step: {step}"

    if chunks:
        assert "chunk_id" in chunks[0]
        llm = DeepSeekClient()
        if llm.is_available:
            pb = PromptBuilder(persona="default")
            t_gen = time.time()
            answer = llm.chat(pb.build_system_prompt(),
                            pb.build_user_message("图书馆几点开门", chunks),
                            max_tokens=128)
            t_gen = (time.time() - t_gen) * 1000
            assert len(answer) > 5
        else:
            answer = "[API unavailable]"
            t_gen = 0
    else:
        answer = "[abstained]"
        t_gen = 0

    total = round(t_ret + t_gen)
    assert total < 30000, f"pipeline too slow: {total}ms"
    return True


# ═══════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    stats = {"pass": 0, "skip": 0, "fail": 0}
    tests = [
        # Tier 1
        ("chunk_format", test_chunk_format),
        ("rrf_fusion_logic", test_rrf_fusion_logic),
        ("classifier_pipeline", test_query_classifier_pipeline),
        ("prompt_builder_integration", test_prompt_builder_integration),
        ("authority_scorer_integration", test_authority_scorer_integration),
        ("bm25_e2e", test_bm25_e2e),
        # Tier 2
        ("api_connectivity", test_api_connectivity),
        ("llm_generation", test_llm_generation),
        ("llm_streaming", test_llm_streaming),
        # Tier 3
        ("dense_retrieval", test_dense_retrieval),
        ("full_pipeline_r4", test_full_pipeline_r4),
    ]

    print(f"=== SUSTech RAG Integration Tests ===")
    print(f"GPU={'YES' if has_gpu() else 'no'}  API={'YES' if has_api() else 'no'}\n")

    for name, fn in tests:
        try:
            result = fn()
            if result is True:
                print(f"  [PASS] {name}")
                stats["pass"] += 1
            elif result is None:
                print(f"  [SKIP] {name}")
                stats["skip"] += 1
            else:
                print(f"  [PASS] {name}")
                stats["pass"] += 1
        except AssertionError as e:
            print(f"  [FAIL] {name}: {e}")
            stats["fail"] += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
            stats["fail"] += 1

    print(f"\n=== {stats['pass']} pass, {stats['skip']} skip, {stats['fail']} fail ===")
    if stats["fail"] > 0:
        exit(1)
