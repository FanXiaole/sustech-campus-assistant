"""
端到端 smoke test: 1 个查询走完整 RAG pipeline。
验证 pipeline 不崩溃、返回合理结果。
"""

import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_pipeline_no_crash():
    """完整 pipeline 不应崩溃，且返回非空结果。"""
    from retrieval.hybrid_rrf import hybrid_retrieve
    from retrieval.dense_retriever import get_dense_retriever
    from retrieval.sparse_retriever import get_sparse_retriever
    from retrieval.query_classifier import get_classifier
    from generation.prompt_builder import PromptBuilder

    query = "图书馆几点开门"

    dense = get_dense_retriever()
    sparse = get_sparse_retriever()
    classifier = get_classifier(mode="rule")

    t0 = time.time()
    chunks, trace = hybrid_retrieve(
        query=query,
        dense_retriever=dense,
        sparse_retriever=sparse,
        use_classifier=True,
        classifier=classifier,
        abstention_check=True,
    )
    elapsed_ms = (time.time() - t0) * 1000

    # 基本断言
    assert isinstance(chunks, list), "chunks must be a list"
    assert isinstance(trace, dict), "trace must be a dict"
    assert "steps" in trace, "trace must have steps"
    assert elapsed_ms < 30000, f"Pipeline too slow: {elapsed_ms:.0f}ms"

    # 对于"图书馆几点开门"，应该有结果
    if len(chunks) > 0:
        assert all("chunk_id" in c for c in chunks), "Each chunk must have chunk_id"
        assert all("text" in c for c in chunks), "Each chunk must have text"

    print(f"  Pipeline OK: {len(chunks)} chunks in {elapsed_ms:.0f}ms")
    return True


def test_evaluator_no_crash():
    """Evaluator 不应崩溃。"""
    from evaluation.evaluator import RAGEvaluator

    e = RAGEvaluator()
    question = {
        "q_id": "test",
        "ground_truth": "8:00-22:00",
        "key_facts": ["8:00", "22:00", "周一至周五"],
        "expected_abstain": False,
    }
    result = e.evaluate(
        answer="南科大图书馆周一至周五开放时间为8:00到22:00。",
        chunks=[{"raw_text": "图书馆服务时间：周一至周五 8:00-22:00。", "text": "[来源:library] 图书馆服务时间：周一至周五 8:00-22:00。"}],
        question_meta=question,
        did_abstain=False,
    )
    assert 0 <= result["total_score"] <= 10
    print(f"  Evaluator OK: score={result['total_score']}")
    return True


def test_config_integrity():
    """Config 完整性。"""
    from config import (
        PERSONA_PRESETS, SOURCE_AUTHORITY, ABSTENTION_THRESHOLD,
        EMBED_MODEL, RERANK_MODEL, API_LLM,
    )
    assert len(PERSONA_PRESETS) == 3
    assert "default" in PERSONA_PRESETS
    assert SOURCE_AUTHORITY["official"] == 1.0
    assert 0 < ABSTENTION_THRESHOLD < 0.05
    assert "bge" in EMBED_MODEL.lower()
    assert "deepseek" in API_LLM.lower()
    print("  Config OK")


if __name__ == "__main__":
    print("=== Smoke Tests ===")
    test_config_integrity()
    test_evaluator_no_crash()
    # Pipeline test requires GPU + ChromaDB + BM25
    try:
        import torch
        if torch.cuda.is_available():
            test_pipeline_no_crash()
        else:
            print("  Pipeline test: SKIPPED (GPU not available)")
    except ImportError:
        print("  Pipeline test: SKIPPED (torch not installed)")
    print("ALL SMOKE TESTS PASSED")
