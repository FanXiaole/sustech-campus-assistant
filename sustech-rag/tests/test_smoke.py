"""
=============================================================================
SUSTech RAG — 扩展测试套件（无需 GPU，纯本地运行）
=============================================================================
覆盖：config / evaluator / classifier / prompt_builder / tokenizer / authority
运行：python tests/test_smoke.py
=============================================================================
"""

import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

def test_config():
    from config import (
        PERSONA_PRESETS, SOURCE_AUTHORITY, ABSTENTION_THRESHOLD,
        EMBED_MODEL, RERANK_MODEL, API_LLM, RRF_K, DENSE_TOP_K,
        BM25_TOP_K, RRF_FUSION_TOP, RERANK_TOP_K,
    )
    assert len(PERSONA_PRESETS) == 3
    assert "default" in PERSONA_PRESETS
    assert SOURCE_AUTHORITY["official"] == 1.0
    assert 0 < ABSTENTION_THRESHOLD < 0.05
    assert "bge" in EMBED_MODEL.lower()
    assert "deepseek" in API_LLM.lower()
    assert RRF_K == 60
    assert DENSE_TOP_K == 50
    assert BM25_TOP_K == 50
    assert RRF_FUSION_TOP == 20
    assert RERANK_TOP_K == 5
    print("  [PASS] config")


# ═══════════════════════════════════════════════════════════════
# Evaluator — 5 维度规则评分
# ═══════════════════════════════════════════════════════════════

def _make_evaluator():
    from evaluation.evaluator import RAGEvaluator
    return RAGEvaluator()


def test_correctness_perfect():
    e = _make_evaluator()
    score, reason = e.score_correctness(
        "图书馆周一至周五 8:00-22:00 开放，周末 9:00-21:00 开放。",
        "周一至周五 8:00-22:00，周末 9:00-21:00",
        ["8:00", "22:00", "周一至周五"],
    )
    assert score == 2, f"expected 2, got {score}: {reason}"
    print("  [PASS] correctness_perfect")


def test_correctness_empty():
    e = _make_evaluator()
    score, _ = e.score_correctness("", "8:00-22:00", ["8:00"])
    assert score == 0
    print("  [PASS] correctness_empty")


def test_grounding_full():
    e = _make_evaluator()
    score, _ = e.score_grounding(
        "图书馆开放时间为周一至周五 8:00 到 22:00。",
        [{"raw_text": "图书馆服务时间：周一至周五 8:00-22:00，周末 9:00-21:00。"}],
    )
    assert score >= 1, f"expected >=1, got {score}"
    print("  [PASS] grounding_full")


def test_grounding_no_chunks():
    e = _make_evaluator()
    score, _ = e.score_grounding("图书馆8点开门。", [])
    assert score == 0
    print("  [PASS] grounding_no_chunks")


def test_completeness_all():
    e = _make_evaluator()
    score, _ = e.score_completeness(
        "南科大图书馆周一至周五 8:00-22:00 开放。",
        ["8:00", "22:00", "周一至周五"],
    )
    assert score == 2, f"expected 2, got {score}"
    print("  [PASS] completeness_all")


def test_completeness_partial():
    e = _make_evaluator()
    score, _ = e.score_completeness(
        "南科大图书馆8点开门。",
        ["8:00", "22:00", "周一至周五", "周末"],
    )
    assert score <= 1
    print("  [PASS] completeness_partial")


def test_traceability_explicit():
    e = _make_evaluator()
    score, _ = e.score_traceability(
        "根据图书馆网站信息，开放时间为8:00-22:00。来源：图书馆官网。",
        [{"raw_text": "test"}],
    )
    assert score >= 1
    print("  [PASS] traceability_explicit")


def test_traceability_none():
    e = _make_evaluator()
    score, _ = e.score_traceability("8点到10点。", [{"raw_text": "test"}])
    assert score == 0
    print("  [PASS] traceability_none")


def test_abstention_correct_refuse():
    e = _make_evaluator()
    score, _ = e.score_abstention(
        "未找到相关信息，建议直接访问官网。",
        expected_abstain=True, did_abstain=True,
    )
    assert score == 2, f"expected 2, got {score}"
    print("  [PASS] abstention_correct_refuse")


def test_abstention_should_refuse_but_answered():
    e = _make_evaluator()
    score, _ = e.score_abstention(
        "清华大学的图书馆开放时间是8:00-22:00。",
        expected_abstain=True, did_abstain=False,
    )
    assert score == 0, f"expected 0, got {score}"
    print("  [PASS] abstention_should_refuse")


def test_abstention_normal_answer():
    e = _make_evaluator()
    score, _ = e.score_abstention(
        "南科大图书馆8:00-22:00开放。",
        expected_abstain=False, did_abstain=False,
    )
    assert score == 2
    print("  [PASS] abstention_normal")


def test_full_evaluation():
    e = _make_evaluator()
    result = e.evaluate(
        answer="南科大图书馆周一至周五开放时间为8:00到22:00。",
        chunks=[{"raw_text": "图书馆服务时间：周一至周五 8:00-22:00。",
                 "text": "[来源:library] 图书馆服务时间：周一至周五 8:00-22:00。"}],
        question_meta={
            "q_id": "test", "ground_truth": "8:00-22:00",
            "key_facts": ["8:00", "22:00", "周一至周五"],
            "expected_abstain": False,
        },
    )
    assert 0 <= result["total_score"] <= 10
    assert "correctness" in result["dimensions"]
    assert "grounding" in result["dimensions"]
    assert "completeness" in result["dimensions"]
    assert "traceability" in result["dimensions"]
    assert "abstention" in result["dimensions"]
    print(f"  [PASS] full_evaluation (score={result['total_score']})")


def test_bootstrap_ci():
    from evaluation.evaluator import RAGEvaluator
    # 两组有明显差异的分数
    a = [8, 7, 9, 8, 7, 8, 9, 7]
    b = [3, 4, 3, 5, 4, 3, 4, 4]
    ci = RAGEvaluator.bootstrap_compare(a, b)
    assert ci["significant"] is True, f"expected significant diff, got p={ci['p_value']}"
    assert ci["observed_diff"] > 0
    print(f"  [PASS] bootstrap_ci (diff={ci['observed_diff']}, p={ci['p_value']})")


def test_bootstrap_no_diff():
    from evaluation.evaluator import RAGEvaluator
    import random
    random.seed(0)
    a = [random.gauss(5, 1) for _ in range(50)]
    b = [random.gauss(5, 1) for _ in range(50)]
    ci = RAGEvaluator.bootstrap_compare(a, b)
    # 无差异时 p 应该较大
    assert ci["p_value"] > 0.01, f"expected p>0.01, got {ci['p_value']}"
    print(f"  [PASS] bootstrap_no_diff (p={ci['p_value']})")


# ═══════════════════════════════════════════════════════════════
# Query Classifier — 规则模式
# ═══════════════════════════════════════════════════════════════

def test_classifier_factual_simple():
    from retrieval.query_classifier import get_classifier
    qc = get_classifier(mode="rule")
    q_type, conf = qc.classify("图书馆几点开门")
    assert q_type == "FACTUAL_SIMPLE", f"got {q_type}"
    print(f"  [PASS] classifier_factual_simple ({q_type})")


def test_classifier_procedural():
    from retrieval.query_classifier import get_classifier
    qc = get_classifier(mode="rule")
    q_type, _ = qc.classify("如何办理校园卡")
    assert q_type == "PROCEDURAL", f"got {q_type}"
    print(f"  [PASS] classifier_procedural ({q_type})")


def test_classifier_comparative():
    from retrieval.query_classifier import get_classifier
    qc = get_classifier(mode="rule")
    q_type, _ = qc.classify("理工学院和医学院有什么区别")
    assert q_type == "COMPARATIVE", f"got {q_type}"
    print(f"  [PASS] classifier_comparative ({q_type})")


def test_classifier_oos_other_school():
    from retrieval.query_classifier import get_classifier
    qc = get_classifier(mode="rule")
    q_type, _ = qc.classify("清华大学校长是谁")
    assert q_type == "OUT_OF_SCOPE", f"got {q_type}"
    print(f"  [PASS] classifier_oos ({q_type})")


def test_classifier_temporal():
    from retrieval.query_classifier import get_classifier
    qc = get_classifier(mode="rule")
    q_type, _ = qc.classify("2026年招生政策有什么变化")
    assert q_type == "TEMPORAL", f"got {q_type}"
    print(f"  [PASS] classifier_temporal ({q_type})")


def test_classifier_weights():
    from retrieval.query_classifier import get_classifier
    qc = get_classifier(mode="rule")
    for q_type in ["FACTUAL_SIMPLE", "FACTUAL_COMPLEX", "COMPARATIVE",
                   "PROCEDURAL", "TEMPORAL", "OUT_OF_SCOPE"]:
        d, s = qc.get_rrf_weights(q_type)
        assert 0 <= d <= 2.0
        assert 0 <= s <= 2.0
    # OOS 应该返回零权重
    d, s = qc.get_rrf_weights("OUT_OF_SCOPE")
    assert d == 0 and s == 0
    print("  [PASS] classifier_weights")


def test_classifier_cache():
    from retrieval.query_classifier import get_classifier
    qc = get_classifier(mode="rule")
    t1, _ = qc.classify("图书馆几点开门")
    t2, _ = qc.classify("图书馆几点开门")
    assert t1 == t2  # 缓存命中，结果应一致
    print("  [PASS] classifier_cache")


# ═══════════════════════════════════════════════════════════════
# Prompt Builder — 3 种人格
# ═══════════════════════════════════════════════════════════════

def test_prompt_builder_default():
    from generation.prompt_builder import PromptBuilder
    pb = PromptBuilder(persona="default")
    sp = pb.build_system_prompt()
    assert "南方科技大学" in sp
    assert "参考资料" in sp or "基于" in sp
    print("  [PASS] prompt_default")


def test_prompt_builder_personas():
    from generation.prompt_builder import PromptBuilder
    for pid in ["default", "unhinged", "sexy"]:
        pb = PromptBuilder(persona=pid)
        sp = pb.build_system_prompt()
        assert len(sp) > 50
        um = pb.build_user_message("测试问题", [{
            "source_family": "library", "url": "https://test.edu",
            "raw_text": "这是一条测试资料。",
        }])
        assert "测试问题" in um
        assert "测试资料" in um
    print("  [PASS] prompt_all_personas")


def test_prompt_builder_unknown_persona_fallback():
    from generation.prompt_builder import PromptBuilder
    pb = PromptBuilder(persona="nonexistent")
    assert pb.persona == "default"
    print("  [PASS] prompt_unknown_fallback")


def test_prompt_builder_estimate_tokens():
    from generation.prompt_builder import PromptBuilder
    pb = PromptBuilder()
    n = pb.estimate_tokens("你好世界" * 100)
    assert 0 < n < 10000
    print(f"  [PASS] prompt_estimate_tokens ({n})")


# ═══════════════════════════════════════════════════════════════
# Tokenizer
# ═══════════════════════════════════════════════════════════════

def test_tokenizer():
    from indexing.tokenizer import load_stopwords, tokenize
    sw = load_stopwords()
    assert len(sw) > 100, f"too few stopwords: {len(sw)}"
    tokens = tokenize("南方科技大学图书馆几点开门", sw)
    assert len(tokens) >= 2, f"too few tokens: {tokens}"
    # 停用词应该被过滤
    assert "的" not in tokens
    assert "了" not in tokens
    # 单字应该被过滤
    for t in tokens:
        assert len(t) >= 2, f"single-char token leaked: '{t}'"
    print(f"  [PASS] tokenizer ({len(tokens)} tokens)")


# ═══════════════════════════════════════════════════════════════
# Authority Scorer
# ═══════════════════════════════════════════════════════════════

def test_authority_scorer():
    from retrieval.authority_scorer import AuthorityScorer
    scorer = AuthorityScorer()

    chunks = [
        {"chunk_id": "A", "text": "官方信息", "rerank_score": 0.85,
         "source_family": "official", "crawled_at": "2026-05-15T10:00:00+00:00"},
        {"chunk_id": "B", "text": "手册信息", "rerank_score": 0.85,
         "source_family": "manual", "crawled_at": "2026-05-15T10:00:00+00:00"},
    ]
    scored = scorer.score("测试查询", chunks, q_type="FACTUAL_COMPLEX")

    assert len(scored) == 2
    assert "final_score" in scored[0]
    assert "authority_score" in scored[0]
    # official 应该排在 manual 前面（同等 rerank 分数时）
    assert scored[0]["source_family"] == "official", \
        f"expected official first, got {scored[0]['source_family']}"
    print("  [PASS] authority_scorer")


def test_authority_temporal_freshness():
    from retrieval.authority_scorer import AuthorityScorer
    scorer = AuthorityScorer()

    chunks = [
        {"chunk_id": "A", "text": "新页面", "rerank_score": 0.80,
         "source_family": "news", "crawled_at": "2026-05-20T10:00:00+00:00"},
        {"chunk_id": "B", "text": "旧页面", "rerank_score": 0.80,
         "source_family": "news", "crawled_at": "2020-01-15T10:00:00+00:00"},
    ]
    scored = scorer.score("最新政策", chunks, q_type="TEMPORAL")
    # 新页面 freshness 更高，应排前面
    assert scored[0]["chunk_id"] == "A", \
        f"expected newer doc first, got {scored[0]['chunk_id']}"
    print("  [PASS] authority_temporal_freshness")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== SUSTech RAG Test Suite ===\n")

    print("-- Config --")
    test_config()

    print("\n-- Evaluator --")
    test_correctness_perfect()
    test_correctness_empty()
    test_grounding_full()
    test_grounding_no_chunks()
    test_completeness_all()
    test_completeness_partial()
    test_traceability_explicit()
    test_traceability_none()
    test_abstention_correct_refuse()
    test_abstention_should_refuse_but_answered()
    test_abstention_normal_answer()
    test_full_evaluation()
    test_bootstrap_ci()
    test_bootstrap_no_diff()

    print("\n-- Query Classifier --")
    test_classifier_factual_simple()
    test_classifier_procedural()
    test_classifier_comparative()
    test_classifier_oos_other_school()
    test_classifier_temporal()
    test_classifier_weights()
    test_classifier_cache()

    print("\n-- Prompt Builder --")
    test_prompt_builder_default()
    test_prompt_builder_personas()
    test_prompt_builder_unknown_persona_fallback()
    test_prompt_builder_estimate_tokens()

    print("\n-- Tokenizer --")
    test_tokenizer()

    print("\n-- Authority Scorer --")
    test_authority_scorer()
    test_authority_temporal_freshness()

    print("\n=== ALL TESTS PASSED ===")
