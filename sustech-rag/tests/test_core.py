"""
=============================================================================
SUSTech RAG — 核心函数单元测试
=============================================================================
测试三个最关键的函数：
  - rrf_fusion: 核心融合算法
  - classify_by_rules: 查询分类准确性
  - _fact_in_text: 评分中的事实匹配

运行：pytest tests/ -v  或  python -m pytest tests/ -v
=============================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


# ======================================================================
# Test: RRF Fusion
# ======================================================================

class TestRRFFusion:
    """测试 RRF 融合算法的正确性。"""

    def test_basic_fusion(self):
        from retrieval.hybrid_rrf import rrf_fusion

        dense = [
            {"chunk_id": "A", "text": "doc A", "score": 0.95},
            {"chunk_id": "B", "text": "doc B", "score": 0.80},
            {"chunk_id": "C", "text": "doc C", "score": 0.60},
        ]
        bm25 = [
            {"chunk_id": "B", "text": "doc B", "score": 8.5},
            {"chunk_id": "A", "text": "doc A", "score": 6.0},
            {"chunk_id": "D", "text": "doc D", "score": 5.0},
        ]

        result = rrf_fusion(dense, bm25, k=60, top_n=10)

        # A appears in both → should rank highest
        assert len(result) >= 3
        assert result[0]["chunk_id"] in ("A", "B")  # A or B should be #1

    def test_empty_inputs(self):
        from retrieval.hybrid_rrf import rrf_fusion

        # Both empty
        result = rrf_fusion([], [], k=60, top_n=5)
        assert len(result) == 0

        # One empty
        dense = [{"chunk_id": "X", "text": "x", "score": 0.9}]
        result = rrf_fusion(dense, [], k=60, top_n=5)
        assert len(result) == 1
        assert result[0]["chunk_id"] == "X"

    def test_weighted_fusion(self):
        from retrieval.hybrid_rrf import rrf_fusion

        dense = [{"chunk_id": "A", "text": "a", "score": 0.9}]
        bm25 = [{"chunk_id": "B", "text": "b", "score": 10.0}]

        # Dense-heavy → A should win
        result_dense = rrf_fusion(dense, bm25, k=60, dense_weight=10.0, sparse_weight=0.1, top_n=5)
        assert result_dense[0]["chunk_id"] == "A"

        # Sparse-heavy → B should win
        result_sparse = rrf_fusion(dense, bm25, k=60, dense_weight=0.1, sparse_weight=10.0, top_n=5)
        assert result_sparse[0]["chunk_id"] == "B"

    def test_rrf_score_boundaries(self):
        """RRF 分数范围合理性检验。"""
        from retrieval.hybrid_rrf import rrf_fusion

        # rank=1 in both → RRF ≈ 1/(60+1) + 1/(60+1) ≈ 0.0328
        dense = [{"chunk_id": "X", "text": "t", "score": 0.99}]
        bm25 = [{"chunk_id": "X", "text": "t", "score": 9.99}]
        result = rrf_fusion(dense, bm25, k=60, top_n=5)

        assert len(result) == 1
        assert 0.03 < result[0]["rrf_score"] < 0.04


# ======================================================================
# Test: Query Classifier (Rule-based)
# ======================================================================

class TestQueryClassifier:
    """测试规则查询分类器的准确性。"""

    def test_factual_simple(self):
        from retrieval.query_classifier import classify_by_rules, QueryType

        q_type, _ = classify_by_rules("图书馆几点开门？")
        assert q_type in (QueryType.FACTUAL_SIMPLE, QueryType.PROCEDURAL)

    def test_oos_detection(self):
        from retrieval.query_classifier import classify_by_rules, QueryType

        q_type, _ = classify_by_rules("清华大学校长是谁？")
        assert q_type == QueryType.OUT_OF_SCOPE

        q_type, _ = classify_by_rules("港科大图书馆几点开门")
        assert q_type == QueryType.OUT_OF_SCOPE

    def test_procedural(self):
        from retrieval.query_classifier import classify_by_rules, QueryType

        q_type, _ = classify_by_rules("如何办理校园卡？")
        assert q_type == QueryType.PROCEDURAL

    def test_comparative(self):
        from retrieval.query_classifier import classify_by_rules, QueryType

        q_type, _ = classify_by_rules("理工学院和医学院有什么区别？")
        assert q_type == QueryType.COMPARATIVE

    def test_temporal(self):
        from retrieval.query_classifier import classify_by_rules, QueryType

        q_type, _ = classify_by_rules("2026年最新的招生政策有什么变化？")
        assert q_type == QueryType.TEMPORAL

    def test_sustech_query_not_oos(self):
        """SUSTech 相关的查询不应被判为 OOS。"""
        from retrieval.query_classifier import classify_by_rules, QueryType

        for q in [
            "南科大计算机系怎么样",
            "图书馆在哪儿",
            "致仁书院有什么活动",
            "sustech校园卡怎么充值",
        ]:
            q_type, _ = classify_by_rules(q)
            assert q_type != QueryType.OUT_OF_SCOPE, f"'{q}' should not be OOS"

    def test_weight_assignment(self):
        """验证不同查询类型获取正确的权重。"""
        from retrieval.query_classifier import (
            QueryType, STRATEGY_WEIGHTS
        )
        # OOS 的权重为 (0, 0) → 不检索
        assert STRATEGY_WEIGHTS[QueryType.OUT_OF_SCOPE] == (0.0, 0.0)
        # 事实简单查询偏向 sparse
        assert STRATEGY_WEIGHTS[QueryType.FACTUAL_SIMPLE][1] > \
               STRATEGY_WEIGHTS[QueryType.FACTUAL_SIMPLE][0]
        # 复杂查询偏向 dense
        assert STRATEGY_WEIGHTS[QueryType.FACTUAL_COMPLEX][0] > \
               STRATEGY_WEIGHTS[QueryType.FACTUAL_COMPLEX][1]


# ======================================================================
# Test: Fact-in-Text Matching
# ======================================================================

class TestFactInText:
    """测试 evaluator 中的事实匹配逻辑。"""

    def test_exact_match(self):
        from evaluation.evaluator import RAGEvaluator

        assert RAGEvaluator._fact_in_text("图书馆", "南科大图书馆开放时间")
        assert not RAGEvaluator._fact_in_text("食堂", "南科大图书馆开放时间")

    def test_number_matching_strict(self):
        """多年份数字必须精确匹配，不能把 '2' 当成 '2026'。"""
        from evaluation.evaluator import RAGEvaluator

        # 2026 必须完整出现
        assert RAGEvaluator._fact_in_text("2026", "招生政策2026年")
        assert not RAGEvaluator._fact_in_text("2026", "今年是2月开学")
        # 多位数字都需要匹配
        assert RAGEvaluator._fact_in_text("8:00", "开放时间8:00")
        assert not RAGEvaluator._fact_in_text("8:00", "早上8点20分")

    def test_partial_match(self):
        """长 fact 的部分匹配。"""
        from evaluation.evaluator import RAGEvaluator

        # "周一至周五" 出现在答案中 → OK
        assert RAGEvaluator._fact_in_text(
            "周一至周五",
            "图书馆周一至周五开放，时间为8:00到22:00"
        )
        # 精确包含 → OK
        assert RAGEvaluator._fact_in_text(
            "8:00-22:00",
            "开放时间8:00-22:00，周末9:00-21:00"
        )

    def test_single_digit_loose(self):
        """单数字宽松匹配。"""
        from evaluation.evaluator import RAGEvaluator

        # 单数字 "6" 可以宽松匹配
        assert RAGEvaluator._fact_in_text("6个书院", "学校有6个书院")


# ======================================================================
# Test: Config Integrity
# ======================================================================

class TestConfig:
    """测试配置文件的完整性。"""

    def test_source_authority_keys(self):
        from config import SOURCE_AUTHORITY
        assert len(SOURCE_AUTHORITY) >= 7
        assert SOURCE_AUTHORITY["official"] == 1.0
        assert 0 <= SOURCE_AUTHORITY["unknown"] <= 1.0

    def test_persona_presets(self):
        from config import PERSONA_PRESETS
        # 只有 3 种人格
        assert len(PERSONA_PRESETS) == 3
        assert "default" in PERSONA_PRESETS
        assert "unhinged" in PERSONA_PRESETS
        assert "sexy" in PERSONA_PRESETS
        # casual 和 academic 已被删除
        assert "casual" not in PERSONA_PRESETS
        assert "academic" not in PERSONA_PRESETS

    def test_abstention_threshold_range(self):
        from config import ABSTENTION_THRESHOLD
        # 阈值应在 RRF 分数有效范围内 [0, 0.033]
        assert 0 < ABSTENTION_THRESHOLD < 0.05


# ======================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
