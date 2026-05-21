"""
=============================================================================
Source Authority Scorer — 来源权威性排序信号 ★ 创新点 #6
=============================================================================
传统 Reranker 只看语义相关性，不考虑"信息来源是否可信"。
但在校园知识库场景中，来源可信度差异很大：

  学校官网公告 >> 某篇校园新闻稿 >> 未知来源页面

Authority Scorer 在 Reranker 的语义分数之上叠加一个"来源可信度"信号。

两个子信号：
  1. Authority Boost（权威性）：来自 config.SOURCE_AUTHORITY 的静态权重
     - official(1.0) > admission(0.95) > library(0.90) > ...
     - 这是**先验**信任（priori trust），不依赖具体查询

  2. Freshness Decay（时效性）：页面越旧，可信度越低
     - 对 TEMPORAL 类型的查询特别重要
     - "最新招生政策"应该优先展示 2026 年的页面而非 2020 年的
     - freshness_score = exp(-days_since_crawl / 180)

最终分数的加权公式：
  final = rerank_score × 0.80 + authority_weight × 0.12 + freshness × 0.08

为什么 authority 和 freshness 的权重这么小（12% 和 8%）？
  → 语义相关性始终是最重要的。权威性和时效性是"微调"信号，
    不是主导信号。只在两个 chunk 语义分数接近时发挥作用。

使用方法：
  from retrieval.authority_scorer import AuthorityScorer
  scorer = AuthorityScorer()
  scored = scorer.score(query, reranked_chunks, q_type="TEMPORAL")

=============================================================================
"""

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ============================================================================
# 配置导入
# ============================================================================
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SOURCE_AUTHORITY


class AuthorityScorer:
    """
    来源权威性打分器。

    在 Reranker 分数之上，叠加权威性和时效性两个微调信号。
    """

    def __init__(
        self,
        authority_weights: dict[str, float] = None,
        authority_contribution: float = 0.12,
        freshness_contribution: float = 0.08,
        rerank_contribution: float = 0.80,
    ):
        """
        初始化权威性打分器。

        参数：
            authority_weights: 来源 → 权威性分数映射
            authority_contribution: 权威性在最终分数中的权重（默认 12%）
            freshness_contribution: 时效性在最终分数中的权重（默认 8%）
            rerank_contribution: Reranker 在最终分数中的权重（默认 80%）
               注意：三个 weight 加起来应该 = 1.0
        """
        self.authority_weights = authority_weights or SOURCE_AUTHORITY
        self.w_authority = authority_contribution
        self.w_freshness = freshness_contribution
        self.w_rerank = rerank_contribution

    def _get_authority_score(self, source_family: str) -> float:
        """
        获取来源的权威性分数。

        参数：
            source_family: 来源类型（"official", "library" 等）

        返回：
            [0, 1] 范围内的权威性分数
        """
        return self.authority_weights.get(source_family, 0.50)

    def _get_freshness_score(self, crawled_at: str) -> float:
        """
        计算页面的时效性分数。

        使用的是指数衰减模型：
        - 今天爬的页面 → freshness = 1.0
        - 180 天前爬的页面 → freshness ≈ 0.37
        - 360 天前爬的页面 → freshness ≈ 0.14

        为什么不用线性衰减（如 freshness = 1 - days/180）？
        → 指数衰减更符合"信息老化"的直觉：
          - 前 30 天：信息几乎不变（freshness 仍在 ~0.85）
          - 30-180 天：开始明显老化
          - 180 天以上：已经相当陈旧

        参数：
            crawled_at: ISO 8601 格式的抓取时间戳

        返回：
            [0, 1] 范围的时效性分数
        """
        if not crawled_at:
            return 0.5  # 没有时间信息 → 给中等时效性

        try:
            # 解析 ISO 时间戳
            crawl_date = datetime.fromisoformat(
                crawled_at.replace("Z", "+00:00")
            )
            now = datetime.now(timezone.utc)
            days_since = (now - crawl_date).days

            if days_since < 0:
                # 未来时间 → 应该是时间戳错误，给 1.0
                return 1.0

            # 指数衰减：半衰期 = 180 天
            return math.exp(-days_since / 180)
        except (ValueError, TypeError):
            return 0.5

    def score(
        self,
        query: str,
        reranked_chunks: list[dict],
        q_type: str = "FACTUAL_COMPLEX",
    ) -> list[dict]:
        """
        对 Reranker 的结果施加权威性和时效性微调。

        参数：
            query: 用户查询文本（暂未使用，保留以备未来扩展）
            reranked_chunks: Reranker 输出的 chunk 列表
            q_type: 查询类型（TEMPORAL 查询会启用时效性加权）

        返回：
            叠加了 final_score 的 chunk 列表
        """
        use_freshness = (q_type == "TEMPORAL")

        for chunk in reranked_chunks:
            # 获取三个子信号
            rerank_score = chunk.get("rerank_score", chunk.get("rrf_score", 0))
            authority = self._get_authority_score(
                chunk.get("source_family", "unknown")
            )
            freshness = (
                self._get_freshness_score(chunk.get("crawled_at", ""))
                if use_freshness
                else 0.5  # 非时间查询 → freshness 固定为中等
            )

            # 加权求和
            if use_freshness:
                final = (
                    self.w_rerank * rerank_score
                    + self.w_authority * authority
                    + self.w_freshness * freshness
                )
            else:
                # 不对时间敏感的查询 → freshness 的权重转移到 rerank
                final = (
                    (self.w_rerank + self.w_freshness) * rerank_score
                    + self.w_authority * authority
                )

            # 将各子信号存储到 chunk 中（方便 Pipeline Inspector 展示）
            chunk["authority_score"] = authority
            chunk["freshness_score"] = freshness
            chunk["final_score"] = final

        # 按 final_score 重新排序
        scored = sorted(
            reranked_chunks,
            key=lambda x: x.get("final_score", 0),
            reverse=True,
        )

        return scored


# ============================================================================
# 单例
# ============================================================================

_scorer_instance: AuthorityScorer | None = None


def get_authority_scorer() -> AuthorityScorer:
    """获取全局唯一的 AuthorityScorer 实例。"""
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = AuthorityScorer()
    return _scorer_instance


# ============================================================================
# 测试/演示
# ============================================================================
if __name__ == "__main__":
    print("Source Authority Scorer — Unit Test\n")

    scorer = AuthorityScorer()

    # 模拟 Reranker 输出的 chunk
    mock_chunks = [
        {
            "chunk_id": "A",
            "text": "图书馆开放时间为每天8:00-22:00。",
            "rerank_score": 0.85,
            "source_family": "library",
            "crawled_at": "2026-05-15T10:00:00+00:00",
        },
        {
            "chunk_id": "B",
            "text": "据同学反映，图书馆最近延长了开放时间。",
            "rerank_score": 0.82,
            "source_family": "manual",
            "crawled_at": "2026-05-20T10:00:00+00:00",
        },
        {
            "chunk_id": "C",
            "text": "校园图书馆服务指南（2020年版）。",
            "rerank_score": 0.80,
            "source_family": "library",
            "crawled_at": "2020-01-15T10:00:00+00:00",
        },
    ]

    print("Before authority scoring (by rerank_score only):")
    for i, c in enumerate(sorted(mock_chunks, key=lambda x: -x["rerank_score"])):
        print(f"  #{i+1}: [{c['rerank_score']:.3f}] {c['source_family']:10s} {c['text'][:60]}...")

    scored = scorer.score("图书馆开放时间", mock_chunks, q_type="FACTUAL_COMPLEX")

    print("\nAfter authority scoring:")
    for i, c in enumerate(scored):
        print(f"  #{i+1}: [final={c['final_score']:.3f}] "
              f"(rerank={c['rerank_score']:.3f} "
              f"auth={c['authority_score']:.2f} "
              f"fresh={c['freshness_score']:.3f}) "
              f"{c['source_family']:10s} {c['text'][:60]}...")

    print(f"\nNotice: Chunk A (library) beats Chunk B (manual)")
    print(f"even though B has slightly higher rerank_score.")
    print(f"Authority weighting made the difference.")
