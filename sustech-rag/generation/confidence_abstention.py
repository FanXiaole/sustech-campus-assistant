"""
=============================================================================
Confidence Abstention — 证据不足时拒答 ★ 创新点 #7
=============================================================================
这是我们的"安全网"模块。在 RAG 系统中，拒答（abstention）比瞎编（hallucination）
更有价值——特别是在校园场景中，错误的考试日期、错误的办公时间会造成实际困扰。

两级检测机制：

  Stage 1 — 分数检测（快速，在 LLM 调用之前）：
    检查 RRF 融合后的最高分数。如果最高分仍低于阈值，
    说明检索结果与查询相关度很低 → 直接拒答，不浪费 LLM 调用。

  Stage 2 — LLM 自我检查（精准，在 LLM 生成之后）：
    LLM 生成答案后，再次调用 LLM 检查"这个答案中的每个 claim
    是否在提供的参考资料中有依据"。如果发现无依据的 claim，
    给答案追加免责声明。

两种 abstention 模式：
  - "score_only"：只做 Stage 1（快，零成本）
  - "llm_check"：只做 Stage 2（精准，但有额外 API 调用成本）
  - "both"：两阶段都做（最安全，推荐用于生产环境）

使用方法：
  from generation.confidence_abstention import ConfidenceChecker
  checker = ConfidenceChecker()
  decision = checker.check_stage1(rrf_scores)  # Stage 1
  verified_answer = checker.check_stage2(answer, chunks, llm_fn)  # Stage 2

=============================================================================
"""

import json
import time
from pathlib import Path

# ============================================================================
# 配置导入
# ============================================================================
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ABSTENTION_THRESHOLD


# ============================================================================
# 弃权消息模板
# ============================================================================

# Stage 1 的拒答回复（检索分数太低）
ABSTAIN_MESSAGE_SCORE = (
    "抱歉，根据现有校园资料，未找到与您问题相关的信息。\n\n"
    "建议您：\n"
    "1. 换一种更具体的表述方式重新提问\n"
    "2. 直接访问南方科技大学官方网站 (sustech.edu.cn) 查询\n"
    "3. 联系相关部门获取权威信息\n\n"
    "🔍 当前检索置信度过低，为避免提供不准确的信息，系统已自动启用拒答保护。"
)

# Stage 2 的拒答回复（LLM 发现 hallucination）
ABSTAIN_MESSAGE_HALLUCINATION = (
    "⚠️ 以下回答的部分内容可能超出了参考资料的覆盖范围，已在文中标注。\n"
    "请以南方科技大学官方信息为准。\n\n"
)

# Stage 2 的自我检查 prompt
GROUNDING_CHECK_PROMPT = """你是一个严格的事实核查员。请检查以下AI生成的回答中，每个论断是否在提供的参考资料中有明确依据。

回答：
{answer}

参考资料：
{chunks}

请逐一检查回答中的关键论断，判断其是否能在参考资料中找到支持。
请只输出一个JSON对象（不要加任何其他文字）：
{{
  "grounded": true/false,
  "unsupported_claims": ["无依据的论断1", "无依据的论断2"],
  "overall_confidence": 0.0-1.0
}}

如果回答中的所有论断都能在参考资料中找到依据，grounded=true。
如果存在任何编造、推测或无依据的陈述，grounded=false。"""


class ConfidenceChecker:
    """
    置信度检查器。

    负责在 RAG pipeline 的两个关键节点检查检索质量和生成质量。
    """

    def __init__(
        self,
        threshold: float = None,
        mode: str = "score_only",
    ):
        """
        初始化置信度检查器。

        参数：
            threshold: Stage 1 的拒答阈值（低于此分数 → 拒答）
            mode: "score_only" | "llm_check" | "both"
        """
        self.threshold = threshold or ABSTENTION_THRESHOLD
        self.mode = mode

        # 统计数据（用于实验报告）
        self.stats = {
            "total_queries": 0,
            "stage1_abstentions": 0,
            "stage2_flags": 0,
            "total_abstentions": 0,
        }

    # ─── Stage 1: 分数检测 ─────────────────────────────────

    def check_stage1(self, rrf_scores: list[float]) -> tuple[bool, float]:
        """
        Stage 1: 基于 RRF 分数的快速拒答判断。

        这个检查在 LLM 调用之前执行，目的是：
        - 如果检索质量明显很差，直接拒答，不浪费 LLM 调用
        - 避免 LLM 在没有充分证据时"强行编造"

        参数：
            rrf_scores: RRF 融合后的分数列表（来自 hybrid_rrf.py）

        返回：
            (should_abstain: bool, max_score: float)
        """
        if not rrf_scores:
            return (True, 0.0)

        max_score = max(rrf_scores)

        # 判断逻辑
        should_abstain = max_score < self.threshold

        return (should_abstain, max_score)

    # ─── Stage 2: LLM 自我检查 ─────────────────────────────

    def check_stage2(
        self,
        answer: str,
        chunks: list[dict],
        llm_fn,
    ) -> dict:
        """
        Stage 2: LLM 自我检查——验证生成内容的依据。

        这个检查在 LLM 生成答案后执行，目的是：
        - 发现 LLM "忽略上下文"而编造的内容
        - 为生成的答案提供质量保证

        参数：
            answer: LLM 生成的答案
            chunks: 用于生成的检索 chunk
            llm_fn: LLM 调用函数

        返回：
            {"grounded": bool, "unsupported_claims": [...], "overall_confidence": float}
        """
        # 构建 chunks 文本
        chunks_text = ""
        for i, chunk in enumerate(chunks, 1):
            raw = chunk.get("raw_text", chunk.get("text", ""))
            chunks_text += f"[{i}] {raw[:300]}\n"

        # 构建检查 prompt
        user_prompt = GROUNDING_CHECK_PROMPT.format(
            answer=answer,
            chunks=chunks_text,
        )

        try:
            response = llm_fn(
                "你是一个严格的事实核查员。请只输出JSON。",
                user_prompt,
            )

            # 解析 JSON
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()

            result = json.loads(response)
            return result

        except Exception:
            # LLM 自我检查失败 → 保守处理：标记为未确认
            return {
                "grounded": False,
                "unsupported_claims": ["自我检查失败，无法确认"],
                "overall_confidence": 0.5,
            }

    # ─── 综合检查 ────────────────────────────────────────

    def full_check(
        self,
        answer: str,
        rrf_scores: list[float],
        chunks: list[dict],
        llm_fn=None,
    ) -> dict:
        """
        执行完整的置信度检查（根据 self.mode 决定用哪些阶段）。

        这是外部调用的主入口。

        参数：
            answer: LLM 生成的答案（Stage 2 需要）
            rrf_scores: RRF 分数列表（Stage 1 需要）
            chunks: 检索 chunk 列表（Stage 2 需要）
            llm_fn: LLM 调用函数（Stage 2 需要）

        返回：
            {
                "should_abstain": bool,
                "abstain_message": str or None,
                "stage1_result": {...},
                "stage2_result": {...} or None,
            }
        """
        self.stats["total_queries"] += 1
        result = {
            "should_abstain": False,
            "abstain_message": None,
            "stage1_result": None,
            "stage2_result": None,
        }

        # ── Stage 1 ──
        if self.mode in ("score_only", "both"):
            should_abstain, max_score = self.check_stage1(rrf_scores)
            result["stage1_result"] = {
                "should_abstain": should_abstain,
                "max_rrf_score": max_score,
                "threshold": self.threshold,
            }

            if should_abstain:
                result["should_abstain"] = True
                result["abstain_message"] = ABSTAIN_MESSAGE_SCORE
                self.stats["stage1_abstentions"] += 1
                self.stats["total_abstentions"] += 1
                return result

        # ── Stage 2 ──
        if self.mode in ("llm_check", "both") and answer and llm_fn:
            stage2_result = self.check_stage2(answer, chunks, llm_fn)
            result["stage2_result"] = stage2_result

            if not stage2_result.get("grounded", True):
                result["should_abstain"] = True
                # 追加免责声明（而不是完全拒答）
                disclaimer = (
                    f"{ABSTAIN_MESSAGE_HALLUCINATION}"
                    f"未找到依据的内容：{stage2_result.get('unsupported_claims', [])}"
                )
                result["abstain_message"] = disclaimer + "\n\n原回答：\n" + answer
                self.stats["stage2_flags"] += 1
                self.stats["total_abstentions"] += 1

        return result

    def get_stats(self) -> dict:
        """获取弃权统计数据（用于实验报告）。"""
        total = max(self.stats["total_queries"], 1)
        return {
            **self.stats,
            "abstention_rate": self.stats["total_abstentions"] / total,
            "stage1_rate": self.stats["stage1_abstentions"] / total,
            "stage2_rate": self.stats["stage2_flags"] / total,
        }


# ============================================================================
# 测试/演示
# ============================================================================
if __name__ == "__main__":
    print("Confidence Abstention — Unit Test\n")

    checker = ConfidenceChecker(threshold=0.35, mode="score_only")

    # 模拟高分场景（应该不拒答）
    high_scores = [1.2, 0.8, 0.6, 0.5, 0.4]
    abstain, max_s = checker.check_stage1(high_scores)
    print(f"High scores: max={max_s:.4f}, abstain={abstain} "
          f"(expected: False)")

    # 模拟低分场景（应该拒答）
    low_scores = [0.02, 0.015, 0.01]
    abstain, max_s = checker.check_stage1(low_scores)
    print(f"Low scores:  max={max_s:.4f}, abstain={abstain} "
          f"(expected: True)")

    print(f"\nThreshold: {checker.threshold}")
    print(f"When max RRF score < {checker.threshold}, system will refuse to answer.")
    print(f"This prevents hallucination when retrieval quality is poor.")
