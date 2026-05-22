"""
=============================================================================
SUSTech RAG Evaluator — 5 维度评分系统
=============================================================================
这是评测的核心模块。对 RAG 系统输出的每个答案进行五个维度的打分。

五个维度（每个 0-2 分，总分 0-10）：
  Dim 1 — Correctness（正确性）: 答案是否正确、包含关键事实
  Dim 2 — Grounding（依据性）: 每个 claim 是否可追溯到检索 chunk
  Dim 3 — Completeness（完整性）: 是否覆盖所有 key_facts
  Dim 4 — Traceability（可追溯性）: 是否明确引用来源
  Dim 5 — Abstention Quality（拒答质量）★ 我们的独有维度:
          对于 expected_abstain=True 的问题 → 应该拒答
          对于 expected_abstain=False 的问题 → 应该回答

评分方式：
  - 主方案：基于规则的自动化评分（keyword matching + heuristics）
  - 可选方案：LLM 辅助评分（对于复杂的主观维度，如 Grounding）
  - 目前实现规则评分，LLM 辅助评分作为未来扩展

使用方法：
  from evaluation.evaluator import evaluate_answer
  scores = evaluate_answer(answer, chunks, question_meta)
=============================================================================
"""

import json
import re
from pathlib import Path

# ============================================================================
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class RAGEvaluator:
    """
    五维度 RAG 评测器。

    每个评分方法返回 (score: int, explanation: str)。
    score ∈ {0, 1, 2}，explanation 是对评分依据的中文说明。
    """

    def __init__(self):
        self.scores_log = []  # 记录本次实验的所有评分

    def reset(self):
        """清空评分日志——每次实验前必须调用，避免状态污染。"""
        self.scores_log = []

    # ─── Dim 1: Correctness ──────────────────────────────────

    def score_correctness(
        self, answer: str, ground_truth: str, key_facts: list[str]
    ) -> tuple[int, str]:
        """
        评分维度 1：正确性 —— 答案是否与 ground_truth 一致。

        基于两个信号加权：
        1. key_facts 命中率 (权重 0.6)
        2. ground_truth 语义重叠度 (权重 0.4) —— 用短词 2-gram 衡量
        """
        if not answer or len(answer.strip()) < 5:
            return (0, "答案为空或过短")

        answer_lower = answer.lower()

        # key_facts 命中
        if key_facts:
            hits = sum(1 for f in key_facts if self._fact_in_text(f, answer_lower))
            fact_rate = hits / len(key_facts)
        else:
            fact_rate = 0.5

        # ground_truth 语义重叠（2-gram 比较，区别于 completeness 的 key_fact 检查）
        if ground_truth:
            gt_words = re.findall(r'[一-鿿]{2,4}|\d{2,}', ground_truth)
            if gt_words:
                gt_hits = sum(1 for w in gt_words if w in answer_lower)
                gt_overlap = gt_hits / len(gt_words)
            else:
                gt_overlap = 0.5
        else:
            gt_overlap = 0.5

        # 加权综合
        combined = fact_rate * 0.6 + gt_overlap * 0.4

        if combined >= 0.7:
            return (2, f"正确 (key_facts={fact_rate:.0%}, overlap={gt_overlap:.0%})")
        elif combined >= 0.4:
            return (1, f"部分正确 (key_facts={fact_rate:.0%}, overlap={gt_overlap:.0%})")
        else:
            return (0, f"不正确 (key_facts={fact_rate:.0%}, overlap={gt_overlap:.0%})")

    # ─── Dim 2: Grounding ─────────────────────────────────────

    def score_grounding(
        self, answer: str, chunks: list[dict]
    ) -> tuple[int, str]:
        """
        评分维度 2：证据依据性。

        检查答案中的关键信息（短词语、数字、实体）是否在 chunk 文本中出现。
        使用 2-gram 中文短词匹配，而非长句匹配，避免 LLM 改写导致的不匹配。
        """
        if not chunks:
            return (0, "无检索结果，答案无依据")

        # 构建所有 chunk 的合并文本（去除元数据前缀 [来源:xxx][域名:xxx]）
        all_context = ""
        for c in chunks:
            txt = c.get("raw_text", c.get("text", ""))
            # 去除 "[...]" 元数据前缀
            if txt.startswith("[来源:") or txt.startswith("[背景:"):
                idx = txt.find("] ")
                if idx > 0:
                    # 可能有多个前缀 [背景:xxx][来源:xxx]
                    while txt.startswith("[") and "] " in txt[:30]:
                        end = txt.find("] ") + 2
                        txt = txt[end:]
            all_context += txt + " "

        # 提取答案中的中文 2-grams（短词）和数字
        cjk_words = re.findall(r'[一-鿿]{2,4}', answer)
        numbers = re.findall(r'\d{2,}', answer)

        # 合并为检查项
        check_items = cjk_words + numbers
        if not check_items:
            return (1, "答案中无可检测的中文短语或数字")

        # 计算命中率
        supported = sum(1 for item in check_items if item in all_context)
        support_rate = supported / len(check_items)

        if support_rate >= 0.6:
            return (2, f"答案高度有据 ({supported}/{len(check_items)} 项可追溯)")
        elif support_rate >= 0.3:
            return (1, f"答案部分有据 ({supported}/{len(check_items)} 项可追溯)")
        else:
            return (0, f"答案缺乏依据 ({supported}/{len(check_items)} 项可追溯)")

    # ─── Dim 3: Completeness ──────────────────────────────────

    def score_completeness(
        self, answer: str, key_facts: list[str]
    ) -> tuple[int, str]:
        """
        评分维度 3：完整性。

        评分标准：
          2: 所有 key_facts 都在答案中
          1: >50% key_facts 覆盖
          0: <50% key_facts 覆盖
        """
        if not key_facts:
            return (1, "未定义 key_facts，默认给 1 分")

        hits = sum(
            1 for fact in key_facts
            if self._fact_in_text(fact, answer.lower())
        )
        hit_rate = hits / len(key_facts)

        if hit_rate >= 0.9:
            return (2, f"完整覆盖 ({hits}/{len(key_facts)} 关键事实)")
        elif hit_rate >= 0.5:
            return (1, f"部分覆盖 ({hits}/{len(key_facts)} 关键事实)")
        else:
            return (0, f"覆盖不足 ({hits}/{len(key_facts)} 关键事实)")

    # ─── Dim 4: Traceability ──────────────────────────────────

    def score_traceability(
        self, answer: str, chunks: list[dict]
    ) -> tuple[int, str]:
        """
        评分维度 4：来源可追溯性。

        评分标准：
          2: 明确引用来源（如"根据图书馆网站..."）
          1: 隐式有据（内容来自 chunk 但未明确引用）
          0: 完全无来源标识

        检测方法：在答案中搜索引用标记和来源关键词。
        """
        # 检测显式引用
        citation_patterns = [
            r'根据.*网站', r'来自.*官网', r'来源[：:].*',
            r'参考.*资料', r'.*指出', r'.*显示',
            r'\[.*\]',  # [来源:library] 这种格式
        ]

        explicit_citations = 0
        for pattern in citation_patterns:
            if re.search(pattern, answer):
                explicit_citations += 1

        if explicit_citations >= 2:
            return (2, f"明确引用来源 ({explicit_citations} 处)")
        elif explicit_citations >= 1:
            return (1, "有一处来源引用，但不够详细")
        else:
            # 检查是否至少提到了信息来源类型
            source_keywords = ["官网", "图书馆", "网站", "手册", "公告",
                              "通知", "介绍", "页面"]
            if any(kw in answer for kw in source_keywords):
                return (1, "隐式提及来源，但未明确引用")
            return (0, "未引用任何来源")

    # ─── Dim 5: Abstention Quality ────────────────────────────

    def score_abstention(
        self, answer: str, expected_abstain: bool, did_abstain: bool
    ) -> tuple[int, str]:
        """
        评分维度 5：拒答质量 ★ 我们的独有维度。

        评分标准：
        对于 expected_abstain=True 的问题（范围外/幻觉诱导）：
          2: 正确拒答并解释原因
          1: 部分拒答（表示不确定但仍给出猜测）
          0: 自信地给出幻觉答案

        对于 expected_abstain=False 的问题（正常问题）：
          2: 正常回答，没有不必要地拒答
          1: 过度保守（不必要的迟疑）
          0: 错误拒绝了本应回答的问题
        """
        if expected_abstain:
            # 期望拒答
            if did_abstain:
                # 检查是否解释了拒答原因
                if any(kw in answer for kw in ["未找到", "无法", "不在", "超出", "建议直接"]):
                    return (2, "正确拒答并解释原因")
                return (1, "拒答但未充分解释")
            else:
                return (0, "应拒答但未拒答（可能产生幻觉）")
        else:
            # 期望回答
            if did_abstain:
                return (0, "错误拒绝了一个本应回答的问题")
            else:
                # 检查是否过度保守
                hedge_keywords = ["可能", "或许", "不确定", "建议核实", "仅供参考"]
                hedge_count = sum(1 for kw in hedge_keywords if kw in answer)
                if hedge_count >= 3:
                    return (1, f"回答但过度保守 ({hedge_count} 处迟疑)")
                return (2, "正常回答，无不当拒答")

    # ─── LLM 辅助 Grounding (替代方案) ──────────────────────

    def score_grounding_llm(
        self, answer: str, chunks: list[dict], llm_fn=None
    ) -> tuple[int, str]:
        """
        使用 LLM 进行更精确的 grounding 检查。

        相比 2-gram 规则版，LLM 版能理解语义改写，不会因词汇差异而误判。
        需要额外 API 调用。仅在 `llm_fn` 可用时启用。
        """
        if not chunks or llm_fn is None:
            return self.score_grounding(answer, chunks)

        chunks_text = ""
        for i, c in enumerate(chunks, 1):
            txt = c.get("raw_text", c.get("text", ""))
            if txt.startswith("[来源:") or txt.startswith("[背景:"):
                while txt.startswith("[") and "] " in txt[:30]:
                    txt = txt[txt.find("] ") + 2:]
            chunks_text += f"[{i}] {txt[:300]}\n"

        prompt = f"""请检查以下回答中的每个关键论断是否在参考资料中有明确依据。

回答：{answer}

参考资料：
{chunks_text}

请回复JSON：
{{"grounded": true/false, "unsupported": ["无依据的论断"], "score": 0/1/2}}

评分标准：2=全部有据, 1=部分有据, 0=无依据。只输出JSON。"""

        try:
            response = llm_fn("你是事实核查员。只输出JSON。", prompt)
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            result = json.loads(response.strip())
            score = int(result.get("score", 1))
            return (score, result.get("unsupported", []))
        except Exception:
            return self.score_grounding(answer, chunks)  # fallback

    # ─── 综合评分 ─────────────────────────────────────────────

    def evaluate(
        self,
        answer: str,
        chunks: list[dict],
        question_meta: dict,
        did_abstain: bool = False,
    ) -> dict:
        """
        对单个回答执行完整的五维度评分。

        参数：
            answer: LLM 生成的回答
            chunks: 用于生成回答的检索 chunk
            question_meta: 测试题目的元数据（含 ground_truth, key_facts 等）
            did_abstain: 系统是否触发了拒答

        返回：
            {
                "q_id": str,
                "total_score": int (0-10),
                "dimensions": {
                    "correctness": {"score": int, "explanation": str},
                    "grounding": {"score": int, "explanation": str},
                    "completeness": {"score": int, "explanation": str},
                    "traceability": {"score": int, "explanation": str},
                    "abstention": {"score": int, "explanation": str},
                },
                "metrics": {
                    "keyword_hit_rate": float,
                    "answer_length": int,
                    "num_chunks_used": int,
                },
            }
        """
        q_id = question_meta.get("q_id", "unknown")
        ground_truth = question_meta.get("ground_truth", "")
        key_facts = question_meta.get("key_facts", [])
        expected_abstain = question_meta.get("expected_abstain", False)

        # 五维度评分
        dim1 = self.score_correctness(answer, ground_truth, key_facts)
        dim2 = self.score_grounding(answer, chunks)
        dim3 = self.score_completeness(answer, key_facts)
        dim4 = self.score_traceability(answer, chunks)
        dim5 = self.score_abstention(answer, expected_abstain, did_abstain)

        dimensions = {
            "correctness": {"score": dim1[0], "explanation": dim1[1]},
            "grounding": {"score": dim2[0], "explanation": dim2[1]},
            "completeness": {"score": dim3[0], "explanation": dim3[1]},
            "traceability": {"score": dim4[0], "explanation": dim4[1]},
            "abstention": {"score": dim5[0], "explanation": dim5[1]},
        }

        total = sum(d["score"] for d in dimensions.values())

        # 额外指标
        keyword_hits = sum(
            1 for fact in key_facts
            if self._fact_in_text(fact, answer.lower())
        )
        hit_rate = keyword_hits / max(len(key_facts), 1)

        result = {
            "q_id": q_id,
            "total_score": total,
            "dimensions": dimensions,
            "metrics": {
                "keyword_hit_rate": round(hit_rate, 3),
                "answer_length": len(answer),
                "num_chunks_used": len(chunks),
            },
        }

        self.scores_log.append(result)
        return result

    def get_aggregate(self) -> dict:
        """计算所有已评分问题的汇总统计。"""
        if not self.scores_log:
            return {}

        n = len(self.scores_log)
        dim_avgs = {}
        for dim in ["correctness", "grounding", "completeness", "traceability", "abstention"]:
            scores = [s["dimensions"][dim]["score"] for s in self.scores_log]
            dim_avgs[dim] = {
                "mean": round(sum(scores) / n, 2),
                "std": round(self._std(scores), 2),
                "total_2": sum(1 for s in scores if s == 2),
                "total_1": sum(1 for s in scores if s == 1),
                "total_0": sum(1 for s in scores if s == 0),
            }

        totals = [s["total_score"] for s in self.scores_log]
        return {
            "num_questions": n,
            "dimensions": dim_avgs,
            "total_score": {
                "mean": round(sum(totals) / n, 2),
                "std": round(self._std(totals), 2),
                "max_possible": 10,
            },
            "avg_keyword_hit_rate": round(
                sum(s["metrics"]["keyword_hit_rate"] for s in self.scores_log) / n, 3
            ),
        }

    # ─── 统计显著性 ──────────────────────────────────────────

    @staticmethod
    def bootstrap_compare(scores_a: list[float], scores_b: list[float], n_bootstrap: int = 10000) -> dict:
        """
        Bootstrap 检验两组实验分数的差异是否显著。

        返回 p-value 和 95% CI。p < 0.05 表示差异显著。
        """
        import random
        random.seed(42)

        diff = [a - b for a, b in zip(scores_a, scores_b)]
        obs_mean = sum(diff) / len(diff)

        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = [random.choice(diff) for _ in range(len(diff))]
            bootstrap_means.append(sum(sample) / len(sample))

        bootstrap_means.sort()
        ci_low = bootstrap_means[int(n_bootstrap * 0.025)]
        ci_high = bootstrap_means[int(n_bootstrap * 0.975)]
        p_value = sum(1 for m in bootstrap_means if abs(m) >= abs(obs_mean)) / n_bootstrap

        return {
            "observed_diff": round(obs_mean, 3),
            "ci_95": [round(ci_low, 3), round(ci_high, 3)],
            "p_value": round(p_value, 3),
            "significant": p_value < 0.05,
        }

    # ─── 辅助方法 ─────────────────────────────────────────────

    @staticmethod
    def _fact_in_text(fact: str, text: str) -> bool:
        """检查一个 key_fact 是否出现在文本中（宽松匹配但不丢精度）。"""
        # 如果 fact 整体出现 → 最佳匹配
        if fact.lower() in text:
            return True
        # 如果 fact 包含多位数（如年份、时间），检查完整数字串
        numbers = re.findall(r'\d{2,}', fact)  # 至少2位数字才要求精确匹配
        if numbers:
            all_nums_found = all(num in text for num in numbers)
            if not all_nums_found:
                return False  # 关键数字缺失 → 不算命中
        # 单个数字（0-9）宽松匹配
        single_digits = re.findall(r'(?<!\d)\d(?!\d)', fact)
        if single_digits:
            if not any(d in text for d in single_digits):
                return False
        # 文本片段匹配：较长 fact 检查至少三分之二
        if len(fact) > 4:
            chunk_size = max(2, len(fact) // 2)
            matches = 0
            checks = 0
            for i in range(0, len(fact) - chunk_size + 1, max(1, chunk_size // 2)):
                sub = fact[i:i + chunk_size]
                if len(sub) >= 2:
                    checks += 1
                    if sub in text:
                        matches += 1
            # 至少一半的片段出现在文本中
            return checks > 0 and matches / checks >= 0.5
        return False

    @staticmethod
    def _std(values: list) -> float:
        """计算标准差（无偏估计）。"""
        if len(values) <= 1:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5


# ============================================================================
# 便捷函数
# ============================================================================

def evaluate_answer(
    answer: str,
    chunks: list[dict],
    question: dict,
    did_abstain: bool = False,
) -> dict:
    """对单个回答的便捷评分函数。"""
    evaluator = RAGEvaluator()
    return evaluator.evaluate(answer, chunks, question, did_abstain)


if __name__ == "__main__":
    # 演示评分
    evaluator = RAGEvaluator()

    mock_question = {
        "q_id": "test_001",
        "ground_truth": "图书馆周一至周五 8:00-22:00，周末 9:00-21:00",
        "key_facts": ["图书馆", "8:00", "22:00", "周一至周五"],
        "expected_abstain": False,
    }

    mock_answer = "根据图书馆网站的信息，南科大图书馆的工作日开放时间为早上8:00到晚上10:00。"

    mock_chunks = [{
        "raw_text": "图书馆服务时间：周一至周五 8:00-22:00，周末 9:00-21:00。",
    }]

    result = evaluator.evaluate(mock_answer, mock_chunks, mock_question)
    print(json.dumps(result, ensure_ascii=False, indent=2))
