"""
=============================================================================
SUSTech Error Analyzer — 失败案例分类分析
=============================================================================
系统性的错误分析是 RAG 研究中最有价值的环节之一。
这个模块把失败案例归入预定义的类别，并为每个类别提供"诊断 → 修复建议"。

五种失败类别：
  1. retrieval_failure  — 正确答案在知识库中但未被检索到
  2. context_overflow   — 上下文过长，噪声淹没信号
  3. missing_knowledge  — 知识库中根本不存在所需信息
  4. hallucination      — LLM 忽略检索到的上下文，自己编造
  5. over_refusal       — 本应回答的问题被错误拒答

分析方法：
  - 对评测结果中有低分维度的问题，分析是哪种失败
  - 统计每种失败的频率
  - 为每种失败提供具体的改进建议

使用方法：
  from evaluation.error_analyzer import analyze_errors
  report = analyze_errors(eval_results)
=============================================================================
"""

import json
from pathlib import Path

# ============================================================================
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR


class ErrorAnalyzer:
    """失败案例分类分析器。"""

    def __init__(self):
        self.categories = {
            "retrieval_failure": {
                "name": "检索失败",
                "description": "正确答案存在于知识库中，但检索系统没有将其排在 top-K",
                "indicator": "grounding_score=0 且 关键词在其他 chunk 中存在",
                "fix": "调整 chunk 大小、改进 RRF 权重、增加 top-K、优化 HyDE prompt",
            },
            "context_overflow": {
                "name": "上下文稀释",
                "description": "检索了太多不相关的 chunk，噪声淹没了有用信息",
                "indicator": "有部分正确信息但被噪声干扰，completeness < 1",
                "fix": "减小 top-K、提高 reranker 阈值、优化 chunk 质量",
            },
            "missing_knowledge": {
                "name": "知识缺失",
                "description": "问题涉及的知识不在当前知识库中",
                "indicator": "所有 chunk 的 RRF 分数都 < 0.35",
                "fix": "扩展数据源、增加新的爬取目标、添加人工整理的 FAQ",
            },
            "hallucination": {
                "name": "幻觉编造",
                "description": "LLM 忽略检索上下文，编造不存在的信息",
                "indicator": "correctness=0 且 grounding=0 但答案很长",
                "fix": "加强 prompt 约束、降低 temperature、增加 abstention 阈值",
            },
            "over_refusal": {
                "name": "过度拒答",
                "description": "系统对本应回答的问题执行了拒答",
                "indicator": "abstention=0 且 expected_abstain=False 且 did_abstain=True",
                "fix": "调低 abstention 阈值、改进 query classifier 避免误判 OUT_OF_SCOPE",
            },
        }

    def classify_failure(
        self, eval_result: dict, did_abstain: bool
    ) -> str | None:
        """
        将一个评测结果分类到一个失败类别。

        返回 None 表示没有明显失败（总分 >= 6 分视为通过）。

        参数：
            eval_result: RAGEvaluator.evaluate() 的输出
            did_abstain: 系统是否触发了拒答
        """
        dims = eval_result.get("dimensions", {})

        c = dims.get("correctness", {}).get("score", 1)
        g = dims.get("grounding", {}).get("score", 1)
        comp = dims.get("completeness", {}).get("score", 1)
        a = dims.get("abstention", {}).get("score", 1)

        total = eval_result.get("total_score", 5)

        # 总分 >= 7 → 无失败
        if total >= 7:
            return None

        # 总分 5-6 → 部分失败，记录但不分类
        if total >= 5:
            return "partial"

        # 拒答相关
        if a == 0 and not did_abstain and c == 0:
            return "over_refusal"

        # 幻觉
        if c == 0 and g == 0:
            return "hallucination"

        # 检索失败
        if g == 0:
            return "retrieval_failure"

        # 知识缺失
        if c == 0 and comp <= 1:
            return "missing_knowledge"

        # 上下文稀释
        if comp <= 1 and len(eval_result.get("chunks", [])) > 5:
            return "context_overflow"

        # 默认：检索相关
        return "retrieval_failure"

    def analyze(
        self, eval_results: list[dict], did_abstain_list: list[bool] = None
    ) -> dict:
        """
        分析一批评测结果，输出完整的错误分析报告。

        参数：
            eval_results: 评测结果列表
            did_abstain_list: 每个问题是否触发了拒答

        返回：
            错误分析报告字典
        """
        if did_abstain_list is None:
            did_abstain_list = [False] * len(eval_results)

        # 分类统计
        category_counts = {}
        category_examples = {}
        total_failures = 0

        for i, (result, did_abstain) in enumerate(
            zip(eval_results, did_abstain_list)
        ):
            category = self.classify_failure(result, did_abstain)

            if category is None:
                continue  # 通过，无失败

            total_failures += 1

            if category not in category_counts:
                category_counts[category] = 0
                category_examples[category] = []
            category_counts[category] += 1

            # 保存最多 3 个示例
            if len(category_examples[category]) < 3:
                category_examples[category].append({
                    "q_id": result.get("q_id", ""),
                    "total_score": result.get("total_score", 0),
                    "dimensions": result.get("dimensions", {}),
                })

        # 构建报告
        total = max(len(eval_results), 1)
        report = {
            "total_questions": len(eval_results),
            "total_failures": total_failures,
            "pass_rate": round(1.0 - total_failures / total, 3),
            "by_category": {},
        }

        for cat_id, cat_info in self.categories.items():
            count = category_counts.get(cat_id, 0)
            report["by_category"][cat_id] = {
                "name": cat_info["name"],
                "count": count,
                "percentage": round(count / total, 3),
                "description": cat_info["description"],
                "indicator": cat_info["indicator"],
                "fix": cat_info["fix"],
                "examples": category_examples.get(cat_id, []),
            }

        # Partial failures
        partial_count = category_counts.get("partial", 0)
        if partial_count:
            report["by_category"]["partial"] = {
                "name": "部分缺陷",
                "count": partial_count,
                "percentage": round(partial_count / total, 3),
                "description": "总分偏低但有部分正确信息",
                "fix": "综合优化：改进检索质量 + prompt 工程",
                "examples": category_examples.get("partial", []),
            }

        return report

    def print_report(self, report: dict):
        """以可读格式打印错误分析报告。"""
        print(f"\n{'='*60}")
        print(f"ERROR ANALYSIS REPORT")
        print(f"{'='*60}")
        print(f"Total questions: {report['total_questions']}")
        print(f"Failures: {report['total_failures']}")
        print(f"Pass rate: {report['pass_rate']:.1%}")
        print(f"\n{'─'*60}")
        print(f"Failure breakdown:")
        print(f"{'─'*60}")

        for cat_id, cat_data in sorted(
            report["by_category"].items(),
            key=lambda x: -x[1]["count"],
        ):
            print(f"\n  {cat_data['name']} ({cat_id})")
            print(f"    Occurrences: {cat_data['count']} ({cat_data['percentage']:.1%})")
            print(f"    Cause: {cat_data['description']}")
            print(f"    Fix: {cat_data['fix']}")
            if cat_data.get("examples"):
                print(f"    Example q_ids: {[e['q_id'] for e in cat_data['examples']]}")

        print(f"{'='*60}\n")

    def save_report(self, report: dict, path: Path = None):
        """保存错误分析报告到 JSON。"""
        if path is None:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            path = RESULTS_DIR / "error_analysis.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Error analysis saved to: {path}")


if __name__ == "__main__":
    analyzer = ErrorAnalyzer()
    analyzer.print_report(analyzer.categories)
