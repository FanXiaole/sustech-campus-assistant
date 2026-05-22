"""
=============================================================================
LLM-based Evaluator — 使用 DeepSeek 进行语义级五维评分
=============================================================================
与规则版 evaluator.py 并行运行，对比两者的评分差异。
LLM 版优势：理解语义改写，不会因词汇差异误判（如"巴士"vs"校车"）。

使用方法（在 GPU 服务器上，需要 DEEPSEEK_API_KEY）：
  python evaluation/llm_evaluator.py --experiment R4

输出：
  results/{exp}/llm_scores.json — LLM 评分结果
  results/{exp}/llm_vs_rule_comparison.json — LLM vs 规则版对比
=============================================================================
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR, DATA_DIR


EVAL_PROMPT = """你是 RAG 系统评测专家。请对以下问答进行五维度评分。

【用户问题】{question}
【参考答案 (ground_truth)】{ground_truth}
【关键事实 (key_facts)】{key_facts}
【期望拒答 (expected_abstain)】{expected_abstain}

【系统回答】{answer}

【检索到的资料】
{chunks}

评分标准（每维 0/1/2 分）：
1. correctness（正确性）：答案事实是否正确？与 ground_truth 是否一致？
   2=完全正确，1=部分正确，0=错误
2. grounding（依据性）：回答中的论断是否能在资料中找到依据？
   2=全部有据，1=部分有据，0=无依据
3. completeness（完整性）：是否覆盖所有 key_facts？
   2=全部覆盖，1=部分覆盖，0=几乎未覆盖
4. traceability（可追溯性）：是否引用了来源？
   2=明确引用，1=隐式提及，0=无引用
5. abstention（拒答质量）：对于应该拒答的问题是否正确拒答？
   2=正确处理（该拒则拒，该答则答），1=处理不当，0=完全错误

只输出JSON（不要加其他文字）：
{{"correctness": {{"score": 0-2, "reason": "简短理由"}},
 "grounding": {{"score": 0-2, "reason": "简短理由"}},
 "completeness": {{"score": 0-2, "reason": "简短理由"}},
 "traceability": {{"score": 0-2, "reason": "简短理由"}},
 "abstention": {{"score": 0-2, "reason": "简短理由"}},
 "overall_assessment": "一句话总结"}}"""


def load_test_set() -> list[dict]:
    path = DATA_DIR / "test_set_v2.json"
    if not path.exists():
        path = DATA_DIR / "test_set.json"
    with open(path) as f:
        return json.load(f)


def run_llm_evaluation(experiment_id: str):
    """对指定实验的每道题用 LLM 评分。"""
    from generation.llm_api import DeepSeekClient

    # 加载规则版评分结果
    scores_path = RESULTS_DIR / experiment_id / "scores_per_question.json"
    if not scores_path.exists():
        print(f"ERROR: No scores found for {experiment_id}")
        return

    with open(scores_path) as f:
        rule_scores = json.load(f)

    test_set = load_test_set()
    llm = DeepSeekClient()
    if not llm.is_available:
        print("ERROR: DeepSeek API not available")
        return

    llm_results = []
    comparison = []
    dims = ["correctness", "grounding", "completeness", "traceability", "abstention"]

    for i, (q, rule) in enumerate(zip(test_set, rule_scores)):
        if q["q_id"] != rule["q_id"]:
            print(f"  WARNING: q_id mismatch {q['q_id']} vs {rule['q_id']}")
            continue

        print(f"[{i+1}/{len(test_set)}] {q['q_id']}: {q['question'][:50]}", end=" ", flush=True)

        # 使用保存的实际答案
        answer_text = rule.get("answer", "[答案不可用]")
        if answer_text == "[答案不可用]":
            print("no answer saved, skipping")
            continue

        # 构建检索资料上下文（从 rule dimensions 提取）
        chunks_info = f"规则评分参考: correctness={rule['dimensions']['correctness']['score']}, grounding={rule['dimensions']['grounding']['score']}, completeness={rule['dimensions']['completeness']['score']}"

        prompt = EVAL_PROMPT.format(
            question=q["question"],
            ground_truth=q.get("ground_truth", ""),
            key_facts=json.dumps(q.get("key_facts", []), ensure_ascii=False),
            expected_abstain=q.get("expected_abstain", False),
            answer=answer_text,
            chunks=chunks_info,
        )

        try:
            result = llm.chat_json(
                "你是RAG评测专家。只输出JSON。", prompt,
                temperature=0.1, max_tokens=512)

            if "error" in result:
                print(f"LLM error: {result['error']}")
                continue

            llm_total = sum(result[d]["score"] for d in dims if d in result)
            rule_total = rule["total_score"]

            llm_results.append({"q_id": q["q_id"], "llm_total": llm_total, "dimensions": result})
            comparison.append({
                "q_id": q["q_id"],
                "rule_total": rule_total,
                "llm_total": llm_total,
                "delta": llm_total - rule_total,
            })

            print(f"rule={rule_total} llm={llm_total} Δ={llm_total-rule_total:+d}")

        except Exception as e:
            print(f"error: {e}")

        time.sleep(0.3)

    # 保存
    out_dir = RESULTS_DIR / experiment_id
    with open(out_dir / "llm_scores.json", "w") as f:
        json.dump(llm_results, f, ensure_ascii=False, indent=2)

    if comparison:
        avg_rule = sum(c["rule_total"] for c in comparison) / len(comparison)
        avg_llm = sum(c["llm_total"] for c in comparison) / len(comparison)
        avg_delta = sum(c["delta"] for c in comparison) / len(comparison)

        comp_summary = {
            "experiment": experiment_id,
            "num_questions": len(comparison),
            "avg_rule_score": round(avg_rule, 2),
            "avg_llm_score": round(avg_llm, 2),
            "avg_delta": round(avg_delta, 2),
            "per_question": comparison,
            "llm_higher": sum(1 for c in comparison if c["delta"] > 0),
            "llm_lower": sum(1 for c in comparison if c["delta"] < 0),
            "same": sum(1 for c in comparison if c["delta"] == 0),
        }

        with open(out_dir / "llm_vs_rule_comparison.json", "w") as f:
            json.dump(comp_summary, f, ensure_ascii=False, indent=2)

        print(f"\nLLM vs Rule Comparison ({experiment_id}):")
        print(f"  Avg Rule Score: {avg_rule:.2f}")
        print(f"  Avg LLM Score:  {avg_llm:.2f}")
        print(f"  Avg Delta:      {avg_delta:+.2f}")
        print(f"  LLM higher: {comp_summary['llm_higher']}, lower: {comp_summary['llm_lower']}, same: {comp_summary['same']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", "-e", required=True, help="Experiment ID (e.g., R4)")
    args = parser.parse_args()
    run_llm_evaluation(args.experiment)
