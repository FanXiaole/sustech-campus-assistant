"""
=============================================================================
Bootstrap CI Runner — 实验显著性检验
=============================================================================
对已完成的实验进行 pairwise bootstrap 检验，判断组间差异是否显著。

使用方法（在 GPU 服务器上）：
  python evaluation/run_bootstrap.py

输出：
  results/bootstrap_ci.json — 所有实验对的 p-value 和 95% CI
=============================================================================
"""

import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RESULTS_DIR
from evaluation.evaluator import RAGEvaluator


def load_scores(exp_id: str) -> list[float]:
    """加载某个实验的逐题总分。"""
    path = RESULTS_DIR / exp_id / "scores_per_question.json"
    if not path.exists():
        print(f"  SKIP {exp_id}: no scores file at {path}")
        return []
    with open(path) as f:
        data = json.load(f)
    return [s["total_score"] for s in data]


def main():
    experiments = ["R0", "R1", "R2", "R3", "R4", "E1", "E2", "E3", "E4", "A1", "A2"]
    available = [e for e in experiments if (RESULTS_DIR / e).exists()]

    if len(available) < 2:
        print("Need at least 2 experiments with results. Found:", available)
        return

    print(f"Found {len(available)} experiments: {available}\n")

    # 关键对比对
    pairs = [
        ("R1", "R0", "Dense vs No RAG"),
        ("R3", "R1", "Hybrid RRF vs Dense"),
        ("R4", "R3", "Reranker vs Hybrid RRF"),
        ("R4", "R1", "Reranker vs Dense"),
        ("E1", "R3", "HyDE vs Hybrid RRF"),
        ("E2", "R3", "Enrichment vs Hybrid RRF"),
        ("E3", "R3", "Classifier vs Hybrid RRF"),
        ("E4", "R4", "Full Innovation vs Best"),
        ("A1", "R4", "Small Chunks vs Default"),
        ("A2", "R4", "Large Chunks vs Default"),
    ]

    results = {}
    print(f"{'Comparison':<40} {'Diff':>8} {'CI Low':>8} {'CI High':>8} {'p-val':>8} {'Sig?':>6}")
    print("-" * 82)

    for exp_a, exp_b, label in pairs:
        if exp_a not in available or exp_b not in available:
            continue
        scores_a = load_scores(exp_a)
        scores_b = load_scores(exp_b)
        if not scores_a or not scores_b:
            continue

        ci = RAGEvaluator.bootstrap_compare(scores_a, scores_b)
        results[f"{exp_a}_vs_{exp_b}"] = {
            "label": label,
            **ci,
        }

        sig = "YES" if ci["significant"] else "no"
        print(f"{label:<40} {ci['observed_diff']:>+8.3f} {ci['ci_95'][0]:>8.3f} {ci['ci_95'][1]:>8.3f} {ci['p_value']:>8.3f} {sig:>6}")

    # 保存
    out_path = RESULTS_DIR / "bootstrap_ci.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {out_path}")

    # 汇总真正显著的发现
    significant = [(k, v) for k, v in results.items() if v["significant"]]
    if significant:
        print("\nStatistically significant differences (p < 0.05):")
        for k, v in significant:
            print(f"  {v['label']}: diff={v['observed_diff']:+.3f}, p={v['p_value']:.3f}")
    else:
        print("\nNo statistically significant differences found.")


if __name__ == "__main__":
    main()
