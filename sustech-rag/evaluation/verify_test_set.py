"""
=============================================================================
Test Set Verifier — 验证 ground_truth 是否与知识库实际内容一致
=============================================================================
因为测试集在爬取前编写，部分 ground_truth 可能与实际知识库内容不符。
此脚本逐题检查：
1. key_facts 中的每个事实是否在知识库中出现
2. 整体 ground_truth 的核心断言是否有检索结果支撑
3. 标记"高风险"题目（ground_truth 可能错误）

使用方法（在 GPU 服务器上）：
  python evaluation/verify_test_set.py

输出：
  data/test_set_verified.json — 每个题目的验证结果
=============================================================================
"""

import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR


def load_doc_texts() -> list[str]:
    """加载所有清洗后文档的纯文本。"""
    processed = DATA_DIR / "processed" / "processed_docs.jsonl"
    if not processed.exists():
        print(f"Warning: {processed} not found, trying chunks...")
        processed = DATA_DIR / "chunks" / "chunks_default.jsonl"

    texts = []
    with open(processed) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
                text = doc.get("clean_text", doc.get("raw_text", doc.get("text", "")))
                texts.append(text)
            except json.JSONDecodeError:
                continue
    return texts


def verify_fact(fact: str, all_text: str) -> bool:
    """检查一个事实是否在全文集合中出现。"""
    return fact.lower() in all_text.lower()


def verify_question(q: dict, all_text: str, doc_texts: list[str]) -> dict:
    """验证单个题目的 ground_truth。"""
    q_id = q["q_id"]
    key_facts = q.get("key_facts", [])
    ground_truth = q.get("ground_truth", "")

    result = {
        "q_id": q_id,
        "question": q["question"],
        "facts_checked": {},
        "issues": [],
    }

    # 检查每个 key_fact
    for fact in key_facts:
        if not fact:
            continue
        found = verify_fact(fact, all_text)
        result["facts_checked"][fact] = found
        if not found:
            result["issues"].append(f"key_fact '{fact}' 在知识库中未找到")

    # 检查 ground_truth 核心数字/实体
    import re
    gt_numbers = re.findall(r'\d{2,}', ground_truth)
    for num in gt_numbers:
        if num not in all_text:
            result["issues"].append(f"ground_truth 中的数字 '{num}' 在知识库中未找到")

    # 检查是否有检索结果（用关键词搜索）
    question_words = q["question"][:20]  # 用问题前几个词作为搜索线索
    matching_docs = sum(1 for d in doc_texts if any(
        kw in d for kw in q["question"].split()[:5] if len(kw) >= 2
    ))
    result["matching_docs_count"] = matching_docs

    if matching_docs == 0:
        result["issues"].append("知识库中完全找不到相关文档")

    # 风险等级
    if len(result["issues"]) >= 3:
        result["risk"] = "high"
    elif len(result["issues"]) >= 1:
        result["risk"] = "medium"
    else:
        result["risk"] = "low"

    return result


def main():
    print("Loading document corpus...")
    doc_texts = load_doc_texts()
    all_text = " ".join(doc_texts)
    print(f"Loaded {len(doc_texts)} documents, {len(all_text)} chars total\n")

    # 加载测试集
    test_path = DATA_DIR / "test_set.json"
    with open(test_path) as f:
        test_set = json.load(f)

    print(f"Verifying {len(test_set)} questions...\n")

    results = []
    high_risk = []
    by_risk = {"high": 0, "medium": 0, "low": 0}

    for q in test_set:
        if q.get("expected_abstain"):
            # OOS 问题不需要验证 ground_truth
            results.append({
                "q_id": q["q_id"],
                "question": q["question"],
                "risk": "n/a (OOS)",
                "issues": [],
            })
            continue

        r = verify_question(q, all_text, doc_texts)
        results.append(r)
        by_risk[r["risk"]] = by_risk.get(r["risk"], 0) + 1

        if r["risk"] == "high":
            high_risk.append(r)
            print(f"🔴 HIGH RISK: {r['q_id']} — {q['question'][:60]}")
            for issue in r["issues"]:
                print(f"   ⚠ {issue}")
        elif r["risk"] == "medium":
            print(f"🟡 MEDIUM: {r['q_id']} — {', '.join(r['issues'])}")

    # 汇总
    print(f"\n{'='*60}")
    print(f"Verification Summary")
    print(f"{'='*60}")
    print(f"Total questions (non-OOS): {len([r for r in results if r['risk'] != 'n/a (OOS)'])}")
    print(f"  Low risk:    {by_risk.get('low', 0)}")
    print(f"  Medium risk: {by_risk.get('medium', 0)}")
    print(f"  High risk:   {by_risk.get('high', 0)}")

    if high_risk:
        print(f"\n🔴 HIGH RISK questions need manual review:")
        for r in high_risk:
            print(f"  {r['q_id']}: {r['question'][:80]}")
            for issue in r["issues"]:
                print(f"    - {issue}")

    # 保存
    out_path = DATA_DIR / "test_set_verified.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
