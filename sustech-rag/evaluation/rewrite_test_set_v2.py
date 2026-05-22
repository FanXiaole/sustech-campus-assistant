"""
=============================================================================
Test Set Rewriter v2 — BM25 + LLM，无需 GPU
=============================================================================
使用 BM25（轻量级，不需要 GPU）检索相关文档，然后 LLM 验证并重写。
"""

import json
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR


OOS_GROUND_TRUTH = "知识库无信息"
OOS_NOTE_BM25 = "BM25 无结果"
OOS_NOTE_LLM = "LLM 判定知识库无答案"

REWRITE_PROMPT = """你是校园知识库审核员。基于以下信息修正一道测试题的参考答案。

题目：{question}
原参考答案：{original_gt}
原关键事实：{original_kf}

知识库中检索到的相关资料：
{chunks}

如果资料包含答案，基于资料重写 ground_truth 和 key_facts。
如果资料不包含答案，标记 answerable=false。
key_facts 必须使用资料中实际出现的词语（2-6字短词），不要编造。

只输出JSON：
{{"answerable": true/false, "new_ground_truth": "...", "new_key_facts": ["...", "..."], "note": "..."}}"""


def load_bm25_and_chunks():
    """加载 BM25 索引和对应的 chunks。"""
    idx_path = DATA_DIR.parent / "index_store" / "bm25_index.pkl"
    with open(idx_path, "rb") as f:
        bundle = pickle.load(f)

    bm25_model = bundle["index"]  # BM25Okapi 对象
    bm25_chunks = bundle["chunks"]  # 原始 chunks 列表
    return bm25_model, bm25_chunks


def tokenize(text: str) -> list[str]:
    """Jieba 分词。"""
    import jieba
    return [w.strip() for w in jieba.cut(text) if len(w.strip()) >= 1]


def main():
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("ERROR: DEEPSEEK_API_KEY not set")
        sys.exit(1)

    from generation.llm_api import DeepSeekClient

    print("Loading BM25 index and chunks...")
    bm25, chunks = load_bm25_and_chunks()
    print(f"Loaded {len(chunks)} chunks, BM25 ready\n")

    with open(DATA_DIR / "test_set.json") as f:
        test_set = json.load(f)

    llm = DeepSeekClient()
    new_test_set = []
    stats = {"answerable": 0, "not_answerable": 0, "oos": 0}

    for i, q in enumerate(test_set):
        q_id = q["q_id"]

        if q.get("expected_abstain"):
            new_test_set.append({**q, "verified": True, "verification_note": "OOS (original)"})
            stats["oos"] += 1
            continue

        print(f"[{i+1}/{len(test_set)}] {q_id}: {q['question'][:60]}", end=" ", flush=True)

        # BM25 search: get scores for all docs, pick top-5
        tokens = tokenize(q["question"])
        scores = bm25.get_scores(tokens)
        # 按分数降序排列，取 top-5 索引
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
        results = [chunks[idx] for idx in top_indices if scores[idx] > 0]

        if not results:
            new_q = {**q, "ground_truth": OOS_GROUND_TRUTH, "key_facts": [],
                     "expected_abstain": True, "verified": True,
                     "verification_note": OOS_NOTE_BM25}
            new_test_set.append(new_q)
            stats["not_answerable"] += 1
            print("-> no BM25 results -> OOS")
            continue

        # Build context for LLM
        ctx = ""
        for j, c in enumerate(results, 1):
            txt = c.get("raw_text", c.get("text", ""))[:350]
            src = c.get("source_family", "unknown")
            ctx += f"[{j}] src:{src}\n{txt}\n\n"

        prompt = REWRITE_PROMPT.format(
            question=q["question"],
            original_gt=q.get("ground_truth", ""),
            original_kf=json.dumps(q.get("key_facts", []), ensure_ascii=False),
            chunks=ctx,
        )

        try:
            resp = llm.chat("你是事实审核员。只输出JSON，不要加任何其他文字。",
                          prompt, temperature=0.1, max_tokens=400)
            result = llm.chat_json(
                "你是事实审核员。只输出JSON，不要加任何其他文字。",
                prompt, temperature=0.1, max_tokens=400)

            new_q = {**q, "verified": True}
            if result.get("answerable", True):
                new_q["ground_truth"] = result.get("new_ground_truth", q["ground_truth"])
                new_q["key_facts"] = result.get("new_key_facts", q["key_facts"])
                new_q["verification_note"] = result.get("note", "")
                stats["answerable"] += 1
                print(f"-> answerable | kf={new_q['key_facts']}")
            else:
                new_q["ground_truth"] = OOS_GROUND_TRUTH
                new_q["key_facts"] = []
                new_q["expected_abstain"] = True
                new_q["verification_note"] = result.get("note", OOS_NOTE_LLM)
                stats["not_answerable"] += 1
                print("-> NOT answerable")

            new_test_set.append(new_q)

        except Exception as e:
            print(f"-> LLM error: {e}")
            new_test_set.append({**q, "verified": False, "verification_note": str(e)})

        time.sleep(0.3)

    # Save
    out_path = DATA_DIR / "test_set_v2.json"
    with open(out_path, "w") as f:
        json.dump(new_test_set, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"Done. Saved to {out_path}")
    print(f"  Answerable:     {stats['answerable']}")
    print(f"  Newly OOS:      {stats['not_answerable']}")
    print(f"  Original OOS:   {stats['oos']}")

    # Show changes
    print(f"\nKey fact changes:")
    for old, new in zip(test_set, new_test_set):
        if old.get("key_facts") != new.get("key_facts"):
            print(f"  {new['q_id']}:")
            print(f"    Old: {old.get('key_facts')}")
            print(f"    New: {new.get('key_facts')}")
        if old.get("expected_abstain") != new.get("expected_abstain"):
            print(f"  {new['q_id']}: expected_abstain {old.get('expected_abstain')} -> {new.get('expected_abstain')}")


if __name__ == "__main__":
    main()
