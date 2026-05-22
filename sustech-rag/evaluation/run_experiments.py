"""
=============================================================================
SUSTech Experiment Runner — 全实验矩阵执行器
=============================================================================
⚠️ PARTIAL GPU REQUIRED — R3, R4, E1, E4 等配置需要 GPU。
   运行前请确保 AutoDL RTX 6000 Blackwell 实例已启动。

实验矩阵（10 个配置）：
  Baseline:
    R0: no_rag           → 纯 LLM，零检索
    R1: dense_only       → bge-m3 top-5 → LLM
    R2: bm25_only        → BM25 top-5 → LLM

  Core RAG:
    R3: hybrid_rrf       → Dense(50)+BM25(50) → RRF → top-5 → LLM
    R4: hybrid_full      → R3 + Reranker → top-5 → LLM

  Innovation ablations:
    E1: hyde_hybrid      → HyDE + R3
    E2: enriched_hybrid  → Contextual enrichment chunks + R3
    E3: classified_route → Query classifier routing + R3
    E4: full_innovation  → All innovations combined ★

  Chunk size ablations:
    A1: chunks_small     → R4 with 300-char chunks
    A2: chunks_large     → R4 with 900-char chunks

每个实验的产出：
  results/{experiment_id}/
    ├── scores_per_question.json
    ├── aggregate_scores.json
    ├── examples/  (5 个随机 QA 示例)
    ├── latency_stats.json
    └── pipeline_traces.json

使用方法：
  python evaluation/run_experiments.py --experiments R0,R1,R2  # 只跑 baseline
  python evaluation/run_experiments.py --all                     # 全跑

=============================================================================
"""

import json
import os
import time
from pathlib import Path

# ============================================================================
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.hybrid_rrf import hybrid_retrieve
from retrieval.dense_retriever import get_dense_retriever

from config import (
    ABSTAIN_MESSAGE,
    CHUNK_DIR,
    DATA_DIR,
    RESULTS_DIR,
    DENSE_TOP_K,
    BM25_TOP_K,
    RRF_FUSION_TOP,
    RERANK_TOP_K,
)


class ExperimentRunner:
    """
    实验矩阵执行器。

    负责加载组件、按配置运行评测、收集结果。
    """

    def __init__(self):
        self.results_dir = RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 这些组件在需要时才加载（懒加载）
        self._test_set = None
        self._dense_retriever = None
        self._dense_retrievers: dict[str, object] = {}  # 按 collection 缓存
        self._sparse_retriever = None
        self._reranker = None
        self._authority_scorer = None
        self._classifier = None
        self._prompt_builder = None
        self._llm_client = None

    # ─── 懒加载属性 ───────────────────────────────────────

    @property
    def test_set(self):
        if self._test_set is None:
            # 优先使用 v2 验证版测试集
            test_path_v2 = DATA_DIR / "test_set_v2.json"
            test_path = DATA_DIR / "test_set.json"
            use_path = test_path_v2 if test_path_v2.exists() else test_path
            if not use_path.exists():
                raise FileNotFoundError(
                    f"Test set not found. "
                    f"Run evaluation/test_set_builder.py first."
                )
            with open(use_path, "r", encoding="utf-8") as f:
                self._test_set = json.load(f)
            label = "v2 (verified)" if use_path == test_path_v2 else "v1 (original)"
            print(f"[Runner] Loaded {len(self._test_set)} test questions ({label})")
        return self._test_set

    @property
    def dense_retriever(self):
        if self._dense_retriever is None:
            from retrieval.dense_retriever import get_dense_retriever
            self._dense_retriever = get_dense_retriever()
        return self._dense_retriever

    @property
    def sparse_retriever(self):
        if self._sparse_retriever is None:
            from retrieval.sparse_retriever import get_sparse_retriever
            self._sparse_retriever = get_sparse_retriever()
        return self._sparse_retriever

    @property
    def reranker(self):
        if self._reranker is None:
            from retrieval.reranker import get_reranker
            self._reranker = get_reranker()
        return self._reranker

    @property
    def authority_scorer(self):
        if self._authority_scorer is None:
            from retrieval.authority_scorer import get_authority_scorer
            self._authority_scorer = get_authority_scorer()
        return self._authority_scorer

    @property
    def classifier(self):
        if self._classifier is None:
            from retrieval.query_classifier import get_classifier
            self._classifier = get_classifier(mode="rule")
        return self._classifier

    @property
    def prompt_builder(self):
        if self._prompt_builder is None:
            from generation.prompt_builder import PromptBuilder
            self._prompt_builder = PromptBuilder(persona="default")
        return self._prompt_builder

    @property
    def llm_client(self):
        if self._llm_client is None:
            from generation.llm_api import DeepSeekClient
            from generation.llm_local import OllamaClient, LLMFallback
            api = DeepSeekClient()
            local = OllamaClient()
            self._llm_client = LLMFallback(api, local)
        return self._llm_client

    # ─── 实验配置定义 ────────────────────────────────────

    def get_configs(self) -> dict:
        """返回所有实验配置的定义。"""
        return {
            # ── Baselines ──
            "R0": {
                "name": "No RAG (LLM Only)",
                "retrieval": "none",
                "reranker": False,
                "hyde": False,
                "classifier": False,
                "authority": False,
                "collection": "sustech_default",
            },
            "R1": {
                "name": "Dense Only",
                "retrieval": "dense",
                "dense_top_k": RERANK_TOP_K,
                "reranker": False,
                "hyde": False,
                "classifier": False,
                "authority": False,
                "collection": "sustech_default",
            },
            "R2": {
                "name": "BM25 Only",
                "retrieval": "bm25",
                "bm25_top_k": RERANK_TOP_K,
                "reranker": False,
                "hyde": False,
                "classifier": False,
                "authority": False,
                "collection": "sustech_default",
            },

            # ── Core RAG ──
            "R3": {
                "name": "Hybrid RRF",
                "retrieval": "hybrid",
                "reranker": False,
                "hyde": False,
                "classifier": False,
                "authority": False,
                "collection": "sustech_default",
            },
            "R4": {
                "name": "Hybrid + Reranker",
                "retrieval": "hybrid",
                "reranker": True,
                "hyde": False,
                "classifier": False,
                "authority": False,
                "collection": "sustech_default",
            },

            # ── Innovation Ablations ──
            "E1": {
                "name": "HyDE + Hybrid RRF",
                "retrieval": "hybrid",
                "reranker": False,
                "hyde": True,
                "classifier": False,
                "authority": False,
                "collection": "sustech_default",
            },
            "E2": {
                "name": "Enriched + Hybrid RRF",
                "retrieval": "hybrid",
                "reranker": False,
                "hyde": False,
                "classifier": False,
                "authority": False,
                "collection": "sustech_enriched",
            },
            "E3": {
                "name": "Classifier + Hybrid RRF",
                "retrieval": "hybrid",
                "reranker": False,
                "hyde": False,
                "classifier": True,
                "authority": False,
                "collection": "sustech_default",
            },
            "E4": {
                "name": "Full Innovation Stack",
                "retrieval": "hybrid",
                "reranker": True,
                "hyde": True,
                "classifier": True,
                "authority": True,
                "collection": "sustech_default",
            },

            # ── Authority standalone ablation ──
            "E5": {
                "name": "R4 + Authority Scorer",
                "retrieval": "hybrid",
                "reranker": True,
                "hyde": False,
                "classifier": False,
                "authority": True,
                "collection": "sustech_default",
            },

            # ── Chunk Size Ablations ──
            "A1": {
                "name": "R4 + Small Chunks (300)",
                "retrieval": "hybrid",
                "reranker": True,
                "hyde": False,
                "classifier": False,
                "authority": False,
                "collection": "sustech_small",
            },
            "A2": {
                "name": "R4 + Large Chunks (900)",
                "retrieval": "hybrid",
                "reranker": True,
                "hyde": False,
                "classifier": False,
                "authority": False,
                "collection": "sustech_large",
            },
        }

    # ─── 执行单个实验 ────────────────────────────────────

    def run_experiment(self, exp_id: str, config: dict) -> dict:
        """
        执行一个实验配置，对所有 50 个问题进行评测。

        参数：
            exp_id: 实验标识（"R0", "R3" 等）
            config: 实验配置字典

        返回：
            该实验的汇总结果
        """
        print(f"\n{'='*60}")
        print(f"Running: {exp_id} — {config['name']}")
        print(f"{'='*60}")

        eval_results = []
        latency_stats = []
        did_abstain_list = []
        pipeline_traces = []

        from evaluation.evaluator import RAGEvaluator
        evaluator = RAGEvaluator()  # 每个实验独立 evaluator

        for q in self.test_set:
            query = q["question"]
            q_id = q["q_id"]

            t_start = time.time()

            # ── 检索阶段 ──
            chunks = []
            trace = {}

            retrieval_mode = config["retrieval"]

            if retrieval_mode == "none":
                # R0: No retrieval
                chunks = []
                trace = {"mode": "none"}

            elif retrieval_mode == "dense":
                # R1: Dense only
                chunks = self.dense_retriever.search(query, top_k=config.get("dense_top_k", DENSE_TOP_K))
                trace = {"mode": "dense", "top_k": len(chunks)}

            elif retrieval_mode == "bm25":
                # R2: BM25 only
                chunks = self.sparse_retriever.search(
                    query, top_k=config.get("bm25_top_k", BM25_TOP_K)
                )
                trace = {"mode": "bm25", "top_k": len(chunks)}

            elif retrieval_mode == "hybrid":
                # R3/R4/E1-E4/A1-A2/E5: Full hybrid pipeline

                # 按需加载 dense retriever（按 collection 缓存）
                collection_name = config.get("collection", "sustech_default")
                if collection_name == "sustech_default":
                    dense = self.dense_retriever
                elif collection_name in self._dense_retrievers:
                    dense = self._dense_retrievers[collection_name]
                else:
                    dense = get_dense_retriever(collection_name=collection_name)
                    self._dense_retrievers[collection_name] = dense

                # HyDE LLM function
                hyde_fn = None
                if config.get("hyde"):
                    def hyde_fn(system, user):
                        return self.llm_client.chat(system, user, max_tokens=256)

                chunks, trace = hybrid_retrieve(
                    query=query,
                    dense_retriever=dense,
                    sparse_retriever=self.sparse_retriever,
                    use_hyde=config.get("hyde", False),
                    hyde_llm_fn=hyde_fn,
                    use_classifier=config.get("classifier", False),
                    classifier=self.classifier if config.get("classifier") else None,
                    reranker=self.reranker if config.get("reranker") else None,
                    authority_scorer=self.authority_scorer if config.get("authority") else None,
                    abstention_check=True,
                )

            t_retrieval = (time.time() - t_start) * 1000

            # ── 拒答检测 ──
            did_abstain = len(chunks) == 0

            # ── 生成阶段 ──
            answer = ""
            t_gen_start = time.time()

            if did_abstain:
                answer = ABSTAIN_MESSAGE
            else:
                system_prompt = self.prompt_builder.build_system_prompt()
                user_message = self.prompt_builder.build_user_message(query, chunks)
                try:
                    answer = self.llm_client.chat(system_prompt, user_message)
                except Exception as e:
                    answer = f"[生成失败: {e}]"

            t_generation = (time.time() - t_gen_start) * 1000

            # ── 评测 ──
            score = evaluator.evaluate(
                answer=answer,
                chunks=chunks,
                question_meta=q,
                did_abstain=did_abstain,
            )

            # 记录延迟
            latency_stats.append({
                "q_id": q_id,
                "retrieval_ms": round(t_retrieval),
                "generation_ms": round(t_generation),
                "total_ms": round(t_retrieval + t_generation),
            })

            did_abstain_list.append(did_abstain)
            eval_results.append(score)
            pipeline_traces.append({"q_id": q_id, "trace": trace})

            # 进度
            if (len(eval_results)) % 10 == 0:
                print(f"  [{exp_id}] {len(eval_results)}/{len(self.test_set)} "
                      f"questions evaluated")

        # ── 汇总统计（复用循环里已填充的 evaluator） ──
        aggregate = evaluator.get_aggregate()

        # 保存结果
        exp_dir = self.results_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        with open(exp_dir / "scores_per_question.json", "w", encoding="utf-8") as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=2)

        with open(exp_dir / "aggregate_scores.json", "w", encoding="utf-8") as f:
            json.dump(aggregate, f, ensure_ascii=False, indent=2)

        with open(exp_dir / "latency_stats.json", "w", encoding="utf-8") as f:
            json.dump(latency_stats, f, ensure_ascii=False, indent=2)

        # ── 错误分析 ──
        from evaluation.error_analyzer import ErrorAnalyzer
        analyzer = ErrorAnalyzer()
        error_report = analyzer.analyze(eval_results, did_abstain_list)
        analyzer.save_report(error_report, exp_dir / "error_analysis.json")

        print(f"\n  [{exp_id}] Results:")
        print(f"    Total:   {aggregate['total_score']['mean']:.2f}/10")
        for dim, data in aggregate["dimensions"].items():
            print(f"    {dim}: {data['mean']:.2f} (±{data['std']:.2f})")

        return {
            "exp_id": exp_id,
            "aggregate": aggregate,
            "latency_avg_ms": round(
                sum(l["total_ms"] for l in latency_stats) / len(latency_stats)
            ),
        }

    # ─── 执行所有实验 ────────────────────────────────────

    def run_all(self, experiments: list[str] = None):
        """
        执行指定的实验列表（或全部）。

        参数：
            experiments: 要执行的实验 ID 列表，None = 全部
        """
        all_configs = self.get_configs()

        if experiments is None:
            experiments = list(all_configs.keys())

        print(f"\n{'#'*60}")
        print(f"EXPERIMENT MATRIX RUNNER")
        print(f"{'#'*60}")
        print(f"Experiments to run: {experiments}")
        print(f"Total: {len(experiments)}")
        print(f"{'#'*60}\n")

        results = {}
        for exp_id in experiments:
            if exp_id not in all_configs:
                print(f"Unknown experiment: {exp_id}, skipping")
                continue
            result = self.run_experiment(exp_id, all_configs[exp_id])
            results[exp_id] = result

        # ── 生成对比表 ──
        self._save_comparison_table(results)
        self._print_summary(results)

        return results

    def _save_comparison_table(self, results: dict):
        """生成实验对比表（供 Gradio Demo 使用）。"""
        comparison = {}
        for exp_id, result in results.items():
            agg = result.get("aggregate", {})
            comparison[exp_id] = {
                "total_score": agg.get("total_score", {}).get("mean", 0),
                "dimensions": agg.get("dimensions", {}),
                "latency_avg_ms": result.get("latency_avg_ms", 0),
            }

        path = self.results_dir / "comparison_table.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        print(f"\nComparison table saved to: {path}")

    def _print_summary(self, results: dict):
        """打印所有实验的总结对比表。"""
        print(f"\n{'='*70}")
        print(f"EXPERIMENT SUMMARY")
        print(f"{'='*70}")
        print(f"{'Exp':<6} {'Name':<30} {'Score':>6} {'Latency':>8}")
        print(f"{'─'*70}")

        for exp_id, result in results.items():
            name = self.get_configs().get(exp_id, {}).get("name", exp_id)[:28]
            score = result.get("aggregate", {}).get("total_score", {}).get("mean", 0)
            latency = result.get("latency_avg_ms", 0)
            print(f"{exp_id:<6} {name:<30} {score:>5.2f}/10 {latency:>7.0f}ms")

        print(f"{'='*70}\n")


# ============================================================================
# 命令行入口
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SUSTech RAG Experiment Runner")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--experiments", type=str, default=None,
                       help="Comma-separated list (e.g., R0,R1,R2)")
    parser.add_argument("--baseline", action="store_true",
                       help="Run baseline experiments only (R0,R1,R2)")
    args = parser.parse_args()

    runner = ExperimentRunner()

    if args.baseline:
        experiments = ["R0", "R1", "R2"]
    elif args.experiments:
        experiments = args.experiments.split(",")
    elif args.all:
        experiments = None  # 全部
    else:
        print("Usage:")
        print("  --all            Run all 10 experiments")
        print("  --baseline       Run baseline only (R0,R1,R2)")
        print("  --experiments R0,R3,R4  Run specific experiments")
        print("\nNo experiments specified. Running baseline only.")
        experiments = ["R0", "R1", "R2"]

    runner.run_all(experiments)
