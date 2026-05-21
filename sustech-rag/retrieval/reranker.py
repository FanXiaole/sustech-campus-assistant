"""
=============================================================================
SUSTech Reranker — 基于 bge-reranker-v2-m3 的 Cross-Encoder 重排序
=============================================================================
Reranker 是 RAG pipeline 的"精排"阶段。

Bi-Encoder（Dense Retriever） vs Cross-Encoder（Reranker）的区别：
  Bi-Encoder:
    - 分别 encode query 和 document，然后用点积/余弦计算相似度
    - 快：document embedding 可以预计算、缓存在向量数据库中
    - 但不够精确：query 和 document 在 encode 时"没见过彼此"

  Cross-Encoder:
    - 把 [query, document] 拼接在一起，一次性过模型
    - 慢：每个 query-document pair 都要重新过模型
    - 但更精确：模型可以捕捉 query 和 document 之间的细粒度交互
      → "query 中的'开门'指的是 document 中的'开放时间'"

为什么先用 Bi-Encoder 召回 top-50，再用 Cross-Encoder 精排 top-5？
  → 这就是两阶段检索（retrieve-then-rerank）的经典范式：
    - 粗排：bi-encoder 从 ~15k 文档中快速筛选 50 个候选（~0.01s）
    - 精排：cross-encoder 对 50 个候选精细化排序（~0.5s）
    - 如果 15k 全部过 cross-encoder → ~100s，不现实

模型选择：bge-reranker-v2-m3
  - BAAI 最新的 Cross-Encoder reranker
  - MTEB Reranking 榜单 leader（与 bge-m3 embedding 同一团队）
  - 支持中英混合输入

使用方法：
  from retrieval.reranker import Reranker
  reranker = Reranker()
  reranked = reranker.rerank(query, candidates, top_k=5)

=============================================================================
"""

import time
from pathlib import Path
from typing import Any

import torch

# ============================================================================
# 配置导入
# ============================================================================
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RERANK_MODEL, RERANK_TOP_K


class Reranker:
    """
    Cross-Encoder 重排序器。

    封装 bge-reranker-v2-m3 的加载和推理。
    与 DenseRetriever 一样采用懒加载 + 单例模式。
    """

    def __init__(
        self,
        model_name: str = RERANK_MODEL,
        device: str = None,
        use_fp16: bool = True,
    ):
        """
        初始化 Reranker。

        参数：
            model_name: HuggingFace 模型名称
            device: 运行设备（"cuda", "cpu" 或 None=auto）
            use_fp16: 是否使用半精度（FP16）推理
                FP16 的优势：VRAM 减半、速度翻倍、精度损失 < 0.1%
                在 RTX 5090 上强烈建议开启
        """
        self.model_name = model_name

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.use_fp16 = use_fp16 and device == "cuda"

        print(f"[Reranker] Loading {model_name} on {device} "
              f"(FP16={self.use_fp16})...")
        t_start = time.time()

        # 使用 FlagEmbedding 的官方实现（BAAI 团队维护）
        try:
            from FlagEmbedding import FlagReranker
            self._model = FlagReranker(
                model_name,
                use_fp16=self.use_fp16,
                device=device,
            )
        except ImportError:
            # Fallback: 使用 sentence_transformers 的 CrossEncoder
            print("[Reranker] FlagEmbedding not available, "
                  "using sentence_transformers CrossEncoder instead")
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                model_name,
                device=device,
            )

        t_elapsed = time.time() - t_start
        print(f"[Reranker] Model loaded in {t_elapsed:.1f}s")

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = RERANK_TOP_K,
    ) -> list[dict]:
        """
        对候选 chunk 进行重排序。

        流程：
        1. 构建 [query, chunk_text] 对
        2. 用 Cross-Encoder 对每个 pair 打分
        3. 按分数降序排列
        4. 返回 top-K

        参数：
            query: 用户查询文本
            candidates: RRF 融合后的候选列表
            top_k: 返回数量（默认 5）

        返回：
            重排序后的结果列表（在原 dict 基础上增加 rerank_score 字段）
        """
        if not candidates:
            return []

        t_start = time.time()

        # 构建 query-document pairs
        pairs = []
        for c in candidates:
            # 使用 raw_text（不带元数据前缀）进行 rerank
            # 因为 reranker 需要的是语义内容，不是来源标签
            text = c.get("text", "")
            pairs.append([query, text])

        # 批量推理
        scores = self._model.compute_score(
            pairs,
            normalize=True,
            # ^ normalize=True → 输出 [0, 1] 范围内的分数
            #   （对 reranker 的输出做 sigmoid）
        )

        # 如果只有一个候选，compute_score 返回标量而非列表
        if not isinstance(scores, list):
            scores = [scores]

        # 将分数附加到结果中
        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = float(scores[i]) if i < len(scores) else 0.0

        # 按 rerank_score 降序排列
        sorted_results = sorted(
            candidates,
            key=lambda x: x.get("rerank_score", 0),
            reverse=True,
        )[:top_k]

        elapsed_ms = round((time.time() - t_start) * 1000)
        print(f"[Reranker] Reranked {len(candidates)} → {len(sorted_results)} "
              f"in {elapsed_ms}ms")

        return sorted_results


# ============================================================================
# 单例
# ============================================================================

_reranker_instance: Reranker | None = None


def get_reranker() -> Reranker:
    """获取全局唯一的 Reranker 实例。"""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Reranker()
    return _reranker_instance


# ============================================================================
# 测试
# ============================================================================
if __name__ == "__main__":
    print("Reranker — Unit test requires GPU and model download")
    print("This module will be tested as part of the full pipeline evaluation.\n")

    # 展示 Reranker 在 pipeline 中的位置
    print("Pipeline context:")
    print("  Dense top-50 → BM25 top-50 → RRF top-20 → ★ Reranker top-5 → LLM")
    print("                                              ^^^^^^^^^^^^^^^^")
    print("  Reranker 在这里把 20 个候选精排为 5 个最相关的 chunk")
