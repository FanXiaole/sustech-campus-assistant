"""
=============================================================================
SUSTech Dense Retriever — 基于 bge-m3 的稠密向量检索
=============================================================================
稠密检索是 RAG 系统的核心组件。它的工作原理是：
  1. 用 Embedding 模型将查询文本转换为一个高维向量（bge-m3: 1024 维）
  2. 在向量数据库中找到与查询向量最相似的文档向量
  3. 返回相似度最高的 top-K 个 chunk

与稀疏检索（BM25）的区别：
  - BM25 依赖精确的词汇匹配（"图书馆" ≠ "图书室"）
  - Dense 依赖语义相似度（"图书馆" ≈ "图书收藏处" ≈ "借书的地方"）
  - BM25 对专有名词和数字更好，Dense 对语义变体更好
  → 所以我们用 RRF 把两者融合（hybrid_rrf.py）

模型选择：bge-m3
  - BAAI 开发的多语言 embedding 模型
  - 支持中英混合输入（南科大网站恰好是中英混合场景）
  - MTEB 中文检索榜单的长期 leader
  - 支持 instruction prefix（查询端可以添加指令前缀提升效果）

使用方法：
  from retrieval.dense_retriever import DenseRetriever
  retriever = DenseRetriever()
  results = retriever.search("图书馆几点开门", top_k=50)

=============================================================================
"""

import json
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

# ============================================================================
# 配置导入
# ============================================================================
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    CHUNK_DIR,
    DENSE_TOP_K,
    EMBED_MODEL,
    INDEX_DIR,
)

# ============================================================================
# 指令前缀
# ============================================================================
# bge-m3 支持 instruction prefix：在 query 前面添加指令来定制检索行为
# 这是 bge 系列模型的一个重要特性，可以零样本地定制检索目标
#
# 为什么 query 需要指令但 document 不需要？
# → 因为 query 是"任务描述"（"检索相关信息、检索问题"），
#    document 是"被动匹配"（只需要被找到，不需要理解任务）
#    Instruction-tuned embedding 在训练时已经学会了这个 asymmetry。

QUERY_INSTRUCTION = "Represent this sentence for retrieving relevant campus information: "
# 注意：bge-m3 的官方文档推荐对 query 用此指令，对 document 用空字符串


class DenseRetriever:
    """
    稠密向量检索器。

    封装了 SentenceTransformer 模型的加载、查询编码和 ChromaDB 查询。
    设计为"先初始化（加载模型和索引）、再查询"的两步模式，
    避免每次查询都重新加载模型（加载模型 ~5-10 秒，不可接受）。
    """

    def __init__(
        self,
        model_name: str = EMBED_MODEL,
        device: str = None,
        collection_name: str = "sustech_default",
    ):
        """
        初始化稠密检索器。

        参数：
            model_name: HuggingFace 模型名称
            device: 运行设备（"cuda", "cpu" 或 None=auto）
            collection_name: ChromaDB collection 名称
                "sustech_default"  → 默认 600-char chunks
                "sustech_small"    → 300-char chunks（消融实验）
                "sustech_large"    → 900-char chunks（消融实验）
                "sustech_enriched" → 带 contextual enrichment
        """
        self.model_name = model_name
        self.collection_name = collection_name

        # 自动选择设备
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        print(f"[DenseRetriever] Loading {model_name} on {device}...")
        t_start = time.time()

        self.model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=True,
            # ^ bge-m3 的官方实现需要 trust_remote_code=True
            #   因为模型卡片中包含自定义的 pooling 和 normalization 代码
        )

        t_elapsed = time.time() - t_start
        print(f"[DenseRetriever] Model loaded in {t_elapsed:.1f}s")
        print(f"[DenseRetriever] Device: {self.model.device}")

        # 延迟加载 ChromaDB（等到第一次 search 时再加载）
        # 原因：ChromaDB 的初始化很快，但可能在不需要时浪费内存
        self._collection = None

    def _ensure_collection_loaded(self):
        """
        确保 ChromaDB collection 已加载（懒加载模式）。

        ChromaDB 的 PersistentClient 在初始化时会读取磁盘上的索引文件。
        对于 sustech_default collection（~15k chunks），加载时间 <1 秒。
        """
        if self._collection is not None:
            return

        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is required for dense retrieval. "
                "Install with: pip install chromadb"
            )

        chroma_path = str(INDEX_DIR / "chroma_db")
        self._client = chromadb.PersistentClient(path=chroma_path)

        try:
            self._collection = self._client.get_collection(self.collection_name)
            print(f"[DenseRetriever] Loaded collection '{self.collection_name}' "
                  f"({self._collection.count()} vectors)")
        except Exception:
            raise RuntimeError(
                f"Collection '{self.collection_name}' not found at {chroma_path}. "
                f"Run indexing/embedder.py first to build the index."
            )

    def encode_query(self, query: str, use_instruction: bool = True) -> list[float]:
        """
        将查询文本编码为 embedding 向量。

        参数：
            query: 查询文本
            use_instruction: 是否添加指令前缀（默认 True）

        返回：
            1024 维的归一化向量（list of float）
        """
        if use_instruction:
            query = QUERY_INSTRUCTION + query

        # encode() 返回 numpy array，转为 Python list
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            # ^ bge-m3 的 embedding 必须 normalize 后才能用余弦相似度
            #   ChromaDB 内部使用余弦距离，等价于归一化向量的欧氏距离
            show_progress_bar=False,
        )
        return embedding.tolist()

    def search(
        self,
        query: str,
        top_k: int = DENSE_TOP_K,
        return_scores: bool = True,
    ) -> list[dict]:
        """
        执行稠密检索。

        参数：
            query: 用户查询文本
            top_k: 返回的候选 chunk 数量（默认 50，为 RRF 融合提供足够的候选）
            return_scores: 是否返回相似度分数

        返回：
            [{chunk_id, text, score, source_family, url, ...}, ...]
            按相似度降序排列
        """
        self._ensure_collection_loaded()

        query_embedding = self.encode_query(query)

        # ChromaDB query 参数
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }
        # include=["distances"] 会返回距离值（cosine 距离，越小越相似）
        if return_scores:
            query_kwargs["include"] = ["distances", "metadatas", "documents"]

        results = self._collection.query(**query_kwargs)

        # 将 ChromaDB 的返回格式转换为我们统一的格式
        formatted_results = []
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        documents = results.get("documents", [[]])[0]

        for i, chunk_id in enumerate(ids):
            distance = distances[i] if i < len(distances) else 1.0
            # ChromaDB 返回的是 cosine distance，转换为 similarity score
            # cosine_distance = 1 - cosine_similarity
            # → similarity = 1 - distance
            similarity = 1.0 - distance

            metadata = metadatas[i] if i < len(metadatas) else {}
            document = documents[i] if i < len(documents) else ""

            formatted_results.append({
                "chunk_id": chunk_id,
                "text": document or metadata.get("text", ""),
                "score": similarity,
                "source_family": metadata.get("source_family", "unknown"),
                "url": metadata.get("url", ""),
                "heading_path": metadata.get("heading_path", ""),
            })

        return formatted_results

    def search_by_embedding(
        self,
        embedding: list[float],
        top_k: int = DENSE_TOP_K,
    ) -> list[dict]:
        """
        直接使用 embedding 向量进行检索（不重新编码查询文本）。

        这个方法用于 HyDE：先由 LLM 生成假设性文档，再 embed 这个文档，
        然后用这个 embedding 直接检索。与 search() 的区别是：
        - search() 接受 text → 调用 encode_query → 检索
        - search_by_embedding() 接受 embedding → 直接检索

        参数：
            embedding: 预先计算好的 embedding 向量
            top_k: 返回数量

        返回：
            检索结果列表
        """
        self._ensure_collection_loaded()

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["distances", "metadatas", "documents"],
        )

        formatted_results = []
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        documents = results.get("documents", [[]])[0]

        for i, chunk_id in enumerate(ids):
            distance = distances[i] if i < len(distances) else 1.0
            similarity = 1.0 - distance
            metadata = metadatas[i] if i < len(metadatas) else {}
            document = documents[i] if i < len(documents) else ""

            formatted_results.append({
                "chunk_id": chunk_id,
                "text": document or metadata.get("text", ""),
                "score": similarity,
                "source_family": metadata.get("source_family", "unknown"),
                "url": metadata.get("url", ""),
                "heading_path": metadata.get("heading_path", ""),
            })

        return formatted_results


# ============================================================================
# 快速创建（单例模式，避免重复加载模型）
# ============================================================================

_retriever_instance: DenseRetriever | None = None


def get_dense_retriever(
    collection_name: str = "sustech_default",
) -> DenseRetriever:
    """
    获取全局唯一的 DenseRetriever 实例。

    为什么需要单例？
    → SentenceTransformer 模型加载 ~5-10 秒，占用 ~2GB VRAM。
      如果每次查询都新建一个实例，不仅慢而且会 OOM。
      单例确保整个进程共享同一个模型实例。

    参数：
        collection_name: ChromaDB collection 名称

    返回：
        DenseRetriever 实例
    """
    global _retriever_instance
    if _retriever_instance is None or _retriever_instance.collection_name != collection_name:
        _retriever_instance = DenseRetriever(collection_name=collection_name)
    return _retriever_instance


# ============================================================================
# 测试入口
# ============================================================================
if __name__ == "__main__":
    retriever = get_dense_retriever()
    test_queries = [
        "图书馆开放时间",
        "计算机系有哪些教授",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        results = retriever.search(q, top_k=3)
        for r in results:
            print(f"  [{r['score']:.4f}] {r['text'][:120]}...")
