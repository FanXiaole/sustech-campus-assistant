"""
=============================================================================
SUSTech Embedder — 批量向量化与 ChromaDB 索引构建
=============================================================================
⚠️ GPU REQUIRED — 运行前需要先启动 AutoDL RTX 5090 实例

功能：将 chunk JSONL 中的所有 chunk 用 bge-m3 编码为向量，存入 ChromaDB。

bge-m3 输出：
  - 维度：1024
  - 归一化：L2-norm（余弦相似度 = 点积）
  - 最大输入长度：8192 tokens（远超 chunk 的 600 字符）

为什么选择 ChromaDB 而不是 FAISS/Milvus/Qdrant？
  - ChromaDB 是 Python-native 的 → 部署最简单，一个 pip install
  - 持久化存储（基于 SQLite3 + HNSW 索引）→ 关闭程序后索引仍在
  - HNSW 参数可调 → construction_ef=200, M=16 适合 1-10 万文档
  - 自带 metadata 过滤 → 可以按 source_family 等字段筛选
  - 对于 15k 文档的规模，ChromaDB 的性能完全够用

FAISS 只在 >100k 文档时才明显优于 ChromaDB，我们的规模不需要。

性能目标（RTX 5090）：
  - ~18,000 chunks/min at batch_size=256
  - 15k chunks → ~12 分钟

使用方法：
  python indexing/embedder.py

=============================================================================
"""

import json
import time
from pathlib import Path
from typing import Any

import torch

# ============================================================================
# 配置导入
# ============================================================================
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    CHUNK_DIR,
    EMBED_MODEL,
    INDEX_DIR,
)


# ============================================================================
# Embedder 类
# ============================================================================

class Embedder:
    """
    bge-m3 批量向量化器。

    设计为可以处理任意 chunk JSONL 并创建对应的 ChromaDB collection。
    支持 default/small/large/enriched 四种 collection。
    """

    def __init__(
        self,
        model_name: str = EMBED_MODEL,
        device: str = None,
        batch_size: int = 32,
    ):
        """
        初始化 Embedder。

        参数：
            model_name: HuggingFace 模型名称
            device: "cuda", "cpu" 或 None（自动选择）
            batch_size: 编码时的 batch 大小
                256 是 RTX 5090 的最优值：
                - 更大（512）→ VRAM 峰值 > 24GB，有 OOM 风险
                - 更小（128）→ GPU 利用率不足，时间翻倍
        """
        from sentence_transformers import SentenceTransformer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.batch_size = batch_size

        if device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[Embedder] GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        else:
            print("[Embedder] WARNING: Running on CPU — this will be VERY slow!")

        print(f"[Embedder] Loading {model_name}...")
        t_start = time.time()

        self.model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": "auto"},
        )
        # 截断超长文本，防止 attention OOM（默认 8192 tokens）
        self.model.max_seq_length = 2048

        t_elapsed = time.time() - t_start
        print(f"[Embedder] Model loaded in {t_elapsed:.1f}s")

        # 编码配置
        self.doc_prompt = ""  # bge-m3: 文档不需要指令前缀
        self.query_prompt = "Represent this sentence for retrieval: "

    def encode_documents(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> list[list[float]]:
        """
        批量编码文档文本。

        参数：
            texts: 文本列表
            show_progress: 是否显示进度条

        返回：
            归一化的 embedding 向量列表
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            # ^ normalize → L2-norm = 1 → 余弦相似度 = 点积
            show_progress_bar=show_progress,
            prompt=self.doc_prompt,
        )
        return embeddings.tolist()

    def encode_query(self, query: str) -> list[float]:
        """编码单个查询（带指令前缀）。"""
        return self.model.encode(
            [f"{self.query_prompt}{query}"],
            normalize_embeddings=True,
        ).tolist()[0]


def build_chroma_collection(
    chunks_path: Path,
    collection_name: str,
    embedder: Embedder,
    chroma_path: Path = None,
) -> dict:
    """
    从 chunk JSONL 构建 ChromaDB collection。

    流程：
    1. 加载 chunk JSONL
    2. 提取所有文本（带元数据前缀的完整 text）
    3. 用 bge-m3 批量编码
    4. 写入 ChromaDB（HNSW 索引）

    参数：
        chunks_path: chunk JSONL 文件路径
        collection_name: ChromaDB collection 名称
        embedder: Embedder 实例
        chroma_path: ChromaDB 持久化目录

    返回：
        统计信息字典
    """
    import chromadb

    if chroma_path is None:
        chroma_path = str(INDEX_DIR / "chroma_db")

    print(f"\n{'='*60}")
    print(f"Building ChromaDB Collection: {collection_name}")
    print(f"{'='*60}")
    print(f"Source: {chunks_path}")

    # ── 加载 chunks ──
    t_load_start = time.time()
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"Loaded {len(chunks):,} chunks in {time.time() - t_load_start:.1f}s")

    # ── 提取文本和元数据 ──
    texts = []
    metadatas = []
    ids = []

    for chunk in chunks:
        texts.append(chunk["text"])  # 带元数据前缀的文本
        metadatas.append({
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "source_family": chunk.get("source_family", "unknown"),
            "url": chunk.get("url", ""),
            "heading_path": chunk.get("heading_path", ""),
            "char_count": chunk.get("char_count", 0),
            "chunk_strategy": chunk.get("chunk_strategy", ""),
        })
        ids.append(chunk["chunk_id"])

    # ── 批量编码 ──
    print(f"Encoding {len(texts):,} chunks (batch_size={embedder.batch_size})...")
    t_encode_start = time.time()
    embeddings = embedder.encode_documents(texts, show_progress=True)
    encode_time = time.time() - t_encode_start
    print(f"Encoded in {encode_time:.1f}s "
          f"({len(embeddings) / encode_time:.0f} chunks/s)")

    # ── 写入 ChromaDB ──
    print(f"Writing to ChromaDB at {chroma_path}...")
    t_write_start = time.time()

    client = chromadb.PersistentClient(path=chroma_path)

    # 如果 collection 已存在，先删除再重建
    # （这样可以确保每次运行 embedder.py 得到的是最新的索引）
    try:
        client.delete_collection(collection_name)
        print(f"  Deleted existing collection '{collection_name}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",
            # ^ 余弦距离 = 1 - 余弦相似度
            #   配合 bge-m3 的 L2-normalized embedding → 等同于欧氏距离

            "hnsw:construction_ef": 200,
            # ^ construction_ef: HNSW 图构建时的搜索宽度
            #   200 是"高召回"的设置（默认 100）
            #   值越大 → 索引越精确，但构建越慢
            #   对于 15k 文档，200 是合理的平衡

            "hnsw:M": 16,
            # ^ M: HNSW 图中每个节点的出度
            #   16 是"中等"设置（默认 16）
            #   值越大 → 搜索越快，但内存越大
            #   对于 1024 维向量，16 是经验最优值
        },
    )

    # 分批写入（每批 1000 条）
    # 为什么要分批？→ ChromaDB 的 add() 在单次传入 >5000 条时
    # 可能会因为内存峰值过高而变慢甚至崩溃
    batch_upsert_size = 1000
    for i in range(0, len(ids), batch_upsert_size):
        batch_end = min(i + batch_upsert_size, len(ids))
        collection.add(
            ids=ids[i:batch_end],
            embeddings=embeddings[i:batch_end],
            metadatas=metadatas[i:batch_end],
            documents=texts[i:batch_end],
        )
        if (i // batch_upsert_size) % 5 == 0:
            print(f"  ... written {batch_end}/{len(ids)} vectors")

    write_time = time.time() - t_write_start
    print(f"Written {len(ids):,} vectors in {write_time:.1f}s")

    # ── 验证：运行几个测试查询 ──
    print(f"\nRunning verification queries...")
    test_queries = [
        "图书馆开放时间",
        "计算机系教授",
        "新生入学流程",
    ]
    for q in test_queries:
        q_embedding = embedder.encode_query(q)
        results = collection.query(
            query_embeddings=[q_embedding],
            n_results=3,
            include=["distances", "documents"],
        )
        print(f"\n  Query: {q}")
        for i, (chunk_id, distance, doc) in enumerate(zip(
            results["ids"][0], results["distances"][0], results["documents"][0]
        )):
            similarity = 1.0 - distance
            preview = doc[:100].replace("\n", " ")
            print(f"    #{i+1} [{similarity:.4f}] {preview}...")

    # ── 统计 ──
    index_size_mb = sum(
        (Path(chroma_path) / f).stat().st_size
        for f in ("chroma.sqlite3",)
        if (Path(chroma_path) / f).exists()
    ) / (1024 * 1024)

    stats = {
        "collection_name": collection_name,
        "total_vectors": collection.count(),
        "index_size_mb": index_size_mb,
        "encode_time_s": encode_time,
        "write_time_s": write_time,
        "total_time_s": encode_time + write_time,
    }

    print(f"\n{'='*60}")
    print(f"COLLECTION BUILT: {collection_name}")
    print(f"  Total vectors: {stats['total_vectors']:,}")
    print(f"  Index size: {stats['index_size_mb']:.1f} MB")
    print(f"  Time: {stats['total_time_s']:.0f}s "
          f"(encode: {stats['encode_time_s']:.0f}s, write: {stats['write_time_s']:.0f}s)")
    print(f"{'='*60}\n")

    return stats


def build_all_collections():
    """
    为三种 chunk 大小变体都构建 ChromaDB collection。

    这是 GPU 阶段的主入口。必须在 AutoDL RTX 5090 上运行。
    """
    # 检查 GPU 是否可用
    if not torch.cuda.is_available():
        print("\n" + "!" * 60)
        print("WARNING: CUDA not available! Embedding will run on CPU.")
        print("For 15k chunks, this will take HOURS instead of minutes.")
        print("Start your AutoDL RTX 5090 instance before proceeding.")
        print("!" * 60 + "\n")
        response = input("Continue on CPU anyway? (y/N): ")
        if response.lower() != "y":
            print("Exiting. Please start GPU and re-run.")
            return

    embedder = Embedder()

    configs = [
        (CHUNK_DIR / "chunks_default.jsonl", "sustech_default"),
        (CHUNK_DIR / "chunks_small.jsonl", "sustech_small"),
        (CHUNK_DIR / "chunks_large.jsonl", "sustech_large"),
    ]

    all_stats = {}
    for chunks_path, collection_name in configs:
        if not chunks_path.exists():
            print(f"SKIPPING: {chunks_path} not found. Run chunker.py first.")
            continue
        stats = build_chroma_collection(
            chunks_path=chunks_path,
            collection_name=collection_name,
            embedder=embedder,
        )
        all_stats[collection_name] = stats

    print("=" * 60)
    print("ALL COLLECTIONS BUILT")
    print("=" * 60)
    for name, stats in all_stats.items():
        print(f"  {name}: {stats['total_vectors']:,} vectors, "
              f"{stats['total_time_s']:.0f}s")
    print()

    return all_stats


if __name__ == "__main__":
    build_all_collections()
