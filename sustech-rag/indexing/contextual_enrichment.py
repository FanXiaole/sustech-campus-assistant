"""
=============================================================================
Contextual Chunk Enrichment — LLM 上下文增强 ★ 创新点 #3
=============================================================================
论文：Anthropic, "Contextual Retrieval" (2024)
  → 在 chunk 前面加上 LLM 生成的"背景描述"，让每个 chunk 都能被独立理解。

问题：Chunk 化之后，上下文信息丢失
  考虑这句话（它本身可能是正确的）：
    "截止日期是 2026 年 12 月 31 日。"

  这句话在检索中可能排名很高（因为"截止日期"匹配了查询），
  但它丢失了关键信息："什么"的截止日期？研究生招生？期末考试？奖学金申请？

  LLM 看到这个 chunk 时也不知道，所以要么编造，要么追问。

解决方案：Contextual Enrichment
  用 LLM 为每个 chunk 生成一段"背景描述"（Situated Context），
  然后把这段描述拼到 chunk 前面：

  Before:
    "截止日期是 2026 年 12 月 31 日。"

  After:
    "[背景：本文介绍2026年研究生招生时间安排，此段说明网上报名的截止日期]
     截止日期是 2026 年 12 月 31 日。"

  现在这个 chunk 在语义上是自包含的，无论检索还是 LLM 都能正确理解。

实现方式（成本优化）：
  1. 使用 DeepSeek API（免费 Qwen2.5-7B）→ 零 GPU 成本
  2. 异步批量调用（asyncio + httpx, concurrency=10）→ 速度最大化
  3. 结果缓存到 JSON → 可恢复、可复用
  4. 之后用这些 enriched chunks 构建第二个 ChromaDB collection → A/B 对比

使用方法：
  python indexing/contextual_enrichment.py

=============================================================================
"""

import asyncio
import json
import os
import time
from pathlib import Path

import httpx

# ============================================================================
# 配置导入
# ============================================================================
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    CHUNK_DIR,
    INDEX_DIR,
    PROC_DIR,
    DEEPSEEK_BASE,
)


# ============================================================================
# Prompt 模板
# ============================================================================

ENRICHMENT_SYSTEM_PROMPT = """你是一个简洁的文档分析师。你的任务是为文档片段生成上下文描述。
请用中文回答。"""

# user prompt 模板：
# {document_context} = 文档前 500 字符
# {chunk_text} = 当前 chunk 的文本（前 200 字符）
ENRICHMENT_USER_TEMPLATE = """这是一份南科大校园文档的摘要：
{document_context}

以下是这份文档中的一段文字：
{chunk_text}

请用一句话（不超过60个中文字符）描述这段文字在整份文档中的位置和作用。
例如："这部分说明了图书馆的工作日开放时间"或"这段列出了研究生招生报名所需的材料"。
只输出这一句话，不要加任何前缀或解释。"""


# ============================================================================
# 异步 API 调用
# ============================================================================

async def generate_context_async(
    client: httpx.AsyncClient,
    api_key: str,
    document_context: str,
    chunk_text: str,
    semaphore: asyncio.Semaphore,
) -> str:
    """
    异步调用 DeepSeek API，为单个 chunk 生成背景描述。

    参数：
        client: httpx 异步客户端（复用连接池）
        api_key: DeepSeek API key
        document_context: 文档的前 500 字符
        chunk_text: 当前 chunk 的文本（前 200 字符）
        semaphore: 并发控制信号量

    返回：
        生成的背景描述文本
    """
    async with semaphore:  # 控制并发数
        user_prompt = ENRICHMENT_USER_TEMPLATE.format(
            document_context=document_context[:500],
            chunk_text=chunk_text[:200],
        )

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": ENRICHMENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 80,
            "temperature": 0.3,
            # ^ 低温度 → 输出更确定、更简洁
        }

        try:
            response = await client.post(
                f"{DEEPSEEK_BASE}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            context = data["choices"][0]["message"]["content"].strip()
            return context

        except Exception as e:
            # API 调用失败 → 返回空字符串（chunk 保持原样）
            return ""


# ============================================================================
# 主处理流程
# ============================================================================

def load_cached_contexts(cache_path: Path) -> dict[str, str]:
    """加载已缓存的上下文描述。"""
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cached_contexts(cache_path: Path, contexts: dict[str, str]):
    """保存上下文描述到缓存文件。"""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(contexts, f, ensure_ascii=False, indent=2)


async def enrich_chunks_async(
    chunks_path: Path,
    processed_docs_path: Path,
    cache_path: Path,
    api_key: str,
    concurrency: int = 10,
) -> list[dict]:
    """
    异步批量处理所有 chunk，生成上下文背景描述。

    处理逻辑：
    1. 加载所有 chunk 和对应的原始文档
    2. 构建 doc_id → document_context 的映射
    3. 对每个 chunk 异步调用 LLM 生成背景描述
    4. 将背景描述拼到 chunk.text 前面
    5. 结果写入缓存

    参数：
        chunks_path: chunk JSONL 路径
        processed_docs_path: 清洗后文档路径（用于获取每个文档的前 500 字）
        cache_path: 上下文缓存文件路径
        api_key: DeepSeek API key
        concurrency: 并发 API 调用数
    """
    # ── 加载数据 ──
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    print(f"Loaded {len(chunks):,} chunks")

    # 构建 doc_id → 文档文本的映射
    doc_contexts = {}
    with open(processed_docs_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                doc_id = doc["doc_id"]
                doc_contexts[doc_id] = doc["text"][:500]

    print(f"Loaded {len(doc_contexts):,} document contexts")

    # ── 加载缓存 ──
    cached_contexts = load_cached_contexts(cache_path)
    print(f"Loaded {len(cached_contexts):,} cached contexts")

    # ── 确定需要处理的 chunk ──
    to_process = []
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        if chunk_id not in cached_contexts:
            to_process.append(chunk)

    print(f"Need to process: {len(to_process):,} chunks "
          f"({len(cached_contexts):,} already cached)")

    if not to_process:
        print("All chunks already have cached contexts. Skipping API calls.")
        return chunks

    # ── 异步批量调用 ──
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:
        tasks = []
        for chunk in to_process:
            doc_id = chunk["doc_id"]
            doc_context = doc_contexts.get(doc_id, "")
            chunk_text = chunk.get("raw_text", chunk.get("text", ""))

            task = generate_context_async(
                client=client,
                api_key=api_key,
                document_context=doc_context,
                chunk_text=chunk_text,
                semaphore=semaphore,
            )
            tasks.append((chunk["chunk_id"], task))

        print(f"Starting {len(tasks):,} API calls (concurrency={concurrency})...")
        t_start = time.time()

        # 分批次执行（避免一次性创建太多协程）
        results = {}
        batch_size = 100
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            batch_coros = [t for _, t in batch]
            batch_ids = [cid for cid, _ in batch]

            batch_results = await asyncio.gather(*batch_coros, return_exceptions=True)

            for chunk_id, result in zip(batch_ids, batch_results):
                if isinstance(result, Exception):
                    # 调用失败 → 空字符串
                    results[chunk_id] = ""
                else:
                    results[chunk_id] = result

            # 每完成一批就打印进度
            done = min(i + batch_size, len(tasks))
            elapsed = time.time() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  {done}/{len(tasks)} chunks ({rate:.1f} chunks/s)")

    t_total = time.time() - t_start
    print(f"Completed in {t_total:.1f}s ({len(tasks) / t_total:.1f} chunks/s)")

    # ── 更新缓存 ──
    cached_contexts.update(results)
    save_cached_contexts(cache_path, cached_contexts)

    # ── 应用到 chunk ──
    enriched_chunks = []
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        context = cached_contexts.get(chunk_id, "")

        enriched = dict(chunk)  # 浅拷贝
        if context:
            # 在 text 前面添加背景描述
            enriched["text"] = f"[背景:{context}] {chunk.get('raw_text', chunk.get('text', ''))}"
        else:
            # 没有背景描述 → 保持原样
            enriched["text"] = chunk.get("text", "")

        enriched_chunks.append(enriched)

    return enriched_chunks


def run_enrichment(
    chunks_path: Path = None,
    output_suffix: str = "default",
) -> Path:
    """
    执行完整的上下文增强流程（同步包装器）。

    参数：
        chunks_path: chunk JSONL 路径
        output_suffix: 输出文件后缀

    返回：
        输出的 enriched chunk JSONL 路径
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("\n" + "!" * 60)
        print("ERROR: DEEPSEEK_API_KEY environment variable not set.")
        print("Set it with: export DEEPSEEK_API_KEY='your-key-here'")
        print("Or get a free key at: https://cloud.siliconflow.cn")
        print("!" * 60 + "\n")
        return None

    if chunks_path is None:
        chunks_path = CHUNK_DIR / f"chunks_{output_suffix}.jsonl"

    processed_docs_path = PROC_DIR / "processed_docs.jsonl"
    cache_path = CHUNK_DIR / "chunk_contexts.json"
    output_path = CHUNK_DIR / f"chunks_{output_suffix}_enriched.jsonl"

    print(f"\n{'='*60}")
    print(f"Contextual Chunk Enrichment")
    print(f"{'='*60}")
    print(f"Source: {chunks_path}")
    print(f"Cache: {cache_path}")
    print(f"Output: {output_path}")

    # 运行异步处理
    enriched_chunks = asyncio.run(
        enrich_chunks_async(
            chunks_path=chunks_path,
            processed_docs_path=processed_docs_path,
            cache_path=cache_path,
            api_key=api_key,
            concurrency=10,
        )
    )

    if enriched_chunks is None:
        return None

    # ── 保存输出 ──
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in enriched_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # ── 统计 ──
    enriched_count = sum(
        1 for c in enriched_chunks if "[背景:" in c.get("text", "")
    )
    print(f"\nEnriched {enriched_count:,}/{len(enriched_chunks):,} chunks "
          f"({100 * enriched_count / max(1, len(enriched_chunks)):.1f}%)")
    print(f"Output saved to: {output_path}")
    print(f"Next step: build ChromaDB collection 'sustech_enriched' from this file")

    return output_path


# ============================================================================
# 示例（演示 Contextual Enrichment 的效果）
# ============================================================================
if __name__ == "__main__":
    print("Contextual Chunk Enrichment — Demonstration")
    print("=" * 60)
    print()
    print("Before enrichment:")
    print('  "申请截止日期为 2026 年 12 月 31 日。"')
    print()
    print("After enrichment:")
    print('  "[背景：本文介绍2026年研究生招生时间安排，此段说明网上报名的截止日期]')
    print('   申请截止日期为 2026 年 12 月 31 日。"')
    print()
    print("Now the chunk is self-contained and unambiguous.")
    print()
    print("To run actual enrichment:")
    print("  1. export DEEPSEEK_API_KEY='your-key'")
    print("  2. python indexing/contextual_enrichment.py")
