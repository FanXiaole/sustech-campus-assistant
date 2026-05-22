"""
=============================================================================
HyDE — Hypothetical Document Embedding ★ 创新点 #1
=============================================================================
论文：Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels"
      (2022, arXiv:2212.10496)

核心思想（一句话）：
  不直接 embed 用户的简短查询，而是让 LLM 先生成一份"假设性的答案文档"，
  然后 embed 这份文档，用它去检索真实文档。

为什么有效？—— Query-Document Embedding Gap
  用户的查询和知识库中的文档用的是"不同的语言"：

  用户查询："图书馆几点开？"          (口语化、简短、问题形式)
  知识库文档："图书馆服务时间：       (书面化、完整、陈述形式)
             周一至周五 8:00-22:00，
             周末 9:00-21:00"

  在 embedding 空间中，这两个文本的向量距离可能很大！
  因为 embedding 模型在训练时学到的是：疑问句和陈述句分布在不同的区域。

  HyDE 的解决方案：
  查询 "图书馆几点开？"
    → LLM 生成假设文档："南科大图书馆的开放时间是每天早上8点到晚上10点，..."
    → embed 这份假设文档
    → 假设文档和真实文档都在"陈述句空间"→ 距离更近 → 检索更准

延迟开销：
  每次查询多一次 LLM 调用（~0.5-1s），但需要额外一次 LLM 调用（~0.5-1s）。

使用方法：
  from retrieval.hyde import hyde_retrieve
  results = hyde_retrieve("图书馆几点开门", llm_fn, embed_fn, collection)

=============================================================================
"""

import time
from pathlib import Path
from typing import Callable

# ============================================================================
# 配置导入
# ============================================================================
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DENSE_TOP_K

# ============================================================================
# HyDE Prompt 模板
# ============================================================================

# System prompt：告诉 LLM 它的任务是"写一份假设的校园资料"
# v2: 更具体的格式要求，模拟官网文档风格，提升与真实文档的 embedding 相似度
HYDE_SYSTEM_PROMPT = (
    "你是南方科技大学（SUSTech）官方校园手册的编写者。"
    "你的任务是根据用户提问，生成一段假想的校园手册条目。"
    "这段文字应模仿学校官网/手册风格：陈述句、信息密度高、包含具体数字和时间。"
    "用中文写，不超过120字。只输出条目内容本身，不加任何前缀。"
)

# User prompt 模板
# v2: 加入风格示例，引导 LLM 生成更接近真实文档格式的假设文本
HYDE_USER_TEMPLATE = (
    "用户想知道：{query}\n\n"
    "请写一段假想的南科大校园手册条目来回答。风格参考：\n"
    "\"图书馆服务时间：周一至周五 8:00-22:00，周末 9:00-21:00。"
    "凭校园卡入馆，可借阅图书30天。\"\n\n"
    "你的假想条目："
)


def generate_hypothetical_doc(
    query: str,
    llm_fn: Callable[[str, str], str],
) -> str:
    """
    用 LLM 生成假设性文档（HyDE 的核心步骤）。

    参数：
        query: 用户原始查询
        llm_fn: LLM 调用函数，签名为 llm_fn(system_prompt, user_prompt) -> str

    返回：
        假设性文档文本（纯文本，不含前缀或解释）
    """
    user_prompt = HYDE_USER_TEMPLATE.format(query=query)
    hyp_doc = llm_fn(HYDE_SYSTEM_PROMPT, user_prompt)
    return hyp_doc.strip()


def hyde_retrieve(
    query: str,
    llm_fn: Callable[[str, str], str],
    embed_fn: Callable[[str], list[float]],
    dense_search_fn: Callable[[list[float], int], list[dict]],
    top_k: int = DENSE_TOP_K,
) -> tuple[list[dict], dict]:
    """
    执行 HyDE 增强的稠密检索。

    完整的 HyDE 流程：
    1. LLM 根据查询生成假设性文档
    2. 将假设性文档 embed 为向量
    3. 用这个向量去 ChromaDB 检索真实文档

    参数：
        query: 用户原始查询
        llm_fn: LLM 调用函数
        embed_fn: embedding 函数（接受文本 → 返回向量）
        dense_search_fn: 稠密检索函数（接受向量 → 返回结果列表）
        top_k: 返回的候选 chunk 数量

    返回：
        (检索结果列表, trace 字典) — trace 包含假设文档和延迟信息
    """
    trace = {
        "method": "HyDE",
        "query": query,
    }

    # 步骤 1：生成假设文档
    t0 = time.time()
    hyp_doc = generate_hypothetical_doc(query, llm_fn)
    trace["hypothetical_doc"] = hyp_doc
    trace["hyde_gen_ms"] = round((time.time() - t0) * 1000)

    # 步骤 2：Embed 假设文档
    t0 = time.time()
    hyp_embedding = embed_fn(hyp_doc)
    trace["hyde_embed_ms"] = round((time.time() - t0) * 1000)

    # 步骤 3：用假设 embedding 检索
    t0 = time.time()
    results = dense_search_fn(hyp_embedding, top_k)
    trace["hyde_search_ms"] = round((time.time() - t0) * 1000)

    trace["total_hyde_ms"] = (
        trace["hyde_gen_ms"] + trace["hyde_embed_ms"] + trace["hyde_search_ms"]
    )

    return results, trace


# ============================================================================
# 对比实验：直接查询 vs HyDE
# ============================================================================

def direct_vs_hyde_example():
    """
    演示直接查询 vs HyDE 查询的区别（仅用于理解和文档）。
    实际运行时需要 LLM API 和 embedding 模型。
    """
    print("=" * 60)
    print("HyDE 效果对比演示")
    print("=" * 60)
    print()
    print("查询: '图书馆几点开门？'")
    print()
    print("直接查询 embedding 空间中的距离：")
    print("  查询向量 ←→ 文档向量 '图书馆服务时间：8:00-22:00'")
    print("  距离较大（疑问句 vs 陈述句）")
    print()
    print("HyDE 方案：")
    print("  1. LLM 生成假设文档：")
    print("     '南科大图书馆每天从早上8点开放到晚上10点，'")
    print("      '周末为上午9点到晚上9点。'")
    print("  2. Embed 假设文档")
    print("  3. 假设文档向量 ←→ 文档向量 '图书馆服务时间：8:00-22:00'")
    print("  距离更小（陈述句 vs 陈述句）✅")
    print("=" * 60)


if __name__ == "__main__":
    direct_vs_hyde_example()
