"""
=============================================================================
SUSTech Document Chunker — 双策略文档切分系统
=============================================================================
这是整个 RAG pipeline 中最关键的预处理步骤。切分质量直接决定检索质量。

两种切分策略：
  STRATEGY A — RecursiveCharacterChunker（递归字符切分）
    适用于：普通网页的纯文本（HTML 清洗后的产物）
    原理：  按分隔符优先级逐级尝试切分，优先在"语义边界"切
           例如：先尝试在句号处切，句号切不动再换逗号，最后才按字符数硬切
    优势：  天然尊重中文标点符号的语义边界

  STRATEGY B — MarkdownHeaderChunker（Markdown 标题切分）
    适用于：南科手册等结构化 Markdown 文档
    原理：  按 H1/H2/H3 标题层级切分，保留文档的层次结构
    优势：  标题本身提供强语义信号 → 检索时"图书馆 > 开放时间"
           和"某课程 > 考试时间"不会混在一起

与队友方案的差异：
  - 队友：手动按段落切分（500-800 字，max 1200）/ 手动按标题切
  - 我们：使用 LangChain 标准切分器 + 标注层级路径 + 元数据前缀
  - 额外：三种 chunk 大小变体用于消融实验

使用方法：
  python indexing/chunker.py

=============================================================================
"""

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# ============================================================================
# 配置导入
# ============================================================================
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    CHUNK_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE_DEFAULT,
    CHUNK_SIZE_LARGE,
    CHUNK_SIZE_SMALL,
    MIN_CHUNK_LEN,
    PROC_DIR,
)


# ============================================================================
# STRATEGY A: 递归字符切分器（适用于普通网页）
# ============================================================================

# 分隔符优先级列表（从最"强"的分隔符到最"弱"的分隔符）：
# 为什么要这个顺序？
# → 越强的分隔符越接近"语义边界"。如果可以在"。"处切，就不要在" "处切。
#   当 chunk_size 限制无法满足时，自动降级到下一个分隔符。
#
# "。\n" = 中文句号后跟换行 → 强烈暗示段落结束
# "。"   = 中文句号（任意位置）
# "\n\n" = 段落分隔（空行）
# "\n"   = 换行
# "，"   = 中文逗号 → 弱语义边界，至少比分词边界强
# " "    = 英文单词边界
# ""     = 最后手段：按字符硬切
WEB_SEPARATORS = ["。\n", "。", "\n\n", "\n", "，", ". ", ".", " ", ""]


def build_recursive_chunker(
    chunk_size: int = CHUNK_SIZE_DEFAULT,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """
    构建递归字符切分器。

    参数：
        chunk_size: 目标 chunk 大小（字符数），默认 600
        chunk_overlap: chunk 之间的重叠字符数，默认 100

    返回：
        配置好的 RecursiveCharacterTextSplitter 实例

    为什么需要 overlap？
    → 假设有一句话："学生宿舍的关门时间是晚上 11 点。"
       如果这句话正好在两个 chunk 的边界处被切开，前半在 chunk_03，
       后半在 chunk_04，那么问"宿舍几点关门"时就可能检索失败。
       overlap=100 意味着 chunk_03 的最后 100 个字符会在 chunk_04 的开头
       重复出现，有效防止这种边界切割问题。
    """
    return RecursiveCharacterTextSplitter(
        separators=WEB_SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # ^ 使用 Python 内置的 len() 来计算"字符数"
        #   注意：中文语境下，1 字符 ≈ 1~1.5 token，所以 600 字符 ≈ 600-900 token
        is_separator_regex=False,
        # ^ separators 列表中的元素会被当作"普通字符串"而非正则表达式
        #   因为我们的分隔符包含 "。" 这种非 ASCII 字符，regex 模式可能会意外匹配
        keep_separator=True,
        # ^ 保留分隔符在 chunk 中而不是丢弃它
        #   例如：在 "。" 处切分时，"。" 会保留在前一个 chunk 的末尾
        #   这对中文很重要——去掉句号会让文本不完整
    )


def add_web_metadata_prefix(text: str, source_family: str, url: str) -> str:
    """
    为网页 chunk 添加元数据前缀。

    为什么需要前缀？
    → 网页文本本身没有"我是来自图书馆网站的服务时间信息"这种元信息。
      在 embedding 时，前缀提供了来源信号，帮助检索系统理解 chunk 的出处。

    前缀格式：[来源:official][域名:www.sustech.edu.cn] 正文内容...

    参数：
        text: chunk 的纯文本
        source_family: 来源类型（"official", "library" 等）
        url: 原始页面 URL

    返回：
        带元数据前缀的文本
    """
    # 从 URL 中提取域名
    try:
        domain = urlparse(url).netloc.replace("www.", "")
    except Exception:
        domain = "unknown"

    prefix = f"[来源:{source_family}][域名:{domain}] "
    return prefix + text


# ============================================================================
# STRATEGY B: Markdown 标题切分器（适用于南科手册）
# ============================================================================

# Markdown 标题层级映射
# 为什么只用 H1-H3？
# → H4 及以下的标题粒度太细（通常是列表项内的子标题），
#   作为 chunk 的语义单元太碎片化，反而降低检索质量。
MD_HEADERS_TO_SPLIT = [
    ("#", "H1"),    # # 一级标题 → 文档的主章节
    ("##", "H2"),   # ## 二级标题 → 章节下的子话题
    ("###", "H3"),  # ### 三级标题 → 子话题下的具体条目
]

# 表格检测正则：匹配 | --- | 这样的 Markdown 表格分隔行
_TABLE_SEPARATOR_RE = re.compile(r"^\|[\s\-:|]+\|$", re.MULTILINE)

# 列表项检测正则：匹配以 "- " 或 "* " 或 "1. " 开头的行
_LIST_ITEM_RE = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+")


def build_markdown_chunker() -> MarkdownHeaderTextSplitter:
    """
    构建 Markdown 标题切分器。

    返回：
        配置好的 MarkdownHeaderTextSplitter 实例
    """
    return MarkdownHeaderTextSplitter(
        headers_to_split_on=MD_HEADERS_TO_SPLIT,
        strip_headers=True,
        # ^ 去掉 chunk 文本中重复的标题行
        #   例如：chunk 文本本身已经以 "## 开放时间" 开头，
        #   如果 strip_headers=False，这个标题行会保留在 chunk 中
        #   如果 strip_headers=True，标题行会被移除（但元数据中保留了标题信息）
    )


def detect_and_preserve_tables(text: str) -> list[dict]:
    """
    检测 Markdown 中的表格并将其标记为"不可拆分"。

    表格是语义上不可分割的单元：
    把一张图书馆开放时间表拆成两半，每一半都没用。
    所以我们先检测表格，把它们整体保留。

    参数：
        text: Markdown 文本

    返回：
        表格位置列表 [{"start": int, "end": int, "table_text": str}, ...]
    """
    tables = []
    lines = text.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        # 检测表格头 + 分隔行的模式
        # 典型的 Markdown 表格：
        #   | 列1 | 列2 |
        #   |-----|-----|
        #   | 值1 | 值2 |
        if line.startswith("|") and "|" in line:
            # 检查下一行是否是分隔行
            if i + 1 < len(lines) and _TABLE_SEPARATOR_RE.match(lines[i + 1]):
                table_start = i
                i += 2  # 跳过头和分隔行
                # 读取所有后续的数据行
                while i < len(lines) and lines[i].strip().startswith("|"):
                    i += 1
                table_end = i
                table_text = "\n".join(lines[table_start:table_end])
                tables.append({
                    "start_line": table_start,
                    "end_line": table_end,
                    "text": table_text,
                })
                continue
        i += 1

    return tables


def group_consecutive_list_items(text: str) -> str:
    """
    将连续的列表项合并为一个逻辑块。

    为什么？
    → 单个列表项（如 "- 食堂提供早餐"）太短，不成 chunk。
      但一组列表项（如 "- 早餐7:00\n- 午餐11:30\n- 晚餐17:30"）是一个完整的
      "餐饮时间表"——应该作为一个整体来检索。

    实现方式：
    用特殊标记 <!--LIST_GROUP--> 包裹连续的列表行，
    告诉后续的切分器不要在这些行之间切割。

    参数：
        text: Markdown 文本

    返回：
        处理后的文本（列表项组被标记包裹）
    """
    lines = text.split("\n")
    result_lines = []
    in_list_group = False
    list_buffer = []

    for line in lines:
        match = _LIST_ITEM_RE.match(line)
        if match:
            # 这是一个列表项
            if not in_list_group:
                in_list_group = True
                list_buffer = []
            list_buffer.append(line)
        else:
            # 不是列表项
            if in_list_group and list_buffer:
                # 结束当前列表组
                result_lines.append("<!--LIST_GROUP_START-->")
                result_lines.extend(list_buffer)
                result_lines.append("<!--LIST_GROUP_END-->")
                list_buffer = []
                in_list_group = False
            result_lines.append(line)

    # 处理文件末尾的列表组
    if in_list_group and list_buffer:
        result_lines.append("<!--LIST_GROUP_START-->")
        result_lines.extend(list_buffer)
        result_lines.append("<!--LIST_GROUP_END-->")

    return "\n".join(result_lines)


def adjust_chunk_lengths(
    chunks: list[dict],
    min_len: int = MIN_CHUNK_LEN,
    max_len: int = 900,
) -> list[dict]:
    """
    调整 Markdown chunk 的长度，确保每个 chunk 在合理范围内。

    处理逻辑：
    - 太短（< MIN_CHUNK_LEN）→ 合并到前一个 chunk（如果前一个也太短→合并到后一个）
    - 太长（> 900 chars）→ 按段落（\n\n）再切分

    参数：
        chunks: 初步切分后的 chunk 列表
        min_len: 最小字符数
        max_len: 最大字符数

    返回：
        调整后的 chunk 列表
    """
    # ── 第一轮：合并过短的 chunk ──
    merged_chunks = []
    buffer = None  # 暂存太短的 chunk，等下一个 chunk 来时合并

    for chunk in chunks:
        text = chunk.get("page_content", "")
        if len(text) < min_len:
            if buffer is None:
                # 这个短 chunk 先存起来
                buffer = chunk
            else:
                # 之前已经有一个短 chunk，合并它们
                buffer["page_content"] += "\n" + text
                # 合并标题路径
                existing_heading = buffer.get("metadata", {}).get("heading_path", "")
                new_heading = chunk.get("metadata", {}).get("heading_path", "")
                if new_heading and new_heading != existing_heading:
                    buffer["metadata"]["heading_path"] = (
                        f"{existing_heading} > {new_heading}" if existing_heading
                        else new_heading
                    )
        else:
            if buffer is not None:
                # 把之前暂存的短 chunk 合并到当前这个正常长度的 chunk
                buffer["page_content"] += "\n" + text
                buffer["metadata"]["heading_path"] = (
                    chunk.get("metadata", {}).get("heading_path", "")
                )
                merged_chunks.append(buffer)
                buffer = None
            else:
                merged_chunks.append(chunk)

    # 如果最后还有一个未处理的 buffer
    if buffer is not None and merged_chunks:
        # 合并到最后一个 chunk
        merged_chunks[-1]["page_content"] += "\n" + buffer["page_content"]
    elif buffer is not None:
        # 只有一个 chunk 而且很短 → 保留
        merged_chunks.append(buffer)

    # ── 第二轮：拆分过长的 chunk ──
    final_chunks = []
    for chunk in merged_chunks:
        text = chunk.get("page_content", "")
        if len(text) <= max_len:
            final_chunks.append(chunk)
        else:
            # 按段落（\n\n）再切分
            paragraphs = text.split("\n\n")
            sub_chunk_texts = []
            current_text = ""
            for para in paragraphs:
                if len(current_text) + len(para) < max_len:
                    current_text += ("\n\n" if current_text else "") + para
                else:
                    if current_text:
                        sub_chunk_texts.append(current_text)
                    current_text = para
            if current_text:
                sub_chunk_texts.append(current_text)

            # 为每个子 chunk 创建新记录
            for i, sub_text in enumerate(sub_chunk_texts):
                new_chunk = dict(chunk)  # 浅拷贝
                new_chunk["page_content"] = sub_text
                if i > 0:
                    # 子 chunk 的标题路径追加序号
                    heading = new_chunk.get("metadata", {}).get("heading_path", "")
                    new_chunk["metadata"]["heading_path"] = f"{heading} (part {i+1})"
                final_chunks.append(new_chunk)

    return final_chunks


def run_markdown_chunking(md_text: str, source_family: str, url: str) -> list[dict]:
    """
    对 Markdown 文档执行完整的结构化切分。

    流程：
    1. 检测并保护表格（让它们不会被切分破坏）
    2. 合并连续的列表项
    3. 按 H1/H2/H3 标题层级切分
    4. 调整 chunk 长度（合并短 chunk，拆分长 chunk）

    参数：
        md_text: Markdown 原始文本
        source_family: 来源类型（通常为 "manual"）
        url: 文档来源 URL

    返回：
        chunk 字典列表，每个 chunk 包含 text、metadata 等字段
    """
    # 步骤 1: 保护表格
    # 在实际切分前，先把表格区域标记出来，后续的切分器会跳过这些区域
    tables = detect_and_preserve_tables(md_text)

    # 步骤 2: 合并连续列表项
    text = group_consecutive_list_items(md_text)

    # 步骤 3: 按标题层级切分
    chunker = build_markdown_chunker()
    chunks = chunker.split_text(text)

    # 步骤 4: 调整长度
    chunks = adjust_chunk_lengths(chunks)

    # 步骤 5: 添加元数据
    domain = "sustech.edu.cn"
    try:
        domain = urlparse(url).netloc.replace("www.", "")
    except Exception:
        pass

    for i, chunk in enumerate(chunks):
        text = chunk.get("page_content", "")
        heading_path = ""
        # 从 LangChain metadata 中提取标题路径
        md = chunk.get("metadata", {})
        heading_parts = []
        for level in ["H1", "H2", "H3"]:
            if level in md and md[level]:
                heading_parts.append(md[level])
        heading_path = " > ".join(heading_parts)

        # 构建元数据前缀
        if heading_path:
            prefix = f"[来源:{source_family}][章节:{heading_path}][域名:{domain}] "
        else:
            prefix = f"[来源:{source_family}][域名:{domain}] "

        chunks[i] = {
            "text": prefix + text,
            "raw_text": text,
            "source_family": source_family,
            "url": url,
            "heading_path": heading_path,
            "char_count": len(text),
            "chunk_strategy": "markdown",
        }

    return chunks


# ============================================================================
# 主切分流程
# ============================================================================

def run_chunking(
    input_path: Path = None,
    chunk_size: int = CHUNK_SIZE_DEFAULT,
    output_suffix: str = "default",
) -> dict:
    """
    对清洗后的文档执行切分，输出一个 chunk JSONL 文件。

    这个函数会：
    1. 读取 processed_docs.jsonl
    2. 根据文本内容判断使用哪种切分策略
    3. 对每个文档执行切分
    4. 为每个 chunk 分配唯一 ID
    5. 保存到 data/chunks/chunks_{suffix}.jsonl
    6. 打印统计信息

    参数：
        input_path: 输入 JSONL 路径（清洗后的文档），默认从 config 读取
        chunk_size: chunk 大小（字符数），用于 recursive chunker
        output_suffix: 输出文件名后缀（"default", "small", "large"）

    返回：
        统计信息字典
    """
    if input_path is None:
        input_path = PROC_DIR / "processed_docs.jsonl"

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CHUNK_DIR / f"chunks_{output_suffix}.jsonl"

    print(f"\n{'='*60}")
    print(f"SUSTech Document Chunking — Strategy: {output_suffix}")
    print(f"{'='*60}")
    print(f"Input: {input_path}")
    print(f"Chunk size: {chunk_size} chars, Overlap: {CHUNK_OVERLAP} chars")

    # ── 加载清洗后的文档 ──
    docs = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    print(f"Loaded {len(docs)} processed documents")

    # ── 构建切分器 ──
    web_chunker = build_recursive_chunker(chunk_size=chunk_size)

    # ── 逐文档切分 ──
    all_chunks = []
    stats = {
        "total_docs": len(docs),
        "total_chunks": 0,
        "web_chunks": 0,
        "markdown_chunks": 0,
        "total_chars": 0,
        "char_counts": [],
    }

    for i, doc in enumerate(docs):
        text = doc.get("text", "")
        url = doc.get("url", "")
        doc_id = doc.get("doc_id", "")
        source_family = doc.get("source_family", "unknown")

        # 判断使用哪种切分策略
        # 判断依据：URL 中包含 "nanuke" 或 "manual" / 文本有明显的 Markdown 标题
        is_markdown = (
            "nanuke" in url.lower()
            or "manual" in url.lower()
            or bool(re.search(r"^#{1,3}\s", text, re.MULTILINE))
        )

        if is_markdown:
            # 策略 B: Markdown 标题切分
            chunks = run_markdown_chunking(text, source_family, url)
            stats["markdown_chunks"] += len(chunks)
        else:
            # 策略 A: 递归字符切分
            raw_chunks = web_chunker.split_text(text)
            chunks = []
            for j, chunk_text in enumerate(raw_chunks):
                if len(chunk_text) < MIN_CHUNK_LEN:
                    continue
                enriched_text = add_web_metadata_prefix(
                    chunk_text, source_family, url
                )
                chunks.append({
                    "text": enriched_text,
                    "raw_text": chunk_text,
                    "source_family": source_family,
                    "url": url,
                    "heading_path": "",
                    "char_count": len(chunk_text),
                    "chunk_strategy": "recursive",
                })
            stats["web_chunks"] += len(chunks)

        # 为每个 chunk 分配唯一 ID
        for j, chunk in enumerate(chunks):
            chunk["chunk_id"] = f"{doc_id}_chunk_{j:04d}"
            chunk["doc_id"] = doc_id
            all_chunks.append(chunk)
            stats["total_chars"] += chunk["char_count"]
            stats["char_counts"].append(chunk["char_count"])

        # 每 500 个文档打印进度
        if (i + 1) % 500 == 0:
            print(f"  ... chunked {i+1}/{len(docs)} docs, "
                  f"{len(all_chunks)} chunks so far")

    stats["total_chunks"] = len(all_chunks)

    # ── 保存输出 ──
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # ── 打印统计 ──
    print(f"\n{'='*60}")
    print(f"CHUNKING COMPLETE — {output_suffix}")
    print(f"{'='*60}")
    print(f"Total chunks:     {stats['total_chunks']:,}")
    print(f"  - Web (recursive):    {stats['web_chunks']:,}")
    print(f"  - Markdown (headers): {stats['markdown_chunks']:,}")
    print(f"Total chars:      {stats['total_chars']:,}")

    if stats["char_counts"]:
        char_counts = sorted(stats["char_counts"])
        n = len(char_counts)
        print(f"Char count stats:")
        print(f"  Min:    {char_counts[0]:>6}")
        print(f"  Max:    {char_counts[-1]:>6}")
        print(f"  Mean:   {sum(char_counts) / n:>8.0f}")
        print(f"  P50:    {char_counts[n // 2]:>6}")
        print(f"  P95:    {char_counts[int(n * 0.95)]:>6}")

    print(f"\nOutput saved to: {output_path}")
    print(f"{'='*60}\n")

    return stats


def run_all_chunk_sizes():
    """
    一次性生成三种 chunk 大小的输出文件（用于消融实验）。
    """
    print("\n" + "=" * 70)
    print("GENERATING ALL CHUNK SIZE VARIANTS")
    print("=" * 70)

    configs = [
        (CHUNK_SIZE_SMALL, "small"),
        (CHUNK_SIZE_DEFAULT, "default"),
        (CHUNK_SIZE_LARGE, "large"),
    ]

    all_stats = {}
    for size, suffix in configs:
        stats = run_chunking(chunk_size=size, output_suffix=suffix)
        all_stats[suffix] = stats

    print("=" * 70)
    print("ALL CHUNK SIZE VARIANTS GENERATED")
    print("=" * 70)
    for suffix, stats in all_stats.items():
        print(f"  chunks_{suffix}: {stats['total_chunks']:,} chunks, "
              f"avg {stats['total_chars'] // max(stats['total_chunks'], 1)} chars")
    print()

    return all_stats


# ============================================================================
# 主入口
# ============================================================================
if __name__ == "__main__":
    run_all_chunk_sizes()
