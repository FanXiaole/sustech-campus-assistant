"""
=============================================================================
SUSTech Document Cleaning Pipeline — 文档清洗管道
=============================================================================
功能：将原始的 HTML JSONL 转换为结构化的清洗文档 JSONL
输入：data/raw/raw_pages.jsonl（scraper.py 的输出）
输出：data/processed/processed_docs.jsonl

清洗管道的 8 个步骤（每个步骤是独立的函数，按顺序依次调用）：
  1. extract_text()       — BeautifulSoup 提取纯文本，移除噪声元素
  2. detect_language()    — 语言检测，只保留中文和英文
  3. normalize_whitespace()— 折叠多余的空白字符
  4. filter_short()       — 过滤掉过短的页面（空壳 / SPA 骨架）
  5. url_normalize()      — URL 规范化，去掉无关参数
  6. dedup_by_url()       — 按 URL 去重（保留最新抓取的那条）
  7. dedup_by_hash()      — 按内容 MD5 去重（近重复检测）
  8. classify_source()    — 根据 URL 模式分类来源类型

设计哲学：
  - 每个步骤是纯函数（输入 → 输出），方便单独测试
  - 链式调用：上一步的输出是下一步的输入
  - 一步一统计：每步完成后打印处理统计（文档数变化、丢弃原因）

使用方法：
  python indexing/cleaner.py

=============================================================================
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse, parse_qs

from bs4 import BeautifulSoup
from langdetect import DetectorFactory, detect, LangDetectException

# ============================================================================
# 统一编码设置 & 配置导入
# ============================================================================
# langdetect 的检测结果有一定随机性（基于概率模型）
# 设置种子保证每次清洗同一份数据得到相同的结果
DetectorFactory.seed = 42

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MIN_CHUNK_LEN,
    PROC_DIR,
    RAW_DIR,
    SOURCE_AUTHORITY,
)

# ============================================================================
# 第 1 步：文本提取
# ============================================================================

# 噪声标签与属性：这些 HTML 元素中通常是导航、广告、版权信息等
# 它们不包含对知识库有用的"正文信息"，需要剥离
NOISE_TAGS = [
    "nav", "footer", "script", "style", "header",
    "noscript", "iframe", "aside", "form",
]
# 噪声类名模式（很多网站的导航/侧栏有特定 class）
NOISE_CLASS_PATTERNS = [
    "nav", "footer", "sidebar", "menu", "breadcrumb",
    "copyright", "social", "share", "comment",
    "advertisement", "banner", "popup", "modal",
]
# aria-hidden="true" 的元素通常用于 CSS 动画/图标，不是正文
# 注意：不能用 dict（重复 key 会相互覆盖）→ 用 list of tuples
NOISE_ATTRS = [
    ("aria-hidden", "true"),
    ("role", "navigation"),
    ("role", "banner"),
    ("role", "contentinfo"),
]


def extract_text(html: str) -> str:
    """
    从原始 HTML 中提取纯文本内容。

    处理策略（按优先级顺序）：
    1. 首先移除 <script> 和 <style> 标签（它们的内容可能被误判为"文本"）
    2. 然后移除导航/页脚/侧栏等常见噪声区域
    3. 用 BeautifulSoup 的 get_text() 提取剩余文本
    4. 合并连续的空白行，但保留段落分隔

    参数：
        html: 原始 HTML 字符串

    返回：
        提取出的纯文本字符串。如果 HTML 无效，返回空字符串。
    """
    if not html or len(html.strip()) < 10:
        # 空页面或太短→不太可能是有效 HTML
        return ""

    try:
        soup = BeautifulSoup(html, "lxml")
        # ^ lxml 解析器比 html.parser 快约 3-5 倍
        #   对于几万页面的处理，这是显著的性能提升
    except Exception:
        # 如果 lxml 也解析不了，这个页面基本不可用
        return ""

    # ── 移除脚本和样式 ──
    # decompose() 会彻底删除标签及其内容
    for tag_name in ["script", "style", "noscript"]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # ── 移除噪声区域 ──
    # 1. 按标签名移除
    for tag_name in NOISE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # 2. 按 CSS class 模式移除（部分匹配）
    #    原因：<div class="sidebar-main"> 和 <div class="sidebar__inner">
    #    都应该被移除，但 class 名不完全相同。
    for element in soup.find_all(True):  # True = 匹配所有标签
        if not hasattr(element, "attrs") or element.attrs is None:
            continue
        classes = element.get("class", [])
        if isinstance(classes, list):
            class_str = " ".join(classes).lower()
        else:
            class_str = str(classes).lower()

        # 检查 class 是否匹配任何噪声模式
        for pattern in NOISE_CLASS_PATTERNS:
            if pattern in class_str:
                element.decompose()
                break

    # 3. 按 aria-* 属性移除
    for attr, value in NOISE_ATTRS:  # list of tuples
        for element in soup.find_all(attrs={attr: value}):
            element.decompose()

    # ── 提取文本 ──
    # separator="\n": 每个块级元素之间插入换行
    # strip=True: 去除每段的首尾空白
    text = soup.get_text(separator="\n", strip=True)

    return text


# ============================================================================
# 第 2 步：语言检测
# ============================================================================

def detect_language(text: str) -> tuple[str, str]:
    """
    检测文本的主要语言。

    我们只保留中文和英文，原因：
    - 南科大官网绝大多数内容是中文，部分有英文对照
    - 其他语言（日文、韩文等）在网站中极少出现，可能是爬虫抓到的
      外部链接或垃圾页面
    - 一个包含多语言的文档就用 langdetect 的 "zh-cn" 或 "en" 判断

    参数：
        text: 清洗后的纯文本

    返回：
        (语言代码, 是否保留) 的元组。
        语言代码如 "zh-cn", "en", "ja" 等。
        是否保留为 True（中文/英文）或 False（其他语言）。
    """
    if not text or len(text) < 50:
        # 太短的文本 langdetect 检测不准
        # 保守策略：保留（因为 filter_short 之后会过滤）
        return ("unknown", True)

    try:
        lang = detect(text)
    except LangDetectException:
        # 检测失败（文本可能是纯数字、纯符号等）
        return ("unknown", True)

    # 中文变体统一为 "zh"
    if lang.startswith("zh"):
        return (lang, True)
    # 英文
    if lang == "en":
        return (lang, True)

    # 其他语言 → 丢弃
    return (lang, False)


# ============================================================================
# 第 3 步：空白规范化
# ============================================================================

# 匹配 3 个或更多连续换行 → 压缩为 2 个（保留段落分隔但去掉空行泛滥）
_RE_MULTIPLE_NEWLINES = re.compile(r"\n{3,}")
# 匹配 tab 和垂直 tab → 替换为单空格
_RE_TABS = re.compile(r"[\t\v]")
# 匹配行内多余空格（2 个或更多）→ 压缩为 1 个
_RE_MULTISPACE = re.compile(r"[^\S\n]{2,}")


def normalize_whitespace(text: str) -> str:
    """
    规范化文本中的空白字符。

    为什么要规范化？
    - HTML 提取后的文本通常有大量空白字符（缩进、空行）
    - 这些空白会影响 chunk 大小的计算精度
    - 但也不要把所有换行都去掉 → 段落边界对后续切分很重要

    处理策略：
    1. tab → space
    2. 多个连续空行 → 最多保留一个空行
    3. 行内多余空格 → 压缩为一个
    4. 去掉首尾空白

    参数：
        text: 原始文本

    返回：
        规范化后的文本
    """
    # tab → space
    text = _RE_TABS.sub(" ", text)
    # 3+ 换行 → 2 换行（一个空行 = 段落分隔）
    text = _RE_MULTIPLE_NEWLINES.sub("\n\n", text)
    # 行内多余空格 → 单空格
    text = _RE_MULTISPACE.sub(" ", text)
    # 首尾空白
    text = text.strip()

    return text


# ============================================================================
# 第 4 步：短文本过滤
# ============================================================================

def check_text_length(text: str, min_len: int = MIN_CHUNK_LEN, max_len: int = 50000) -> tuple[bool, str]:
    """
    过滤异常长度的文档。
    - 过短 (< min_len): 空壳/SPA/登录页/纯图片页
    - 过长 (> max_len): 数据 dump（如 PyPI 镜像列表、JSON 端点）
    返回 (是否保留, 拒绝原因)。
    """
    if len(text) < min_len:
        return (False, "too_short")
    if len(text) > max_len:
        return (False, "too_long")
    return (True, "")


# ============================================================================
# 第 5 步：URL 规范化
# ============================================================================

def normalize_url(url: str) -> str:
    """
    规范化 URL，为去重做准备。

    处理：
    - 去掉 fragment（#section）→ 不改变页面内容
    - 去掉 utm_*, fbclid, gclid 等跟踪参数 → 和内容无关
    - 转为小写（scheme + netloc 部分）→ 统一大小写差异
    - 去掉末尾 /

    参数：
        url: 原始 URL

    返回：
        规范化后的 URL
    """
    parsed = urlparse(url)

    # 过滤 query parameters：只保留可能影响页面内容的参数
    # page=, id=, article=, p= 这些参数影响内容
    # utm_*, fbclid, gclid 等只用于跟踪，不影响内容
    if parsed.query:
        params = parse_qs(parsed.query, keep_blank_values=False)
        # 过滤掉跟踪参数
        clean_params = {
            k: v
            for k, v in params.items()
            if not any(
                k.lower().startswith(prefix)
                for prefix in ["utm_", "fbclid", "gclid", "msclkid", "ref", "source"]
            )
        }
        # 重建 query string
        if clean_params:
            query = "&".join(
                f"{k}={v[0]}" for k, v in clean_params.items() if v
            )
        else:
            query = ""
    else:
        query = ""

    # 去掉末尾 /（除非是根路径）
    path = parsed.path.rstrip("/") if parsed.path != "/" else "/"

    normalized = urlunparse((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        path,
        parsed.params,
        query,
        "",  # fragment 为空
    ))
    return normalized


# ============================================================================
# 第 6 步：按 URL 去重
# ============================================================================

def dedup_by_url(docs: list[dict]) -> list[dict]:
    """
    按 URL 去重：同一个 URL 的多条记录，只保留 crawed_at 最新的那条。

    为什么同一个 URL 会有多条记录？
    - 同一个页面可能通过不同的路径被发现（比如从首页和从栏目页）
    - cron 定期爬取会产生重复
    - normalize_url 之后部分 URL 会合并

    参数：
        docs: 文档列表

    返回：
        去重后的文档列表
    """
    seen: dict[str, dict] = {}

    for doc in docs:
        url = doc["url"]
        if url not in seen:
            seen[url] = doc
        else:
            # 保留 crawed_at 更新的
            existing_time = seen[url].get("crawled_at", "")
            new_time = doc.get("crawled_at", "")
            if new_time > existing_time:
                seen[url] = doc

    return list(seen.values())


# ============================================================================
# 第 7 步：按内容哈希去重
# ============================================================================

def dedup_by_hash(docs: list[dict]) -> list[dict]:
    """
    按内容 MD5 哈希去重：检测完全重复或高度相似的页面。

    现实中的重复场景：
    - 同一个公告被转载到多个栏目（URL 不同，内容完全相同）
    - 分页导航的第一页和最后一页的 header/footer 完全相同
    - 模板页面：同一个模板生成的页面，只有标题不同
      （MD5 不会匹配，这是预期行为）

    使用 MD5 而不是 simhash/MinHash？
    → 对于"完全相同的文本"，MD5 是最快最精确的。
      如果需要模糊匹配（>95% 相似），后续可以考虑 MinHash。
      但目前精确去重已经能解决大部分问题。

    参数：
        docs: 文档列表

    返回：
        去重后的文档列表
    """
    seen_hashes: set[str] = set()
    unique_docs: list[dict] = []

    for doc in docs:
        text_hash = doc.get("text_hash")
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            unique_docs.append(doc)

    return unique_docs


# ============================================================================
# 第 8 步：来源分类
# ============================================================================

# URL 模式到来源类型的映射表
# 为什么用 URL 模式而不是网站内容来判断？
# → 速度快、无歧义：lib.sustech.edu.cn 一定是图书馆网站，
#    不需要分析页面内容。
SOURCE_PATTERNS = {
    "admission": [
        "admit.sustech.edu.cn",
        "zs.sustech.edu.cn",
        "admission.sustech.edu.cn",
    ],
    "library": [
        "lib.sustech.edu.cn",
        "library.sustech.edu.cn",
    ],
    "department": [
        # 院系子域名：如 cse.sustech.edu.cn, math.sustech.edu.cn
        # 不是所有子域名都是院系 → 用排除法（不是其他类型的子域名就是院系）
    ],
    "news": [
        "newshub.sustech.edu.cn",
        "news.sustech.edu.cn",
    ],
    "official": [
        "www.sustech.edu.cn",
        "sustech.edu.cn",
    ],
}


def classify_source(url: str) -> str:
    """
    根据 URL 将文档归入一个来源类型。

    分类逻辑（优先级从高到低）：
    1. 精确匹配 known patterns → 直接返回对应的 source_family
    2. 包含 sustech.edu.cn 子域名 → department（院系）
    3. 包含 sustech.edu.cn 但不匹配任何 rules → official（默认官方）
    4. 不包含 sustech.edu.cn → unknown（外部链接，不应出现但保留兜底）

    参数：
        url: 规范化后的 URL

    返回：
        来源类型字符串，对应 config.SOURCE_AUTHORITY 的 key
    """
    url_lower = url.lower()
    parsed = urlparse(url_lower)
    domain = parsed.netloc

    # 第 1 轮：精确匹配
    for source_type, patterns in SOURCE_PATTERNS.items():
        if source_type == "department":
            continue  # department 用排除法
        for pattern in patterns:
            if pattern in domain:
                return source_type

    # 第 2 轮：判断是否是院系子域名
    # 院系子域名的特征：xxx.sustech.edu.cn 且不是 www/lib 等已知服务
    if domain.endswith(".sustech.edu.cn") and domain.count(".") >= 2:
        # 排除已知的非院系子域名
        non_dept_subdomains = [
            "www", "lib", "library", "admit", "zs", "news", "newshub",
            "mail", "sso", "auth", "idp", "portal", "vpn",
        ]
        subdomain_part = domain.split(".")[0].lower()
        if subdomain_part not in non_dept_subdomains:
            return "department"

    # 第 3 轮：包含 sustech.edu.cn 但未匹配特定模式
    if "sustech.edu.cn" in domain:
        return "official"

    # 第 4 轮：外部域名
    return "unknown"


# ============================================================================
# 主清洗管道
# ============================================================================

def run_cleaning(input_path: Path = None, output_path: Path = None) -> dict:
    """
    执行完整的文档清洗管道。

    这个函数把所有 8 个步骤串联起来，并记录每一步的统计信息。
    统计信息的目的是：
    - 了解数据质量（多少页面是垃圾？什么样的垃圾最多？）
    - 方便后续 debug（某个 chunk 的来源文档在哪个阶段被丢弃了？）

    参数：
        input_path: 输入 JSONL 路径，默认为 data/raw/raw_pages.jsonl
        output_path: 输出 JSONL 路径，默认为 data/processed/processed_docs.jsonl

    返回：
        统计信息字典，包含每一步过滤掉的文档数量
    """
    if input_path is None:
        web_path = RAW_DIR / "raw_pages.jsonl"
        manual_path = RAW_DIR / "manual_docs.jsonl"
    else:
        web_path = input_path
        manual_path = None
    if output_path is None:
        PROC_DIR.mkdir(parents=True, exist_ok=True)
        output_path = PROC_DIR / "processed_docs.jsonl"

    # ── 统计计数器 ──
    stats = {
        "total_input": 0,
        "removed_empty_text": 0,
        "removed_wrong_lang": 0,
        "removed_too_short": 0,
        "removed_too_long": 0,
        "removed_dup_url": 0,
        "removed_dup_hash": 0,
        "final_count": 0,
        "total_chars": 0,
    }

    # ── 加载原始数据（网页 + 南科手册） ──
    raw_docs = []

    # 网页数据
    web_count = 0
    if web_path and web_path.exists():
        with open(web_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    doc["_format"] = "html"  # 标记原始格式
                    raw_docs.append(doc)
                    web_count += 1
                except json.JSONDecodeError:
                    continue

    # 南科手册 Markdown 数据
    manual_count = 0
    if manual_path and manual_path.exists():
        with open(manual_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    doc["_format"] = "markdown"
                    doc["source_type"] = "manual"
                    raw_docs.append(doc)
                    manual_count += 1
                except json.JSONDecodeError:
                    continue

    stats["total_input"] = len(raw_docs)

    print(f"\n{'='*60}")
    print(f"SUSTech Document Cleaning Pipeline")
    print(f"{'='*60}")
    print(f"Input: {web_count} web pages + {manual_count} manual docs "
          f"= {stats['total_input']} total")

    # ── 逐步骤处理 ──
    processed_docs = []

    for i, raw in enumerate(raw_docs):
        url = raw.get("url", "")
        crawled_at = raw.get("crawled_at", "")
        fmt = raw.get("_format", "html")

        # ── 步骤 1: 提取文本（HTML→纯文本 / Markdown→直接使用） ──
        if fmt == "markdown":
            text = raw.get("markdown", "")
            if not text.strip():
                stats["removed_empty_text"] += 1
                continue
        else:
            html = raw.get("html", "")
            text = extract_text(html)
            if not text:
                stats["removed_empty_text"] += 1
                continue

        # ── 步骤 2: 语言检测 ──
        lang, keep = detect_language(text)
        if not keep:
            stats["removed_wrong_lang"] += 1
            continue

        # ── 步骤 3: 空白规范化 ──
        text = normalize_whitespace(text)

        # ── 步骤 4: 长度过滤（太短 = 空壳，太长 = 数据 dump） ──
        keep, reason = check_text_length(text)
        if not keep:
            if reason == "too_long":
                stats["removed_too_long"] += 1
            else:
                stats["removed_too_short"] += 1
            continue

        # ── 步骤 5: URL 规范化 ──
        normalized_url = normalize_url(url)

        # ── 步骤 6-8 在批次后统一处理 ──
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()

        # ── 步骤 8: 来源分类 ──
        if raw.get("source_type") == "manual":
            source_family = "manual"
        else:
            source_family = classify_source(normalized_url)

        # 构建 doc_id：URL 的 SHA256 前 16 个字符
        # 为什么是前 16 位？→ 256 位 = 64 hex chars，太长了
        # 前 16 位 hex = 64 bits，对于几万文档的集合，碰撞概率 < 10^-10
        doc_id = hashlib.sha256(normalized_url.encode("utf-8")).hexdigest()[:16]

        doc = {
            "doc_id": doc_id,
            "url": normalized_url,
            "text": text,
            "source_family": source_family,
            "char_count": len(text),
            "text_hash": text_hash,
            "crawled_at": crawled_at,
        }

        processed_docs.append(doc)

        # 每 500 条打印进度
        if (i + 1) % 500 == 0:
            print(f"  ... processed {i+1}/{stats['total_input']} documents")

    after_steps_1_5 = len(processed_docs)
    print(f"\n[Step 1-5] After text extraction, language filter, "
          f"whitespace normalization, short filter: {after_steps_1_5} docs")
    print(f"  - Empty/irrelevant HTML removed: {stats['removed_empty_text']}")
    print(f"  - Wrong language removed: {stats['removed_wrong_lang']}")
    print(f"  - Too short (< {MIN_CHUNK_LEN} chars): {stats['removed_too_short']}")
    print(f"  - Too long (> 50000 chars): {stats['removed_too_long']}")

    # ── 步骤 6: URL 去重 ──
    before_url_dedup = len(processed_docs)
    processed_docs = dedup_by_url(processed_docs)
    stats["removed_dup_url"] = before_url_dedup - len(processed_docs)
    print(f"[Step 6] After URL dedup: {len(processed_docs)} docs "
          f"(removed {stats['removed_dup_url']} duplicates)")

    # ── 步骤 7: 内容哈希去重 ──
    before_hash_dedup = len(processed_docs)
    processed_docs = dedup_by_hash(processed_docs)
    stats["removed_dup_hash"] = before_hash_dedup - len(processed_docs)
    print(f"[Step 7] After content hash dedup: {len(processed_docs)} docs "
          f"(removed {stats['removed_dup_hash']} near-duplicates)")

    # ── 统计 ──
    stats["final_count"] = len(processed_docs)
    stats["total_chars"] = sum(d["char_count"] for d in processed_docs)

    # ── 来源分布统计 ──
    source_counts: dict[str, int] = {}
    for doc in processed_docs:
        src = doc["source_family"]
        source_counts[src] = source_counts.get(src, 0) + 1

    # ── 写入输出 ──
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in processed_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # ── 打印最终摘要 ──
    total_mb = sum(len(d["text"].encode("utf-8")) for d in processed_docs) / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"CLEANING COMPLETE")
    print(f"{'='*60}")
    print(f"Final document count: {stats['final_count']}")
    print(f"Total text: {total_mb:.1f} MB ({stats['total_chars']:,} chars)")
    print(f"Removed (empty text): {stats['removed_empty_text']}")
    print(f"Removed (wrong lang): {stats['removed_wrong_lang']}")
    print(f"Removed (too short): {stats['removed_too_short']}")
    print(f"Removed (too long): {stats['removed_too_long']}")
    print(f"Removed (dup URL): {stats['removed_dup_url']}")
    print(f"Removed (dup hash): {stats['removed_dup_hash']}")
    print(f"\nSource distribution:")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        weight = SOURCE_AUTHORITY.get(src, 0.5)
        bar = "█" * (count // max(1, stats["final_count"] // 40))
        print(f"  {src:<15} {count:>6} docs  [weight={weight:.2f}]  {bar}")
    print(f"\nOutput saved to: {output_path}")
    print(f"{'='*60}\n")

    return stats


# ============================================================================
# 主入口
# ============================================================================
if __name__ == "__main__":
    run_cleaning()
