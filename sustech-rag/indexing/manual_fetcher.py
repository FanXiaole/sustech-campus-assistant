"""
=============================================================================
SUSTech Manual Fetcher — 南科助手文档爬取器
=============================================================================
爬取目标：https://github.com/SUSTech-CRA/sustech-online-ng/tree/master/docs
这是南科大学生维护的校园生活指南（"南科手册"），内容涵盖：
  - 校园设施（食堂、宿舍、图书馆、体育场馆）
  - 学术事务（选课、考试、学分、科研资源）
  - 生活服务（校园卡、医疗、交通巴士、快递）
  - 新生指南（入学流程、校园导航、常见问题）
  - 紧急信息（安全指引、联系方式）

价值：这是最权威的"非官方"校园信息来源，比官网更贴近学生实际需求。
      且采用 Markdown 格式 → 非常适合我们的 chunker.py 的 Strategy B。

爬取方式：通过 GitHub API 获取 docs/ 目录下的所有 .md 文件，
         然后通过 raw.githubusercontent.com 下载原始 Markdown 内容。

为什么不直接用 git clone？
  - 轻量级：只下载需要的 docs/ 目录，不下载整个 repo
  - 可更新：可以定期拉取最新的 commit
  - 可追溯：每个文件都有 GitHub 的 source URL

使用方法：python indexing/manual_fetcher.py
=============================================================================
"""

import json
import time
from pathlib import Path

import httpx

# ============================================================================
# 配置导入
# ============================================================================
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RAW_DIR

# GitHub 仓库配置
REPO_OWNER = "SUSTech-CRA"
REPO_NAME = "sustech-online-ng"
DOCS_PATH = "docs"  # 只爬 docs/ 目录
BRANCH = "master"

# GitHub API endpoints
GITHUB_API = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"
GITHUB_RAW = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}"


def fetch_file_list() -> list[dict]:
    """
    通过 GitHub API 获取 docs/ 目录下的所有 .md 文件列表。

    返回：
        [{"name": "campus-life.md", "path": "docs/campus-life.md",
          "download_url": "https://raw.githubusercontent.com/...", ...}, ...]
    """
    # GitHub API 的 Contents endpoint 可以直接列出目录内容
    # 不需要 token（公开仓库可以匿名访问）
    url = f"{GITHUB_API}/contents/{DOCS_PATH}?ref={BRANCH}"

    print(f"Fetching file list from GitHub API...")
    print(f"URL: {url}")

    response = httpx.get(url, timeout=30.0, follow_redirects=True)
    response.raise_for_status()
    files = response.json()

    md_files = []
    for f in files:
        if f["type"] != "file":
            continue
        name = f["name"].lower()
        # 只下载 .md 文件
        if name.endswith(".md"):
            md_files.append({
                "name": f["name"],
                "path": f["path"],
                "download_url": f["download_url"],
                "size": f["size"],
                "html_url": f["html_url"],
                "sha": f["sha"],
            })

    print(f"Found {len(md_files)} markdown files in {REPO_NAME}/{DOCS_PATH}/")
    return md_files


def download_markdown_files(file_list: list[dict], output_dir: Path) -> list[dict]:
    """
    下载所有 Markdown 文件，保存为 JSONL 格式。

    保存格式与 scraper.py 保持一致，这样 cleaner.py 可以统一处理：
    {
      "url": "https://raw.githubusercontent.com/...",
      "html": "",  # Markdown 文件没有 HTML
      "markdown": "原始 Markdown 内容",
      "title": "文件名（不含 .md 后缀）",
      "crawled_at": "2026-05-21T...",
      "source_type": "manual",
      "source_url": "https://github.com/...",
    }

    参数：
        file_list: fetch_file_list() 的返回值
        output_dir: 保存目录

    返回：
        下载的文档列表
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    docs = []

    for f in file_list:
        print(f"  Downloading: {f['name']} ({f['size']} bytes)...")

        try:
            response = httpx.get(f["download_url"], timeout=30.0)
            response.raise_for_status()
            markdown_content = response.text

            if not markdown_content.strip():
                print(f"    WARNING: Empty file, skipping")
                continue

            doc = {
                "url": f["download_url"],
                "html": "",  # Markdown 没有 HTML
                "markdown": markdown_content,
                "title": f["name"].replace(".md", ""),
                "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source_type": "manual",
                "source_url": f["html_url"],
                "source_sha": f["sha"],
            }
            docs.append(doc)
            print(f"    OK: {len(markdown_content)} chars")

        except Exception as e:
            print(f"    FAILED: {e}")

    # 保存为 JSONL
    output_path = output_dir / "manual_docs.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(docs)} documents to {output_path}")
    return docs


def run_manual_fetch():
    """
    主入口：爬取南科手册所有文档。

    这个函数在 run_all.sh 中会被调用（在 scraper.py 之后）。
    """
    print(f"\n{'='*60}")
    print(f"SUSTech Manual Fetcher — 南科助手文档")
    print(f"{'='*60}")
    print(f"Source: {REPO_OWNER}/{REPO_NAME}/{DOCS_PATH}")
    print(f"Format: Markdown (.md) files")
    print()

    # 步骤 1: 获取文件列表
    try:
        file_list = fetch_file_list()
    except Exception as e:
        print(f"ERROR: Failed to fetch file list from GitHub: {e}")
        print("This might be a network issue. Try:")
        print(f"  git clone https://github.com/{REPO_OWNER}/{REPO_NAME}.git")
        print(f"  cp -r {REPO_NAME}/{DOCS_PATH}/*.md data/raw/")
        return []

    if not file_list:
        print("No markdown files found. Check the repo structure.")
        return []

    # 步骤 2: 下载所有文件
    docs = download_markdown_files(file_list, RAW_DIR)

    print(f"\n{'='*60}")
    print(f"Manual fetch complete: {len(docs)} documents")
    print(f"Next: run cleaner.py to process these alongside web docs")
    print(f"{'='*60}\n")

    return docs


# ============================================================================
if __name__ == "__main__":
    run_manual_fetch()
