"""
=============================================================================
SUSTech Chinese Tokenizer — Jieba 分词 + 停用词管理（共享模块）
=============================================================================
从 bm25_builder.py 抽取出来，供 bm25_builder 和 sparse_retriever 共享。
避免循环依赖和函数签名不一致的问题。

使用方法：
  from indexing.tokenizer import load_stopwords, tokenize
  stopwords = load_stopwords()
  tokens = tokenize("图书馆几点开门", stopwords)
=============================================================================
"""

from pathlib import Path
from urllib.request import urlretrieve

# ============================================================================
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR

try:
    import jieba
    try:
        jieba.enable_parallel()
    except Exception:
        pass
except ImportError:
    print("ERROR: jieba not installed. Run: pip install jieba")
    raise

# ============================================================================
# 停用词
# ============================================================================

STOPWORDS: set[str] = set()
STOPWORDS_URL = "https://raw.githubusercontent.com/goto456/stopwords/master/cn_stopwords.txt"
STOPWORDS_PATH = DATA_DIR / "stopwords_zh.txt"

_MINIMAL_STOPWORDS = (
    "的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 "
    "你 会 着 没有 看 好 自己 这 他 她 它 们 那 些 么 什么 怎么 哪 "
    "为什么 啊 吧 吗 呢 哦 嗯 哈 呀 哇 啦 而 但 且 或 与 及 被 把 "
    "从 对 以 让 给 向 因 所以 因此 但是 然而 虽然 如果 因为 不过 "
    "可以 这个 那个 这里 那里 哪里 怎么 怎么样 还 没 除了 按照 "
    "关于 由于 经过 然后 接着 最后 目前 最近 已经 即将 马上"
)


def download_stopwords() -> Path:
    """下载中文停用词表，失败则创建极简版。"""
    if STOPWORDS_PATH.exists():
        return STOPWORDS_PATH
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading Chinese stopwords...")
    try:
        urlretrieve(STOPWORDS_URL, STOPWORDS_PATH)
    except Exception:
        print("Download failed, creating minimal stopwords...")
        with open(STOPWORDS_PATH, "w", encoding="utf-8") as f:
            for w in _MINIMAL_STOPWORDS.split():
                f.write(w + "\n")
    return STOPWORDS_PATH


def load_stopwords() -> set[str]:
    """加载停用词到内存 set（O(1) 查找，全局缓存）。"""
    global STOPWORDS
    if STOPWORDS:
        return STOPWORDS
    path = download_stopwords()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word:
                STOPWORDS.add(word)
    print(f"Loaded {len(STOPWORDS)} stopwords")
    return STOPWORDS


def tokenize(text: str, stopwords: set[str]) -> list[str]:
    """
    Jieba 精确模式分词 + 过滤。

    过滤规则：
    - 长度 < 2 → 丢弃（单字无检索区分度）
    - 在停用词表中 → 丢弃（高频功能词）
    - 英文词原样保留（SUSTech, AI 等对校园场景重要）
    """
    tokens = jieba.lcut(text)
    return [
        t.strip() for t in tokens
        if len(t.strip()) >= 2 and t.strip() not in stopwords
    ]
