"""
=============================================================================
SUSTech Campus Web Crawler — 基于 Scrapy 的南科大网站爬虫
=============================================================================
爬取目标：
  1. www.sustech.edu.cn — 学校主站（官方政策、新闻、信息公开）
  2. lib.sustech.edu.cn — 图书馆（服务时间、数据库、借阅规则）
  3. admit.sustech.edu.cn — 招生网站（招生简章、录取信息）
  4. 各院系子站 — 通过主站链接自动发现

技术要点：
  - 遵守 robots.txt（不会爬禁止的路径）
  - 1.5 秒下载延迟（不给学校服务器造成压力）
  - 只跟踪 sustech.edu.cn 子域名
  - 保存为 JSONL 格式（每行一个 JSON 对象，方便后续流式处理）
  - 记录所有失败的 URL 到文件，方便排查

使用方法：
  cd sustech-rag
  scrapy runspider indexing/scraper.py -o data/raw/raw_pages.jsonl

=============================================================================
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor

# ============================================================================
# 导入全局配置
# ============================================================================
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    RAW_DIR,
    SCRAPY_DOWNLOAD_DELAY,
    SCRAPY_SEED_URLS,
    SCRAPY_TARGET_PAGES,
)


class SUSTechSpider(scrapy.Spider):
    """
    南科大校园网站爬虫

    工作流程：
    1. 从 SEED_URLS 开始爬取
    2. 访问每个页面，提取 HTML
    3. 从页面中提取所有内部链接，加入待爬队列
    4. 达到 TARGET_PAGES 后停止
    """

    name = "sustech_crawler"
    # custom_settings 会覆盖 Scrapy 全局 settings.py 的默认值
    # 为什么在这里设置而不是用 settings.py？→ 保持项目结构简洁，一个文件就能跑
    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        # ^ robots.txt 是网站根目录下的文件，告诉爬虫哪些路径可以爬
        #   sustech.edu.cn/robots.txt 通常允许大部分静态页面

        "DOWNLOAD_DELAY": SCRAPY_DOWNLOAD_DELAY,
        # ^ 两次请求之间的等待时间（秒）
        #   1.5 秒意味着每分钟约 40 个请求，不会对服务器造成压力

        "RANDOMIZE_DOWNLOAD_DELAY": True,
        # ^ 在 DOWNLOAD_DELAY 基础上加入随机抖动
        #   避免请求过于规律，模拟人类浏览行为

        "CONCURRENT_REQUESTS": 4,
        # ^ 同时发送的请求数。设为 4 是因为：
        #   太高 = 被服务器封 IP
        #   太低 = 爬太慢
        #   1.5s delay × 4 concurrent ≈ 每分钟 160 个页面

        "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
        # ^ 对同一个域名的并发限制。因为所有子站都是 sustech.edu.cn，
        #   设成 2 避免对同一台服务器造成压力

        "USER_AGENT": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "SUSTech-RAG-Project/1.0 (Educational Research; "
            "contact: student-project@sustech.edu.cn)"
        ),
        # ^ 自定义 User-Agent，明确标识自己的身份
        #   这是一种良好的爬虫礼仪，学校 IT 如果看到异常流量可以联系我们

        "DOWNLOAD_TIMEOUT": 30,
        # ^ 单个页面下载的超时时间（秒）
        #   如果 30 秒内没有响应，放弃这个 URL 并记录到失败列表

        "RETRY_TIMES": 2,
        # ^ 失败后重试次数。只重试 2 次，避免在死链上浪费时间

        "LOG_LEVEL": "INFO",
        # ^ Scrapy 自身的日志级别。INFO 会显示关键进度，DEBUG 会刷屏

        # 输出格式设置
        "FEEDS": {
            str(RAW_DIR / "raw_pages.jsonl"): {
                "format": "jsonlines",
                "encoding": "utf-8",
                "overwrite": True,
                # ^ overwrite=True: 每次运行覆盖旧文件
                #   overwrite=False: 追加到旧文件（适合增量爬取）
            },
        },

        # 自动限速扩展（AutoThrottle）：
        # 根据服务器响应时间自动调整下载延迟
        # 如果服务器响应变慢，自动增加延迟；响应快，自动减少延迟
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": SCRAPY_DOWNLOAD_DELAY,
        "AUTOTHROTTLE_MAX_DELAY": 5.0,
        # ^ 最大不会超过 5 秒延迟

        "CLOSESPIDER_PAGECOUNT": SCRAPY_TARGET_PAGES,
        # ^ 达到目标页面数后自动停止，防止无限爬取
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.page_count = 0
        self.failed_urls = []
        # LinkExtractor 只创建一次（避免每个页面都重新编译正则）
        self._link_extractor = LinkExtractor(
            allow_domains=["sustech.edu.cn"],
            deny=[
                r"\.pdf$", r"\.zip$", r"\.docx?$", r"\.xlsx?$",
                r"\.ppt$", r"\.jpg$", r"\.png$", r"\.gif$", r"\.mp4$",
                r"\.css$", r"\.js$", r"\.ico$", r"\.svg$",
            ],
            deny_extensions=[],
            unique=True,
        )
        # 确保 raw 目录存在
        RAW_DIR.mkdir(parents=True, exist_ok=True)

    def start_requests(self):
        """
        生成初始请求。

        Scrapy 框架在启动时会调用这个方法。
        返回一个 generator，yield 每个初始 URL 的 Request 对象。
        """
        for url in SCRAPY_SEED_URLS:
            self.logger.info(f"Starting crawl from seed: {url}")
            yield scrapy.Request(
                url=url,
                callback=self.parse,
                # ^ callback: 下载完成后调用哪个方法来处理响应
                errback=self.handle_error,
                # ^ errback: 如果下载失败（超时、404 等）调用哪个方法
                meta={"depth": 0},
                # ^ meta: 在请求和响应之间传递自定义数据
            )

    def parse(self, response):
        """
        处理每个下载完成的页面。

        这个方法是整个爬虫的核心逻辑：
        1. 从 response 中提取数据（URL、HTML、状态码等）
        2. 提取页面中的内部链接，生成新的请求
        3. 控制爬取深度和数量

        参数：
            response: Scrapy 的 Response 对象，包含完整的 HTTP 响应
        """
        # ── 第 1 步：检查响应是否有效 ──
        if response.status != 200:
            self.failed_urls.append(f"{response.url} (status={response.status})")
            return

        # 检查内容类型：只保留 HTML 页面，跳过 PDF/图片/视频 等二进制文件
        content_type = response.headers.get("Content-Type", b"").decode("utf-8", errors="ignore")
        if "text/html" not in content_type.lower():
            # 不是 HTML → 跳过，但不记录为"失败"（没有爬取的价值）
            return

        # ── 第 2 步：提取页面数据 ──
        self.page_count += 1

        # 解析页面的 <title> 标签，用于后续的文档分类和显示
        title = response.css("title::text").get()
        if title:
            title = title.strip()
        else:
            title = ""

        # 构建输出记录
        # 每条记录的字段：
        #   url: 页面的规范 URL（去掉锚点和部分参数）
        #   html: 原始 HTML（后续 cleaner.py 会处理）
        #   title: 页面标题
        #   crawled_at: ISO 8601 时间戳（带时区）
        #   status_code: HTTP 状态码
        #   content_type: 响应的 Content-Type
        record = {
            "url": self._normalize_url(response.url),
            "html": response.text,
            "title": title,
            "crawled_at": datetime.now(timezone.utc).isoformat(),
            "status_code": response.status,
            "content_type": content_type,
        }

        # 每爬 100 个页面打印一次进度
        if self.page_count % 100 == 0:
            self.logger.info(
                f"Progress: {self.page_count}/{SCRAPY_TARGET_PAGES} pages crawled. "
                f"Current: {response.url}"
            )

        yield record

        # ── 第 3 步：发现新的链接 ──
        # 使用 __init__ 中预创建的 LinkExtractor（避免每个页面重新编译正则）
        links = self._link_extractor.extract_links(response)

        for link in links:
            # 控制爬取深度：最多深度 5 层
            # 深度 0：种子页面（首页）
            # 深度 1：首页上的链接（各栏目页）
            # 深度 2：栏目页上的链接（文章列表）
            # 深度 3：具体文章页
            # 深度 4-5：文章中的引用链接
            current_depth = response.meta.get("depth", 0)
            if current_depth >= 5:
                continue

            yield scrapy.Request(
                url=link.url,
                callback=self.parse,
                errback=self.handle_error,
                meta={"depth": current_depth + 1},
            )

    def handle_error(self, failure):
        """
        处理下载失败的请求。

        参数：
            failure: Twisted Failure 对象，包含失败的原因
        """
        # 从 failure 中提取原始 URL
        url = failure.request.url if hasattr(failure, "request") else str(failure)
        error_msg = str(failure.value) if hasattr(failure, "value") else "Unknown error"

        self.failed_urls.append(f"{url} ({error_msg})")
        self.logger.warning(f"Failed to crawl: {url} — {error_msg}")

    def closed(self, reason):
        """
        爬虫关闭时的回调。

        参数：
            reason: 关闭原因（如 'finished', 'closespider_pagecount' 等）

        这个方法在爬虫完全停止后被 Scrapy 调用。
        我们利用这个时机：
        1. 打印爬取统计信息
        2. 将失败 URL 列表写入文件
        """
        # 写入失败 URL 列表
        failed_path = RAW_DIR / "failed_urls.txt"
        with open(failed_path, "w", encoding="utf-8") as f:
            f.write(f"# Failed URLs from crawl session\n")
            f.write(f"# Total failed: {len(self.failed_urls)}\n")
            f.write(f"# Closed reason: {reason}\n\n")
            for url in self.failed_urls:
                f.write(f"{url}\n")

        # 终端输出统计摘要
        self.logger.info("=" * 60)
        self.logger.info(f"Crawl finished. Reason: {reason}")
        self.logger.info(f"Total pages crawled: {self.page_count}")
        self.logger.info(f"Total failed URLs: {len(self.failed_urls)}")
        self.logger.info(f"Failed URLs saved to: {failed_path}")
        self.logger.info(f"Output saved to: {RAW_DIR / 'raw_pages.jsonl'}")
        self.logger.info("=" * 60)

    @staticmethod
    def _normalize_url(url: str) -> str:
        """
        URL 规范化：去掉不影响页面内容的 URL 组成部分。

        处理内容：
        - 去掉 fragment（#anchor）：# 后面的内容是页面内跳转，不改变页面内容
        - 去掉部分 query parameter（?utm_source=...）：跟踪参数，在后续 cleaner.py 中处理
        - 去掉末尾的 /
        - 转小写（域名部分）

        为什么在这里就去掉 fragment？
        → 避免同一个页面的不同 anchor 链接被当作"不同的 URL"
           例如：/page#section1 和 /page#section2 其实指向同一个页面

        参数：
            url: 原始 URL 字符串

        返回：
            规范化后的 URL 字符串
        """
        parsed = urlparse(url)
        # 去掉 fragment（锚点）
        normalized = urlunparse((
            parsed.scheme,
            parsed.netloc.lower(),
            parsed.path.rstrip("/") if parsed.path != "/" else "/",
            parsed.params,
            parsed.query,
            "",  # fragment 留空
        ))
        return normalized


# ============================================================================
# 主入口：直接运行此文件时启动爬虫
# ============================================================================
if __name__ == "__main__":
    """
    使用方式：
        python indexing/scraper.py

    这行命令会：
    1. 创建 CrawlerProcess（Scrapy 的进程管理器）
    2. 启动 SUSTechSpider
    3. 阻塞直到爬取完成（或达到 TARGET_PAGES）
    4. 输出 raw_pages.jsonl 到 data/raw/
    """
    process = CrawlerProcess()
    process.crawl(SUSTechSpider)
    process.start()
