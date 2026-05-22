"""
=============================================================================
Query Classifier — 查询类型分类与动态路由 ★ 创新点 #4
=============================================================================
不同的查询需要不同的检索策略。一个一刀切的 pipeline 无法在所有查询类型上
都表现最佳。Query Classifier 的工作就是：

  1. 分析用户的查询意图
  2. 将查询分入预定义的类别
  3. 根据类别选择最优的检索策略和 RRF 权重

实现方案：双模分类器
  - Option A（规则分类器）：基于关键词和启发式规则，快速、零成本
  - Option B（LLM 分类器）：基于大模型，更准确但多一次 API 调用
  - 生产环境使用 Option B + 缓存，实验环境可以切换 Option A

分类类型：
  FACTUAL_SIMPLE   → BM25 优先    (关键词精确匹配)
  FACTUAL_COMPLEX  → Dense + RRF  (语义理解重要)
  COMPARATIVE      → HyDE + Dense (需要理解多个概念)
  PROCEDURAL       → Dense 优先   (语义 > 关键词)
  OUT_OF_SCOPE     → 立即拒答     (不浪费检索资源)
  TEMPORAL         → BM25 + 时效性 (最近的信息更重要)

使用方法：
  from retrieval.query_classifier import QueryClassifier
  qc = QueryClassifier()
  q_type = qc.classify("图书馆几点开门？")

=============================================================================
"""

import json
import time
from pathlib import Path
from typing import Callable

# ============================================================================
# 配置导入
# ============================================================================
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DENSE_TOP_K,
    BM25_TOP_K,
)

# ============================================================================
# 查询类型定义
# ============================================================================

class QueryType:
    """查询类型常量。"""
    FACTUAL_SIMPLE = "FACTUAL_SIMPLE"
    FACTUAL_COMPLEX = "FACTUAL_COMPLEX"
    COMPARATIVE = "COMPARATIVE"
    PROCEDURAL = "PROCEDURAL"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    TEMPORAL = "TEMPORAL"


# 各类型对应的最优检索权重配置
# (dense_weight, sparse_weight)
# 经验教训：第一版权重差异过大（0.5 vs 1.5），导致分类错误时伤害显著
# v2: 缩小权重范围（0.8-1.2），即使分类错误也不会严重偏离默认等权重
STRATEGY_WEIGHTS = {
    QueryType.FACTUAL_SIMPLE: (0.8, 1.2),
    #   BM25 稍重 → 关键词精确匹配有帮助
    #   例如："校历" → "2025-2026学年校历" 这种精确词匹配 BM25 做得最好

    QueryType.FACTUAL_COMPLEX: (1.2, 0.8),
    #   Dense 稍重 → 语义理解更重要，但不过度偏离等权重
    #   例如："计算机系教师的科研方向" →
    #   文档中出现的是"研究方向"、"研究领域"等语义变体

    QueryType.COMPARATIVE: (1.2, 0.8),
    #   Dense 稍重 → 对比类查询涉及理解两个概念的语义关系
    #   例如："理工学院和医学院的区别" → 不太可能有文档直接写"区别"

    QueryType.PROCEDURAL: (1.0, 1.0),
    #   等权重 → 流程类查询既有语义变体也有关键词精确匹配
    #   例如："如何办理借书证" → "借书证"可精确匹配，"办理流程"需语义理解

    QueryType.TEMPORAL: (0.8, 1.2),
    #   BM25 稍重 → 时间敏感信息通常是关键词匹配
    #   例如："2026年招生政策" → 年份是精确匹配的信号

    QueryType.OUT_OF_SCOPE: (0.0, 0.0),
    #   不检索 → 立即拒答
}


# ============================================================================
# Option A: 规则分类器（快速、零成本）
# ============================================================================

# SUSTech 相关的关键词列表
# 用于判断查询是否"在知识库范围内"
SUSTECH_KEYWORDS = [
    "南科大", "南方科技大学", "sustech", "SUSTech",
    "图书馆", "食堂", "宿舍", "教室", "实验室",
    "选课", "课程", "学分", "考试", "成绩", "学位",
    "导师", "教授", "科研", "专业", "学院", "系",
    "招生", "录取", "奖学金", "助学金", "学费",
    "社团", "学生会", "活动", "讲座", "体育",
    "校车", "公交", "巴士", "校园卡", "学生证",
    "借书", "还书", "数据库", "电子资源",
    "学工", "教务", "教务系统", "cas", "sso",
    "书院", "致仁", "树仁", "致诚", "树德", "致新", "树礼",
    "计算机系", "电子系", "材料系", "生物系", "数学系",
    "物理系", "化学系", "海洋系", "环境系", "金融系",
    # 为什么列这么多关键词？→ 防止把"南科大图书馆几点开"误判为 OUT_OF_SCOPE
    # 关键词列表覆盖了主要的校园生活、学术事务、行政服务等领域
]

# 对比类关键词
COMPARATIVE_KEYWORDS = [
    "区别", "对比", "比较", "不同", "哪个好", "更好",
    "差异", "异同", "优缺点", "更推荐", "选择",
    "vs", "VS", "还是", "或者",
]

# 流程类关键词
PROCEDURAL_KEYWORDS = [
    "如何", "怎么", "怎样", "流程", "步骤", "怎么办",
    "办理", "申请", "报名", "注册", "预约", "借用",
    "开通", "绑定", "激活", "充值", "挂失", "解挂",
]

# 时间类关键词
TEMPORAL_KEYWORDS = [
    "最新", "最近", "今年", "这学期", "本学期", "下学期",
    "2025", "2026", "现在", "目前", "当前", "新的",
    "变化", "更新", "调整", "最新政策", "最新招生",
]


# 其他学校列表（防止分类器把"港科大图书馆"判为 SUSTech）
OTHER_SCHOOLS = [
    "清华", "北大", "浙大", "复旦", "上海交大", "中科大",
    "港大", "港中文", "港科大", "麻省", "斯坦福", "哈佛",
    "深大", "哈工大", "中山大学", "武汉大学", "南京大学",
]


def classify_by_rules(query: str) -> tuple[str, float]:
    """
    基于关键词规则的快速查询分类。

    分类逻辑（按优先级顺序）：
    1. 先检查是否 OUT_OF_SCOPE（范围外）
    2. 再检查 TEMPORAL（时间敏感）
    3. 检查 PROCEDURAL（流程类）
    4. 检查 COMPARATIVE（对比类）
    5. 最后区分 FACTUAL_SIMPLE vs FACTUAL_COMPLEX

    返回：(查询类型, 置信度)
    """
    query_lower = query.lower().strip()

    # ── 第 1 层：范围检测 ──
    # 优先检查是否在问其他学校（无论是否含 SUSTech 关键词都判为 OOS）
    asks_other_school = any(school in query for school in OTHER_SCHOOLS)
    if asks_other_school:
        return (QueryType.OUT_OF_SCOPE, 0.90)

    # 检查是否含 SUSTech 相关关键词
    has_sustech = any(kw.lower() in query_lower for kw in SUSTECH_KEYWORDS)
    if not has_sustech and len(query) > 15:
        # 较长的查询但没有 SUSTech 关键词 → 可能是范围外
        return (QueryType.OUT_OF_SCOPE, 0.70)

    # ── 第 2 层：时间敏感检测 ──
    has_temporal = any(kw in query_lower for kw in TEMPORAL_KEYWORDS)
    if has_temporal:
        return (QueryType.TEMPORAL, 0.80)

    # ── 第 3 层：流程检测 ──
    has_procedural = any(kw in query_lower for kw in PROCEDURAL_KEYWORDS)
    if has_procedural:
        return (QueryType.PROCEDURAL, 0.85)

    # ── 第 4 层：对比检测 ──
    has_comparative = any(kw in query_lower for kw in COMPARATIVE_KEYWORDS)
    if has_comparative:
        return (QueryType.COMPARATIVE, 0.85)

    # ── 第 5 层：简单 vs 复杂事实 ──
    # 简短的查询 → 大概率是 simple fact
    if len(query) <= 15:
        return (QueryType.FACTUAL_SIMPLE, 0.75)
    else:
        return (QueryType.FACTUAL_COMPLEX, 0.70)


# ============================================================================
# Option B: LLM 分类器（更准确，但多一次 API 调用）
# ============================================================================

CLASSIFIER_SYSTEM_PROMPT = """你是一个查询意图分类专家。请分析用户的查询，将其归入以下类型之一：

- FACTUAL_SIMPLE: 简单的事实查询，通常可以用一个关键词匹配找到答案
  例如："图书馆几点开门？""食堂在哪里？"

- FACTUAL_COMPLEX: 复杂的事实查询，需要综合多条信息才能回答
  例如："计算机系的教师科研方向有哪些？"

- COMPARATIVE: 对比类查询，需要比较两个或多个事物
  例如："理工学院和医学院的区别？"

- PROCEDURAL: 流程类查询，询问如何做某事
  例如："如何办理借书证？"

- OUT_OF_SCOPE: 与南方科技大学无关的查询
  例如："清华大学校长是谁？"

- TEMPORAL: 时间敏感的查询，需要最新的信息
  例如："最近的招生政策有什么变化？"

请仅回复一个JSON对象，不要加任何其他文字。
{"type": "查询类型", "confidence": 0.0-1.0之间的置信度}"""


def classify_by_llm(
    query: str,
    llm_fn: Callable[[str, str], str],
) -> tuple[str, float]:
    """
    使用 LLM 进行查询分类（Option B）。

    参数：
        query: 用户查询
        llm_fn: LLM 调用函数

    返回：
        (查询类型, 置信度)
    """
    user_prompt = f"请分类以下查询：{query}"

    try:
        response = llm_fn(CLASSIFIER_SYSTEM_PROMPT, user_prompt)
        # 尝试解析 JSON 响应
        # LLM 可能会在 JSON 前后加 markdown 代码块标记
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
            response = response.strip()

        result = json.loads(response)
        q_type = result.get("type", QueryType.FACTUAL_COMPLEX)
        confidence = float(result.get("confidence", 0.7))

        # 验证类型是否合法
        valid_types = set(STRATEGY_WEIGHTS.keys())
        if q_type not in valid_types:
            q_type = QueryType.FACTUAL_COMPLEX

        return (q_type, confidence)

    except Exception:
        # LLM 调用失败 → 降级为规则分类器
        return classify_by_rules(query)


# ============================================================================
# QueryClassifier 主类
# ============================================================================

class QueryClassifier:
    """
    查询分类器（支持规则模式和 LLM 模式）。

    使用方式：
        qc = QueryClassifier(mode="llm", llm_fn=my_llm_function)
        q_type, confidence = qc.classify("图书馆几点开门？")
        weights = qc.get_rrf_weights(q_type)
    """

    def __init__(
        self,
        mode: str = "rule",
        llm_fn: Callable[[str, str], str] = None,
        cache_enabled: bool = True,
    ):
        """
        初始化分类器。

        参数：
            mode: "rule" = 规则分类 / "llm" = LLM 分类
            llm_fn: LLM 调用函数（mode="llm" 时必需）
            cache_enabled: 是否启用分类结果缓存（相同查询不重复分类）
        """
        self.mode = mode
        self.llm_fn = llm_fn
        self.cache_enabled = cache_enabled
        self._cache: dict[str, tuple[str, float]] = {}
        self._max_cache_size = 500  # 防止无界增长

        if mode == "llm" and llm_fn is None:
            print("[QueryClassifier] Warning: LLM mode but no llm_fn provided, "
                  "falling back to rule-based")
            self.mode = "rule"

    def classify(self, query: str) -> tuple[str, float]:
        """
        对查询进行分类。

        参数：
            query: 用户查询文本

        返回：
            (查询类型, 置信度)
        """
        # 检查缓存
        if self.cache_enabled and query in self._cache:
            return self._cache[query]

        t_start = time.time()

        if self.mode == "llm":
            q_type, confidence = classify_by_llm(query, self.llm_fn)
        else:
            q_type, confidence = classify_by_rules(query)

        elapsed_ms = int((time.time() - t_start) * 1000)

        # 缓存结果（LRU：超过 max_size 时清空旧缓存）
        if self.cache_enabled:
            if len(self._cache) >= self._max_cache_size:
                self._cache.clear()
            self._cache[query] = (q_type, confidence)

        return (q_type, confidence)

    def get_rrf_weights(self, q_type: str) -> tuple[float, float]:
        """
        根据查询类型获取 RRF 融合权重。

        参数：
            q_type: 查询类型（来自 classify()）

        返回：
            (dense_weight, sparse_weight)
        """
        return STRATEGY_WEIGHTS.get(
            q_type,
            (1.0, 1.0),  # 默认：等权重融合
        )

    def is_out_of_scope(self, q_type: str) -> bool:
        """判断是否应该立即拒答。"""
        return q_type == QueryType.OUT_OF_SCOPE

    def should_use_hyde(self, q_type: str) -> bool:
        """
        判断是否应该启用 HyDE。

        对比类和复杂事实查询从 HyDE 中受益最大，
        因为它们的语义理解和文档表达差异更大。
        """
        return q_type in (QueryType.COMPARATIVE, QueryType.FACTUAL_COMPLEX)


# ============================================================================
# 单例
# ============================================================================

_classifier_instance: QueryClassifier | None = None


def get_classifier(mode: str = "rule", llm_fn: Callable = None) -> QueryClassifier:
    """获取全局唯一的 QueryClassifier 实例。"""
    global _classifier_instance
    if _classifier_instance is None or (
        mode == "llm" and _classifier_instance.mode != "llm"
    ):
        _classifier_instance = QueryClassifier(mode=mode, llm_fn=llm_fn)
    return _classifier_instance


# ============================================================================
# 测试
# ============================================================================
if __name__ == "__main__":
    qc = QueryClassifier(mode="rule")

    test_queries = [
        ("图书馆几点开门？", QueryType.FACTUAL_SIMPLE),
        ("计算机系有哪些教授？他们的研究方向是什么？", QueryType.FACTUAL_COMPLEX),
        ("理工学院和医学院有什么区别？", QueryType.COMPARATIVE),
        ("如何办理校园卡？", QueryType.PROCEDURAL),
        ("清华大学校长是谁？", QueryType.OUT_OF_SCOPE),
        ("2026年招生政策有什么变化？", QueryType.TEMPORAL),
    ]

    print("=" * 60)
    print("Query Classifier Test (Rule-based)")
    print("=" * 60)
    for query, expected in test_queries:
        q_type, conf = qc.classify(query)
        status = "✅" if q_type == expected else "❌"
        weights = qc.get_rrf_weights(q_type)
        print(f"{status} [{q_type:20s}] (conf={conf:.2f}) "
              f"weights:dense={weights[0]:.1f},sparse={weights[1]:.1f} "
              f"→ {query[:50]}")
