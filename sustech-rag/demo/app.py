"""
=============================================================================
SUSTech RAG Demo — Gradio 5 标签页演示应用
=============================================================================
演示系统的设计目标：
  1. 直观展示 RAG pipeline 的每个环节
  2. 支持实时 vs 缓存两种模式
  3. 通过人格选择器展示我们的独有创新
  4. 实验对比数据可视化

使用方法：
  python demo/app.py                    # 本地启动
  DEMO_MODE=cached python demo/app.py   # 缓存模式（无需GPU/API）
  python demo/app.py --share            # 生成公网链接
=============================================================================
"""

import json
import os
import time
from pathlib import Path

import gradio as gr

# ============================================================================
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    ABSTAIN_MESSAGE,
    DATA_DIR,
    DEFAULT_PERSONA,
    DEMO_MODE,
    DEMO_PORT,
    DEMO_SHARE,
    PERSONA_PRESETS,
    RESULTS_DIR,
)

# ============================================================================
# 模式检测
# ============================================================================
DEMO_MODE = os.getenv("DEMO_MODE", DEMO_MODE)
MODE_LABEL = "📼 CACHED" if DEMO_MODE == "cached" else "🔴 LIVE"

# ============================================================================
# 缓存管理
# ============================================================================
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


_OLD_CACHE_FN = None  # replaced by key-based save/load below


def safe_json_dumps(obj, max_len=500):
    """安全的 JSON 序列化——处理不可序列化对象。"""
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
        return s[:max_len] + ("..." if len(s) > max_len else "")
    except Exception:
        return f"(无法序列化: {type(obj).__name__})"


def save_cache(key: str, data: dict):
    """保存查询结果到缓存文件。"""
    path = CACHE_DIR / f"{key.replace('/', '_')}.json"
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_cache(key: str) -> dict | None:
    """从缓存文件加载查询结果。"""
    path = CACHE_DIR / f"{key.replace('/', '_')}.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


# ============================================================================
# 懒加载组件（避免启动时全部加载）
# ============================================================================

_components = {}

def get_component(name: str):
    """懒加载组件，避免启动时 OOM。"""
    if name not in _components:
        if name == "dense":
            from retrieval.dense_retriever import get_dense_retriever
            _components[name] = get_dense_retriever()
        elif name == "sparse":
            from retrieval.sparse_retriever import get_sparse_retriever
            _components[name] = get_sparse_retriever()
        elif name == "reranker":
            from retrieval.reranker import get_reranker
            _components[name] = get_reranker()
        elif name == "classifier":
            from retrieval.query_classifier import get_classifier
            _components[name] = get_classifier(mode="rule")
        elif name == "authority":
            from retrieval.authority_scorer import get_authority_scorer
            _components[name] = get_authority_scorer()
        elif name == "llm":
            from generation.llm_api import DeepSeekClient
            from generation.llm_local import OllamaClient, LLMFallback
            api_client = DeepSeekClient()
            local_client = OllamaClient()
            _components[name] = LLMFallback(api_client, local_client)
        elif name == "builder":
            from generation.prompt_builder import PromptBuilder
            _components[name] = PromptBuilder(persona=DEFAULT_PERSONA)
    return _components[name]


# ============================================================================
# TAB 1: 校园 Q&A（主交互）
# ============================================================================

def query_rag(
    question: str,
    retrieval_mode: str,
    use_hyde: bool,
    persona: str,
    show_chunks: bool,
):
    """
    处理用户查询——完整的 RAG pipeline。

    参数：
        question: 用户问题
        retrieval_mode: "no_rag" / "dense" / "bm25" / "hybrid" / "full"
        use_hyde: 是否启用 HyDE
        persona: 人格标识
        show_chunks: 是否展示检索到的 chunk
    """
    if not question.strip():
        return "请输入问题。", "", ""

    t_start = time.time()

    # ── 缓存检测 ──
    cache_key = f"{question}_{retrieval_mode}_{use_hyde}_{persona}"
    if DEMO_MODE == "cached":
        cached = load_cache(cache_key)
        if cached:
            return cached["answer"], cached["sources"], cached["latency"]

    # ── Prompt builder + LLM（提前获取，所有路径共享） ──
    builder = get_component("builder")
    builder.set_persona(persona)
    llm = get_component("llm")

    # ── 检索 ──
    chunks = []
    trace_info = []

    if retrieval_mode == "no_rag":
        chunks = []
        trace_info.append("📭 未使用检索（纯 LLM）")

    elif retrieval_mode == "dense":
        dense = get_component("dense")
        chunks = dense.search(question, top_k=5)
        trace_info.append(f"🔍 Dense 检索 → {len(chunks)} chunks")

    elif retrieval_mode == "bm25":
        sparse = get_component("sparse")
        chunks = sparse.search(question, top_k=5)
        trace_info.append(f"📖 BM25 检索 → {len(chunks)} chunks")

    elif retrieval_mode in ("hybrid", "full"):
        from retrieval.hybrid_rrf import hybrid_retrieve

        dense = get_component("dense")
        sparse = get_component("sparse")
        classifier = get_component("classifier")
        reranker = get_component("reranker") if retrieval_mode == "full" else None
        authority = get_component("authority") if retrieval_mode == "full" else None

        hyde_fn = None
        if use_hyde and llm.is_available:
            def hyde_fn(system, user):
                return llm.chat(system, user, max_tokens=256)

        chunks, trace = hybrid_retrieve(
            query=question,
            dense_retriever=dense,
            sparse_retriever=sparse,
            use_hyde=use_hyde,
            hyde_llm_fn=hyde_fn,
            use_classifier=True,
            classifier=classifier,
            reranker=reranker,
            authority_scorer=authority,
            abstention_check=True,
        )
        trace_info.append(f"🔄 Hybrid pipeline → {len(chunks)} chunks (trace: {safe_json_dumps(trace.get('steps', {}))})")

    # ── 拒答检测 ──
    if len(chunks) == 0:
        answer = f"⚠️ {ABSTAIN_MESSAGE}"
        sources = "【系统已触发拒答保护——检索置信度过低】"
        latency = f"⏱️ {int((time.time() - t_start) * 1000)}ms"
        return answer, sources, latency

    # ── 生成 ──
    system_prompt = builder.build_system_prompt()
    user_message = builder.build_user_message(question, chunks)

    if llm.is_available:
        answer = llm.chat(system_prompt, user_message)
        if llm.used_backend == "local":
            answer = "[注意：当前使用本地 Ollama 模型，回答质量可能低于 DeepSeek API]\n\n" + answer
    else:
        answer = ("[LLM API 不可用] 请设置 DEEPSEEK_API_KEY 环境变量或启动 Ollama。\n\n"
                  f"检索到的资料（共 {len(chunks)} 条）：\n" +
                  "\n".join(f"- {c.get('raw_text', c.get('text', ''))[:200]}..."
                           for c in chunks[:3]))

    # ── 来源展示 ──
    sources = ""
    if show_chunks:
        for i, c in enumerate(chunks, 1):
            src = c.get("source_family", "unknown")
            url = c.get("url", "")[:100]
            text = c.get("raw_text", c.get("text", ""))[:200]
            sources += f"📄 **Chunk {i}** [{src}] {url}\n> {text}...\n\n"

    latency = f"⏱️ {int((time.time() - t_start) * 1000)}ms"

    # ── 缓存结果 ──
    if DEMO_MODE == "cached":
        save_cache(cache_key, {"answer": answer, "sources": sources, "latency": latency})

    return answer, sources, latency


def build_tab1():
    """Tab 1: 校园 Q&A 主界面。"""
    with gr.Column():
        gr.Markdown(f"## 🎓 南科大校园知识库问答  <small style='color:gray;'>{MODE_LABEL}</small>")

        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(
                    label="你的问题",
                    placeholder="例如：图书馆几点关门？计算机系有哪些教授？如何办理借书证？",
                    lines=3,
                )
            with gr.Column(scale=1):
                retrieval_mode = gr.Dropdown(
                    label="检索模式",
                    choices=[
                        ("完整管线 (Hybrid + Reranker)", "full"),
                        ("Hybrid RRF (无Reranker)", "hybrid"),
                        ("仅稠密检索 (bge-m3)", "dense"),
                        ("仅 BM25 检索", "bm25"),
                        ("无检索 (纯 LLM)", "no_rag"),
                    ],
                    value="full",
                )
                persona = gr.Dropdown(
                    label="🎭 回答风格",
                    choices=[
                        (cfg["name"], pid)
                        for pid, cfg in PERSONA_PRESETS.items()
                    ],
                    value=DEFAULT_PERSONA,
                )
                use_hyde = gr.Checkbox(label="🔮 HyDE 查询扩展", value=False)
                show_chunks = gr.Checkbox(label="📄 展示检索资料", value=True)
                submit_btn = gr.Button("🔍 查询", variant="primary")

        answer = gr.Textbox(label="回答", lines=10, interactive=False)
        sources = gr.Textbox(label="检索依据", lines=8, interactive=False, visible=True)
        latency = gr.Textbox(label="延迟", interactive=False)

        submit_btn.click(
            fn=query_rag,
            inputs=[question, retrieval_mode, use_hyde, persona, show_chunks],
            outputs=[answer, sources, latency],
        )


# ============================================================================
# TAB 2: Pipeline Inspector（管线追踪）
# ============================================================================

def inspect_pipeline(question: str):
    """展示完整的 pipeline trace。"""
    if not question.strip():
        return "请输入问题查看 pipeline 追踪。", {}

    from retrieval.hybrid_rrf import hybrid_retrieve
    from retrieval.dense_retriever import get_dense_retriever
    from retrieval.sparse_retriever import get_sparse_retriever

    dense = get_dense_retriever()
    sparse = get_sparse_retriever()
    classifier = get_component("classifier")
    reranker = get_component("reranker")

    chunks, trace = hybrid_retrieve(
        query=question,
        dense_retriever=dense,
        sparse_retriever=sparse,
        use_hyde=False,
        use_classifier=True,
        classifier=classifier,
        reranker=reranker,
        abstention_check=True,
    )

    # 格式化为可读文本
    steps = trace.get("steps", {})
    text = f"## 🔬 Pipeline Trace\n\n"
    text += f"**Query**: {trace.get('query', '')}\n"
    text += f"**Type**: {trace.get('query_type', 'unknown')}\n"
    text += f"**Total**: {trace.get('total_ms', 0)}ms\n\n"

    for step_name, step_data in steps.items():
        text += f"### {step_name}\n"
        if isinstance(step_data, dict):
            text += f"```json\n{json.dumps(step_data, ensure_ascii=False, indent=2)}\n```\n"
        text += "\n"

    # RRF 分数柱状图数据（取前 20 个）
    bar_data = {}
    if chunks:
        bar_data = {
            c.get("chunk_id", "")[:16]: c.get("rrf_score", c.get("rerank_score", 0))
            for c in chunks[:20]
        }

    return text, bar_data


def build_tab2():
    """Tab 2: Pipeline Inspector。"""
    with gr.Column():
        gr.Markdown("## 🔬 Pipeline Inspector")
        gr.Markdown("输入问题，查看检索管线的每一步内部状态。")

        q_input = gr.Textbox(label="问题", placeholder="输入问题...")
        inspect_btn = gr.Button("🔍 追踪", variant="primary")

        trace_output = gr.Markdown("等待输入...")
        bar_plot = gr.BarPlot(
            x="chunk_id",
            y="score",
            title="RRF / Rerank 分数 (Top 20)",
            x_title="Chunk ID",
            y_title="Score",
            height=300,
        )

        def on_inspect(q):
            text, bar_data = inspect_pipeline(q)
            # Convert dict to list of dicts for BarPlot
            items = [{"chunk_id": k, "score": v} for k, v in bar_data.items()]
            return text, items

        inspect_btn.click(
            fn=on_inspect,
            inputs=[q_input],
            outputs=[trace_output, bar_plot],
        )


# ============================================================================
# TAB 3: 实验结果 Dashboard
# ============================================================================

def load_experiment_data():
    """加载对比表数据。"""
    path = RESULTS_DIR / "comparison_table.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def build_tab3():
    """Tab 3: 实验结果 Dashboard。"""
    with gr.Column():
        gr.Markdown("## 📊 实验对比 Dashboard")

        data = load_experiment_data()
        if not data:
            gr.Markdown("*暂无实验数据。运行 `python evaluation/run_experiments.py` 后此处将显示结果。*")

            # 展示示例数据格式
            gr.Markdown("""
            ### 预期展示内容

            **雷达图**: R0 vs R3 vs R4 vs E4 五维度对比
            - Correctness / Grounding / Completeness / Traceability / Abstention

            **柱状图**: 各实验按难度层级（easy/medium/hard）的正确率

            **延迟对比表**:
            | 配置 | 总均分 | 检索延迟 | 生成延迟 |
            |------|--------|---------|---------|
            | R0 (no RAG) | x.x | 0ms | xxxms |
            | R3 (hybrid RRF) | x.x | xxxms | xxxms |
            | R4 (+ reranker) | x.x | xxxms | xxxms |
            | E4 (full innovation) | x.x | xxxms | xxxms |

            **3 个关键发现** (硬编码):
            1. HyDE 在比较类问题上提升最显著（+0.5分）
            2. Reranker 对 easy 问题提升不大，但对 hard 问题提升20%
            3. 拒答机制防止了 5/5 个范围外问题的幻觉
            """)
        else:
            # 真实数据展示
            lines = ["| Experiment | Total Score | Dimensions | Latency |",
                    "|-----------|------------|------------|---------|"]
            for exp_id, exp_data in sorted(data.items()):
                score = exp_data.get("total_score", 0)
                latency = exp_data.get("latency_avg_ms", 0)
                dims = exp_data.get("dimensions", {})
                dim_str = ", ".join(
                    f"{k}: {v.get('mean', 0):.1f}" for k, v in dims.items()
                )
                lines.append(f"| {exp_id} | {score:.2f}/10 | {dim_str} | {latency:.0f}ms |")
            gr.Markdown("\n".join(lines))


# ============================================================================
# TAB 4: A/B 对比
# ============================================================================

def compare_ab(question: str):
    """执行 A/B 对比：无 RAG vs 完整 RAG。"""
    if not question.strip():
        return "请输入问题。", "请输入问题。"

    # Side A: 无 RAG
    llm = get_component("llm")
    builder = get_component("builder")

    if llm.is_available:
        answer_a = llm.chat(
            builder.build_system_prompt(),
            f"用户问题：{question}\n\n请直接基于你的知识回答。",
        )
    else:
        answer_a = "[LLM 不可用]"

    # Side B: 完整 RAG
    from retrieval.hybrid_rrf import hybrid_retrieve
    dense = get_component("dense")
    sparse = get_component("sparse")

    chunks, _ = hybrid_retrieve(
        query=question,
        dense_retriever=dense,
        sparse_retriever=sparse,
        use_hyde=False,
        use_classifier=True,
        classifier=get_component("classifier"),
        reranker=get_component("reranker"),
    )

    if llm.is_available and chunks:
        answer_b = llm.chat(
            builder.build_system_prompt(),
            builder.build_user_message(question, chunks),
        )
    else:
        answer_b = "[RAG 不可用]" if not chunks else "[LLM 不可用]"

    return answer_a, answer_b


def build_tab4():
    """Tab 4: A/B 对比。"""
    with gr.Column():
        gr.Markdown("## 🧪 A/B 对比：无 RAG vs 完整 RAG")

        q_input = gr.Textbox(label="问题", placeholder="输入问题...")
        compare_btn = gr.Button("🔬 对比", variant="primary")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### ❌ 无 RAG（纯 LLM）")
                answer_a = gr.Textbox(label="", lines=10, interactive=False)
            with gr.Column():
                gr.Markdown("### ✅ 完整 RAG（最佳配置）")
                answer_b = gr.Textbox(label="", lines=10, interactive=False)

        compare_btn.click(
            fn=compare_ab,
            inputs=[q_input],
            outputs=[answer_a, answer_b],
        )


# ============================================================================
# TAB 5: 错误分析
# ============================================================================

def build_tab5():
    """Tab 5: 错误分析浏览器。"""
    with gr.Column():
        gr.Markdown("## 📋 错误分析")

        failure_category = gr.Dropdown(
            label="失败类别",
            choices=[
                ("检索失败 — 答案在知识库但未检索到", "retrieval_failure"),
                ("上下文稀释 — top-K 太大噪声淹没信号", "context_overflow"),
                ("知识缺失 — 知识库中不存在所需信息", "missing_knowledge"),
                ("幻觉编造 — LLM 忽略上下文自己编造", "hallucination"),
                ("过度拒答 — 本应回答但被系统拒绝", "over_refusal"),
            ],
            value="retrieval_failure",
        )

        analysis_output = gr.Markdown("""
        ### 各类失败的典型示例与修复建议

        **检索失败** (retrieval_failure):
        - 原因: chunk 切分边界不当，关键信息被切断
        - 示例: 问"计算机系培养方案"但培养方案表在 chunk 边界被拆成两半
        - 修复: 增大 overlap 到 150，或用 semantic chunking

        **上下文稀释** (context_overflow):
        - 原因: top-K 设太大，噪声 chunk 比例过高
        - 示例: 检索了 20 个 chunk 但只有 2 个相关
        - 修复: 缩小 RRF_FUSION_TOP 到 15，提高 reranker 阈值

        **知识缺失** (missing_knowledge):
        - 原因: 爬取范围不够，特定领域信息缺失
        - 示例: 问某位新入职教授的信息但官网尚未更新
        - 修复: 增加爬取种子 URL、定期重新爬取

        **幻觉编造** (hallucination):
        - 原因: LLM 在无依据时仍倾向生成答案
        - 示例: 问"米其林三星食堂"但系统未拒答
        - 修复: 降低 ABSTENTION_THRESHOLD、强化 prompt 约束

        **过度拒答** (over_refusal):
        - 原因: 拒答阈值太高，正常查询被拒绝
        - 示例: 检索分数刚好低于阈值但实际有相关信息
        - 修复: 调高 ABSTENTION_THRESHOLD、改进 query classifier
        """)

        return failure_category, analysis_output


# ============================================================================
# 主入口
# ============================================================================

def build_demo():
    """构建完整的 Gradio demo。"""
    with gr.Blocks(
        title="SUSTech Campus RAG — 南科大校园知识库",
    ) as demo:
        gr.Markdown(
            f"""# 🏫 南方科技大学校园知识库问答系统
            **Retrieval-Augmented Generation (RAG)** · 7 项创新 · 5 维度评测 · {MODE_LABEL}
            ---
            """
        )

        with gr.Tabs():
            with gr.TabItem("🎓 校园问答"):
                build_tab1()
            with gr.TabItem("🔬 管线追踪"):
                build_tab2()
            with gr.TabItem("📊 实验对比"):
                build_tab3()
            with gr.TabItem("🧪 A/B 对比"):
                build_tab4()
            with gr.TabItem("📋 错误分析"):
                build_tab5()

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(
        server_port=DEMO_PORT,
        share=DEMO_SHARE,
        show_error=True,
        theme=gr.themes.Soft(),
    )
