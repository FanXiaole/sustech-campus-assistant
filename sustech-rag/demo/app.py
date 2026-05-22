"""
=============================================================================
SUSTech RAG Demo — Gradio 演示应用（v2 精简版）
=============================================================================
3 个标签页：校园问答 · 实验数据 · 管线追踪
特色：流式输出 · 来源权威徽章 · 检索置信度 · 实时延迟拆解
=============================================================================
"""

import json, os, random, time
from pathlib import Path
import gradio as gr

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    ABSTAIN_MESSAGE, DEFAULT_PERSONA, DEMO_MODE, DEMO_PORT, DEMO_SHARE,
    PERSONA_PRESETS, RESULTS_DIR,
)

DEMO_MODE = os.getenv("DEMO_MODE", DEMO_MODE)
MODE_LABEL = "CACHED" if DEMO_MODE == "cached" else "LIVE"

# ── 懒加载 ──
_components = {}

def get_component(name: str):
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
        elif name == "llm":
            from generation.llm_api import DeepSeekClient
            from generation.llm_local import OllamaClient, LLMFallback
            _components[name] = LLMFallback(DeepSeekClient(), OllamaClient())
        elif name == "builder":
            from generation.prompt_builder import PromptBuilder
            _components[name] = PromptBuilder(persona=DEFAULT_PERSONA)
        elif name == "authority":
            from retrieval.authority_scorer import get_authority_scorer
            _components[name] = get_authority_scorer()
    return _components[name]


# ── 来源权威徽章 ──
AUTH_BADGES = {
    "official":   "Official", "admission": "Admission", "library": "Library",
    "department": "Department", "manual": "Manual", "news": "News", "unknown": "?",
}


# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 1 — 校园问答（流式输出 + 来源卡片 + 置信度）           ║
# ╚══════════════════════════════════════════════════════════════╝

def random_question():
    from config import DATA_DIR
    path = DATA_DIR / "test_set_v2.json"
    if not path.exists():
        path = DATA_DIR / "test_set.json"
    if path.exists():
        with open(path) as f:
            qs = json.load(f)
        return random.choice([q["question"] for q in qs if not q.get("expected_abstain")])
    return "图书馆几点开门？"


def run_retrieval(question: str, mode: str, use_hyde: bool):
    """执行检索，返回 (chunks, timings_dict)。"""
    t0 = time.time()

    if mode == "no_rag":
        return [], {"retrieval": 0}

    if mode == "dense":
        dense = get_component("dense")
        chunks = dense.search(question, top_k=5)
        return chunks, {"retrieval": round((time.time() - t0) * 1000)}

    if mode == "bm25":
        sparse = get_component("sparse")
        chunks = sparse.search(question, top_k=5)
        return chunks, {"retrieval": round((time.time() - t0) * 1000)}

    # hybrid / full
    from retrieval.hybrid_rrf import hybrid_retrieve
    dense = get_component("dense")
    sparse = get_component("sparse")
    classifier = get_component("classifier")
    reranker = get_component("reranker") if mode == "full" else None
    authority = get_component("authority") if mode == "full" else None

    hyde_fn = None
    if use_hyde:
        llm = get_component("llm")
        if llm.is_available:
            def hyde_fn(s, u): return llm.chat(s, u, max_tokens=256)

    chunks, trace = hybrid_retrieve(
        query=question, dense_retriever=dense, sparse_retriever=sparse,
        use_hyde=use_hyde, hyde_llm_fn=hyde_fn,
        use_classifier=True, classifier=classifier,
        reranker=reranker, authority_scorer=authority, abstention_check=True,
    )

    steps = trace.get("steps", {})
    timings = {}
    for step in ["dense", "bm25", "rrf", "reranker", "authority"]:
        sd = steps.get(step, {})
        if isinstance(sd, dict) and "time_ms" in sd:
            timings[step] = sd["time_ms"]
    timings["total"] = trace.get("total_ms", 0)
    return chunks, timings


def stream_answer(question: str, mode: str, use_hyde: bool, persona: str):
    """流式生成回答 + 来源卡片。"""
    if not question.strip():
        yield "", "", "", ""
        return

    chunks, timings = run_retrieval(question, mode, use_hyde)

    # 拒答
    if not chunks:
        yield (f"⚠️ {ABSTAIN_MESSAGE}",
               "<div style='color:#ef4444'>检索置信度过低，已触发拒答保护</div>",
               _fmt_timings(timings),
               "classify→retrieve→abstain")
        return

    # 构建来源卡片
    sources_html = _build_source_cards(chunks)

    # 流式生成
    builder = get_component("builder")
    builder.set_persona(persona)
    system_prompt = builder.build_system_prompt()
    user_message = builder.build_user_message(question, chunks)

    llm = get_component("llm")
    if not llm.is_available:
        yield ("[LLM 不可用] 请设置 DEEPSEEK_API_KEY",
               sources_html, _fmt_timings(timings), "retrieve→generate(failed)")
        return

    trace_text = _build_trace_text(mode, use_hyde, timings, len(chunks))

    answer_parts = []
    for token in llm.stream_chat(system_prompt, user_message):
        answer_parts.append(token)
        full = "".join(answer_parts)
        if llm.used_backend == "local" and len(answer_parts) <= 3:
            full = "[注意: 使用本地 Ollama 模型]\n\n" + full
        yield full, sources_html, _fmt_timings(timings), trace_text

    # 最终输出
    yield "".join(answer_parts), sources_html, _fmt_timings(timings), trace_text


def _build_source_cards(chunks: list[dict]) -> str:
    """构建来源卡片 HTML。"""
    html = '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:4px">'
    for i, c in enumerate(chunks[:5], 1):
        src = c.get("source_family", "unknown")
        badge = AUTH_BADGES.get(src, "?")
        rrf = c.get("rrf_score", 0)
        rerank = c.get("rerank_score", 0)
        if rerank:
            score_str = f'<span style="color:#22c55e">{rerank:.3f}</span>'
        elif rrf:
            score_str = f'<span style="color:#94a3b8">{rrf:.4f}</span>'
        else:
            score_str = ""
        html += (
            f'<div style="background:#1e293b;border-radius:8px;padding:8px 12px;'
            f'min-width:170px;flex:1;border-left:3px solid #3b82f6">'
            f'<div style="font-size:11px;color:#94a3b8">{badge} #{i} {score_str}</div>'
            f'<div style="font-size:13px;margin-top:4px;line-height:1.4">'
            f'{c.get("raw_text", c.get("text", ""))[:120]}...</div>'
            f'</div>'
        )
    html += '</div>'
    return html


def _build_trace_text(mode: str, use_hyde: bool, timings: dict, n_chunks: int) -> str:
    """构建管线 trace 文本。"""
    parts = [f"mode={mode} hyde={use_hyde} chunks={n_chunks}"]
    for k, v in timings.items():
        parts.append(f"{k}:{v:.0f}ms" if v >= 1 else f"{k}:<1ms")
    return " | ".join(parts)


def _fmt_timings(timings: dict) -> str:
    """格式化时间信息为可读字符串。"""
    parts = []
    for k, v in timings.items():
        if isinstance(v, (int, float)):
            parts.append(f"{k}:{v:.0f}ms" if v >= 1 else f"{k}:<1ms")
    return " · ".join(parts)


def build_tab1():
    with gr.Column():
        gr.Markdown("## 校园知识库问答")

        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(
                    label="", placeholder="输入你的问题，例如：图书馆几点关门？计算机系有哪些教授？",
                    lines=2, show_label=False, elem_id="question-box")
            with gr.Column(scale=1):
                submit_btn = gr.Button("查询", variant="primary", size="lg")

        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Radio(
                    [("完整管线", "full"), ("Hybrid RRF", "hybrid"),
                     ("仅 Dense", "dense"), ("仅 BM25", "bm25"), ("无检索", "no_rag")],
                    value="full", label="检索模式", interactive=True)
            with gr.Column(scale=1):
                persona = gr.Radio(
                    [(cfg["name"], pid) for pid, cfg in PERSONA_PRESETS.items()],
                    value=DEFAULT_PERSONA, label="回答风格")
            with gr.Column(scale=1):
                use_hyde = gr.Checkbox(label="HyDE 查询扩展", value=False)
                surprise_btn = gr.Button("随机提问", size="sm")

        answer = gr.Textbox(label="回答", lines=12, interactive=False, elem_id="answer-box")
        sources = gr.HTML(label="检索来源")
        timing = gr.Textbox(label="延迟", interactive=False, elem_id="timing-box")
        trace_box = gr.Textbox(label="管线 Trace", interactive=False, visible=False,
                               lines=3, elem_id="trace-box")

        submit_btn.click(
            fn=stream_answer, inputs=[question, mode, use_hyde, persona],
            outputs=[answer, sources, timing, trace_box], queue=True)
        question.submit(
            fn=stream_answer, inputs=[question, mode, use_hyde, persona],
            outputs=[answer, sources, timing, trace_box], queue=True)
        surprise_btn.click(fn=random_question, inputs=[], outputs=[question])


# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 2 — 实验数据（矩阵总览 + Bootstrap + LLM Eval）        ║
# ╚══════════════════════════════════════════════════════════════╝

def load_experiment_data():
    result = {}
    for fn, key in [("comparison_table.json", "comparison"), ("bootstrap_ci.json", "bootstrap")]:
        path = RESULTS_DIR / fn
        if path.exists():
            with open(path) as f:
                result[key] = json.load(f)
    llm_path = RESULTS_DIR / "R4" / "llm_vs_rule_comparison.json"
    if llm_path.exists():
        with open(llm_path) as f:
            result["llm_eval"] = json.load(f)
    return result


def build_tab2():
    with gr.Column():
        gr.Markdown("## 实验数据")

        data = load_experiment_data()
        if not data:
            gr.Markdown("*暂无数据*")
            return

        comp = data.get("comparison", {})
        boot = data.get("bootstrap", {})
        llm_eval = data.get("llm_eval", {})

        with gr.Tabs():
            with gr.TabItem("实验矩阵"):
                if comp:
                    lines = ["| ID | 配置 | 总分 | 延迟 |", "|---|---|---|---|"]
                    names = {"R0":"No RAG","R1":"Dense","R2":"BM25","R3":"Hybrid RRF",
                             "R4":"+Reranker","E1":"+HyDE","E2":"+Enriched","E3":"+Classifier",
                             "E4":"Full Stack","E5":"+Authority","A1":"Small(300)","A2":"Large(900)"}
                    for eid, ed in sorted(comp.items()):
                        s = ed.get("total_score", 0)
                        l = ed.get("latency_avg_ms", 0)
                        name = names.get(eid, eid)
                        lines.append(f"| **{eid}** | {name} | {s:.2f}/10 | {l:.0f}ms |")
                    gr.Markdown("\n".join(lines))

            with gr.TabItem("Bootstrap 显著性"):
                if boot:
                    lines = ["| 对比 | Δ | 95% CI | p | 显著 |",
                             "|---|---|---|---|---|"]
                    for key, ci in sorted(boot.items()):
                        sig = "YES" if ci.get("significant") else "-"
                        lines.append(
                            f"| {ci.get('label', key)} | {ci['observed_diff']:+.3f} | "
                            f"[{ci['ci_95'][0]:.2f}, {ci['ci_95'][1]:.2f}] | "
                            f"{ci['p_value']:.3f} | {sig} |")
                    gr.Markdown("\n".join(lines))

            with gr.TabItem("LLM vs 规则评分"):
                if llm_eval:
                    gr.Markdown(f"""
                    | 指标 | 规则评分 | LLM 评分 | 差异 |
                    |---|---|---|---|
                    | 平均总分 | {llm_eval['avg_rule_score']:.2f} | {llm_eval['avg_llm_score']:.2f} | **+{llm_eval['avg_delta']:.2f}** |
                    | LLM 更高 | — | {llm_eval['llm_higher']}/50 | — |
                    | LLM 更低 | — | {llm_eval['llm_lower']}/50 | — |
                    | 相同 | — | {llm_eval['same']}/50 | — |

                    **结论**: LLM 语义评分系统性高于规则评分 {llm_eval['avg_delta']:.1f} 分，证实规则版因术语不匹配导致低估。
                    """)


# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 3 — 管线追踪                                          ║
# ╚══════════════════════════════════════════════════════════════╝

def inspect_pipeline(question: str):
    if not question.strip():
        return "请输入问题。", None

    from retrieval.hybrid_rrf import hybrid_retrieve
    dense = get_component("dense")
    sparse = get_component("sparse")

    chunks, trace = hybrid_retrieve(
        query=question, dense_retriever=dense, sparse_retriever=sparse,
        use_hyde=False, use_classifier=True, classifier=get_component("classifier"),
        reranker=get_component("reranker"), abstention_check=True,
    )

    steps = trace.get("steps", {})
    lines = [f"## Query: {trace.get('query', '')}",
             f"**Type**: {trace.get('query_type', '?')} | **Total**: {trace.get('total_ms', 0)}ms",
             ""]

    for name, data in steps.items():
        if isinstance(data, dict):
            ms = data.get("time_ms", "?")
            lines.append(f"### {name} ({ms}ms)")
            if "top_3" in data:
                for r in data["top_3"]:
                    lines.append(f"- `{r.get('id','?')}` score={r.get('score',r.get('rrf_score',r.get('rerank_score','?')))}")
            if "skipped" in data:
                lines.append("*(skipped)*")
            lines.append("")

    conf = trace.get("confidence", {})
    lines.append(f"**Confidence**: max_rrf={conf.get('max_rrf_score', 0):.4f}, "
                 f"threshold={conf.get('threshold', 0)}, "
                 f"abstain={'YES' if conf.get('should_abstain') else 'no'}")

    bar_data = None
    if chunks:
        bar_data = [
            {"Chunk": c.get("chunk_id", "")[:12],
             "RRF": round(c.get("rrf_score", 0), 4),
             "Rerank": round(c.get("rerank_score", 0), 4)}
            for c in chunks[:15]
        ]

    return "\n".join(lines), bar_data


def build_tab3():
    with gr.Column():
        gr.Markdown("## 管线追踪")
        with gr.Row():
            q_input = gr.Textbox(label="问题", placeholder="输入问题查看检索管线内部状态...",
                                scale=3)
            inspect_btn = gr.Button("追踪", variant="primary", scale=1)

        trace_md = gr.Markdown("等待输入...")
        bar = gr.BarPlot(
            x="Chunk", y="RRF", title="Chunk RRF Scores (Top 15)",
            height=280)

        inspect_btn.click(
            fn=inspect_pipeline, inputs=[q_input],
            outputs=[trace_md, bar])


# ╔══════════════════════════════════════════════════════════════╗
# ║  主入口                                                     ║
# ╚══════════════════════════════════════════════════════════════╝

CSS = """
#question-box textarea { font-size: 16px; }
#answer-box textarea { font-size: 14px; line-height: 1.6; }
#timing-box textarea { font-size: 12px; color: #94a3b8; }
#trace-box textarea { font-size: 11px; font-family: monospace; }
footer { visibility: hidden; }
"""


def build_demo():
    with gr.Blocks(title="SUSTech RAG") as demo:
        gr.Markdown(f"""
        # SUSTech Campus RAG
        **7 innovations · 11 experiments · Bootstrap CI · LLM evaluation**
        """)

        with gr.Tabs():
            with gr.TabItem("Q&A"):
                build_tab1()
            with gr.TabItem("Experiments"):
                build_tab2()
            with gr.TabItem("Pipeline"):
                build_tab3()

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(
        server_port=DEMO_PORT,
        server_name="0.0.0.0",
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        css=CSS,
    )
