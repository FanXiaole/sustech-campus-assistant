"""
=============================================================================
SUSTech Prompt Builder — 上下文拼接、Prompt 模板、★ 人格定制
=============================================================================
这个模块负责构建发送给 LLM 的完整 prompt。

核心职责：
  1. 从检索结果中提取文本，拼接成 LLM 的上下文窗口
  2. 根据选定的 persona（人格），生成对应的 system prompt
  3. 构建最终的用户消息（上下文 + 用户问题）
  4. 管理 token 预算（不超出 LLM 的上下文窗口限制）

★ 人格定制系统（我们 vs 队友的核心差异化功能）：

设计哲学：事实层和人格层分离
  ┌──────────────────────────────────────────┐
  │  System Prompt                            │
  │  ┌──────────────────────────────────────┐ │
  │  │ 1. 角色设定（固定）                  │ │
  │  │    "你是南科大校园知识助手..."       │ │
  │  ├──────────────────────────────────────┤ │
  │  │ 2. 行为约束（固定）                  │ │
  │  │    "只基于提供的资料回答..."         │ │
  │  ├──────────────────────────────────────┤ │
  │  │ 3. ★ 风格指令（可切换，来自 Persona） │ │
  │  │    "请用极度浮夸的语气回答..."        │ │
  │  └──────────────────────────────────────┘ │
  └──────────────────────────────────────────┘

  人格只影响第 3 层（表达方式），不影响第 1、2 层（事实准确性的硬约束）。
  这样即使选择"疯狂模式"，LLM 也不会编造事实——只会用更浮夸的措辞
  来表达相同的事实。

使用方法：
  from generation.prompt_builder import PromptBuilder
  builder = PromptBuilder(persona="unhinged")
  system_prompt = builder.build_system_prompt()
  user_message = builder.build_user_message(query, chunks)

=============================================================================
"""

from pathlib import Path

# ============================================================================
# 配置导入
# ============================================================================
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DEFAULT_PERSONA,
    PERSONA_PRESETS,
    RERANK_TOP_K,
)


class PromptBuilder:
    """
    Prompt 构建器。

    负责把检索结果 + 用户查询 + 人格设定组装成 LLM 可理解的 prompt。
    """

    # ── 固定的角色设定（所有 persona 共享） ──
    BASE_SYSTEM_PROMPT = """你是南方科技大学（SUSTech）的校园知识助手。你的知识来源于学校官方网站、院系介绍、图书馆信息、招生资料和校园手册。

重要规则：
1. 只基于我提供的"参考资料"来回答问题。如果资料中没有相关信息，请明确说"根据现有资料无法确定"。
2. 不要编造任何信息。校园信息的准确性非常重要——错误的开放时间、错误的申请截止日期会给师生带来实际困扰。
3. 如果参考资料中的信息前后矛盾，请指出矛盾所在。
4. 在回答中尽量提及信息来源（如"根据图书馆网站..."）。
5. 用中文回答。"""

    # ── 用户消息模板 ──
    USER_MESSAGE_TEMPLATE = """参考资料：
{context}

用户问题：{query}

请基于以上参考资料回答问题。"""

    def __init__(self, persona: str = None):
        """
        初始化 Prompt 构建器。

        参数：
            persona: 人格标识（"default", "unhinged", "sexy"）
        """
        self.persona = persona or DEFAULT_PERSONA

    @property
    def persona_config(self) -> dict:
        """获取当前人格的配置。"""
        return PERSONA_PRESETS.get(
            self.persona,
            PERSONA_PRESETS[DEFAULT_PERSONA],
        )

    def set_persona(self, persona: str):
        """
        切换人格。

        参数：
            persona: 人格标识
        """
        if persona not in PERSONA_PRESETS:
            print(f"[PromptBuilder] Unknown persona '{persona}', "
                  f"falling back to '{DEFAULT_PERSONA}'")
            persona = DEFAULT_PERSONA
        self.persona = persona

    def list_personas(self) -> list[dict]:
        """
        列出所有可用的人格及其描述。

        返回：
            [{"id": "default", "name": "标准模式",
              "description": "..."}, ...]
        """
        return [
            {
                "id": pid,
                "name": cfg["name"],
                "description": cfg["description"],
            }
            for pid, cfg in PERSONA_PRESETS.items()
        ]

    def build_system_prompt(self) -> str:
        """
        构建完整的 system prompt。

        组装逻辑：
        1. 基础角色设定（固定）
        2. 行为约束（固定）
        3. 风格指令（来自当前选择的 persona）

        返回：
            完整的 system prompt 字符串
        """
        style = self.persona_config["style_instruction"]

        # 拼接
        full_prompt = (
            f"{self.BASE_SYSTEM_PROMPT}\n\n"
            f"【回答风格】{style}"
        )

        return full_prompt

    def build_context(self, chunks: list[dict]) -> str:
        """
        将检索到的 chunk 拼接成 LLM 的上下文。

        拼接格式（每个 chunk）：
        [来源:{source_family}] [URL:{url}]
        {chunk_text}

        chunk 之间用分隔线隔开，方便 LLM 区分不同来源。

        参数：
            chunks: 检索/重排序后的 chunk 列表

        返回：
            拼接好的上下文字符串
        """
        if not chunks:
            return "（未找到相关校园资料）"

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source_family", "unknown")
            url = chunk.get("url", "")
            # 优先用 raw_text（干净的文本），避免元数据前缀干扰阅读
            text = chunk.get("raw_text", chunk.get("text", ""))

            # 每个 chunk 的格式
            part = (
                f"--- 参考资料 {i} ---\n"
                f"[来源:{source}] [URL:{url}]\n"
                f"{text}"
            )
            context_parts.append(part)

        return "\n\n".join(context_parts)

    def build_user_message(
        self,
        query: str,
        chunks: list[dict],
        max_chunks: int = RERANK_TOP_K,
    ) -> str:
        """
        构建完整的用户消息（上下文 + 问题）。

        参数：
            query: 用户原始查询
            chunks: 检索到的 chunk 列表
            max_chunks: 最多使用的 chunk 数量（超出会被截断）

        返回：
            完整的用户消息字符串
        """
        # 限制 chunk 数量，防止超出 LLM 上下文窗口
        limited_chunks = chunks[:max_chunks]

        context = self.build_context(limited_chunks)

        message = self.USER_MESSAGE_TEMPLATE.format(
            context=context,
            query=query,
        )

        return message

    def estimate_tokens(self, text: str) -> int:
        """
        粗略估计文本的 token 数量。

        中英文混合场景下的经验估计：
        - 中文：1 个字符 ≈ 1-1.5 个 token
        - 英文：1 个单词 ≈ 1.3 个 token
        - 取保守估计：1 个字符 ≈ 1.5 个 token

        这只是粗略估计！精确计数需要用 tokenizer（如 tiktoken）。
        但我们不需要精确计数——只需确保不超过 8K tokens 的硬限制。

        参数：
            text: 待估计的文本

        返回：
            估计的 token 数
        """
        # 保守估计（实际 token 数通常少于此）
        chinese_chars = sum(1 for c in text if '一' <= c <= '鿿')
        other_chars = len(text) - chinese_chars

        # 中文字符 ≈ 1.5 token，其他 ≈ 0.3 token（按单词算）
        estimated = int(chinese_chars * 1.5 + other_chars * 0.3)
        return estimated

    def is_within_token_budget(
        self,
        system_prompt: str,
        user_message: str,
        budget: int = 7000,
    ) -> bool:
        """
        检查 prompt 是否在 token 预算内。

        7B 模型通常有 8K tokens 的上下文窗口。
        我们保留 1K 给输出 token，所以输入预算为 7K。

        参数：
            system_prompt: system prompt
            user_message: 用户消息（含上下文）
            budget: token 预算

        返回：
            True 表示在预算内
        """
        total_estimated = self.estimate_tokens(system_prompt) + \
                          self.estimate_tokens(user_message)
        return total_estimated <= budget


# ============================================================================
# 人格效果演示
# ============================================================================

def demo_personas():
    """
    展示同一问题在不同人格下的回答风格差异。
    这只是 prompt 层面的演示，不涉及实际 LLM 调用。
    """
    query = "图书馆几点关门？"
    chunks = [{
        "source_family": "library",
        "url": "https://lib.sustech.edu.cn",
        "raw_text": "图书馆服务时间：周一至周五 8:00-22:00，周末 9:00-21:00。",
    }]

    print("=" * 60)
    print("★ Persona Customization Demo")
    print("=" * 60)
    print(f"Query: {query}")
    print(f"Chunks: 1 chunk from library")
    print()

    for pid, cfg in PERSONA_PRESETS.items():
        builder = PromptBuilder(persona=pid)
        system = builder.build_system_prompt()
        user = builder.build_user_message(query, chunks)

        print(f"--- {cfg['name']} ({pid}) ---")
        print(f"Style instruction: {cfg['style_instruction'][:80]}...")
        print()

    print("The actual answer will vary based on LLM generation.")
    print("But the system prompt ensures the persona expression")
    print("while keeping facts grounded in the retrieved chunks.")


if __name__ == "__main__":
    demo_personas()
