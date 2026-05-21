"""
=============================================================================
SUSTech LLM API Client — DeepSeek API 远程 LLM 调用
=============================================================================
这是主要的 LLM 调用入口。使用 DeepSeek 的 OpenAI 兼容 API。

为什么选择 DeepSeek？
  1. OpenAI 兼容接口 → 代码几乎不需要修改
  2. DeepSeek-V4 中文能力极强，校园场景问答质量高
  3. 性价比高（价格远低于 GPT-4，质量接近）
  4. API 端点：https://api.deepseek.com/v1

Fallback 策略：
  如果 DeepSeek API 不可用（网络问题 / 余额不足 / 超时），
  自动降级为本地 Ollama（generation/llm_local.py）。
  这样既保证了 demo 的可靠性，又控制了成本。

使用方法：
  from generation.llm_api import DeepSeekClient
  client = DeepSeekClient()
  answer = client.chat("你是一个助手", "图书馆几点开门？")

=============================================================================
"""

import json
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

# ============================================================================
# 配置导入
# ============================================================================
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    API_LLM,
    DEEPSEEK_BASE as API_BASE,
    RERANK_TOP_K,
)

# ============================================================================
# DeepSeek Client
# ============================================================================

class DeepSeekClient:
    """
    DeepSeek API 的 OpenAI 兼容客户端。

    支持：
    - chat(): 标准对话（system + user + context）
    - chat_json(): 要求 LLM 输出 JSON（用于 Query Classifier）
    - stream(): 流式输出（用于 Gradio demo 的实时显示）
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        base_url: str = None,
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        """
        初始化 DeepSeek 客户端。

        参数：
            api_key: API 密钥（默认从 DEEPSEEK_API_KEY 环境变量读取）
            model: 模型名称（默认 deepseek-chat）
            base_url: API 端点（默认 api.deepseek.com/v1）
            timeout: 单次请求超时时间（秒）
            max_retries: 失败后重试次数
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.model = model or API_LLM
        self.base_url = base_url or API_BASE
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_key:
            print("[DeepSeek] WARNING: No API key set. "
                  "Set DEEPSEEK_API_KEY env variable.")
            self._client = None
        else:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
            print(f"[DeepSeek] Ready. Model: {self.model}")

    @property
    def is_available(self) -> bool:
        """检查 API 是否可用。"""
        return self._client is not None

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        seed: int = 42,
    ) -> str:
        """
        发送对话请求。

        参数：
            system_prompt: 系统提示（角色设定、风格指令等）
            user_message: 用户消息（问题 + 检索到的上下文）
            temperature: 生成温度。0.3 = 偏确定（适合事实型问答），
                        0.7+ = 偏创意（适合人格定制模式）
            max_tokens: 最大输出 token 数
            seed: 随机种子（可复现性）

        返回：
            LLM 的回复文本
        """
        if not self.is_available:
            raise RuntimeError(
                "DeepSeek API not available. "
                "Set DEEPSEEK_API_KEY or check network."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                )
                return response.choices[0].message.content

            except Exception as e:
                if attempt < self.max_retries:
                    print(f"[DeepSeek] Retry {attempt+1}/{self.max_retries} "
                          f"after error: {e}")
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    raise RuntimeError(
                        f"DeepSeek API failed after {self.max_retries+1} attempts: {e}"
                    )

    def chat_json(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.1,
        max_tokens: int = 256,
    ) -> dict:
        """
        请求 LLM 返回 JSON 格式的输出。

        用于 Query Classifier（需要结构化输出）。

        参数：
            system_prompt: 系统提示
            user_message: 用户消息
            temperature: 极低温度（需要确定的 JSON 输出）
            max_tokens: JSON 不需要太长

        返回：
            解析后的 JSON dict
        """
        # 在 system prompt 末尾追加 JSON 输出指令
        json_instruction = "\n请只输出一个JSON对象，不要加任何其他文字、markdown代码块标记或注释。"
        raw = self.chat(
            system_prompt=system_prompt + json_instruction,
            user_message=user_message,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # 尝试解析 JSON
        # LLM 有时会在 JSON 外面包 markdown 代码块标记
        raw = raw.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            if len(parts) >= 2:
                raw = parts[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # 最后一次尝试：用正则提取 {...}
            import re
            match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {"error": "JSON parse failed", "raw": raw}

    def stream_chat(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ):
        """
        流式对话（Generator）。

        用于 Gradio demo 的实时逐字显示。

        参数：
            同 chat()

        Yields:
            每次 yield 一个 token（str）
        """
        if not self.is_available:
            yield "[API不可用] 请检查 DEEPSEEK_API_KEY 设置。"
            return

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        except Exception as e:
            yield f"\n\n[API错误: {e}]"


# ============================================================================
# 轻量级 LLM 函数包装器
# ============================================================================

def create_llm_fn(
    client: DeepSeekClient = None,
    model: str = "deepseek-chat",
) -> callable:
    """
    创建一个标准的 LLM 调用函数（签名兼容 retrieval/hyde.py 等模块）。

    这个包装器的作用是：让检索模块不需要关心 LLM 的具体实现细节。
    它们只需要一个 llm_fn(system_prompt, user_prompt) → str 的函数。

    参数：
        client: DeepSeekClient 实例（None = 自动创建）
        model: 使用的模型（HyDE 和 enrichment 用 7B 就够了）

    返回：
        callable(system_prompt, user_message) -> str
    """
    if client is None:
        client = DeepSeekClient(model=model)

    def llm_fn(system_prompt: str, user_message: str) -> str:
        return client.chat(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.3,
            max_tokens=256,  # HyDE 和 enrichment 的回复都很短
        )

    return llm_fn


# ============================================================================
# 单例
# ============================================================================

_client_instance: DeepSeekClient | None = None


def get_client(model: str = None) -> DeepSeekClient:
    """获取全局唯一的 DeepSeekClient 实例。"""
    global _client_instance
    if _client_instance is None:
        _client_instance = DeepSeekClient(model=model)
    return _client_instance


# ============================================================================
# 测试
# ============================================================================
if __name__ == "__main__":
    client = get_client()
    if client.is_available:
        response = client.chat(
            "你是一个南科大校园助手。",
            "图书馆几点开门？请用中文简短回答。",
        )
        print(f"Response: {response}")
    else:
        print("API not available. Set DEEPSEEK_API_KEY to test.")
