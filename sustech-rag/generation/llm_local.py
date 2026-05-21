"""
=============================================================================
SUSTech Local LLM — Ollama 本地推理（Fallback）
=============================================================================
这是 DeepSeek API 的本地备选方案。当 API 不可用时自动切换。

为什么用 Ollama？
  - 一键安装，自动管理模型下载和更新
  - 兼容 OpenAI API 格式（通过 Ollama 内置的 HTTP server）
  - Qwen2.5:7B 在 RTX 5090 上运行流畅（~16GB VRAM）
  - 离线可用（不需要网络）

Ollama 的安装和启动：
  # 安装 Ollama
  curl -fsSL https://ollama.com/install.sh | sh

  # 拉取模型
  ollama pull qwen2.5:7b

  # 启动服务（默认 http://localhost:11434）
  ollama serve

使用场景：
  - 开发阶段（本地快速测试，不需要消耗 API 额度）
  - Demo 备份（API 宕机时自动切换，保证演示可进行）
  - 隐私敏感（所有数据留在本地）

使用方法：
  from generation.llm_local import OllamaClient
  client = OllamaClient()
  answer = client.chat("你是一个助手", "图书馆几点开门？")

=============================================================================
"""

import time
from pathlib import Path

from openai import OpenAI

# ============================================================================
# 配置导入
# ============================================================================
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import LOCAL_LLM


class OllamaClient:
    """
    Ollama 本地 LLM 客户端。

    通过 Ollama 的内置 OpenAI 兼容 HTTP API 调用本地模型。
    端口默认 11434。
    """

    def __init__(
        self,
        model: str = None,
        base_url: str = "http://localhost:11434/v1",
        timeout: float = 120.0,
    ):
        """
        初始化 Ollama 客户端。

        参数：
            model: Ollama 模型名称（如 "qwen2.5:7b"）
            base_url: Ollama 的 OpenAI 兼容 endpoint
            timeout: 超时时间（本地模型可能比 API 慢）
        """
        self.model = model or LOCAL_LLM
        self.base_url = base_url

        try:
            self._client = OpenAI(
                api_key="ollama",  # Ollama 不验证 API key，但 OpenAI 库要求非空
                base_url=self.base_url,
                timeout=timeout,
            )
            # 测试连接
            self._client.models.list()
            self._available = True
            print(f"[Ollama] Connected. Model: {self.model}")
        except Exception as e:
            print(f"[Ollama] NOT available: {e}")
            print(f"[Ollama] Make sure Ollama is running: ollama serve")
            print(f"[Ollama] And model is pulled: ollama pull {self.model}")
            self._client = None
            self._available = False

    @property
    def is_available(self) -> bool:
        """检查 Ollama 是否可用。"""
        return self._available

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """
        发送对话请求到本地 Ollama。

        参数：
            同 DeepSeekClient.chat()
        """
        if not self.is_available:
            raise RuntimeError("Ollama is not available.")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def stream_chat(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ):
        """
        流式对话。

        注意：Ollama 的流式输出可能比 API 慢，
        因为本地模型的推理速度受限于 GPU 算力。
        """
        if not self.is_available:
            yield "[Ollama不可用] 请确保 Ollama 服务已启动。"
            return

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

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


# ============================================================================
# Fallback 逻辑：API → Local
# ============================================================================

class LLMFallback:
    """
    具有自动 fallback 的 LLM 客户端。

    优先使用 API（质量更高、速度更快），
    API 不可用时自动切换到本地 Ollama。

    使用方式：
        llm = LLMFallback(api_client, local_client)
        answer = llm.chat(system_prompt, user_message)
        # 自动选择可用的后端
    """

    def __init__(self, api_client, local_client: OllamaClient = None):
        self.api_client = api_client
        self.local_client = local_client or OllamaClient()
        self.used_backend = None  # 记录上次使用的后端（供 debug）

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        **kwargs,
    ) -> str:
        """尝试 API → 失败则用本地。"""
        # 尝试 API
        if self.api_client and self.api_client.is_available:
            try:
                result = self.api_client.chat(system_prompt, user_message, **kwargs)
                self.used_backend = "api"
                return result
            except Exception as e:
                print(f"[LLMFallback] API failed: {e}. Falling back to Ollama.")

        # Fallback 到本地
        if self.local_client and self.local_client.is_available:
            result = self.local_client.chat(system_prompt, user_message, **kwargs)
            self.used_backend = "local"
            return result

        raise RuntimeError("Both API and local LLM are unavailable.")

    def stream_chat(self, system_prompt: str, user_message: str, **kwargs):
        """流式输出的 fallback。"""
        if self.api_client and self.api_client.is_available:
            try:
                self.used_backend = "api"
                yield from self.api_client.stream_chat(system_prompt, user_message, **kwargs)
                return
            except Exception as e:
                print(f"[LLMFallback] API stream failed: {e}")

        if self.local_client and self.local_client.is_available:
            self.used_backend = "local"
            yield from self.local_client.stream_chat(system_prompt, user_message, **kwargs)
            return

        yield "[错误] 所有 LLM 后端均不可用。"


# ============================================================================
# 测试
# ============================================================================
if __name__ == "__main__":
    client = OllamaClient()
    if client.is_available:
        response = client.chat(
            "你是一个南科大校园助手。",
            "图书馆几点开门？请用中文简短回答。",
        )
        print(f"Response: {response}")
    else:
        print("Ollama not available. Start it with: ollama serve")
