"""OpenAICompatLLM —— 可选后端：对接任意 OpenAI 兼容服务。

说明：框架默认使用自研小模型（LocalLLM），本后端仅作为可选项，
便于：1) 对比评测（小模型 vs 大模型在同一基准上的差距）；
      2) 演示 Web/RAG 时获得更高质量的医学回答。
可对接 DeepSeek 官方 API（base_url=https://api.deepseek.com）、
vLLM/Ollama 等任意 OpenAI 兼容端点。需要 pip install openai。
"""
from __future__ import annotations

import os
from typing import List, Optional

from .base import BaseLLM


class OpenAICompatLLM(BaseLLM):
    """OpenAI 兼容后端（懒加载 openai 库）。"""

    name = "openai-compat"

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 model: str = "deepseek-chat", max_new_tokens: int = 512,
                 temperature: float = 0.7):
        try:
            from openai import OpenAI  # 可选依赖，用到才导入
        except ImportError as e:
            raise ImportError("使用 OpenAI 兼容后端需要安装 openai 库：pip install openai") from e

        self.client = OpenAI(
            base_url=base_url or os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
            api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
        )
        self.model_name = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(
        self,
        messages: List[dict],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens or self.max_new_tokens,
            temperature=self.temperature if temperature is None else temperature,
            stream=False,
        )
        return resp.choices[0].message.content or ""
