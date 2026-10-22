"""LLM 后端抽象层。

框架的模型接入统一通过 BaseLLM.generate(messages) 完成：
- LocalLLM：默认后端，加载自研小模型（MedMiniLLM + 自训练分词器）；
- OpenAICompatLLM（可选）：对接任意 OpenAI 兼容服务
  （含 DeepSeek 官方 API、vLLM 等），用于对比评测或在演示时
  获得更高质量的医学回答；默认情况下不启用、不依赖 openai 库。

新增后端只需实现 generate() 一个方法。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional


class BaseLLM(ABC):
    """LLM 后端基类。"""

    name: str = "base"

    @abstractmethod
    def generate(
        self,
        messages: List[dict],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """给定 messages（DeepSeek 风格 [{role, content}, ...]），返回生成的文本。"""


def load_llm(kind: str = "local", **kwargs) -> BaseLLM:
    """后端工厂。

    Args:
        kind: "local"（自研模型）或 "openai"（OpenAI 兼容服务，可选）
        kwargs: 各后端所需的参数（如 model_dir、base_url、api_key、model）
    """
    if kind == "local":
        from .local import LocalLLM
        return LocalLLM(**kwargs)
    if kind == "openai":
        from .openai_compat import OpenAICompatLLM
        return OpenAICompatLLM(**kwargs)
    raise ValueError(f"未知 LLM 后端: {kind}")
