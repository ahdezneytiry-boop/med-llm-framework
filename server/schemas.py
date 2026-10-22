"""请求/响应模型（Pydantic v2）。

/v1/chat/completions 与 /v1/models 采用 OpenAI 兼容协议：
任何使用 OpenAI SDK 的客户端都能直接接入本框架（DeepSeek 官方
API 同样采用该协议），便于与现有医疗信息系统集成。
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# OpenAI 兼容协议
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "medmini"
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False      # 演示实现仅支持非流式


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    usage: dict


class ModelItem(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "medllm"


# ---------------------------------------------------------------------------
# 医学业务接口
# ---------------------------------------------------------------------------
class ExtractRequest(BaseModel):
    text: str = Field(..., description="病历/报告原始文本")
    use_llm: bool = True


class ExtractResponse(BaseModel):
    entities: Dict[str, List[str]]


class RagAskRequest(BaseModel):
    question: str
    top_k: int = 3


class RagAskResponse(BaseModel):
    question: str
    answer: str
    sources: List[dict]


class PreConsultStartResponse(BaseModel):
    session_id: str
    reply: str
    state: dict


class PreConsultChatRequest(BaseModel):
    session_id: str
    text: str


class PreConsultChatResponse(BaseModel):
    reply: str
    state: dict
    emergency: Optional[str] = None
    summary: Optional[str] = None
    finished: bool = False
