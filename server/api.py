"""FastAPI 服务 —— 对外暴露两类接口：

A. OpenAI 兼容协议（与 DeepSeek 官方 API 同协议，客户端零改造接入）：
   - POST /v1/chat/completions   对话补全（输出侧护栏：禁用绝对化表述 + 免责声明）
   - GET  /v1/models             模型列表
   - GET  /health                健康检查

B. 医学业务接口：
   - POST /rag/ask               RAG 健康问答（返回带来源引用的回答）
   - POST /extract               病历信息抽取（结构化 JSON）
   - POST /preconsult/start      开启预问诊会话
   - POST /preconsult/chat       预问诊对话（多轮，结束时返回问诊摘要）

所有对话接口统一经过安全护栏：输入 PII 脱敏、急危重症升级急诊建议、
输出附加免责声明。
"""
from __future__ import annotations

import time
import uuid
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import schemas as s

# 会话存储：内存字典（演示级；生产环境应替换为 Redis/数据库）
_SESSIONS: Dict[str, dict] = {}


def create_app(stack: Dict, name: str = "medmini") -> FastAPI:
    """应用工厂：传入 boot.load_stack 组装好的应用栈。

    Args:
        stack: {llm, rag, extractor, agent, guard}（rag 可为 None）
    """
    llm = stack["llm"]
    rag = stack.get("rag")
    extractor = stack["extractor"]
    agent = stack["agent"]
    guard = stack["guard"]

    app = FastAPI(title="MedMiniLLM API", version="0.1.0",
                  description="类 DeepSeek 架构的医学大模型框架服务（演示级）")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])    # 演示环境放开跨域

    # ------------------------------------------------------------------
    # OpenAI 兼容接口
    # ------------------------------------------------------------------
    @app.get("/health")
    def health():
        return {"status": "ok", "model": llm.name, "rag_ready": rag is not None}

    @app.get("/v1/models", response_model=dict)
    def list_models():
        return {"object": "list", "data": [s.ModelItem(id=name).model_dump()]}

    @app.post("/v1/chat/completions", response_model=s.ChatCompletionResponse)
    def chat_completions(req: s.ChatCompletionRequest):
        messages = [m.model_dump() for m in req.messages]
        raw = llm.generate(messages, max_new_tokens=req.max_tokens,
                           temperature=req.temperature)
        # 输出侧护栏：拦截违规表述并附加免责声明
        checked = guard.check_output(raw)
        content = checked.text if checked.allowed else checked.reason
        return s.ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:16]}",
            created=int(time.time()),
            model=req.model,
            choices=[{"index": 0,
                      "message": {"role": "assistant", "content": content},
                      "finish_reason": "stop"}],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    # ------------------------------------------------------------------
    # 医学业务接口
    # ------------------------------------------------------------------
    @app.post("/rag/ask", response_model=s.RagAskResponse)
    def rag_ask(req: s.RagAskRequest):
        if rag is None:
            raise HTTPException(503, "知识库未构建，请先运行 scripts/build_kb.py")
        # 输入护栏：脱敏 + 恶意拦截 + 紧急识别
        checked = guard.check_input(req.question)
        if checked.action == "emergency":
            return s.RagAskResponse(question=req.question,
                                    answer=guard.wrap_answer(checked.reason), sources=[])
        if not checked.allowed:
            return s.RagAskResponse(question=req.question, answer=checked.reason, sources=[])
        ans = rag.ask(checked.text, top_k=req.top_k)
        return s.RagAskResponse(question=req.question,
                                answer=guard.wrap_answer(ans.answer), sources=ans.sources)

    @app.post("/extract", response_model=s.ExtractResponse)
    def extract(req: s.ExtractRequest):
        # 抽取场景只做 PII 脱敏（病历文本本身就可能包含症状词，
        # 不触发"紧急问答升级"，那是对话类接口的职责）
        from medllm.guard.desensitize import desensitize
        text = desensitize(req.text)
        return s.ExtractResponse(entities=extractor.extract(text, max_new_tokens=256))

    @app.post("/preconsult/start", response_model=s.PreConsultStartResponse)
    def preconsult_start():
        session_id = uuid.uuid4().hex[:16]
        out = agent.start()
        _SESSIONS[session_id] = out["state"]
        return s.PreConsultStartResponse(session_id=session_id, reply=out["reply"],
                                         state=out["state"])

    @app.post("/preconsult/chat", response_model=s.PreConsultChatResponse)
    def preconsult_chat(req: s.PreConsultChatRequest):
        state = _SESSIONS.get(req.session_id)
        if state is None:
            raise HTTPException(404, "会话不存在或已过期，请重新 start")
        out = agent.reply(state, req.text)
        _SESSIONS[req.session_id] = out["state"]
        return s.PreConsultChatResponse(**out)

    return app
