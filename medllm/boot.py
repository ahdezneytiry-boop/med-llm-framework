"""组件装配 —— serve / chat / eval / web 共用的启动引导。

职责：从训练产物目录构建 LLM，从知识库目录构建 RAG，
再组装信息抽取器、预问诊 Agent 与安全护栏，输出一个完整"应用栈"。
"""
from __future__ import annotations

from typing import Dict, Optional

from .llm.base import load_llm
from .rag.retriever import RagPipeline
from .rag.store import VectorStore
from .rag.embedder import make_embedder


def load_model_dir(model_dir: str, device: Optional[str] = None, **gen_kw):
    """加载自研模型目录，返回 LocalLLM。"""
    return load_llm("local", model_dir=model_dir, device=device, **gen_kw)


def load_stack(
    model_dir: str,
    kb_dir: str = "data/vector_store",
    provider: str = "local",
    openai_kw: Optional[dict] = None,
    **gen_kw,
) -> Dict:
    """构建完整应用栈。

    Args:
        model_dir: 训练产物目录（含 model_config.json / tokenizer.json / last.pt）
        kb_dir:    RAG 向量库目录（由 scripts/build_kb.py 生成）
        provider:  LLM 后端："local" 或 "openai"
        openai_kw: provider=openai 时的连接参数

    Returns:
        {llm, rag, extractor, agent, guard}
    """
    from .agent.preconsult import PreConsultAgent
    from .extraction import MedicalExtractor
    from .guard.guardrails import Guardrails

    # 1. LLM 后端
    if provider == "openai":
        llm = load_llm("openai", **(openai_kw or {}))
    else:
        llm = load_model_dir(model_dir, **gen_kw)

    # 2. RAG（向量库不存在时返回 None，应用层按无知识库模式运行）
    rag = None
    store = VectorStore.load_if_exists(kb_dir)
    if store is not None:
        embedder = make_embedder(store.meta.get("embedder", "tfidf"), store.meta)
        rag = RagPipeline(store=store, embedder=embedder, llm=llm)

    # 3. 医学应用组件
    guard = Guardrails()
    extractor = MedicalExtractor(llm=llm)
    agent = PreConsultAgent(llm=llm, guard=guard)
    return {"llm": llm, "rag": rag, "extractor": extractor, "agent": agent, "guard": guard}
