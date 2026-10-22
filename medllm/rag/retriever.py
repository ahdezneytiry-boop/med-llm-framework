"""RAG 问答流水线：检索 -> 组装带引用的提示词 -> LLM 生成。

降低医学问答幻觉的关键设计（全部体现在提示词中）：
1. 明确要求"只依据提供的资料回答"；
2. 逐条标注引用来源（资料编号 + 标题），让回答可追溯、可核查；
3. 资料中找不到答案时，要求模型明确说"资料中未提及"，并建议就医，
   而不是凭空编造。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..llm.base import BaseLLM
from .store import VectorStore

SYSTEM_PROMPT = (
    "你是演示医学问答助手。请严格依据下方提供的资料回答问题，"
    "并在每一条结论后标注引用的资料编号（如[1][2]）。"
    "资料中没有的信息不要编造，明确回答\u201c资料中未提及\u201d；"
    "如果用户描述的症状涉及急危重症，请建议立即就医。"
)


@dataclass
class RagAnswer:
    """RAG 回答结果。"""
    question: str
    answer: str
    sources: List[dict] = field(default_factory=list)   # [{doc_id, title, section, text, score}]


class RagPipeline:
    def __init__(self, store: VectorStore, embedder, llm: BaseLLM, top_k: int = 3):
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.top_k = top_k

    # ------------------------------------------------------------------
    def ask(self, question: str, top_k: Optional[int] = None,
            max_new_tokens: int = 256, temperature: Optional[float] = None) -> RagAnswer:
        """RAG 问答：检索 top-k 资料片段，组装提示词后交给 LLM 生成。"""
        top_k = top_k or self.top_k
        query_vec = self.embedder.embed([question])[0]
        hits = self.store.search(query_vec, top_k=top_k)

        # 组装上下文：资料编号 [1]..[k]
        context_parts = []
        sources = []
        for i, (chunk, score) in enumerate(hits, start=1):
            context_parts.append(
                f"[{i}]《{chunk['title']}》{chunk.get('section') or ''}\n{chunk['text']}")
            sources.append({
                "doc_id": chunk["doc_id"], "title": chunk["title"],
                "section": chunk.get("section", ""),
                "text": chunk["text"][:200], "score": round(score, 4),
            })
        context = "\n\n".join(context_parts) if context_parts else "（知识库为空）"

        user_prompt = f"资料：\n{context}\n\n问题：{question}\n请依据资料回答并标注引用编号。"
        answer = self.llm.generate(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return RagAnswer(question=question, answer=answer, sources=sources)
