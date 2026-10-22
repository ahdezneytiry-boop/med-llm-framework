"""文档分块：把长文档切成适合检索/上下文窗口的片段。

策略（经典 RAG 分块）：
1. 优先按 Markdown 标题层级切分（保留章节语义）；
2. 单段过长时按句号/换行再切；
3. 相邻块保留 overlap 字符重叠，避免关键句被切断在边界。
每块附带元数据（文档 id、标题、章节），用于回答时标注引用来源。
"""
from __future__ import annotations

import re
from typing import List

from .corpus import Document

# 句边界正则：中文句号/分号/换行等
_SENT_SPLIT = re.compile(r"(?<=[。；！？\n])")


def chunk_markdown(doc: Document, max_chars: int = 400, overlap: int = 40) -> List[dict]:
    """把一篇文档切成若干块。

    Returns:
        [{text, doc_id, title, section, index}, ...]
    """
    chunks: List[dict] = []
    section = ""
    buffer = ""

    def flush() -> None:
        nonlocal buffer
        text = buffer.strip()
        if text:
            chunks.append({
                "text": text,
                "doc_id": doc.doc_id,
                "title": doc.title,
                "section": section,
                "index": len(chunks),
            })
        buffer = ""

    # 逐行解析：一级标题更新 section，其余内容累积到 buffer
    for line in doc.text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("# "):          # 新章节
            flush()
            section = stripped[2:].strip()
            continue
        if stripped.startswith("#"):           # 次级标题并入章节信息
            flush()
            section = (section + " / " + stripped.lstrip("#").strip()).strip(" /")
            continue
        buffer += stripped + "\n"

        # 超过阈值时切分
        if len(buffer) >= max_chars:
            flush()
    flush()

    # 对仍然过长的块按句边界二次切分，并保留 overlap
    final: List[dict] = []
    for ch in chunks:
        if len(ch["text"]) <= max_chars:
            final.append(ch)
            continue
        sentences = [s for s in _SENT_SPLIT.split(ch["text"]) if s.strip()]
        piece = ""
        for s in sentences:
            if len(piece) + len(s) > max_chars and piece:
                final.append({**ch, "text": piece.strip()})
                piece = piece[-overlap:] if overlap else ""   # 重叠窗口
            piece += s
        if piece.strip():
            final.append({**ch, "text": piece.strip()})
    return final
