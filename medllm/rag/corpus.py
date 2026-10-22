"""文档加载：从目录读取知识库文档（.md / .txt）。

知识库文档结构约定：
- 文件名为文档标题；
- 一级标题（#）视为章节，随分块一起保留，用于回答引用来源。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Document:
    """一篇知识库文档。"""
    doc_id: str      # 文档唯一标识（文件名，不含扩展名）
    title: str       # 标题（文件第一行 # 标题，否则取文件名）
    path: str
    text: str        # 正文（保留标题层级结构）


def load_documents(kb_dir: str, extensions: tuple = (".md", ".txt")) -> List[Document]:
    """递归加载目录下所有文档。"""
    docs: List[Document] = []
    for path in sorted(Path(kb_dir).rglob("*")):
        if path.suffix.lower() not in extensions:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        title = path.stem
        # 尝试取首个一级标题作为标题
        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        docs.append(Document(
            doc_id=path.stem,
            title=title,
            path=str(path),
            text=text,
        ))
    return docs
