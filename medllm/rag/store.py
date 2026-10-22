"""极简向量库：numpy 余弦相似度检索 + 磁盘持久化。

设计：知识库规模（几百~几千块）下，numpy 内积检索足够快；
数据量大时可无缝替换为 FAISS / Milvus / Qdrant（接口保持不变：
add / search / save / load）。
持久化格式：chunks.json（块与元数据）+ vectors.npy（向量矩阵）
+ meta.json（嵌入器信息，恢复查询侧嵌入器用）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


class VectorStore:
    def __init__(self):
        self.chunks: List[dict] = []
        self.vectors: Optional[np.ndarray] = None
        self.meta: dict = {}

    # ------------------------------------------------------------------
    def add(self, chunks: List[dict], vectors: np.ndarray) -> None:
        """批量写入块与向量。"""
        self.chunks.extend(chunks)
        if self.vectors is None:
            self.vectors = vectors.astype(np.float32)
        else:
            self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])

    def search(self, query_vec: np.ndarray, top_k: int = 3,
               min_score: float = 0.0) -> List[Tuple[dict, float]]:
        """余弦相似度检索，返回 [(块, 相似度), ...]（降序）。"""
        if self.vectors is None or len(self.chunks) == 0:
            return []
        q = query_vec.astype(np.float32).reshape(1, -1)
        q = q / (np.linalg.norm(q) + 1e-8)
        scores = (self.vectors @ q.T).reshape(-1)
        order = np.argsort(-scores)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in order
                if float(scores[i]) >= min_score]

    def __len__(self) -> int:
        return len(self.chunks)

    # ------------------------------------------------------------------
    def save(self, out_dir: str) -> None:
        """保存为 chunks.json + vectors.npy + meta.json。"""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "chunks.json").write_text(
            json.dumps(self.chunks, ensure_ascii=False, indent=1), encoding="utf-8")
        np.save(out / "vectors.npy", self.vectors)
        (out / "meta.json").write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, in_dir: str) -> "VectorStore":
        """从目录加载向量库（目录必须存在）。"""
        store = cls()
        base = Path(in_dir)
        store.chunks = json.loads((base / "chunks.json").read_text(encoding="utf-8"))
        store.vectors = np.load(base / "vectors.npy").astype(np.float32)
        meta_path = base / "meta.json"
        store.meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return store

    @classmethod
    def load_if_exists(cls, in_dir: str) -> Optional["VectorStore"]:
        """目录不存在时返回 None（应用层按"无知识库"模式运行）。"""
        base = Path(in_dir)
        if not (base / "chunks.json").exists():
            return None
        return cls.load(in_dir)
