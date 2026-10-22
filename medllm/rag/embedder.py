"""嵌入器：把文本块/查询编码为向量。

提供两种实现（接口一致，可互换）：
1. TFIDFEmbedder（默认）：字符 bigram + TF-IDF 加权的轻量稀疏嵌入。
   优点：零外部依赖、对中文友好（按字符切分，无分词问题）、可解释；
   缺点：无语义泛化能力 —— 演示/原型用途足够，生产应替换为医学领域嵌入模型。
2. ModelEmbedder（可选）：用自研模型的隐层均值池化作为嵌入，
   把"训练得到的模型"直接接入检索链路，体现全栈闭环。

实现新嵌入器只需实现 fit / embed 两个方法。
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np


def _char_bigrams(text: str) -> List[str]:
    """字符 bigram：把文本切成连续两字符组合（中文场景常用特征）。"""
    text = " " + text.replace("\n", " ").replace("\r", " ") + " "
    return [text[i:i + 2] for i in range(len(text) - 1)]


class TFIDFEmbedder:
    """字符 bigram + TF-IDF 嵌入器（演示级，接口与专业嵌入模型对齐）。"""

    name = "tfidf"

    def __init__(self, max_features: int = 65536):
        self.max_features = max_features
        self.vocab: List[str] = []      # 保留的 bigram 词表
        self.idf: np.ndarray = None     # 每个 bigram 的 IDF 权重

    # ------------------------------------------------------------------
    def fit(self, texts: List[str]) -> "TFIDFEmbedder":
        """在语料上构建 bigram 词表与 IDF。

        IDF = log(N / (1 + df))：常见 bigram 权重低，罕见 bigram 权重高，
        让"高血压""糖化"这类有区分度的词组在相似度中占主导。
        """
        df = Counter()
        n_docs = len(texts)
        for t in texts:
            for g in set(_char_bigrams(t)):
                df[g] += 1
        # 按文档频率排序取前 max_features 个
        self.vocab = [g for g, _ in df.most_common(self.max_features)]
        vocab_index = {g: i for i, g in enumerate(self.vocab)}
        counts = np.zeros(len(self.vocab), dtype=np.float32)
        for g, c in df.items():
            if g in vocab_index:
                counts[vocab_index[g]] = c
        self.idf = np.log(n_docs / (1.0 + counts)).astype(np.float32)
        return self

    def embed(self, texts: List[str]) -> np.ndarray:
        """把文本编码为 TF-IDF 向量（L2 归一化，便于余弦相似度）。"""
        assert self.idf is not None, "请先调用 fit()"
        mat = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        vocab_index = {g: i for i, g in enumerate(self.vocab)}
        for r, t in enumerate(texts):
            for g, c in Counter(_char_bigrams(t)).items():
                if g in vocab_index:
                    mat[r, vocab_index[g]] = (1.0 + math.log(c)) * self.idf[vocab_index[g]]
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (mat / norms).astype(np.float32)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab, "idf": self.idf.tolist() if self.idf is not None else None,
                       "max_features": self.max_features}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "TFIDFEmbedder":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        emb = cls(max_features=d.get("max_features", 65536))
        emb.vocab = d["vocab"]
        emb.idf = np.asarray(d["idf"], dtype=np.float32) if d["idf"] else None
        return emb


class ModelEmbedder:
    """用自研模型的隐层均值池化作为嵌入（可选，展示模型在检索链路中的闭环）。"""

    name = "model"

    def __init__(self, model, tokenizer, device: str = "cpu"):
        import torch
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self._torch = torch

    def fit(self, texts: List[str]) -> "ModelEmbedder":
        return self                                    # 无需拟合

    def embed(self, texts: List[str]) -> np.ndarray:
        torch = self._torch
        self.model.eval()
        vecs = []
        with torch.no_grad():
            for t in texts:
                ids = torch.tensor([self.tokenizer.encode_whole(t, add_bos=True)],
                                   device=self.device)
                ids = ids[:, : self.model.cfg.max_seq_len]
                logits, _, _ = self.model(ids)
                # 用输出 logits 的均值池化作为粗粒度语义向量
                v = logits.mean(dim=1).float().cpu().numpy()
                vecs.append(v / (np.linalg.norm(v) + 1e-8))
        return np.vstack(vecs).astype(np.float32)


def make_embedder(kind: str, meta: dict = None, **kwargs):
    """嵌入器工厂：按向量库元信息恢复对应的嵌入器。"""
    meta = meta or {}
    if kind == "model":
        return ModelEmbedder(**kwargs)
    emb = TFIDFEmbedder()
    if meta.get("embedder_path"):
        return TFIDFEmbedder.load(meta["embedder_path"])
    if meta.get("vocab"):
        emb.vocab = meta["vocab"]
        emb.idf = np.asarray(meta["idf"], dtype=np.float32)
    return emb
