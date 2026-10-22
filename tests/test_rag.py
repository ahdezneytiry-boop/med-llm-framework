"""RAG 单元测试：分块、TF-IDF 嵌入、向量库检索。"""
import numpy as np
import pytest

from medllm.rag.chunker import chunk_markdown
from medllm.rag.corpus import Document
from medllm.rag.embedder import TFIDFEmbedder
from medllm.rag.store import VectorStore


@pytest.fixture(scope="module")
def store():
    """构造一个 3 文档的迷你知识库。"""
    docs = [
        Document(doc_id="hypertension", title="高血压科普（演示）", path="x.md",
                 text="# 高血压科普\n\n高血压的诊断标准是诊室血压大于等于140/90毫米汞柱。\n\n"
                      "患者应低盐饮食并定期监测血压。"),
        Document(doc_id="diabetes", title="糖尿病科普（演示）", path="x.md",
                 text="# 糖尿病科普\n\n空腹血糖大于等于7.0毫摩尔每升可考虑糖尿病。\n\n"
                      "糖化血红蛋白反映近两到三个月的平均血糖。"),
        Document(doc_id="fever", title="发热科普（演示）", path="x.md",
                 text="# 发热科普\n\n成人体温超过38.5度可考虑使用退热药。\n\n"
                      "持续高热不退应立即就医。"),
    ]
    chunks = []
    for d in docs:
        chunks.extend(chunk_markdown(d))
    emb = TFIDFEmbedder().fit([c["text"] for c in chunks])
    vecs = emb.embed([c["text"] for c in chunks])
    s = VectorStore()
    s.add(chunks, vecs)
    return s, emb


def test_chunking_keeps_metadata(store):
    s, _ = store
    assert len(s.chunks) >= 3
    doc_ids = {c["doc_id"] for c in s.chunks}
    assert doc_ids == {"hypertension", "diabetes", "fever"}


def test_retrieval_hits_relevant_doc(store):
    """查询"高血压"应命中高血压文档（TF-IDF 关键词匹配能力）。"""
    s, emb = store
    q = emb.embed(["高血压的诊断标准是什么？"])[0]
    hits = s.search(q, top_k=2)
    assert hits, "检索结果为空"
    assert any("hypertension" == h[0]["doc_id"] for h in hits)
    assert hits[0][1] > 0.0                     # 相似度为正


def test_retrieval_fever_query(store):
    """查询"退热药"应命中发热文档。"""
    s, emb = store
    q = emb.embed(["发烧了要不要吃退热药"])[0]
    hits = s.search(q, top_k=3)
    assert any("fever" == h[0]["doc_id"] for h in hits)


def test_store_persistence(store, tmp_path):
    """向量库保存/加载往返一致。"""
    s, _ = store
    out = tmp_path / "store"
    s.save(str(out))
    s2 = VectorStore.load(str(out))
    assert len(s2) == len(s)
    assert s2.chunks == s.chunks
    assert np.allclose(s2.vectors, s.vectors)
