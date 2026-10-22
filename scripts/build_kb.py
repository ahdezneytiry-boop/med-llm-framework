"""构建 RAG 向量知识库。

运行：python scripts/build_kb.py --kb knowledge --out data/vector_store
流程：加载文档 -> 分块 -> 嵌入（默认 TF-IDF）-> 持久化向量库。
"""
import argparse
import os
import sys
import time

# 项目根目录加入模块搜索路径（脚本可直接运行，无需 pip install）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medllm.rag.chunker import chunk_markdown
from medllm.rag.corpus import load_documents
from medllm.rag.embedder import TFIDFEmbedder
from medllm.rag.store import VectorStore


def main():
    parser = argparse.ArgumentParser(description="构建 RAG 向量知识库")
    parser.add_argument("--kb", default="knowledge", help="知识库文档目录")
    parser.add_argument("--out", default="data/vector_store", help="向量库输出目录")
    parser.add_argument("--max_chars", type=int, default=400, help="分块最大字符数")
    parser.add_argument("--overlap", type=int, default=40, help="分块重叠字符数")
    args = parser.parse_args()

    # 1. 加载文档并分块
    docs = load_documents(args.kb)
    if not docs:
        raise SystemExit(f"目录 {args.kb} 下没有找到文档（.md/.txt）")
    chunks = []
    for doc in docs:
        chunks.extend(chunk_markdown(doc, max_chars=args.max_chars, overlap=args.overlap))
    print(f"[文档] {len(docs)} 篇 -> {len(chunks)} 块")

    # 2. 嵌入并持久化
    embedder = TFIDFEmbedder().fit([c["text"] for c in chunks])
    vectors = embedder.embed([c["text"] for c in chunks])

    store = VectorStore()
    store.meta = {"embedder": "tfidf", "vocab": embedder.vocab,
                  "idf": embedder.idf.tolist(), "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    store.add(chunks, vectors)
    store.save(args.out)
    print(f"[完成] 向量库已保存到 {args.out}（{len(store)} 块，维度 {vectors.shape[1]}）")


if __name__ == "__main__":
    main()
