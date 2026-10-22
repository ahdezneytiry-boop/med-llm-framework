"""训练 BPE 分词器（字节级，中文按单字预切分）。

运行：python scripts/train_tokenizer.py \
        --corpus data/demo/pretrain_corpus.txt \
        --out checkpoints/pretrain/tokenizer.json \
        --vocab 4096
"""
import argparse
import os
import sys
import time

# 项目根目录加入模块搜索路径（脚本可直接运行，无需 pip install）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medllm.core.tokenizer import ByteLevelBPE


def main():
    parser = argparse.ArgumentParser(description="训练字节级 BPE 分词器")
    parser.add_argument("--corpus", default="data/demo/pretrain_corpus.txt", help="训练语料")
    parser.add_argument("--out", default="checkpoints/pretrain/tokenizer.json", help="输出路径")
    parser.add_argument("--vocab", type=int, default=4096, help="目标词表大小")
    args = parser.parse_args()

    with open(args.corpus, "r", encoding="utf-8") as f:
        text = f.read()

    tok = ByteLevelBPE()
    t0 = time.time()
    tok.train([text], vocab_size=args.vocab)
    tok.save(args.out)

    print(f"训练完成：词表大小 {tok.vocab_size}，耗时 {time.time() - t0:.1f}s")
    print(f"已保存到 {args.out}")

    # 自检：中文样本往返编解码
    sample = "高血压患者应低盐饮食并定期监测血压。"
    print(f"编码自检: '{sample}' -> {len(tok.encode(sample))} tokens")
    print(f"解码自检: {tok.decode(tok.encode(sample))}")


if __name__ == "__main__":
    main()
