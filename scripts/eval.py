"""运行内置演示基准评测。

运行：python scripts/eval.py --model checkpoints/sft
评测三项：RAG 问答（关键词命中率）、病历抽取（P/R/F1）、预问诊（槽位填充）。
"""
import argparse
import json
import os
import sys

# 项目根目录加入模块搜索路径（脚本可直接运行，无需 pip install）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medllm.boot import load_stack
from medllm.eval.benchmark import run_all


def main():
    parser = argparse.ArgumentParser(description="运行内置演示基准")
    parser.add_argument("--model", default="checkpoints/sft", help="训练产物目录")
    parser.add_argument("--kb", default="data/vector_store", help="RAG 向量库目录")
    parser.add_argument("--provider", default="local", choices=["local", "openai"],
                        help="LLM 后端")
    parser.add_argument("--labeled", default="data/demo/extraction_labeled.jsonl",
                        help="抽取评测标注文件")
    parser.add_argument("--max_extract_samples", type=int, default=8,
                        help="抽取评测最多样本数（0=全部；CPU 演示建议限制，模型通道逐条生成较慢）")
    args = parser.parse_args()

    stack = load_stack(model_dir=args.model, kb_dir=args.kb, provider=args.provider)
    report = run_all(stack, max_extract_samples=args.max_extract_samples)

    print("\n" + "=" * 60)
    print("评测报告（演示基准，非临床评测）")
    print("=" * 60)
    print("\n[RAG 问答]")
    rag = report["rag"]
    if "error" in rag:
        print(f"  ⚠️ {rag['error']}")
    else:
        print(f"  关键词命中率: {rag['mean_keyword_hit']}")
        print(f"  带来源引用:   {rag['with_sources']}")

    print("\n[病历信息抽取]")
    ext = report["extraction"]
    if "error" in ext:
        print(f"  ⚠️ {ext['error']}")
    else:
        print(f"  样本数: {ext['samples']}")
        print(f"  P={ext['precision']}  R={ext['recall']}  F1={ext['f1']}")
        for field, f1 in ext["per_field_f1"].items():
            print(f"    - {field}: F1={f1}")

    print("\n[智能预问诊]")
    pc = report["preconsult"]
    print(f"  对话轮数: {pc['turns']}  槽位填充: {pc['slots_filled']}/{pc['total_slots']}"
          f"  生成摘要: {'是' if pc['has_summary'] else '否'}")

    out = "data/eval_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n完整报告已保存到 {out}")


if __name__ == "__main__":
    main()
