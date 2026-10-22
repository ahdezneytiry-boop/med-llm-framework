"""生成全部演示数据（合成语料 + SFT 数据 + 抽取评测标注）。

运行：python scripts/make_demo_data.py [--out data/demo]
"""
import argparse
import os
import sys
from pathlib import Path

# 项目根目录加入模块搜索路径（脚本可直接运行，无需 pip install）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medllm.data.synthetic import build_all


def main():
    parser = argparse.ArgumentParser(description="生成合成演示数据")
    parser.add_argument("--out", default="data/demo", help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    files = build_all(args.out, seed=args.seed)
    print("演示数据生成完成（全部为合成数据，不含真实患者信息）：")
    for name, path in files.items():
        n_lines = sum(1 for _ in open(path, encoding="utf-8"))
        print(f"  [{name}] {path}  ({n_lines} 行, {Path(path).stat().st_size} bytes)")


if __name__ == "__main__":
    main()
