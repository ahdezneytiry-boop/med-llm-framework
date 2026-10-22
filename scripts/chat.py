"""命令行聊天 —— 与训练好的模型对话（调试/演示用）。

运行：python scripts/chat.py --model checkpoints/sft
交互命令：输入内容回车发送；输入 quit/exit 退出。
"""
import argparse
import os
import sys

# 项目根目录加入模块搜索路径（脚本可直接运行，无需 pip install）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medllm.boot import load_stack


def main():
    parser = argparse.ArgumentParser(description="与 MedMiniLLM 命令行对话")
    parser.add_argument("--model", default="checkpoints/sft", help="训练产物目录")
    parser.add_argument("--provider", default="local", choices=["local", "openai"],
                        help="LLM 后端")
    parser.add_argument("--kb", default="data/vector_store", help="RAG 向量库目录")
    parser.add_argument("--prompt", default="", help="单次提问模式（不进入交互循环）")
    args = parser.parse_args()

    stack = load_stack(model_dir=args.model, kb_dir=args.kb, provider=args.provider)
    llm = stack["llm"]

    if args.prompt:
        print(llm.generate([{"role": "user", "content": args.prompt}]))
        return

    print("=" * 60)
    print("MedMiniLLM 命令行聊天（输入 quit 退出）")
    print("提示：小规模演示模型能力有限；医学回答仅供参考。")
    print("=" * 60)
    history = []
    while True:
        try:
            user = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user.lower() in ("quit", "exit", "q"):
            break
        history.append({"role": "user", "content": user})
        answer = llm.generate(history, max_new_tokens=256)
        print(f"\n助手: {answer}")
        history.append({"role": "assistant", "content": answer})
        if len(history) > 12:      # 控制上下文长度
            history = history[-12:]


if __name__ == "__main__":
    main()
