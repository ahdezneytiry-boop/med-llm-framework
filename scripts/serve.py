"""启动 REST 服务（FastAPI，OpenAI 兼容协议 + 医学业务接口）。

运行：python scripts/serve.py --model checkpoints/sft --port 8000
接口文档（自动生成）：http://127.0.0.1:8000/docs

OpenAI 兼容调用示例（任何 OpenAI SDK 客户端可直接接入）：
    curl http://127.0.0.1:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"medmini","messages":[{"role":"user","content":"你好"}]}'
"""
import argparse
import os
import sys

# 项目根目录加入模块搜索路径（脚本可直接运行，无需 pip install）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn

from medllm.boot import load_stack
from server.api import create_app


def main():
    parser = argparse.ArgumentParser(description="启动 MedMiniLLM REST 服务")
    parser.add_argument("--model", default="checkpoints/sft", help="训练产物目录")
    parser.add_argument("--kb", default="data/vector_store", help="RAG 向量库目录")
    parser.add_argument("--provider", default="local", choices=["local", "openai"],
                        help="LLM 后端：local=自研模型，openai=OpenAI 兼容服务（需环境变量 OPENAI_API_KEY）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    stack = load_stack(model_dir=args.model, kb_dir=args.kb, provider=args.provider)
    app = create_app(stack)
    print(f"服务启动：http://{args.host}:{args.port}  （接口文档 /docs）")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
