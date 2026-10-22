"""Gradio Web 界面 —— 四个功能标签页：

1. 健康问答（RAG）：输入问题 -> 检索知识库 -> 回答 + 引用来源（可点击核对）；
2. 病历信息抽取：粘贴病历/报告文本 -> 结构化字段表；
3. 智能预问诊：多轮对话式问诊，全程护栏保护，结束时给出问诊摘要；
4. 关于与声明：项目说明与医疗免责声明。

运行：python app/web.py --model checkpoints/sft --kb data/vector_store
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gradio as gr

from medllm.boot import load_stack
from medllm.guard.guardrails import DISCLAIMER


def build_ui(stack) -> gr.Blocks:
    """构建 Gradio 界面（依赖外部组装好的应用栈）。"""
    rag = stack.get("rag")
    extractor = stack["extractor"]
    agent = stack["agent"]
    guard = stack["guard"]

    with gr.Blocks(title="MedMiniLLM 医学大模型框架演示") as demo:
        gr.Markdown(
            "# MedMiniLLM —— 类 DeepSeek 医学大模型框架（演示）\n"
            "自研小规模 Transformer（MLA + DeepSeekMoE）驱动的医学信息应用：\n"
            "RAG 健康问答 / 病历信息抽取 / 智能预问诊 / 安全护栏。\n\n"
            "> ⚠️ 本项目为教学演示，不构成任何医疗建议。"
        )

        # ============================= Tab 1：RAG 健康问答 =============================
        with gr.Tab("健康问答（RAG）"):
            with gr.Row():
                qa_question = gr.Textbox(label="问题", lines=2, placeholder="例：高血压的诊断标准是多少？")
            qa_btn = gr.Button("查询")
            qa_answer = gr.Markdown(label="回答")
            qa_sources = gr.Markdown(label="引用来源")
            qa_status = gr.Textbox(label="护栏状态", interactive=False)

            def _qa(question: str):
                checked = guard.check_input(question)
                if checked.action == "emergency":
                    return guard.wrap_answer(checked.reason), "", "⚠️ 紧急识别：已建议急诊"
                if not checked.allowed:
                    return checked.reason, "", "⛔ 输入已拦截"
                if rag is None:
                    return "知识库未构建。请先运行 `python scripts/build_kb.py`。", "", "知识库缺失"
                ans = rag.ask(checked.text)
                src_md = "\n\n".join(
                    f"- **[{i+1}]《{s['title']}》{s.get('section') or ''}**（相似度 {s['score']}）"
                    for i, s in enumerate(ans.sources)
                ) or "（未检索到相关资料）"
                return guard.wrap_answer(ans.answer), src_md, "✅ 正常（已脱敏 + 免责声明）"

            qa_btn.click(_qa, inputs=[qa_question], outputs=[qa_answer, qa_sources, qa_status])

        # ============================= Tab 2：病历信息抽取 =============================
        with gr.Tab("病历信息抽取"):
            ext_text = gr.Textbox(label="病历/报告文本", lines=8,
                                  placeholder="例：患者男，62岁，主诉头晕3天，既往高血压病史，长期服用氨氯地平（5mg 每日一次）。血压165/95mmHg（偏高）。")
            ext_btn = gr.Button("抽取")
            ext_out = gr.JSON(label="结构化抽取结果")

            def _extract(text: str):
                if not text.strip():
                    return {"提示": "请先输入文本"}
                entities = extractor.extract(text)
                entities["_提示"] = "抽取结果仅供参考；输入已做 PII 脱敏处理。"
                return entities

            ext_btn.click(_extract, inputs=[ext_text], outputs=[ext_out])

        # ============================= Tab 3：智能预问诊 =============================
        with gr.Tab("智能预问诊"):
            pc_chatbot = gr.Chatbot(label="问诊对话", height=420)
            pc_input = gr.Textbox(label="您的回复", placeholder="描述您的情况……")
            pc_send = gr.Button("发送")
            pc_reset = gr.Button("重新开始")
            pc_summary = gr.Markdown(label="问诊摘要")
            pc_state = gr.State()

            def _pc_start():
                out = agent.start()
                return out["state"], [(None, out["reply"])], ""

            def _pc_reply(text: str, state):
                if not text.strip():
                    return state, None, ""
                out = agent.reply(state, text)
                summary_md = ("**问诊摘要**\n" + out["summary"]) if out["summary"] else ""
                if out["emergency"]:
                    summary_md = "🚨 " + out["emergency"]
                history = [(None, h["content"]) if h["role"] == "assistant"
                           else (h["content"], None) for h in out["state"]["history"]]
                return out["state"], history, summary_md

            pc_send.click(_pc_reply, inputs=[pc_input, pc_state],
                          outputs=[pc_state, pc_chatbot, pc_summary]).then(
                lambda: "", outputs=[pc_input])
            pc_reset.click(_pc_start, outputs=[pc_state, pc_chatbot, pc_summary])
            demo.load(_pc_start, outputs=[pc_state, pc_chatbot, pc_summary])

        # ============================= Tab 4：声明 =============================
        with gr.Tab("关于与声明"):
            gr.Markdown(
                "## 项目说明\n"
                "- 核心模型：自研 ~17M 参数 Transformer，参考 DeepSeek-V2/V3 的 "
                "MLA（多头潜变量注意力）、DeepSeekMoE（细粒度专家 + 共享专家）、"
                "无辅助损失负载均衡；\n"
                "- 全栈流水线：自训练 BPE 分词 → 预训练 → SFT → 评测 → 服务（OpenAI 兼容）；\n"
                "- 应用层：RAG（TF-IDF 检索）、病历抽取、预问诊 Agent、安全护栏。\n\n"
                f"{DISCLAIMER}\n\n"
                "所有演示数据（知识库、病历、问答对）均为合成或简化科普内容，"
                "与真实个体无关。"
            )

    return demo


def main():
    parser = argparse.ArgumentParser(description="启动 MedMiniLLM Web 界面")
    parser.add_argument("--model", default="checkpoints/sft", help="训练产物目录（model_config.json + last.pt）")
    parser.add_argument("--kb", default="data/vector_store", help="RAG 向量库目录")
    parser.add_argument("--provider", default="local", choices=["local", "openai"],
                        help="LLM 后端：local=自研模型，openai=OpenAI 兼容服务（需环境变量）")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="生成公网分享链接")
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(f"模型目录不存在：{args.model}（请先运行训练脚本）")

    stack = load_stack(model_dir=args.model, kb_dir=args.kb, provider=args.provider)
    demo = build_ui(stack)
    demo.launch(server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
