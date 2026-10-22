# 架构文档 —— MedMiniLLM 全栈工作流

## 总体架构

```
                    ┌──────────────────────────────────────────────┐
                    │              医学应用层（app / server）        │
                    │  Gradio Web 界面 │ FastAPI(OpenAI 兼容协议)     │
                    │  ① RAG 健康问答  ② 病历抽取  ③ 预问诊 Agent      │
                    │  ④ 安全护栏（脱敏/紧急识别/免责声明）            │
                    └───────────────┬──────────────────────────────┘
                                    │ BaseLLM 统一接口
                    ┌───────────────┴──────────────────────────────┐
                    │         LLM 接入层（medllm/llm）               │
                    │  LocalLLM（自研模型） / OpenAICompatLLM（可选） │
                    └───────────────┬──────────────────────────────┘
                                    │
                    ┌───────────────┴──────────────────────────────┐
                    │       核心模型（medllm/core，类 DeepSeek-V3）   │
                    │  自训练 BPE │ MLA │ DeepSeekMoE │ RMSNorm │ RoPE│
                    └───────┬──────────────────────┬───────────────┘
                            │                      │
              ┌─────────────▼──────┐      ┌────────▼──────────┐
              │ 数据管线（data）     │      │ 训练/推理（train/infer）│
              │ 合成语料 → 预训练    │      │ Trainer → SFT      │
              │ messages → SFT     │      │ KV缓存生成 → 评测   │
              └────────────────────┘      └───────────────────┘
                            │                      │
                    知识库（knowledge）    向量库（data/vector_store）
```

## 全栈流水线（一条命令跑通）

```bash
# 1. 生成合成演示数据
python scripts/make_demo_data.py
# 2. 训练分词器 + 预训练（自动训练 BPE，输出检查点）
python scripts/pretrain.py
# 3. SFT 监督微调（指令跟随）
python scripts/sft.py
# 4. 构建 RAG 向量知识库
python scripts/build_kb.py
# 5. 评测（RAG / 抽取 / 预问诊）
python scripts/eval.py
# 6. 部署：REST 服务 + Web 界面
python scripts/serve.py        # http://127.0.0.1:8000/docs
python app/web.py              # http://127.0.0.1:7860
```

## 模块职责一览

| 模块 | 职责 | 关键文件 |
| --- | --- | --- |
| `medllm/core` | 模型本体：BPE、RoPE、MLA、DeepSeekMoE、Transformer | `model.py`, `attention.py`, `moe.py`, `tokenizer.py` |
| `medllm/data` | 数据管线：对话模板、数据集、合成语料 | `chat_format.py`, `dataset.py`, `synthetic.py` |
| `medllm/train` | 训练：AdamW + 余弦调度 + 梯度累积 + 检查点 | `trainer.py`, `scheduler.py` |
| `medllm/infer` | 推理：KV 缓存自回归生成 + 采样策略 | `generate.py` |
| `medllm/llm` | LLM 后端抽象（本地/OpenAI 兼容） | `base.py`, `local.py` |
| `medllm/rag` | RAG：加载/分块/嵌入/向量库/检索问答 | `retriever.py` 等 |
| `medllm/agent` | 预问诊槽位状态机 | `preconsult.py`, `slots.py` |
| `medllm/guard` | 安全护栏：脱敏/紧急识别/免责声明 | `guardrails.py`, `desensitize.py` |
| `medllm/eval` | 指标与内置基准 | `metrics.py`, `benchmark.py` |
| `server` | FastAPI（OpenAI 兼容 + 业务接口） | `api.py`, `schemas.py` |
| `app` | Gradio Web 界面 | `web.py` |
| `scripts` | 各阶段命令行入口 | `pretrain.py`, `sft.py`, `serve.py` 等 |

## 关键设计决策

1. **模型规模**：~17M 参数、fp32、CPU 可训练 —— 保证教学/演示
   在无 GPU 环境下 30 分钟内跑通全流程；
2. **自训练 BPE**：词表随语料生长（演示语料 -> 4096），
   中文按单字预切分，避免第三方 tokenizer 依赖；
3. **MLA 缓存压缩**：生成时每层只缓存潜向量 c_KV 与 K_pe，
   这是类 DeepSeek 架构"长上下文低成本推理"的根基；
4. **双通道抽取**：规则通道保证基础字段可用，模型通道补充语义归纳，
   二者合并 —— 小模型下依然稳定产出结构化结果；
5. **护栏前置**：所有对话入口先脱敏、再紧急识别、输出统一附加
   免责声明；急危重症表述直接升级急诊建议并终止常规问答。

## 扩展方向（Roadmap）

- 数据：接入真实医学语料（需授权）与 RLAIF/DPO 对齐；
- 模型：FP8/INT8 量化、多 token 预测（MTP，V3 已采用）、
  更长的上下文（YaRN）；
- 检索：TF-IDF 替换为医学领域嵌入模型 + FAISS/重排序；
- 部署：多机并行（V3 的 DualPipe 流水线）、vLLM 推理；
- 合规：接入执业医师审核与人工兜底流程。
