# MedMiniLLM —— 类 DeepSeek 的医学大模型全栈框架（教学/演示级）

一个**从零构建**的小规模大语言模型框架，参考 DeepSeek-V2 / V3 的开源技术方案
（MLA 多头潜变量注意力、DeepSeekMoE 细粒度专家、无辅助损失负载均衡），
面向**医学信息**场景，覆盖 **数据 → 训练 → 评测 → 部署 → 应用** 的完整流水线。

> ⚠️ **免责声明**：本项目为教学/演示用途。全部数据为合成演示数据，
> 模型规模小、能力有限，所有输出**不构成诊断或治疗建议**。

## 特性一览

| 层次 | 内容 |
| --- | --- |
| 核心模型 | 自研 ~17M 参数 Transformer：自训练字节级 BPE（中文按单字预切分）、MLA（KV 潜向量压缩缓存）、DeepSeekMoE（16 专家 / top-4 / 共享专家）、无辅助损失负载均衡、RMSNorm、RoPE |
| 训练 | 预训练（因果 LM）+ SFT（DeepSeek 风格 `messages` 数据，仅 assistant 参与损失）、AdamW + warmup/余弦调度、梯度累积、断点续训、CPU 可跑 |
| 推理 | KV 缓存自回归生成（温度/top-p/top-k/重复惩罚）、统一 LLM 后端抽象（本地模型 / 可选 OpenAI 兼容服务） |
| 医学应用 | ① RAG 健康问答（带来源引用、反幻觉提示词）② 病历信息抽取（规则 + 模型双通道）③ 智能预问诊 Agent（槽位状态机 + 问诊摘要）④ 安全护栏（PII 脱敏、急危重症识别转急诊、免责声明） |
| 部署 | FastAPI 服务（**OpenAI 兼容协议**，与 DeepSeek 官方 API 同协议）+ Gradio Web 界面 |
| 评测 | 内置演示基准：RAG 关键词命中率、抽取 P/R/F1、预问诊槽位填充率 |

## 目录结构

```
med-llm-framework/
├── configs/              # 模型结构与训练配置（yaml，全中文注释）
├── medllm/               # 核心库
│   ├── core/             #   BPE 分词器 / RoPE / MLA / DeepSeekMoE / 模型主体
│   ├── data/             #   对话模板 / 数据集 / 合成演示数据生成器
│   ├── train/            #   训练器 / 学习率调度
│   ├── infer/            #   KV 缓存生成 / 聊天封装
│   ├── llm/              #   LLM 后端抽象（local / openai 兼容）
│   ├── rag/              #   文档加载 / 分块 / 嵌入 / 向量库 / 检索问答
│   ├── agent/            #   预问诊槽位状态机
│   ├── guard/            #   PII 脱敏 / 紧急识别 / 免责声明护栏
│   ├── eval/             #   指标 / 内置基准
│   ├── extraction.py     #   病历信息抽取（规则 + 模型双通道）
│   └── boot.py           #   应用栈装配
├── server/               # FastAPI（OpenAI 兼容 + 业务接口）
├── app/                  # Gradio Web 界面（4 个功能标签页）
├── scripts/              # 命令行入口（数据/训练/推理/服务/评测）
├── knowledge/            # 演示知识库（简化科普，标注非医疗建议）
├── docs/                 # 架构文档 + 与 DeepSeek 技术方案对照表
└── tests/                # 单元测试（分词器/模型/RAG/护栏）
```

## 快速开始（全流程约 10~30 分钟，CPU 即可）

```bash
# 0. 安装依赖（torch 请按平台安装对应版本）
pip install -r requirements.txt

# 1. 生成合成演示数据（语料 / SFT 数据 / 评测标注）
python scripts/make_demo_data.py

# 2. 预训练（自动训练 BPE 分词器 + 300 步演示训练）
python scripts/pretrain.py

# 3. SFT 监督微调（指令跟随）
python scripts/sft.py

# 4. 构建 RAG 向量知识库
python scripts/build_kb.py

# 5. 跑一遍内置基准评测
python scripts/eval.py

# 6a. 启动 REST 服务（OpenAI 兼容协议）
python scripts/serve.py                     # 接口文档 http://127.0.0.1:8000/docs
# 6b. 或启动 Web 界面
python app/web.py                           # http://127.0.0.1:7860
```

命令行对话：`python scripts/chat.py`

### OpenAI 兼容调用示例（与 DeepSeek 官方 API 同协议）

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"medmini","messages":[{"role":"user","content":"高血压的诊断标准是什么？"}]}'
```

## 单元测试

```bash
pytest tests/ -v
```

## 关键设计说明

- **为什么自研小模型而不是调用 API？** 本框架定位是"构建类 DeepSeek 的框架"：
  训练、推理、服务的每一环都是自己的代码，规模缩小到 CPU 可演示；
- **MLA 的意义**：KV 缓存压缩为潜向量，是 DeepSeek 系列"长上下文低成本推理"的根基，
  代码见 `medllm/core/attention.py`；
- **无辅助损失负载均衡**：专家偏置随负载自适应修正，避免辅助损失损害模型质量，
  代码见 `medllm/core/moe.py`；
- **护栏前置**：所有对话入口先脱敏、紧急表述直接转急诊建议、输出统一附加免责声明。

更多细节见 `docs/architecture.md`（架构图与设计决策）与
`docs/deepseek_reference.md`（与 DeepSeek 论文逐项对照）。

## 扩展路线

- [ ] 真实医学语料（需授权）与 RLAIF/DPO 对齐
- [ ] FP8 量化、多 token 预测（MTP）、YaRN 长上下文
- [ ] 医学领域嵌入模型 + FAISS + 重排序
- [ ] vLLM 推理、多卡流水线并行
- [ ] 执业医师审核与人工兜底流程

## 参考

- DeepSeek-V2 / V3 技术报告与开源仓库：`github.com/deepseek-ai`
- 本项目所有代码为独立实现，仅参考论文公开的技术思路
