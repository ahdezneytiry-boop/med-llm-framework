# 与 DeepSeek 开源技术方案的对照表

本框架参考 DeepSeek-V2 / DeepSeek-V3 技术报告与开源代码
（deepseek-ai/DeepSeek-V2、DeepSeek-V3 等仓库）实现了一个**小规模教学版**。
下表逐项对照，便于理解每个组件的出处与取舍。

| DeepSeek 技术点 | DeepSeek 论文/代码中的做法 | 本框架实现 | 备注 |
| --- | --- | --- | --- |
| **MLA** 多头潜变量注意力 | KV 压缩为低秩潜向量 c_KV（rank 512），缓存只存 c_KV + 解耦 K_pe，推理缓存约为 MHA 的 1/13；Q 侧同样降维-升维 | 完整实现同构结构：W_DKV→W_UK/W_UV/W_KR，Q 的 W_DQ→W_UQ，RoPE 只施加在解耦维度 | `medllm/core/attention.py`；rank 64，规模缩小 |
| **DeepSeekMoE** 细粒度专家 | 256 个路由专家、每 token 激活 8 个，专家更小更专；1 个共享专家 | 16 个专家、top-4 激活、1 个共享专家，SwiGLU 结构一致 | `medllm/core/moe.py` |
| **无辅助损失负载均衡**（auxiliary-loss-free） | 每个专家维护偏置 b_i，过载 -γ、欠载 +γ（γ=0.001），b_i 只参与 top-k 选择不参与权重 | 完全一致：`expert_token_counts` 统计 + `update_bias()` 每步修正 | 避免了 Mixtral/GShard 式辅助损失对质量的损害 |
| **sigmoid 门控 + top-k 归一化** | 门控用 sigmoid，选中专家权重按原始分数归一化 | 一致 | `moe.py` forward |
| **RMSNorm** | 全部归一化用 RMSNorm | 一致（含 MLA 内部低秩空间的 RMSNorm） | `model.py` |
| **RoPE** | 标准旋转位置编码，只作用于解耦维度 | 一致（复数实现） | `core/rope.py` |
| **对话数据格式** | 开源 SFT 数据为 `{"messages":[{"role","content"}]}` | 一致；chat template 用 `<\|im_start\|>/<\|im_end\|>` | `medllm/data/chat_format.py` |
| **SFT 损失掩码** | 仅 assistant 部分计算损失 | 一致（labels=-100） | `chat_format.py` |
| **预训练调度** | warmup + 余弦衰减（V3 用 WSD 变体） | warmup + cosine | `train/scheduler.py` |
| **OpenAI 兼容 API** | DeepSeek 官方 API 为 OpenAI 兼容协议 | 完全一致：`/v1/chat/completions`、`/v1/models` | `server/api.py` |

## 未实现（受规模与演示定位限制）

| 技术点 | 说明 |
| --- | --- |
| FP8 混合精度训练 | V3 的核心训练技巧；本框架 CPU 演示用 fp32 |
| 多 token 预测（MTP） | V3 提出的训练目标，可用作扩展练习 |
| DualPipe / EPLB / 专家并行 | 大规模分布式训练基础设施，小模型不需要 |
| GRPO / RLAIF 强化学习 | 演示规模未包含对齐阶段，README 中有扩展指引 |
| YaRN 长上下文扩展 | 当前 max_seq_len=512，够演示使用 |

## 进一步阅读

- DeepSeek-V2 论文：*DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*（MLA、DeepSeekMoE）
- DeepSeek-V3 论文：*DeepSeek-V3 Technical Report*（无辅助损失负载均衡、FP8、MTP）
- 开源仓库：`github.com/deepseek-ai/DeepSeek-V2`、`github.com/deepseek-ai/DeepSeek-V3`
