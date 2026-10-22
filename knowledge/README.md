# 演示知识库说明

> ⚠️ **重要声明**：本目录下的所有文档均为**演示用简化科普内容**，
> 用于测试 RAG 检索链路，**不构成任何诊断或治疗建议**，
> 更不能替代执业医师的诊疗意见。真实医疗问题请咨询正规医疗机构。

文档列表：

| 文件 | 主题 |
| --- | --- |
| hypertension.md | 高血压的简化科普（演示） |
| diabetes.md | 糖尿病的简化科普（演示） |
| fever.md | 发热的简化科普（演示） |
| checkup_interpretation.md | 体检报告常见指标解读（演示） |

使用方式：

```bash
# 把本目录构建为向量知识库，供 RAG 问答检索
python scripts/build_kb.py --kb knowledge --out data/vector_store
```
