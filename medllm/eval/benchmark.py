"""内置演示基准 —— 开箱即用的评测集。

三类评测（均不依赖外部数据）：
1. RAG 问答：问题 + 参考答案关键词，评测检索-生成链路的要点覆盖；
2. 信息抽取：带标注的合成病历，计算字段级 P/R/F1；
3. 预问诊：模拟一段标准问诊对话，评测槽位填充完整率与摘要质量。

说明：演示基准用于框架自检与教学，不等同于临床评测。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .metrics import extraction_prf, keyword_hit

# RAG 问答基准：与 knowledge/ 演示知识库对应
DEMO_QA = [
    {"question": "高血压的诊断标准是多少？", "keywords": ["140", "90"]},
    {"question": "高血压患者饮食上要注意什么？", "keywords": ["低盐", "饮食"]},
    {"question": "糖尿病如何诊断？", "keywords": ["血糖", "7.0"]},
    {"question": "糖化血红蛋白的意义是什么？", "keywords": ["糖化血红蛋白", "月"]},
    {"question": "成人发热多少度建议用退热药？", "keywords": ["38.5"]},
    {"question": "什么情况下的发热需要立即就医？", "keywords": ["就医", "呼吸困难"]},
    {"question": "转氨酶升高说明什么？", "keywords": ["肝", "损伤"]},
    {"question": "血脂偏高怎么办？", "keywords": ["饮食", "复查"]},
]

# 预问诊基准对话：标准回复，Agent 应能填满全部 6 个槽位
PRECONSULT_DIALOGUE = [
    "我这两天有点头晕，站起来的时候更明显。",
    "大概两天了。",
    "有高血压，五年了。",
    "在吃氨氯地平。",
    "没有过敏的。",
    "我父亲也有高血压。",
]


# ---------------------------------------------------------------------------
def run_rag_benchmark(rag) -> Dict:
    """RAG 问答评测：关键词命中率 + 来源引用情况。"""
    if rag is None:
        return {"error": "知识库未构建，请先运行 scripts/build_kb.py"}
    hits, with_sources = [], 0
    details = []
    for item in DEMO_QA:
        ans = rag.ask(item["question"])
        hit = keyword_hit(ans.answer, item["keywords"])
        hits.append(hit)
        if ans.sources:
            with_sources += 1
        details.append({"question": item["question"], "keyword_hit": round(hit, 4),
                        "has_sources": bool(ans.sources), "answer": ans.answer[:100]})
    return {"mean_keyword_hit": round(sum(hits) / len(hits), 4),
            "with_sources": f"{with_sources}/{len(DEMO_QA)}", "details": details}


def run_extraction_benchmark(extractor, labeled_path: str = "data/demo/extraction_labeled.jsonl",
                             max_samples: int = 0) -> Dict:
    """信息抽取评测：字段级 P/R/F1。

    Args:
        max_samples: 最多评测的样本数（0 表示全部）。CPU 演示下模型通道
                     逐条生成较慢，冒烟评测时建议限制在 5~10 条。
    """
    samples = []
    for line in Path(labeled_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            samples.append(json.loads(line))
    if not samples:
        return {"error": f"标注数据不存在：{labeled_path}，请先运行 scripts/make_demo_data.py"}
    if max_samples > 0:
        samples = samples[:max_samples]

    total_p = total_r = total_tp = total_fp = total_fn = 0
    per_field_agg: Dict[str, list] = {}
    for s in samples:
        pred = extractor.extract(s["text"])
        p, r, f1, per_field = extraction_prf(pred, s["entities"])
        # 反向聚合微指标
        for field, m in per_field.items():
            per_field_agg.setdefault(field, []).append(m["f1"])
        # 重新累计宏指标（与 metrics 口径一致）
        for field in s["entities"]:
            p_set, g_set = set(pred.get(field, [])), set(s["entities"][field])
            total_tp += len(p_set & g_set)
            total_fp += len(p_set - g_set)
            total_fn += len(g_set - p_set)
    tp, fp, fn = total_tp, total_fp, total_fn
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"samples": len(samples),
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "per_field_f1": {k: round(sum(v) / len(v), 4) for k, v in per_field_agg.items()}}


def run_preconsult_benchmark(agent) -> Dict:
    """预问诊评测：模拟标准对话，检查槽位填充完整率与是否正常产出摘要。"""
    first = agent.start()
    state = first["state"]
    reply_text = first["reply"]
    for user_text in PRECONSULT_DIALOGUE:
        out = agent.reply(state, user_text)
        state = out["state"]
        reply_text = out["reply"]
        if out["emergency"] or out["finished"]:
            break
    filled = {k: v for k, v in state["filled"].items() if v not in ("未提供", "未采集")}
    return {"turns": state["turn"], "slots_filled": len(filled),
            "total_slots": 6, "has_summary": bool(state["summary"]),
            "summary": state["summary"]}


def run_all(stack: Dict, max_extract_samples: int = 0) -> Dict:
    """依次运行三类评测，返回汇总报告。"""
    return {
        "rag": run_rag_benchmark(stack.get("rag")),
        "extraction": run_extraction_benchmark(stack["extractor"],
                                               max_samples=max_extract_samples),
        "preconsult": run_preconsult_benchmark(stack["agent"]),
    }
