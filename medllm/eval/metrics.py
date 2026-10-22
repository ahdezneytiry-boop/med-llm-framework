"""评测指标。

演示基准下用轻量、可解释的指标：
- char_f1：字符级 F1（中文生成评价常用，粒度比 BLEU 更适配短回答）；
- keyword_hit：关键词命中率（RAG 回答是否覆盖参考答案要点）；
- extraction_prf：抽取任务字段级 P / R / F1（集合匹配口径）。
"""
from __future__ import annotations

from typing import Dict, List


def _char_ngrams(text: str, n: int = 1) -> List[str]:
    text = "".join(text.split())
    return [text[i:i + n] for i in range(len(text) - n + 1)]


def char_f1(pred: str, ref: str) -> float:
    """字符级（unigram）F1。"""
    p, r = _char_ngrams(pred), _char_ngrams(ref)
    if not p or not r:
        return 0.0
    common = sum(1 for g in p if g in r)          # 多重集近似（演示级精度足够）
    precision = common / len(p)
    recall = common / len(r)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def keyword_hit(pred: str, keywords: List[str]) -> float:
    """关键词命中率：回答覆盖的参考要点比例。"""
    if not keywords:
        return 0.0
    return sum(1 for k in keywords if k in pred) / len(keywords)


def extraction_prf(pred: Dict[str, List[str]], gold: Dict[str, List[str]]):
    """抽取任务字段级 P/R/F1（集合匹配口径），返回 (p, r, f1, per_field)。"""
    tp = fp = fn = 0
    per_field = {}
    for field in gold:
        p_set, g_set = set(pred.get(field, [])), set(gold.get(field, []))
        tp_f = len(p_set & g_set)
        fp_f = len(p_set - g_set)
        fn_f = len(g_set - p_set)
        p_f = tp_f / (tp_f + fp_f) if tp_f + fp_f else 1.0
        r_f = tp_f / (tp_f + fn_f) if tp_f + fn_f else 1.0
        f1_f = 2 * p_f * r_f / (p_f + r_f) if p_f + r_f else 1.0
        per_field[field] = {"precision": round(p_f, 4), "recall": round(r_f, 4), "f1": round(f1_f, 4)}
        tp += tp_f
        fp += fp_f
        fn += fn_f
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f1, 4), per_field
