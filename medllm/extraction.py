"""病历/报告信息抽取 —— 规则通道 + 模型通道双路合并。

设计动机：小规模自研模型的结构化输出能力有限，因此采用双通道：
1. 规则通道（必选）：基于医学词典的确定性匹配，
   保证症状/诊断/药品/剂量等基础字段始终可用、可复现；
2. 模型通道（可选）：结构化抽取提示词 + JSON 解析，
   由 LLM 补充规则难以覆盖的复杂语义（如"血糖控制不佳"的归纳）；
最终以"规则优先、模型补充"的策略合并两路结果。
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from .data.synthetic import DISEASE_DICT, DRUG_DICT, DRUG_DOSE, EXAM_DICT, SYMPTOM_DICT
from .llm.base import BaseLLM

# 抽取字段定义（与合成演示数据的标注口径一致）
FIELDS = ["症状", "诊断", "药品", "剂量", "检查项目", "异常指标"]

EXTRACT_SYSTEM = (
    "你是病历信息抽取助手。请从用户提供的病历文本中抽取以下字段，"
    "严格只输出一个 JSON 对象（不要输出任何其他文字）："
    '{"症状": [..], "诊断": [..], "药品": [..], "剂量": [..], '
    '"检查项目": [..], "异常指标": [..]}。'
    "文本中未出现的字段输出空列表。"
)

_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def parse_json_lenient(text: str) -> Optional[dict]:
    """宽容解析 LLM 输出中的 JSON（容忍前后杂讯与代码块包裹）。"""
    m = _JSON_BLOCK.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class MedicalExtractor:
    """病历信息抽取器。"""

    def __init__(self, llm: Optional[BaseLLM] = None, use_llm: bool = True):
        self.llm = llm if use_llm else None

    # ------------------------------------------------------------------
    def extract(self, text: str, max_new_tokens: int = 128) -> Dict[str, List[str]]:
        """抽取主入口：规则通道打底，模型通道补充，合并去重。"""
        result = self.rule_extract(text)
        if self.llm is not None:
            llm_result = self.llm_extract(text, max_new_tokens=max_new_tokens)
            if llm_result:
                result = self._merge(result, llm_result)
        return result

    # ------------------------------------------------------------------
    def rule_extract(self, text: str) -> Dict[str, List[str]]:
        """规则通道：词典匹配（确定性、零延迟）。"""
        result = {f: [] for f in FIELDS}
        seen: Dict[str, set] = {f: set() for f in FIELDS}

        def add(field: str, value: str):
            if value and value not in seen[field]:
                seen[field].add(value)
                result[field].append(value)

        for term in SYMPTOM_DICT:
            if term in text:
                add("症状", term)
        for term in DISEASE_DICT:
            if term in text:
                add("诊断", term)
        for drug in DRUG_DICT:
            if drug in text:
                add("药品", drug)
                dose = DRUG_DOSE.get(drug)
                if dose and dose in text:
                    add("剂量", dose)
        for exam in EXAM_DICT:
            if exam in text:
                add("检查项目", exam)
        # 异常指标：识别"(xxx)"注释或"偏高/偏低"字样
        for m in re.finditer(r"([\u4e00-\u9fffA-Za-z]+)(?:偏高|偏低|异常)", text):
            add("异常指标", m.group(0))
        for m in re.finditer(r"（(偏高|偏低|异常)）", text):
            # 提取注释前的指标名（演示级启发式）
            prefix = text[max(0, m.start() - 12):m.start()]
            pm = re.search(r"([\u4e00-\u9fff]{2,6})(?:\d|$)", prefix)
            if pm:
                add("异常指标", pm.group(1) + m.group(1))
        return result

    # ------------------------------------------------------------------
    def llm_extract(self, text: str, max_new_tokens: int = 128) -> Optional[Dict[str, List[str]]]:
        """模型通道：提示词 + JSON 解析，失败时静默回退到规则结果。"""
        try:
            raw = self.llm.generate(
                [{"role": "system", "content": EXTRACT_SYSTEM},
                 {"role": "user", "content": "请抽取：\n" + text}],
                max_new_tokens=max_new_tokens, temperature=0.2,
            )
            parsed = parse_json_lenient(raw)
            if not parsed:
                return None
            clean = {f: [] for f in FIELDS}
            for f in FIELDS:
                vals = parsed.get(f, [])
                if isinstance(vals, list):
                    clean[f] = [str(v) for v in vals if str(v).strip()]
                elif isinstance(vals, str) and vals.strip():
                    clean[f] = [vals.strip()]
            return clean
        except Exception:
            return None

    # ------------------------------------------------------------------
    @staticmethod
    def _merge(rule: Dict[str, List[str]], llm: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """合并策略：规则结果优先保序，模型补充规则缺失的字段值。"""
        merged = {}
        for f in FIELDS:
            out = list(rule.get(f, []))
            for v in llm.get(f, []):
                if v and v not in out:
                    out.append(v)
            merged[f] = out
        return merged
