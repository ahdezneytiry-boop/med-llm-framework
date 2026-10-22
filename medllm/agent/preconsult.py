"""智能预问诊 Agent —— 槽位填充状态机。

工作流程：
1. 按固定顺序询问缺失槽位（主诉 -> 病程 -> 既往史 -> 用药 -> 过敏 -> 家族史）；
2. 每轮用 LLM 把用户回复归纳进槽位；小模型归纳失败时回退到
   "否认/提供" 关键词启发式（"没有/无/否认" 记为否定，否则记为原话）；
3. 全程受安全护栏保护：一旦出现急危重症表述，立即转急诊建议并终止问诊；
4. 全部槽位处理完毕或达到最大轮次后，生成结构化问诊摘要
   （供医生接诊前快速浏览）。

会话状态为纯 dict（可序列化），服务层可跨请求保存/恢复，
实现多轮对话的无状态后端。
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from ..guard.guardrails import Guardrails
from ..llm.base import BaseLLM
from .slots import PRE_CONSULT_SLOTS

# 归纳槽位回复的提示词：让模型输出 JSON {"value": "..", "status": "answered|denied|unclear"}
SLOT_PARSE_SYSTEM = (
    "你是预问诊信息归纳助手。根据用户回复，把信息归纳进指定槽位。"
    '严格只输出一个 JSON：{"value": "归纳后的内容", "status": "answered|denied|unclear"}。'
    "用户明确否认（没有/无/未发现）时 status=denied、value=否认；"
    "回复与问题完全无关或无法归纳时 status=unclear；其余情况 status=answered。"
)

DENY_WORDS = ["没有", "否认", "无", "没得过", "没吃过", "不会", "不过敏"]


class PreConsultAgent:
    """预问诊 Agent。"""

    def __init__(self, llm: BaseLLM, guard: Optional[Guardrails] = None,
                 max_turns: int = 12, max_retry_per_slot: int = 2):
        self.llm = llm
        self.guard = guard or Guardrails()
        self.max_turns = max_turns
        self.max_retry = max_retry_per_slot

    # ------------------------------------------------------------------
    # 会话管理
    # ------------------------------------------------------------------
    def start(self, greeting: bool = True) -> Dict:
        """创建新会话，返回初始状态与开场语。"""
        state = {
            "turn": 0,                       # 已交互轮数
            "filled": {},                    # 已填槽位 {key: value}
            "retries": {},                   # 各槽位追问次数
            "slot_index": 0,                 # 当前询问的槽位下标
            "history": [],                   # 对话记录 [{role, content}]
            "summary": None,                 # 问诊摘要（结束时生成）
            "emergency": None,               # 急危重症提示（命中时设置）
            "finished": False,
        }
        reply = "您好，我是预问诊助手。在就诊前先帮您收集一些基本信息，"
        reply += "请如实描述，涉及急危重症请直接前往急诊。" if greeting else ""
        first_q = PRE_CONSULT_SLOTS[0].question
        state["history"].append({"role": "assistant", "content": first_q})
        return {"state": state, "reply": (reply + "\n\n" + first_q) if greeting else first_q}

    # ------------------------------------------------------------------
    def reply(self, state: Dict, user_text: str) -> Dict:
        """处理用户回复，返回 {reply, state, emergency, summary, finished}。

        这是 Agent 的核心循环：护栏检查 -> 槽位归纳 -> 推进状态机 -> 生成摘要。
        """
        # 1. 输入护栏：脱敏 + 恶意拦截 + 紧急识别
        guard_result = self.guard.check_input(user_text)
        if guard_result.action == "block":
            state["history"].append({"role": "user", "content": user_text})
            return {"reply": guard_result.reason, "state": state,
                    "emergency": None, "summary": None, "finished": False}
        if guard_result.action == "emergency":
            state["emergency"] = guard_result.reason
            state["finished"] = True
            state["history"].append({"role": "user", "content": user_text})
            return {"reply": guard_result.reason, "state": state,
                    "emergency": guard_result.reason, "summary": None, "finished": True}

        clean = guard_result.text
        state["turn"] += 1
        state["history"].append({"role": "user", "content": user_text})

        # 2. 归纳当前槽位
        slot = PRE_CONSULT_SLOTS[state["slot_index"]]
        value, status = self._parse_slot(clean, slot)
        retries = state["retries"].get(slot.key, 0)

        if status == "denied":
            state["filled"][slot.key] = "否认"
            state["slot_index"] += 1
            reply_text = "好的，已记录。"
        elif status == "answered":
            state["filled"][slot.key] = value
            state["slot_index"] += 1
            reply_text = f"已记录：{slot.name}——{value}。"
        elif retries < self.max_retry:
            # 归纳失败：追问一次
            state["retries"][slot.key] = retries + 1
            reply_text = slot.follow_up or f"不好意思没太理解，{slot.question}"
        else:
            # 追问次数用尽：跳过该槽位，标记"未提供"
            state["filled"][slot.key] = "未提供"
            state["slot_index"] += 1
            reply_text = "好的，这个问题先跳过。"

        # 3. 推进状态机：未完成则问下一槽位；完成或超轮次则生成摘要
        if state["slot_index"] < len(PRE_CONSULT_SLOTS) and state["turn"] < self.max_turns:
            next_slot = PRE_CONSULT_SLOTS[state["slot_index"]]
            reply_text += "\n\n" + next_slot.question
        else:
            state["finished"] = True
            state["summary"] = self.summarize(state)
            reply_text += "\n\n问诊完成，以下是本次问诊摘要：\n" + state["summary"]
            reply_text += "\n\n（以上信息仅用于辅助就诊，请以医生面诊为准。）"

        state["history"].append({"role": "assistant", "content": reply_text})
        return {"reply": reply_text, "state": state,
                "emergency": state["emergency"], "summary": state["summary"],
                "finished": state["finished"]}

    # ------------------------------------------------------------------
    # 槽位归纳
    # ------------------------------------------------------------------
    def _parse_slot(self, text: str, slot) -> tuple:
        """归纳用户回复：优先 LLM 结构化归纳，失败回退关键词启发式。"""
        if self.llm is not None:
            try:
                raw = self.llm.generate(
                    [{"role": "system", "content": SLOT_PARSE_SYSTEM},
                     {"role": "user", "content": f"槽位：{slot.name}\n用户回复：{text}"}],
                    max_new_tokens=128, temperature=0.2,
                )
                # 宽容解析 JSON
                start, end = raw.find("{"), raw.rfind("}")
                if start != -1 and end > start:
                    parsed = json.loads(raw[start:end + 1])
                    status = parsed.get("status", "unclear")
                    value = str(parsed.get("value", "")).strip()
                    if status in ("answered", "denied") and value:
                        return value[:120], status
            except Exception:
                pass   # 模型解析失败 -> 走启发式回退

        # 关键词启发式（小模型下的兜底策略）
        if any(w in text for w in DENY_WORDS) and len(text) <= 30:
            return "否认", "denied"
        if len(text) >= 2:
            return text.strip()[:120], "answered"
        return "", "unclear"

    # ------------------------------------------------------------------
    def summarize(self, state: Dict) -> str:
        """生成结构化问诊摘要（规则模板 + LLM 润色可选）。"""
        parts = []
        for slot in PRE_CONSULT_SLOTS:
            value = state["filled"].get(slot.key, "未采集")
            parts.append(f"{slot.name}：{value}")
        summary = "；".join(parts) + "。"
        # 尝试用 LLM 润色为一段通顺的叙述（失败则用模板原文）
        if self.llm is not None:
            try:
                polished = self.llm.generate(
                    [{"role": "system",
                      "content": "你是问诊摘要整理助手，把结构化信息整理成一段通顺的"
                                 "问诊摘要，不要添加任何原文没有的信息。"},
                     {"role": "user", "content": summary}],
                    max_new_tokens=200, temperature=0.3,
                )
                if polished.strip():
                    return polished.strip()
            except Exception:
                pass
        return summary
