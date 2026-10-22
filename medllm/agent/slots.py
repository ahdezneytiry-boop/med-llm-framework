"""预问诊槽位定义。

槽位（slot）是预问诊 Agent 需要采集的信息字段，顺序即提问顺序：
先问主诉（最核心），再依次补充病程、既往史、用药史、过敏史、家族史。
每个槽位附带默认提问语，Agent 按顺序逐槽询问，必要时追问一次。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Slot:
    """一个问诊槽位。"""
    key: str            # 槽位名（英文键，用于结构化摘要）
    name: str           # 槽位中文名（面向用户）
    question: str       # 默认提问语
    follow_up: str = "" # 追问语（用户回答含糊时使用）


# 标准问诊槽位顺序（演示级，非临床完整问诊流程）
PRE_CONSULT_SLOTS: List[Slot] = [
    Slot("chief_complaint", "主诉", "请问您今天主要哪里不舒服？",
         "能再具体描述一下不舒服的感觉吗？比如部位、性质（胀痛/刺痛/隐痛）。"),
    Slot("duration", "病程", "这个情况持续多久了？",
         "是从什么时候开始的呢？大致持续了几天、几周还是几个月？"),
    Slot("history", "既往史", "您有没有高血压、糖尿病等慢性病史？",
         "之前确诊过哪些疾病？正在治疗吗？"),
    Slot("medication", "用药史", "目前正在服用哪些药物？",
         "包括保健品在内，方便说一下药名和用量吗？"),
    Slot("allergy", "过敏史", "您对什么药物或食物过敏吗？",
         "比如青霉素、头孢类或其他药物？"),
    Slot("family", "家族史", "直系亲属中有没有类似的疾病史？",
         "比如父母、兄弟姐妹中有没有类似情况？"),
]
