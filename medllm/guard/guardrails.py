"""安全与合规护栏 —— 医学场景的"三道防线"。

1. 输入护栏：先做 PII 脱敏，再拦截恶意/违规请求（如伪造病历、
   索要管制药品、自伤自残意图等），命中即拒绝并引导求助；
2. 紧急识别：识别急危重症表述（胸痛、呼吸困难、意识丧失等），
   立即给出急诊建议，而不是继续常规问答 —— 这是医学助手最重要的安全机制；
3. 输出护栏：拦截"保证治愈"等绝对化表述，并在每个回答后
   附加免责声明。

注意：所有规则均为演示级实现，真实部署必须经过执业医师与
合规评审，并建立人工兜底流程。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .desensitize import desensitize

# 急危重症/高危表述：命中即建议立即就医或呼叫急救
EMERGENCY_PATTERNS = [
    "胸痛", "胸闷伴大汗", "呼吸困难", "喘不上气", "意识丧失", "昏迷", "晕厥",
    "大出血", "呕血", "咯血", "抽搐", "剧烈腹痛", "自杀", "割腕", "服毒",
    "溺水", "触电", "严重烧伤", "高热惊厥", "窒息", "休克",
]

# 明确拒绝的恶意请求（演示级黑名单）
BLOCK_INPUT_PATTERNS = [
    "伪造病历", "造假病历", "开处方药", "卖我处方", "管制药品", "毒品",
    "教我自杀", "怎么杀人", "制造爆炸", "制造毒药",
]

# 输出中禁止的绝对化表述（医学上不允许承诺疗效）
OUTPUT_BAN_PATTERNS = ["保证治愈", "包治", "根治", "100%有效", "绝无副作用"]

DISCLAIMER = (
    "\n\n——\n"
    "【免责声明】本回答由演示系统生成，内容仅供健康知识参考，"
    "不构成诊断或治疗建议。如有不适，请及时前往正规医疗机构就诊。"
)


@dataclass
class GuardResult:
    """护栏判定结果。"""
    allowed: bool                 # 是否放行
    action: str                   # answer / emergency / block
    text: str = ""                # 处理后（脱敏/改写）的文本
    reason: str = ""              # 拦截/升级原因
    matches: List[str] = None     # 命中的模式


class Guardrails:
    """安全与合规护栏（演示级规则实现）。"""

    def __init__(self, emergency_patterns: List[str] = None,
                 block_patterns: List[str] = None,
                 ban_patterns: List[str] = None):
        self.emergency_patterns = emergency_patterns or EMERGENCY_PATTERNS
        self.block_patterns = block_patterns or BLOCK_INPUT_PATTERNS
        self.ban_patterns = ban_patterns or OUTPUT_BAN_PATTERNS

    # ------------------------------------------------------------------
    def check_input(self, text: str) -> GuardResult:
        """输入护栏：脱敏 -> 恶意请求拦截 -> 紧急识别。"""
        if not text or not text.strip():
            return GuardResult(False, "block", text, "输入为空")

        clean = desensitize(text)                     # 第 1 步：PII 脱敏

        blocked = [p for p in self.block_patterns if p in clean]
        if blocked:                                   # 第 2 步：恶意请求
            return GuardResult(False, "block", clean,
                               f"检测到违规请求（{', '.join(blocked)}），已拒绝回答。"
                               "如您正处于危机中，请拨打 110 / 120 或联系当地心理援助热线。",
                               blocked)

        emergencies = [p for p in self.emergency_patterns if p in clean]
        if emergencies:                               # 第 3 步：紧急识别
            return GuardResult(
                False, "emergency", clean,
                f"检测到可能的急危重症表述（{', '.join(emergencies)}）。"
                "请立即前往急诊就医或拨打 120，本系统无法替代急救处理。",
                emergencies,
            )
        return GuardResult(True, "answer", clean)

    # ------------------------------------------------------------------
    def check_output(self, text: str) -> GuardResult:
        """输出护栏：拦截绝对化表述，并附加免责声明。"""
        banned = [p for p in self.ban_patterns if p in text]
        if banned:
            return GuardResult(
                False, "block", text,
                f"输出包含违规表述（{', '.join(banned)}），已拦截。"
                "医学回答不允许承诺疗效。",
                banned,
            )
        return GuardResult(True, "answer", text + DISCLAIMER)

    def wrap_answer(self, text: str) -> str:
        """给最终回答附加免责声明（供 RAG/问答出口统一调用）。"""
        return text.rstrip() + DISCLAIMER
