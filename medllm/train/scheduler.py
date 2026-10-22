"""学习率调度：线性预热 + 余弦衰减。

DeepSeek-V3 预训练同样采用 warmup + cosine 的策略
（论文 2.1.3 节提到 lr warmup 与 cosine decay）。
"""
from __future__ import annotations

import math


def cosine_lr(step: int, warmup_steps: int, total_steps: int, min_ratio: float = 0.1) -> float:
    """返回当前步的学习率系数（相对峰值 lr 的比例）。

    Args:
        step:         当前步数（从 0 开始）
        warmup_steps: 预热步数（线性上升到峰值）
        total_steps:  总训练步数（余弦衰减的终点）
        min_ratio:    衰减下限（峰值 lr 的 10%，避免学习率降到 0）
    """
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
