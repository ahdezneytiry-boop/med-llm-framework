"""RoPE（旋转位置编码）。

DeepSeek-V3 使用标准的 RoPE，且只施加在 Q/K 的"解耦维度"上
（见 attention.py 中 MLA 的说明）。
本实现采用复数形式：预计算旋转因子 freqs_cis，前向时直接复数相乘，
比实数 cos/sin 表实现更简洁、数值更稳定。
"""
from __future__ import annotations

import torch


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """预计算旋转因子表。

    Args:
        dim:  RoPE 施加的维度（必须是偶数）
        end:  预计算的最大位置数
        theta: 旋转基频（默认 10000，与原始 RoPE 论文一致）

    Returns:
        freqs_cis: 复数张量，形状 [end, dim // 2]
    """
    assert dim % 2 == 0, "RoPE 维度必须为偶数"
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)                                   # [end, dim/2]
    return torch.polar(torch.ones_like(freqs), freqs)               # 复数形式


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """对输入施加旋转位置编码。

    Args:
        x:         输入张量，形状 [..., T, dim]（最后一维为 RoPE 维度）
        freqs_cis: 旋转因子，形状 [T, dim//2]

    Returns:
        与 x 同形状的张量
    """
    # 视作复数（相邻两维一组），与旋转因子相乘后还原
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    out = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return out.type_as(x)
