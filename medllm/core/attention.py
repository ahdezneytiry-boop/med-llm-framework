"""MLA —— 多头潜变量注意力（Multi-head Latent Attention）。

DeepSeek-V2/V3 的核心创新，动机与做法：
1. 动机：标准 MHA 的 KV 缓存随序列长度线性增长，长文本推理显存压力大；
2. 做法：每层的 KV 不直接缓存，而是先压缩成一个低秩潜向量 c_KV
   （维度 kv_lora_rank << n_heads × head_dim），缓存只保存 c_KV 与
   施加 RoPE 的解耦键 K_pe，使推理缓存约为标准 MHA 的 1/10（V3 约 1/13）；
3. Q 侧同样先降维再升维（W_DQ -> W_UQ），减少激活内存；
4. RoPE 只施加在"解耦维度"（q_pe / k_pe），与"非 RoPE 维度"拼接后计算注意力。

本实现与 V3 论文图 2 对应：
    c_Q = RMSNorm(W_DQ h)          q = W_UQ c_Q -> (q_nope, q_pe)
    c_KV = RMSNorm(W_DKV h)        (k_nope, v) = W_UKV c_KV   k_pe = RoPE(W_KR c_KV)
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rotary_emb


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """函数式 RMSNorm（对 MLA 内部的低秩空间做归一化用）。"""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


# 每层缓存的类型：(c_KV 潜向量 [b, s, kv_lora_rank], K_pe [b, s, 1, rope_dim])
KV_CACHE = Tuple[torch.Tensor, torch.Tensor]


class MLA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.hidden // cfg.n_heads
        self.qk_nope_head_dim = self.head_dim - cfg.qk_rope_head_dim  # 非 RoPE 维度
        self.qk_rope_head_dim = cfg.qk_rope_head_dim                  # RoPE 解耦维度
        self.kv_lora_rank = cfg.kv_lora_rank                          # 潜向量维度
        self.scale = self.head_dim ** -0.5                            # 注意力缩放

        # ---- Q 通路：降维 -> 升维（V3 式，节省激活内存）----
        self.wq_a = nn.Linear(cfg.hidden, self.n_heads * self.qk_nope_head_dim, bias=False)
        self.wq_b = nn.Linear(self.n_heads * self.qk_nope_head_dim, self.n_heads * self.head_dim, bias=False)

        # ---- KV 通路：压缩为潜向量 c_KV ----
        self.wkv_a = nn.Linear(cfg.hidden, self.kv_lora_rank, bias=False)
        self.kv_a_norm = nn.RMSNorm(self.kv_lora_rank, eps=cfg.norm_eps)
        self.wkv_b = nn.Linear(
            self.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.head_dim),  # k_nope + v
            bias=False,
        )

        # ---- 解耦 RoPE 键：从潜向量直接生成 K_pe ----
        self.w_kr = nn.Linear(self.kv_lora_rank, self.qk_rope_head_dim, bias=False)

        self.wo = nn.Linear(self.n_heads * self.head_dim, cfg.hidden, bias=False)

    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KV_CACHE] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        position: int = 0,
    ):
        """MLA 前向。

        Args:
            h:         输入 [b, s, hidden]
            mask:      加性因果掩码 [s, s]（训练时）或 None（生成步）
            kv_cache:  上一步的缓存 (c_KV, K_pe)，生成时传入
            freqs_cis: RoPE 旋转因子表 [position+seq, rope_dim/2]
            position:  本段 token 的起始绝对位置（生成时为缓存长度）

        Returns:
            (输出 [b, s, hidden], 新缓存)
        """
        bsz, seq, _ = h.shape

        # ---- Q：降维 -> RMSNorm -> 升维 -> 拆分为 nope/pe（只处理当前 token）----
        q = self.wq_b(_rms_norm(self.wq_a(h)))
        q = q.view(bsz, seq, self.n_heads, self.head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # ---- KV：先压缩当前 token 为潜向量，再与缓存拼接 ----
        c_kv_new = self.kv_a_norm(self.wkv_a(h))
        if kv_cache is not None:
            c_kv = torch.cat([kv_cache[0], c_kv_new], dim=1)   # 完整潜向量（全部位置）
        else:
            c_kv = c_kv_new

        # ---- 从"完整潜向量"升维出 K_nope / V（MLA 的核心：缓存只存潜向量，
        #      使用时再升维，覆盖包括历史在内的全部位置）----
        kv_out = self.wkv_b(c_kv)
        k_nope, v = torch.split(
            kv_out,
            [self.n_heads * self.qk_nope_head_dim, self.n_heads * self.head_dim],
            dim=-1,
        )
        k_nope = k_nope.view(bsz, -1, self.n_heads, self.qk_nope_head_dim)
        v = v.view(bsz, -1, self.n_heads, self.head_dim)

        # ---- 解耦 K_pe：只对当前 token 施加 RoPE，再与缓存拼接 ----
        k_pe = self.w_kr(c_kv_new).view(bsz, seq, 1, self.qk_rope_head_dim)  # 所有头共享
        if freqs_cis is not None:
            pos_slice = freqs_cis[position:position + seq]
            q_pe = apply_rotary_emb(q_pe.transpose(1, 2), pos_slice).transpose(1, 2)   # [b,s,h,d]
            k_pe = apply_rotary_emb(k_pe.transpose(1, 2), pos_slice).transpose(1, 2)   # [b,s,1,d]
        if kv_cache is not None:
            k_pe = torch.cat([kv_cache[1], k_pe], dim=1)
        new_cache: KV_CACHE = (c_kv, k_pe)

        # ---- 拼接 nope/pe，计算注意力 ----
        q = torch.cat([q_nope, q_pe], dim=-1).transpose(1, 2)                          # [b,h,s,d]
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1).transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores.float(), dim=-1).type_as(q)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, seq, -1)
        return self.wo(out), new_cache
