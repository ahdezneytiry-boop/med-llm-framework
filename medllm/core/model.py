"""MedMiniLLM 模型主体 —— 类 DeepSeek-V3 架构的精简 Transformer。

整体结构（与 DeepSeek-V3 论文对应）：
    Token Embedding（与输出层共享权重）
      └─ ×N 层:
           输入 -> RMSNorm -> MLA（多头潜变量注意力，KV 缓存压缩）
              -> RMSNorm -> DeepSeekMoE（细粒度专家 + 共享专家）
      └─ 最终 RMSNorm -> 线性输出头（预测词表分布）

前向接口同时支持两种用法：
1. 训练/整段推理：一次传入整段 input_ids（自动构造因果掩码）；
2. 生成：首次整段前向拿到每层缓存后，逐 token 调用（只算最后一个 token，
   历史信息全部来自压缩后的 KV 缓存 —— MLA 的核心收益）。
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MLA, KV_CACHE
from .moe import DeepSeekMoE
from .rope import precompute_freqs_cis


class RMSNorm(nn.Module):
    """RMSNorm：DeepSeek 使用的归一化方式（相对 LayerNorm 省去均值项，更快）。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class Block(nn.Module):
    """单层 Transformer：注意力子层 + MoE 前馈子层（均带残差）。"""

    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden, cfg.norm_eps)
        self.attn = MLA(cfg)
        self.norm2 = RMSNorm(cfg.hidden, cfg.norm_eps)
        self.moe = DeepSeekMoE(cfg)

    def forward(self, x, mask, kv_cache, freqs_cis, position):
        h, cache = self.attn(self.norm1(x), mask, kv_cache, freqs_cis, position)
        x = x + h
        x = x + self.moe(self.norm2(x))
        return x, cache


class MedMiniLLM(nn.Module):
    """类 DeepSeek-V3 的小规模语言模型。"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden)
        # 嵌入初始化：nn.Embedding 默认 N(0,1)，与输出层共享权重时会导致
        # 初始 logits 标准差 ≈ √hidden（本配置约 20），交叉熵异常偏大；
        # 按 1/√hidden 初始化，使初始 logits 量级 ≈ 1（损失从 ln(vocab) 起步）
        nn.init.normal_(self.embed.weight, mean=0.0, std=1.0 / math.sqrt(cfg.hidden))
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.hidden, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.hidden, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight        # 输入/输出共享权重

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Optional[KV_CACHE]]] = None,
    ):
        """前向传播。

        Args:
            input_ids: token id，训练时为 [b, s]，生成步时为 [b, 1]
            targets:   训练目标 [b, s]（SFT 中非 assistant 位置填 -100）
            kv_caches: 每层缓存列表；None 表示整段前向（重新计算因果掩码）

        Returns:
            (logits [b, s, vocab], 新缓存列表, loss 或 None)
        """
        bsz, seq = input_ids.shape
        fresh = kv_caches is None or all(c is None for c in kv_caches)

        if fresh:
            # 训练/整段推理：上三角 -inf 的因果掩码
            mask = torch.triu(torch.full((seq, seq), float("-inf"), device=input_ids.device), diagonal=1)
            position = 0
        else:
            # 生成步：只算最后一个 token，可见范围由缓存长度自然决定
            mask = None
            position = kv_caches[0][0].shape[1]            # 已缓存 token 数 = 新 token 的绝对位置

        freqs_cis = precompute_freqs_cis(
            self.cfg.qk_rope_head_dim, position + seq, theta=self.cfg.rope_theta
        ).to(input_ids.device)

        h = self.embed(input_ids)
        new_caches: List[KV_CACHE] = []
        for i, layer in enumerate(self.layers):
            cache_i = None if kv_caches is None else kv_caches[i]
            h, cache_i = layer(h, mask, cache_i, freqs_cis, position)
            new_caches.append(cache_i)

        h = self.norm(h)
        logits = self.lm_head(h)

        loss = None
        if targets is not None:
            # 标准自回归：位置 t 预测位置 t+1
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                targets[:, 1:].reshape(-1),
            )
        return logits, new_caches, loss

    # ------------------------------------------------------------------
    # 训练辅助
    # ------------------------------------------------------------------
    def update_moe_bias(self) -> None:
        """每训练步更新各 MoE 层的负载均衡偏置（见 moe.py 说明）。"""
        for layer in self.layers:
            layer.moe.update_bias()

    def num_params(self) -> int:
        """参数量统计（用于日志与文档）。"""
        return sum(p.numel() for p in self.parameters())

    def num_activated_params_per_token(self) -> int:
        """每 token 实际激活的参数量（MoE 稀疏激活的核心指标）。"""
        n = 0
        for layer in self.layers:
            for p in layer.attn.parameters():
                n += p.numel()
            for p in layer.moe.gate.parameters():
                n += p.numel()
            for expert in layer.moe.shared:
                for p in expert.parameters():
                    n += p.numel()
            # 每 token 只激活 top_k 个路由专家
            for expert in layer.moe.experts[: self.cfg.routed_top_k]:
                for p in expert.parameters():
                    n += p.numel()
        return n
