"""DeepSeekMoE —— 细粒度专家 + 共享专家 + 无辅助损失负载均衡。

参考 DeepSeek-V3 的 MoE 设计（论文第 2.1.2 节）：
1. 细粒度专家：把标准 FFN 专家切分为更多、更小的专家（V3 为 256 个，
   本实现 16 个），每 token 只激活 top-k（V3 为 8，这里为 4）。
   参数总量不变的前提下，每个专家更"专"，知识组合更灵活；
2. 共享专家（1 个）：始终激活，用于捕获各上下文通用的知识，
   缓解路由专家之间的参数冗余；
3. 无辅助损失负载均衡（auxiliary-loss-free）：
   传统 MoE（如 Mixtral/GShard）会给损失函数附加负载均衡惩罚项，
   但这可能损害模型质量；V3 改为给每个专家维护一个可学习的偏置 b_i：
   训练时统计各专家接收的 token 数，对"过载"专家把 b_i 减小 γ，
   对"欠载"专家把 b_i 增大 γ。b_i 只参与 top-k 选择、不参与门控权重，
   推理时无需额外开销即可自然均衡负载。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """单个专家 FFN：SwiGLU 结构（与 DeepSeek-V3 的 FFN 一致）。"""

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, inter_dim, bias=False)
        self.up_proj = nn.Linear(dim, inter_dim, bias=False)
        self.down_proj = nn.Linear(inter_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DeepSeekMoE(nn.Module):
    """细粒度 MoE 层（路由专家 + 共享专家）。"""

    def __init__(self, cfg):
        super().__init__()
        self.n_routed = cfg.n_routed_experts
        self.top_k = cfg.routed_top_k
        self.gamma = 0.001   # 负载均衡偏置修正步长（V3 论文取值）

        # 路由专家（细粒度）与共享专家
        self.experts = nn.ModuleList(
            [Expert(cfg.hidden, cfg.moe_inter_dim) for _ in range(self.n_routed)]
        )
        self.shared = nn.ModuleList(
            [Expert(cfg.hidden, cfg.moe_inter_dim) for _ in range(cfg.n_shared_experts)]
        )
        self.gate = nn.Linear(cfg.hidden, self.n_routed, bias=False)

        # 专家偏置：无辅助损失负载均衡的核心（只影响 top-k 选择，不影响权重）
        self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed))
        # 本训练步各专家接收的 token 数（每步清零，供训练器更新偏置）
        self.register_buffer("expert_token_counts", torch.zeros(self.n_routed), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, dim = x.shape
        xf = x.reshape(-1, dim)                                # [N, dim]

        # ---- 门控：V3 使用 sigmoid 而非 softmax ----
        scores = torch.sigmoid(self.gate(xf).float())

        # ---- top-k 选择：偏置参与排序（负载均衡），原始分数参与加权 ----
        scores_for_choice = scores + self.e_score_correction_bias
        topk_idx = torch.topk(scores_for_choice, self.top_k, dim=-1).indices     # [N, k]
        topk_weights = torch.gather(scores, 1, topk_idx)
        topk_weights = topk_weights / (topk_weights.sum(-1, keepdim=True) + 1e-8)  # 归一化

        # ---- 统计负载（供训练后更新偏置）----
        counts = torch.zeros(self.n_routed, device=x.device)
        counts.scatter_add_(0, topk_idx.reshape(-1),
                            torch.ones_like(topk_idx.reshape(-1), dtype=counts.dtype))
        self.expert_token_counts.add_(counts)

        # ---- 逐专家计算（稀疏激活：只计算被选中的 token-专家组合）----
        out = xf.new_zeros(xf.shape)
        for k in range(self.top_k):
            idx_k, w_k = topk_idx[:, k], topk_weights[:, k].unsqueeze(-1)
            for e in range(self.n_routed):
                sel = idx_k == e
                if sel.any():
                    out[sel] += w_k[sel] * self.experts[e](xf[sel])

        # ---- 共享专家：始终激活，直接相加 ----
        for expert in self.shared:
            out = out + expert(xf)
        return out.reshape(bsz, seq, dim)

    def update_bias(self) -> None:
        """训练每步调用：过载专家偏置 -γ，欠载专家偏置 +γ（V3 无辅助损失均衡）。"""
        avg = self.expert_token_counts.mean().clamp(min=1e-8)
        self.e_score_correction_bias.add_(
            self.gamma * torch.sign(avg - self.expert_token_counts)
        )
        self.expert_token_counts.zero_()
