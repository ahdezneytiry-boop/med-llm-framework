"""模型与训练配置。

架构参数命名与含义对齐 DeepSeek-V3 技术报告，但数值缩小到
可在普通 CPU 上演示的规模（约 17M 参数）。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict

import yaml


@dataclass
class ModelConfig:
    """模型结构配置（类 DeepSeek-V3 的精简版）。

    各字段含义：
    - hidden:              隐藏层维度（V3=7168）
    - n_layers:            Transformer 层数（V3=61）
    - n_heads:             注意力头数（V3=128）
    - kv_lora_rank:        KV 潜向量维度，MLA 缓存压缩核心参数（V3=512）
    - qk_rope_head_dim:    Q/K 中施加 RoPE 的解耦维度（V3=64）
    - n_routed_experts:    DeepSeekMoE 细粒度路由专家数（V3=256）
    - routed_top_k:        每 token 激活专家数（V3=8）
    - moe_inter_dim:       专家 FFN 中间维度（V3=2048，SwiGLU）
    - n_shared_experts:    始终激活的共享专家数（V3=1）
    """

    vocab_size: int = 4096
    hidden: int = 384
    n_layers: int = 8
    n_heads: int = 6
    kv_lora_rank: int = 64
    qk_rope_head_dim: int = 32
    n_routed_experts: int = 16
    routed_top_k: int = 4
    moe_inter_dim: int = 128
    n_shared_experts: int = 1
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_seq_len: int = 512
    tie_word_embeddings: bool = True

    # ------------------------------------------------------------------
    def __post_init__(self):
        """合法性校验：维度必须自洽。"""
        assert self.hidden % self.n_heads == 0, "hidden 必须能被 n_heads 整除"
        assert self.qk_rope_head_dim < self.hidden // self.n_heads, "RoPE 维度必须小于头维度"
        assert self.routed_top_k <= self.n_routed_experts, "top_k 不能超过专家总数"

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        """从字典构建（过滤未知字段，便于 yaml 直接传入）。"""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        """从 yaml 配置文件构建。"""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainConfig:
    """训练超参数（预训练与 SFT 共用）。"""

    data: str = "data/demo/pretrain_corpus.txt"
    out_dir: str = "checkpoints/pretrain"
    pretrained: str = ""              # SFT 时用于初始化的预训练检查点
    seq_len: int = 256
    batch_size: int = 4
    grad_accum: int = 2
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 20
    max_steps: int = 300
    log_interval: int = 10
    save_interval: int = 100
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        """从 yaml 配置文件构建。"""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})
