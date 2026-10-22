"""LocalLLM —— 默认后端：加载自研小模型。

从训练产物目录加载（model_config.json + tokenizer.json + last.pt），
组装 MedMiniLLM 与自训练 BPE 分词器，提供统一 generate 接口。
"""
from __future__ import annotations

from typing import List, Optional

import torch

from ..core.config import ModelConfig
from ..core.model import MedMiniLLM
from ..core.tokenizer import ByteLevelBPE
from ..infer.generate import GenerationConfig, chat
from ..train.trainer import pick_device
from .base import BaseLLM


class LocalLLM(BaseLLM):
    """自研小模型后端。"""

    name = "medmini-local"

    def __init__(self, model_dir: str, device: Optional[str] = None,
                 max_new_tokens: int = 256, temperature: float = 0.7):
        import json
        from pathlib import Path

        model_dir = Path(model_dir)
        # 1. 加载模型配置、分词器、权重
        cfg = ModelConfig.from_dict(
            json.loads((model_dir / "model_config.json").read_text(encoding="utf-8"))
        )
        self.tokenizer = ByteLevelBPE.load(str(model_dir / "tokenizer.json"))
        cfg.vocab_size = self.tokenizer.vocab_size     # 以分词器实际词表为准
        self.model = MedMiniLLM(cfg)
        ckpt = torch.load(str(model_dir / "last.pt"), map_location="cpu")
        self.model.load_state_dict(ckpt["model_state"], strict=False)

        self.device = device or pick_device()
        self.model.to(self.device).eval()
        self.gen_cfg = GenerationConfig(max_new_tokens=max_new_tokens, temperature=temperature)
        self.model_config = cfg

    def generate(
        self,
        messages: List[dict],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        cfg = GenerationConfig(
            max_new_tokens=max_new_tokens or self.gen_cfg.max_new_tokens,
            temperature=self.gen_cfg.temperature if temperature is None else temperature,
            top_p=self.gen_cfg.top_p,
            top_k=self.gen_cfg.top_k,
            repetition_penalty=self.gen_cfg.repetition_penalty,
        )
        return chat(self.model, self.tokenizer, messages, cfg, device=self.device)

    def count_tokens(self, text: str) -> int:
        """统计 token 数（服务层 usage 字段用）。"""
        return len(self.tokenizer.encode(text))
