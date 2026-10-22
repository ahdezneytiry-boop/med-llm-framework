"""训练器 —— 预训练与 SFT 共用。

每步流程：
    前向 -> 损失 -> 反向（梯度累积）-> 梯度裁剪 -> AdamW 更新
    -> 余弦学习率调度 -> MoE 负载均衡偏置更新 -> 日志/检查点。

设备自动选择 CUDA > MPS > CPU；CPU 上以 fp32 训练（演示规模完全可行）。
检查点保存：模型权重 + 配置 + 优化器状态 + 训练进度（支持断点续训）。
"""
from __future__ import annotations

import json
import time
from itertools import cycle
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from ..core.config import ModelConfig, TrainConfig
from .scheduler import cosine_lr


def pick_device() -> str:
    """自动选择训练设备。"""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Trainer:
    def __init__(self, model, cfg: TrainConfig, device: Optional[str] = None):
        self.model = model
        self.cfg = cfg
        self.device = device or pick_device()
        self.model.to(self.device)
        # AdamW：主流 LLM 训练器（含 DeepSeek 社区实现）的标准选择
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95), eps=1e-8,
        )
        self.step = 0            # 已完成的优化步数
        self.tokens_seen = 0     # 累计处理 token 数

    # ------------------------------------------------------------------
    def train(self, loader, max_steps: Optional[int] = None) -> None:
        """执行训练主循环。

        Args:
            loader:    数据加载器（每个 batch 含 input_ids 与 labels）
            max_steps: 最大训练步数（默认取配置）
        """
        model, opt, cfg = self.model, self.optimizer, self.cfg
        max_steps = max_steps or cfg.max_steps
        model.train()

        pbar = tqdm(cycle(loader), desc="训练", total=max_steps * cfg.grad_accum)
        running_loss, t0 = 0.0, time.time()
        accum = 0

        # 用 cycle 无限循环数据加载器：训练时长完全由 max_steps 控制，
        # 而非一个 epoch 的批次数（避免小数据集提前结束训练）
        for batch in pbar:
            if self.step >= max_steps:
                break
            input_ids = batch["input_ids"].to(self.device)
            labels = batch.get("labels")
            if labels is not None:
                labels = labels.to(self.device)

            # ---- 前向 + 反向（梯度累积：等效扩大批大小）----
            _, _, loss = model(input_ids, targets=labels)
            (loss / cfg.grad_accum).backward()
            accum += 1

            if accum < cfg.grad_accum:
                continue

            # ---- 参数更新 ----
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 梯度裁剪防爆炸
            opt.step()
            opt.zero_grad(set_to_none=True)
            accum = 0
            self.step += 1
            self.tokens_seen += input_ids.numel()

            # ---- 学习率调度与 MoE 负载均衡偏置 ----
            lr = cfg.lr * cosine_lr(self.step, cfg.warmup_steps, max_steps)
            for g in opt.param_groups:
                g["lr"] = lr
            model.update_moe_bias()      # DeepSeek-V3 无辅助损失负载均衡（见 moe.py）

            running_loss += loss.item()

            # ---- 日志与检查点 ----
            if self.step % cfg.log_interval == 0:
                elapsed = max(time.time() - t0, 1e-6)
                pbar.set_postfix({
                    "loss": f"{running_loss / cfg.log_interval:.4f}",
                    "lr": f"{lr:.2e}",
                    "tok/s": f"{self.tokens_seen / elapsed:.0f}",
                })
                running_loss, t0 = 0.0, time.time()
            if cfg.save_interval and self.step % cfg.save_interval == 0:
                self.save(Path(cfg.out_dir) / f"ckpt_step{self.step}.pt")

        self.save(Path(cfg.out_dir) / "last.pt")

    # ------------------------------------------------------------------
    # 检查点读写
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        """保存检查点（权重 + 配置 + 优化器 + 进度），并在同目录落一份模型配置 json。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        model_cfg = self.model.cfg if hasattr(self.model, "cfg") else None
        torch.save({
            "model_state": self.model.state_dict(),
            "model_config": model_cfg.to_dict() if model_cfg else None,
            "optimizer_state": self.optimizer.state_dict(),
            "step": self.step,
            "tokens_seen": self.tokens_seen,
        }, path)
        if model_cfg:
            (path.parent / "model_config.json").write_text(
                json.dumps(model_cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )

    @staticmethod
    def load_model_weights(model, ckpt_path: str, device: Optional[str] = None) -> None:
        """从检查点恢复模型权重（忽略优化器与进度，用于 SFT 初始化/推理加载）。"""
        ckpt = torch.load(ckpt_path, map_location=device or "cpu")
        state = ckpt["model_state"]
        # 词表大小不一致时丢弃 embedding/lm_head（支持换词表续训）
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[提示] 权重部分加载：缺少 {len(missing)} 项、多余 {len(unexpected)} 项"
                  f"（多为词表大小变化所致，可忽略）")
