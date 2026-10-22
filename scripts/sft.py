"""SFT（监督微调）脚本 —— 指令跟随。

以预训练检查点初始化权重，在 DeepSeek 风格 messages 数据上微调，
仅 assistant 部分参与损失计算。

运行：python scripts/sft.py --config configs/sft.yaml
"""
import argparse
import os
import random
import sys
from pathlib import Path

# 项目根目录加入模块搜索路径（脚本可直接运行，无需 pip install）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from medllm.core.config import ModelConfig, TrainConfig
from medllm.core.model import MedMiniLLM
from medllm.core.tokenizer import ByteLevelBPE
from medllm.data.dataset import build_sft_loader
from medllm.train.trainer import Trainer, pick_device


def main():
    parser = argparse.ArgumentParser(description="SFT 监督微调 MedMiniLLM")
    parser.add_argument("--config", default="configs/sft.yaml", help="SFT 配置")
    parser.add_argument("--model_config", default="configs/model_tiny.yaml", help="模型结构配置")
    parser.add_argument("--steps", type=int, default=0, help="覆盖配置中的最大步数")
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    mcfg = ModelConfig.from_yaml(args.model_config)
    if args.steps:
        cfg.max_steps = args.steps

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # 1. 分词器：从预训练目录复制（词表必须一致），
    #    并先于模型构建——模型 embedding 维度依赖实际词表大小
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_tok = Path(cfg.pretrained).parent / "tokenizer.json"
    dst_tok = out_dir / "tokenizer.json"
    if src_tok.exists():
        tokenizer = ByteLevelBPE.load(str(src_tok))
        tokenizer.save(str(dst_tok))
        print(f"[分词器] 已从 {src_tok} 复制到 {dst_tok}")
    elif dst_tok.exists():
        tokenizer = ByteLevelBPE.load(str(dst_tok))
    else:
        raise SystemExit("找不到分词器：请先运行预训练脚本生成 tokenizer.json")
    mcfg.vocab_size = tokenizer.vocab_size

    # 2. 模型：用预训练权重初始化（结构配置一致）
    device = pick_device()
    model = MedMiniLLM(mcfg)
    if cfg.pretrained and Path(cfg.pretrained).exists():
        Trainer.load_model_weights(model, cfg.pretrained, device=device)
        print(f"[权重] 已从 {cfg.pretrained} 初始化")
    else:
        print("[警告] 未提供预训练检查点，从随机权重开始 SFT")

    # 3. 数据 + 训练
    loader = build_sft_loader(cfg.data, tokenizer, cfg.seq_len, cfg.batch_size)
    print(f"[数据] {len(loader.dataset)} 条 SFT 样本")
    trainer = Trainer(model, cfg, device=device)
    trainer.train(loader, max_steps=cfg.max_steps)
    print(f"SFT 完成：{trainer.step} 步，检查点保存在 {out_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
