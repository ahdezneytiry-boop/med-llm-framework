"""预训练脚本 —— 因果语言建模。

运行：python scripts/pretrain.py --config configs/pretrain.yaml \
                                --model_config configs/model_tiny.yaml
首次运行请先：python scripts/make_demo_data.py（生成语料）
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
from torch.utils.data import DataLoader

from medllm.core.config import ModelConfig, TrainConfig
from medllm.core.model import MedMiniLLM
from medllm.core.tokenizer import ByteLevelBPE
from medllm.data.dataset import PretrainDataset
from medllm.train.trainer import Trainer, pick_device


def main():
    parser = argparse.ArgumentParser(description="预训练 MedMiniLLM")
    parser.add_argument("--config", default="configs/pretrain.yaml", help="训练配置")
    parser.add_argument("--model_config", default="configs/model_tiny.yaml", help="模型结构配置")
    parser.add_argument("--tokenizer", default="", help="分词器路径（默认输出到 out_dir/tokenizer.json）")
    parser.add_argument("--steps", type=int, default=0, help="覆盖配置中的最大步数（调试用）")
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    mcfg = ModelConfig.from_yaml(args.model_config)
    if args.steps:
        cfg.max_steps = args.steps

    # 固定随机种子（可复现）
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # 1. 分词器：优先加载已有，否则在语料上现场训练并保存
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_path = Path(args.tokenizer) if args.tokenizer else out_dir / "tokenizer.json"
    if tok_path.exists():
        tokenizer = ByteLevelBPE.load(str(tok_path))
        print(f"[分词器] 已加载 {tok_path}（词表 {tokenizer.vocab_size}）")
    else:
        print("[分词器] 未找到，在训练语料上现场训练……")
        tokenizer = ByteLevelBPE()
        tokenizer.train([Path(cfg.data).read_text(encoding="utf-8")], vocab_size=mcfg.vocab_size)
        tokenizer.save(str(tok_path))
        print(f"[分词器] 训练完成并保存到 {tok_path}")

    mcfg.vocab_size = tokenizer.vocab_size    # 以分词器实际词表为准

    # 2. 模型 + 数据 + 训练器
    device = pick_device()
    model = MedMiniLLM(mcfg)
    print(f"[模型] 总参数 {model.num_params() / 1e6:.2f}M，"
          f"每 token 激活 {model.num_activated_params_per_token() / 1e6:.2f}M"
          f"（MoE 稀疏激活），设备 {device}")

    ds = PretrainDataset(cfg.data, tokenizer, cfg.seq_len)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    print(f"[数据] {len(ds)} 个训练块")

    trainer = Trainer(model, cfg, device=device)
    trainer.train(loader, max_steps=cfg.max_steps)
    print(f"预训练完成：{trainer.step} 步，检查点保存在 {out_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
