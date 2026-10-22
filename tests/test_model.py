"""模型单元测试：前向形状、因果掩码、训练收敛、KV 缓存一致性、生成。"""
import torch
import pytest

from medllm.core.config import ModelConfig
from medllm.core.model import MedMiniLLM
from medllm.core.tokenizer import ByteLevelBPE
from medllm.infer.generate import GenerationConfig, generate


def tiny_cfg(**overrides) -> ModelConfig:
    """构造快速测试用的小模型配置。"""
    base = dict(
        vocab_size=300, hidden=64, n_layers=2, n_heads=4,
        kv_lora_rank=16, qk_rope_head_dim=8,
        n_routed_experts=8, routed_top_k=2, moe_inter_dim=32,
        n_shared_experts=1, max_seq_len=128,
    )
    base.update(overrides)
    return ModelConfig(**base)


def tiny_tokenizer() -> ByteLevelBPE:
    tok = ByteLevelBPE()
    tok.train(["高血压患者需要定期监测血压。", "hello world"], vocab_size=300)
    return tok


def test_forward_shapes_and_loss():
    """前向输出形状正确，损失为有限标量。"""
    cfg = tiny_cfg()
    model = MedMiniLLM(cfg)
    ids = torch.randint(1, cfg.vocab_size, (2, 16))
    logits, caches, loss = model(ids, targets=ids)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert len(caches) == cfg.n_layers
    assert torch.isfinite(loss)


def test_causal_mask():
    """因果性：位置 t 的 logits 不依赖位置 t+1 的输入。"""
    cfg = tiny_cfg()
    model = MedMiniLLM(cfg)
    ids = torch.randint(1, cfg.vocab_size, (1, 8))
    logits, _, _ = model(ids)
    ids2 = ids.clone()
    ids2[:, 7] = (ids2[:, 7] + 1) % cfg.vocab_size   # 只改最后一个 token
    logits2, _, _ = model(ids2)
    # 前面位置的预测不受最后一个 token 影响。
    # 注：atol=1e-4 用于容纳 CPU BLAS 多线程的浮点归约不确定性
    #（真实信息泄漏的量级是 O(1)，远大于该容差）
    assert torch.allclose(logits[:, :6], logits2[:, :6], atol=1e-4)


def test_training_step_reduces_loss():
    """几步梯度下降后损失下降（模型可被训练）。"""
    cfg = tiny_cfg()
    model = MedMiniLLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ids = torch.randint(1, cfg.vocab_size, (2, 16))
    losses = []
    for _ in range(3):
        opt.zero_grad()
        _, _, loss = model(ids, targets=ids)
        loss.backward()
        opt.step()
        model.update_moe_bias()        # 顺带验证 MoE 偏置更新不报错
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_kv_cache_step_consistency():
    """关键正确性检查：带缓存单步前向 == 扩展序列整段前向的末位 logits。

    语义说明：整段前向在位置 t 输出的是"token t+1 的预测"，
    因此增量步（缓存 + 新 token）等价于对"原序列追加该 token"后的
    整段前向在最后一个位置上的结果。
    """
    cfg = tiny_cfg()
    model = MedMiniLLM(cfg)
    ids = torch.randint(1, cfg.vocab_size, (1, 10))
    logits_full, caches, _ = model(ids)                       # 整段前向，填充缓存

    extended = torch.cat([ids, ids[:, -1:]], dim=1)           # 原序列 + 追加 token
    logits_ext, _, _ = model(extended)                        # 扩展序列整段前向

    logits_step, _, _ = model(ids[:, -1:], kv_caches=caches)  # 缓存单步前向
    assert torch.allclose(logits_ext[:, -1], logits_step[:, -1], atol=1e-4)


def test_generate_produces_valid_ids():
    """生成流程：输出长度受限、token id 合法、可解码。"""
    cfg = tiny_cfg()
    model = MedMiniLLM(cfg)
    tok = tiny_tokenizer()
    cfg.vocab_size = tok.vocab_size
    # 重建模型以匹配词表（嵌入层尺寸）
    model = MedMiniLLM(cfg)
    ids = generate(model, tok, torch.tensor([[tok.bos_id]]),
                   GenerationConfig(max_new_tokens=8, temperature=0.0))
    assert 1 <= len(ids) <= 1 + 8
    assert all(0 <= i < cfg.vocab_size for i in ids)
    text = tok.decode(ids)
    assert isinstance(text, str)


def test_moe_sparse_activation():
    """MoE 每 token 只激活 top-k 个路由专家（负载统计正确）。"""
    cfg = tiny_cfg()
    model = MedMiniLLM(cfg)
    ids = torch.randint(1, cfg.vocab_size, (4, 8))
    model(ids)
    for layer in model.layers:
        counts = layer.moe.expert_token_counts
        assert counts.sum() == 4 * 8 * cfg.routed_top_k   # 每个 token 恰好 top-k 次
