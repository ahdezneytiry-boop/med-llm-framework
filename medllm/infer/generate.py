"""自回归文本生成（带 KV 缓存 + MLA 压缩缓存）。

流程：
1. 首次对完整 prompt 做整段前向，得到每层的压缩 KV 缓存
   （MLA 只存潜向量 c_KV 与解耦 K_pe，缓存占用远小于标准 MHA）；
2. 之后每步只送入最后一个 token，前向仅计算增量部分；
3. 采样策略：温度 + top-p（核采样）+ top-k + 重复惩罚；
4. 遇到 <|im_end|>/<|eos|> 或达到最大长度时停止。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from ..core.model import MedMiniLLM
from ..core.tokenizer import ByteLevelBPE


@dataclass
class GenerationConfig:
    """生成采样参数。"""
    max_new_tokens: int = 256
    temperature: float = 0.7      # 0 等价于贪心
    top_p: float = 0.9            # 核采样累积概率阈值
    top_k: int = 0                # 0 表示不启用 top-k
    repetition_penalty: float = 1.05


def apply_repetition_penalty(logits: torch.Tensor, generated: List[int], penalty: float) -> torch.Tensor:
    """对已生成 token 施加重复惩罚（logits 为正则除、为负则乘）。"""
    if penalty == 1.0 or not generated:
        return logits
    logits = logits.clone()
    for t in set(generated):
        idx = torch.tensor(t, device=logits.device)
        v = logits[0, t]
        logits[0, t] = torch.where(v > 0, v / penalty, v * penalty)
    return logits


@torch.no_grad()
def sample_token(logits: torch.Tensor, temperature: float, top_p: float, top_k: int) -> int:
    """按温度/top-p/top-k 从 logits 中采样一个 token。"""
    logits = logits / max(temperature, 1e-6)
    if top_k > 0:                                    # top-k：截断到概率最高的 k 个
        k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, k)
        logits[logits < v[:, [-1]]] = float("-inf")
    if top_p < 1.0:                                  # top-p：核采样
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits.float(), -1)
        cum = torch.cumsum(probs, -1)
        mask = cum - probs > top_p                   # 保留累积不超过 top_p 的部分
        probs[mask] = 0.0
        probs = probs / probs.sum(-1, keepdim=True)
        idx = torch.multinomial(probs, 1)
        return sorted_idx.gather(-1, idx)[0].item()
    return torch.multinomial(torch.softmax(logits.float(), -1), 1)[0].item()


@torch.no_grad()
def generate(
    model: MedMiniLLM,
    tokenizer: ByteLevelBPE,
    input_ids: torch.Tensor,
    cfg: GenerationConfig,
    device: Optional[str] = None,
) -> List[int]:
    """自回归生成，返回完整 token id 列表（含 prompt 部分）。"""
    device = device or input_ids.device
    model.eval()
    input_ids = input_ids.to(device)

    # 首次整段前向：填充全部缓存
    logits, caches, _ = model(input_ids)
    next_logits = logits[:, -1]
    output: List[int] = input_ids[0].tolist()
    stop_ids = set(tokenizer.stop_ids)

    for _ in range(cfg.max_new_tokens):
        logits = apply_repetition_penalty(next_logits, output, cfg.repetition_penalty)
        next_id = sample_token(logits, cfg.temperature, cfg.top_p, cfg.top_k)
        if next_id in stop_ids:
            break
        output.append(next_id)
        # 增量前向：只算最后一个 token，历史来自压缩缓存
        step_ids = torch.tensor([[next_id]], device=device)
        logits, caches, _ = model(step_ids, kv_caches=caches)
        next_logits = logits[:, -1]
    return output


def chat(
    model: MedMiniLLM,
    tokenizer: ByteLevelBPE,
    messages: List[dict],
    cfg: Optional[GenerationConfig] = None,
    device: Optional[str] = None,
) -> str:
    """对话接口：messages -> 模板渲染 -> 编码 -> 生成 -> 解码 assistant 部分。"""
    cfg = cfg or GenerationConfig()
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    prompt_ids = tokenizer.encode_whole(prompt, add_bos=True)
    ids = generate(model, tokenizer, torch.tensor([prompt_ids]), cfg, device=device)
    # 只解码模型新生成的部分（assistant 的正文）
    return tokenizer.decode(ids[len(prompt_ids):]).strip()
