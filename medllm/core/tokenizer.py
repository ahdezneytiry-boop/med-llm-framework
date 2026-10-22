"""字节级 BPE 分词器（自训练，不依赖第三方 tokenizer 库）。

设计参考 DeepSeek 与 GPT 系列的 BPE 方案：
1. 词表以 256 个 UTF-8 字节为起点，通过"统计频次最高的相邻对并合并"
   迭代生成子词，训练完的词表 = 特殊 token + 256 字节 + 若干合并规则；
2. 中文按"单字预切分"（每个汉字/标点为一个基本单元），这是中文 BPE
   的常见做法，避免把整句当作一个不可再分的单元；
3. 对话模板与 DeepSeek 一致：以 <|im_start|>/<|im_end|> 包裹角色，
   system 在最前，模型在 <|im_start|>assistant 之后开始生成。

训练算法（增量统计版）：
- 每个基本单元维护一个 token id 列表；
- 维护全局"相邻对 -> 频次"计数表；
- 每轮选出频次最高的对，只更新包含该对的单元（避免全量重扫）。
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 预切分正则：
#   - ASCII 连续词（英文单词/数字）
#   - 单个 CJK 字符 / 单个全角标点
#   - 空白串
#   - 其他任意单字符
# ---------------------------------------------------------------------------
_CJK = r"\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef"
_UNIT_PAT = re.compile(rf"[A-Za-z0-9]+|[{_CJK}]|\s+|.", re.S)

# 特殊 token 定义（与主流对话模型 chat template 约定一致）
SPECIAL_TOKENS: Tuple[str, ...] = (
    "<|unk|>", "<|bos|>", "<|eos|>", "<|pad|>", "<|im_start|>", "<|im_end|>",
)
# id 空间分配：0..5 特殊 token，6..261 为 256 个字节，262 起为合并出的子词
_N_SPECIAL = len(SPECIAL_TOKENS)
_BYTE_OFFSET = _N_SPECIAL          # 字节 b 对应 id = b + _BYTE_OFFSET
_MERGE_OFFSET = _N_SPECIAL + 256   # 第 i 条合并规则对应 id = i + _MERGE_OFFSET


def _split_units(text: str) -> List[str]:
    """按预切分正则把文本切成基本单元列表。"""
    return _UNIT_PAT.findall(text)


class ByteLevelBPE:
    """自训练字节级 BPE 分词器。"""

    def __init__(self):
        self.merges: List[Tuple[int, int]] = []      # 合并规则列表（id 对，按序）
        self.merge_rank: Dict[Tuple[int, int], int] = {}  # 合并对 -> 序号
        self.specials: Dict[str, int] = {s: i for i, s in enumerate(SPECIAL_TOKENS)}

    # ------------------------------------------------------------------
    # 基本属性
    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        """当前词表大小。"""
        return _MERGE_OFFSET + len(self.merges)

    @property
    def pad_id(self) -> int:
        return self.specials["<|pad|>"]

    @property
    def bos_id(self) -> int:
        return self.specials["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self.specials["<|eos|>"]

    @property
    def stop_ids(self) -> List[int]:
        """生成时的终止 token。"""
        return [self.eos_id, self.specials["<|im_end|>"]]

    def special_id(self, name: str) -> int:
        return self.specials[name]

    def special_text(self, name: str) -> str:
        return name

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------
    def train(self, texts: Iterable[str], vocab_size: int = 4096) -> "ByteLevelBPE":
        """在语料上训练合并规则。

        Args:
            texts:      文本迭代器（可以是单行列表或整篇文本）
            vocab_size: 目标词表大小（包含特殊 token 与字节）
        """
        # 1. 统计基本单元频率（按单元去重统计，BPE 训练的标准加速手段）
        unit_freq: Counter[str] = Counter()
        for text in texts:
            unit_freq.update(_split_units(text))

        # 2. 每个单元初始化为字节 id 列表（byte + _BYTE_OFFSET）
        units: Dict[str, List[int]] = {
            u: [b + _BYTE_OFFSET for b in u.encode("utf-8")] for u in unit_freq
        }

        # 3. 初始全局"相邻对 -> 频次"表
        pair_count: Counter[Tuple[int, int]] = Counter()
        for u, toks in units.items():
            w = unit_freq[u]
            for i in range(len(toks) - 1):
                pair_count[(toks[i], toks[i + 1])] += w

        # 4. 迭代合并：每次选频次最高的对，只增量更新受影响的单元
        self.merges = []
        n_merges = vocab_size - _MERGE_OFFSET
        for _ in range(n_merges):
            if not pair_count:
                break
            best = max(pair_count.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
            new_id = _MERGE_OFFSET + len(self.merges)
            self.merges.append(best)

            # 找出所有包含该对的单元，更新其 token 列表与计数
            affected = [
                u for u, toks in units.items()
                if any(best == (toks[i], toks[i + 1]) for i in range(len(toks) - 1))
            ]
            for u in affected:
                toks, w = units[u], unit_freq[u]
                for i in range(len(toks) - 1):            # 先扣除旧计数
                    pair_count[(toks[i], toks[i + 1])] -= w
                # 单遍扫描应用合并
                new_toks: List[int] = []
                i = 0
                while i < len(toks):
                    if i < len(toks) - 1 and (toks[i], toks[i + 1]) == best:
                        new_toks.append(new_id)
                        i += 2
                    else:
                        new_toks.append(toks[i])
                        i += 1
                units[u] = new_toks
                for i in range(len(new_toks) - 1):        # 再加回新计数
                    pair_count[(new_toks[i], new_toks[i + 1])] += w

            pair_count = Counter({k: v for k, v in pair_count.items() if v > 0})

        # 构建合并对 -> 序号 的查找表（encode 用）
        self.merge_rank = {pair: i for i, pair in enumerate(self.merges)}
        return self

    # ------------------------------------------------------------------
    # 编解码
    # ------------------------------------------------------------------
    def _encode_unit(self, unit: str) -> List[int]:
        """对单个基本单元做贪婪 BPE 编码。

        算法：反复寻找"合并序号最小"的相邻对并合并，直到没有可合并的对。
        合并序号最小 = 优先应用训练时出现频次更高的规则（BPE 标准语义）。
        单元长度通常很小（几十 token），O(n^2) 扫描完全可接受。
        """
        tokens = [b + _BYTE_OFFSET for b in unit.encode("utf-8")]
        while len(tokens) > 1:
            best_rank, best_i = float("inf"), -1
            for i in range(len(tokens) - 1):
                r = self.merge_rank.get((tokens[i], tokens[i + 1]), float("inf"))
                if r < best_rank:
                    best_rank, best_i = r, i
            if best_i < 0:
                break
            tokens[best_i:best_i + 2] = [_MERGE_OFFSET + best_rank]
        return tokens

    def encode(self, text: str) -> List[int]:
        """把文本编码为 token id 列表（不含特殊 token）。"""
        ids: List[int] = []
        for unit in _split_units(text):
            ids.extend(self._encode_unit(unit))
        return ids

    def encode_whole(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """整段文本编码，可选加 <bos>/<eos>。"""
        ids = self.encode(text)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        """把 token id 列表还原为文本。

        合并规则以"id 对"存储，解码时用栈迭代展开（避免深递归），
        展开到底层字节后统一按 UTF-8 还原。
        """
        out_bytes = bytearray()
        for tid in ids:
            if tid < _BYTE_OFFSET:
                continue                                # 特殊 token，解码时丢弃
            stack = [tid]
            while stack:
                cur = stack.pop()
                if cur < _MERGE_OFFSET:                 # 字节
                    out_bytes.append(cur - _BYTE_OFFSET)
                else:                                   # 合并子词，拆回两个部分
                    a, b = self.merges[cur - _MERGE_OFFSET]
                    stack.append(b)
                    stack.append(a)
        return out_bytes.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # 对话模板（DeepSeek 风格）
    # ------------------------------------------------------------------
    def apply_chat_template(self, messages: List[dict], add_generation_prompt: bool = True) -> str:
        """把 messages 渲染为文本。

        messages 格式（DeepSeek 开源 SFT 数据同款）：
            [{"role": "system" | "user" | "assistant", "content": "..."}, ...]
        """
        im_s, im_e = "<|im_start|>", "<|im_end|>"
        parts: List[str] = []
        for m in messages:
            parts.append(f"{im_s}{m['role']}\n{m['content']}{im_e}\n")
        if add_generation_prompt:
            parts.append(f"{im_s}assistant\n")
        return "".join(parts)

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """保存分词器为 json（合并规则 + 特殊 token 映射）。"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"merges": self.merges, "specials": self.specials}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "ByteLevelBPE":
        """从 json 加载分词器。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.merges = [tuple(m) for m in data["merges"]]
        tok.merge_rank = {pair: i for i, pair in enumerate(tok.merges)}
        tok.specials = {k: int(v) for k, v in data["specials"].items()}
        return tok
