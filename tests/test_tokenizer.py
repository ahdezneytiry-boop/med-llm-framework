"""分词器单元测试：BPE 训练、往返编解码、对话模板。"""
import pytest

from medllm.core.tokenizer import ByteLevelBPE

# 小型测试语料（含中英文混合）
_CORPUS = [
    "高血压患者需要定期监测血压。",
    "糖尿病患者注意饮食控制。",
    "发热超过38.5度可以使用退热药。",
    "hypertension management requires monitoring",
    "患者否认药物过敏史。",
]


@pytest.fixture(scope="module")
def tokenizer():
    """在固定语料上训练一个小词表分词器（快）。"""
    tok = ByteLevelBPE()
    tok.train(_CORPUS, vocab_size=300)
    return tok


def test_vocab_size(tokenizer):
    """词表大小 = 特殊token + 256 字节 + 合并数。"""
    assert tokenizer.vocab_size == 262 + len(tokenizer.merges)
    assert tokenizer.vocab_size <= 300


def test_roundtrip_ascii_and_cjk(tokenizer):
    """中英文文本往返编解码无损。"""
    for text in _CORPUS:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_special_ids_distinct(tokenizer):
    """特殊 token id 互不相同且不与普通 token 冲突。"""
    ids = [tokenizer.special_id(s) for s in
           ["<|unk|>", "<|bos|>", "<|eos|>", "<|pad|>", "<|im_start|>", "<|im_end|>"]]
    assert len(set(ids)) == 6
    for t in _CORPUS:
        enc = tokenizer.encode(t)
        assert not any(i < 6 for i in enc)      # 正文不会出现特殊 token


def test_encode_shorter_than_raw_bytes(tokenizer):
    """BPE 应把常用词组合并，编码长度小于原始字节数（中文语料上）。"""
    text = "高血压患者需要定期监测血压。"
    assert len(tokenizer.encode(text)) < len(text.encode("utf-8"))


def test_chat_template(tokenizer):
    """对话模板包含角色标记，且可无损渲染。"""
    messages = [
        {"role": "system", "content": "你是演示助手。"},
        {"role": "user", "content": "你好"},
    ]
    rendered = tokenizer.apply_chat_template(messages)
    assert "<|im_start|>system" in rendered
    assert "<|im_start|>user" in rendered
    assert rendered.endswith("<|im_start|>assistant\n")   # 生成提示
    # 渲染结果作为纯文本应可无损往返编解码
    # （特殊 token 的 id 由 tokenize_chat 显式插入，不经过 encode 文本匹配）
    assert tokenizer.decode(tokenizer.encode(rendered)) == rendered


def test_save_load(tokenizer, tmp_path):
    """保存/加载后行为一致。"""
    p = tmp_path / "tok.json"
    tokenizer.save(str(p))
    tok2 = ByteLevelBPE.load(str(p))
    assert tok2.vocab_size == tokenizer.vocab_size
    for text in _CORPUS:
        assert tok2.encode(text) == tokenizer.encode(text)
