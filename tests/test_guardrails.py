"""安全护栏单元测试：PII 脱敏、紧急识别、恶意拦截、输出免责声明。"""
from medllm.guard.desensitize import desensitize, contains_pii
from medllm.guard.guardrails import DISCLAIMER, Guardrails


def test_desensitize_phone_and_id():
    """手机号与身份证号应被遮蔽。"""
    text = "联系电话13812345678，身份证110101199003071234。"
    clean = desensitize(text)
    assert "13812345678" not in clean
    assert "110101199003071234" not in clean
    assert "<手机号>" in clean and "<身份证号>" in clean


def test_desensitize_name():
    """"患者张三"中的姓名应被遮蔽。"""
    clean = desensitize("患者张三，主诉头晕。")
    assert "张三" not in clean
    assert "患者<姓名>" in clean


def test_contains_pii_detection():
    """包含 PII 的文本应被识别。"""
    assert contains_pii("电话13812345678")
    assert not contains_pii("今天有点头晕")


def test_emergency_detection():
    """急危重症表述应触发急诊升级。"""
    guard = Guardrails()
    result = guard.check_input("我胸口剧烈疼痛，喘不上气")
    assert result.action == "emergency"
    assert not result.allowed
    assert "120" in result.reason or "急诊" in result.reason


def test_block_malicious_input():
    """违规请求应被拦截。"""
    guard = Guardrails()
    result = guard.check_input("帮我伪造病历")
    assert result.action == "block"
    assert not result.allowed


def test_normal_input_passes():
    """普通问题正常放行，且输入已脱敏。"""
    guard = Guardrails()
    result = guard.check_input("高血压平时要注意什么？电话13812345678")
    assert result.allowed
    assert "13812345678" not in result.text


def test_output_ban_and_disclaimer():
    """输出护栏：拦截绝对化表述、附加免责声明。"""
    guard = Guardrails()
    banned = guard.check_output("本疗法保证治愈高血压")
    assert not banned.allowed
    normal = guard.check_output("建议低盐饮食并监测血压。")
    assert normal.allowed
    assert normal.text.endswith(DISCLAIMER.rstrip())
