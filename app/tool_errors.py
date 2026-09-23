"""工具执行后的两种决策：允许模型修正参数，或直接停止。"""


class RetryableError(ValueError):
    """模型修改参数后可能成功；最多给有限次修正机会。"""


class NonRetryableError(Exception):
    """修改参数也不应继续尝试，例如越过 workspace 安全边界。"""
