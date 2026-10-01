"""工具失败后的三种处理：模型改参、代码原地重试、直接停止。"""


class FixableError(ValueError):
    """模型修改参数后可能成功；最多给有限次修正机会。"""


class RetryableError(Exception):
    """临时服务故障且可安全重复；代码原地补试，不重新请求模型。"""


class NonRetryableError(Exception):
    """修改参数也不应继续尝试，例如越过 workspace 安全边界。"""
