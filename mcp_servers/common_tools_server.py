"""一个真实、独立运行的 MCP 2.x stdio Server。

它不是在 Agent 进程里直接 import 后调用。MCP Client 会启动这个文件作为子进程，
双方通过 stdin/stdout 交换协议消息。这正是桌面端连接本地 MCP Server 的常见方式。

重要：stdio 的 stdout 是协议线路，所以这里不能随便 print；需要日志时应写 stderr。
"""

from datetime import datetime

from mcp.server import MCPServer


mcp = MCPServer(
    name="common-tools",
    description="common_agent 的通用演示工具服务器",
)


@mcp.tool()
def add_numbers(a: float, b: float) -> dict[str, float]:
    """准确计算两个数字之和；不要让语言模型自己心算需要可靠结果的加法。"""

    return {"a": a, "b": b, "sum": a + b}


@mcp.tool()
def get_current_time() -> dict[str, str]:
    """读取运行 MCP Server 这台电脑的当前本地时间与时区。"""

    now = datetime.now().astimezone()
    return {
        "iso_time": now.isoformat(timespec="seconds"),
        "timezone": str(now.tzinfo),
    }


if __name__ == "__main__":
    # run(stdio) 会一直等 MCP Client 从标准输入发来协议请求。
    # 当 Client 关闭连接时，子进程随之退出，不需要我们手工结束进程。
    mcp.run(transport="stdio")
