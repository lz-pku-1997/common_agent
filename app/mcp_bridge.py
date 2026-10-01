"""把 MCP 2.x Server 暴露的工具动态转换为 LangChain 工具。

为什么没有直接使用 langchain-mcp-adapters？
截至本项目锁定版本时，最新适配器 0.3.2 要求 mcp<2，而官方 SDK 已到 2.2.0。
这里直接使用官方 Client，并保留动态工具发现，不牺牲协议版本或工具 Schema。
"""

import json
import sys
from pathlib import Path

from langchain_core.tools import StructuredTool
from anyio import BrokenResourceError, ClosedResourceError, EndOfStream
from mcp import Client, StdioServerParameters
from mcp.shared.exceptions import MCPError
from mcp_types import CONNECTION_CLOSED, REQUEST_TIMEOUT

from app.config import PROJECT_ROOT
from app.tool_errors import FixableError, NonRetryableError, RetryableError


MCP_SERVER_PATH = PROJECT_ROOT / "mcp_servers" / "common_tools_server.py"
SAFE_TO_REPEAT = {"add_numbers", "get_current_time"}  # 本项目 Server 的两个工具无写入副作用；新工具不自动获得重试权限。


def is_transport_error(error: Exception) -> bool:
    """只识别连接关闭和等待超时，不把业务错误、代码错误当成临时故障。"""
    if isinstance(error, ExceptionGroup):  # SDK 的任务组可能把多个异常包装在一起。
        return all(is_transport_error(item) for item in error.exceptions)  # 混有未知异常就不自动重试。
    if isinstance(error, MCPError):
        return error.code in {CONNECTION_CLOSED, REQUEST_TIMEOUT}  # 其他协议错误不保证重试有效。
    return isinstance(error, (ConnectionError, TimeoutError, EOFError, BrokenResourceError, ClosedResourceError, EndOfStream))


def create_server_parameters() -> StdioServerParameters:
    """说明 MCP Client 应该如何启动本地 Server 子进程。"""

    return StdioServerParameters(
        # sys.executable 是当前虚拟环境的 python.exe；换电脑后不需要改绝对路径。
        command=sys.executable,
        args=[str(MCP_SERVER_PATH)],
    )


def mcp_result_to_text(result) -> str:
    """把 MCP 的结构化/多段结果转换成模型能阅读的文字。"""

    if result.is_error:
        error_parts = [
            str(block.text)
            for block in result.content
            if hasattr(block, "text")
        ]
        # Server 正常返回的工具错误常是参数不符；给模型有限次改参机会。
        # 连接故障在调用层单独识别；这里不是服务重试入口。
        raise FixableError("MCP 工具执行失败：" + "\n".join(error_parts))

    if result.structured_content is not None:
        return json.dumps(result.structured_content, ensure_ascii=False)

    text_parts = [
        str(block.text)
        for block in result.content
        if hasattr(block, "text")
    ]
    return "\n".join(text_parts)


def create_mcp_coroutine(tool_name: str):
    """为一个 MCP 工具生成异步调用函数。

    这里使用“函数生成函数”，是为了把每个工具自己的 tool_name 固定下来。
    每次实际调用会建立 stdio 连接、启动 Server 子进程、调用工具并自动关闭连接。
    """

    async def call_mcp_tool(**arguments):
        request_sent = False  # 连接尚未建立时，工具不可能已经产生副作用。
        try:
            async with Client(create_server_parameters()) as client:
                request_sent = True  # 从这里起连接断开也不能断言“工具没有执行”。
                result = await client.call_tool(tool_name, arguments)
        except Exception as error:
            if not is_transport_error(error):
                raise  # 永久配置错误、未知错误交给主循环停止；取消信号不由此处捕获。
            if request_sent and tool_name not in SAFE_TO_REPEAT:
                raise NonRetryableError("MCP 连接中断，工具是否已执行不确定，不能自动重复有副作用的操作") from error
            raise RetryableError("MCP 连接关闭或请求超时") from error  # 执行层原地补试，重新建立连接。
        return mcp_result_to_text(result)  # 正常返回的工具错误仍交给模型改参，不原样补试。

    return call_mcp_tool


async def load_mcp_tools() -> list[StructuredTool]:
    """连接 Server 发现工具，再把真实 input_schema 原样交给 LangChain。"""

    if not MCP_SERVER_PATH.exists():
        raise RuntimeError(f"MCP Server 文件不存在：{MCP_SERVER_PATH}")

    async with Client(create_server_parameters()) as client:
        discovery = await client.list_tools()

    langchain_tools: list[StructuredTool] = []
    for discovered_tool in discovery.tools:
        langchain_tools.append(
            StructuredTool(
                name=discovered_tool.name,
                description=discovered_tool.description or "来自 MCP Server 的工具",
                args_schema=discovered_tool.input_schema,
                coroutine=create_mcp_coroutine(discovered_tool.name),
                metadata={"source": "mcp", "server": "common-tools"},
            )
        )
    return langchain_tools
