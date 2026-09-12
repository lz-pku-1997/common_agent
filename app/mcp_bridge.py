"""把 MCP 2.x Server 暴露的工具动态转换为 LangChain 工具。

为什么没有直接使用 langchain-mcp-adapters？
截至本项目锁定版本时，最新适配器 0.3.2 要求 mcp<2，而官方 SDK 已到 2.2.0。
这里直接使用官方 Client，并保留动态工具发现，不牺牲协议版本或工具 Schema。
"""

import json
import sys
from pathlib import Path

from langchain_core.tools import StructuredTool, ToolException
from mcp import Client, StdioServerParameters

from app.config import PROJECT_ROOT


MCP_SERVER_PATH = PROJECT_ROOT / "mcp_servers" / "common_tools_server.py"


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
        raise ToolException("MCP 工具执行失败：" + "\n".join(error_parts))

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
        async with Client(create_server_parameters()) as client:
            result = await client.call_tool(tool_name, arguments)
            return mcp_result_to_text(result)

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
                handle_tool_error=True,
                metadata={"source": "mcp", "server": "common-tools"},
            )
        )
    return langchain_tools
