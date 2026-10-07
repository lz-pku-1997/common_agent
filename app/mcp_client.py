"""读取配置，连接远程 MCP 服务，把选中的工具接入现有 Agent。"""

import json
import re
from contextlib import asynccontextmanager

from langchain_core.tools import StructuredTool
from anyio import BrokenResourceError, ClosedResourceError, EndOfStream
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.shared.exceptions import MCPError
from mcp_types import CONNECTION_CLOSED, REQUEST_TIMEOUT

from app.config import PROJECT_ROOT, require_environment_variable
from app.tool_errors import FixableError, NonRetryableError, RetryableError


MCP_CONFIG_PATH = PROJECT_ROOT / "mcp.json"  # 服务地址和工具清单属于配置，不写死在引擎里。


def is_transport_error(error: Exception) -> bool:
    """只识别连接关闭和等待超时，不把业务错误、代码错误当成临时故障。"""
    if isinstance(error, ExceptionGroup):  # SDK 的任务组可能把多个异常包装在一起。
        return all(is_transport_error(item) for item in error.exceptions)  # 混有未知异常就不自动重试。
    if isinstance(error, MCPError):
        return error.code in {CONNECTION_CLOSED, REQUEST_TIMEOUT}  # 其他协议错误不保证重试有效。
    return isinstance(error, (ConnectionError, TimeoutError, EOFError, BrokenResourceError, ClosedResourceError, EndOfStream))


def expand_environment(value: str) -> str:
    """将配置中的 ${变量名} 换成 .env 中的值；缺失时只报告变量名。"""
    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", lambda match: require_environment_variable(match[1]), value)


def load_mcp_servers() -> dict:
    """只支持远程 URL、可选 headers 和可选 tools 名称列表。"""
    servers = json.loads(MCP_CONFIG_PATH.read_text(encoding="utf-8-sig"))["mcpServers"]
    for server in servers.values():  # 配置由维护者编写；不再重复检查每个 JSON 字段的类型。
        server["url"] = expand_environment(server["url"])  # 密钥只在内存里替换，不写回配置文件。
        server["headers"] = {key: expand_environment(value) for key, value in server.get("headers", {}).items()}
    return servers


@asynccontextmanager
async def connect_mcp(server: dict):
    """官方 SDK 管理握手和协议；上下文退出时关闭连接，不自建服务器。"""
    async with create_mcp_http_client(headers=server["headers"]) as http_client:
        transport = streamable_http_client(server["url"], http_client=http_client)  # 只使用 Streamable HTTP。
        async with Client(transport, read_timeout_seconds=60) as client:
            yield client


def mcp_result_to_text(result) -> str:
    """把 MCP 的结构化/多段结果转换成模型能阅读的文字。"""

    text_parts = []
    for block in result.content:  # 正常结果和错误结果共用这一段文本提取。
        if hasattr(block, "text"):
            text_parts.append(str(block.text))
    text = "\n".join(text_parts)

    if result.is_error:
        # Server 正常返回的工具错误常是参数不符；给模型有限次改参机会。
        # 连接故障在调用层单独识别；这里不是服务重试入口。
        raise FixableError("MCP 工具执行失败：" + text)

    if result.structured_content is not None:
        return json.dumps(result.structured_content, ensure_ascii=False)

    return text


def create_mcp_coroutine(tool_name: str, server: dict):
    """为一个 MCP 工具生成异步调用函数。

    这里使用“函数生成函数”，是为了把每个工具自己的 tool_name 固定下来。
    server 与发现工具时保持一致；每次调用建立连接，结束时自动关闭。
    """

    async def call_mcp_tool(**arguments):
        request_sent = False  # 连接尚未建立时，工具不可能已经产生副作用。
        try:
            async with connect_mcp(server) as client:  # 发现和调用复用相同的 URL 与鉴权配置。
                request_sent = True  # 从这里起连接断开也不能断言“工具没有执行”。
                result = await client.call_tool(tool_name, arguments)
        except Exception as error:
            if not is_transport_error(error):
                raise NonRetryableError(f"MCP 请求失败（{type(error).__name__}），请检查服务地址、鉴权或协议。") from None  # 不把含 Key 的 HTTP 错误 URL 回灌模型。
            if request_sent:  # 没收到结果不代表没执行；远端操作不擅自原样补试。
                raise NonRetryableError("MCP 连接中断，工具是否已执行不确定，不能自动重复有副作用的操作") from error
            raise RetryableError("MCP 连接关闭或请求超时") from error  # 执行层原地补试，重新建立连接。
        return mcp_result_to_text(result)  # 正常返回的工具错误仍交给模型改参，不原样补试。

    return call_mcp_tool


async def load_mcp_tools() -> list[StructuredTool]:
    """连接 Server 发现工具，再把真实 input_schema 原样交给 LangChain。"""

    langchain_tools: list[StructuredTool] = []
    for name, server in load_mcp_servers().items():
        selected = server.get("tools")
        if selected == []:
            continue  # 空清单直接跳过，不为禁用的服务建立连接。
        try:
            async with connect_mcp(server) as client:
                discovery = await client.list_tools()
        except Exception as error:
            raise RuntimeError(f"MCP 服务 {name} 连接或发现失败（{type(error).__name__}）。") from None  # 启动错误同样不输出含密钥的 URL。
        available = {item.name for item in discovery.tools}
        if selected is not None and set(selected) - available:
            raise ValueError(f"MCP 服务 {name} 未提供这些工具：{sorted(set(selected) - available)}")  # 拼错名称在启动时发现。
        for item in discovery.tools:
            if selected is not None and item.name not in selected:
                continue  # 未选工具不注册，模型不可见，执行层也不可调用。
            tool_name = f"{name}__{item.name}"  # 服务前缀隔离同名工具，远端调用仍使用原始名称。
            if len(tool_name) > 64 or not re.fullmatch(r"[A-Za-z0-9_-]+", tool_name):
                raise ValueError(f"MCP 工具名不符合模型接口要求：{tool_name}")
            langchain_tools.append(StructuredTool(
                name=tool_name,
                description=item.description or f"来自 MCP 服务 {name} 的工具",
                args_schema=item.input_schema,
                coroutine=create_mcp_coroutine(item.name, server),  # 异步槽位：装 async def 函数，ainvoke() 走这里（本地工具填的是 func 槽位）。传原始工具名和 server，由闭包记住连哪个服务、调哪个工具。
                metadata={"source": "mcp", "server": name},  # 只保存来源名称，不把含密钥的 URL 交给模型。
            ))
    return langchain_tools
