"""不经过大模型，直接证明 MCP Client/Server/stdio 三者真实贯通。"""

import asyncio

from app.mcp_bridge import load_mcp_tools


async def run_test() -> None:
    tools = await load_mcp_tools()
    tools_by_name = {tool.name: tool for tool in tools}

    assert "add_numbers" in tools_by_name
    assert "get_current_time" in tools_by_name

    result = await tools_by_name["add_numbers"].ainvoke({"a": 37, "b": 58})
    assert '"sum": 95' in result

    print("MCP 真实验收通过：")
    print(f"- Server 动态暴露工具：{sorted(tools_by_name)}")
    print(f"- stdio 调用 add_numbers(37, 58)：{result}")


def main() -> None:
    asyncio.run(run_test())


if __name__ == "__main__":
    main()
