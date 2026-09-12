"""完整端到端验收：一个问题同时走真实 RAG、真实 MCP 和 SQLite。"""

import asyncio
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from uuid import uuid4

from app.agent import build_common_agent
from app.config import DATABASE_PATH, prepare_runtime_directories
from app.display import message_content_to_text
from app.rag_tools import index_knowledge_base


async def run_test() -> None:
    prepare_runtime_directories()
    # uuid4() 每次生成一个随机编号，保证验收使用全新会话，不能被旧的成功数据“蒙混过关”。
    thread_id = f"official-live-smoke-{uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}

    # 先真实建立向量索引。索引属于写入阶段，检索属于问答阶段，两步职责不同。
    index_result = index_knowledge_base.invoke({"relative_directory": "knowledge"})

    async with AsyncSqliteSaver.from_conn_string(str(DATABASE_PATH)) as saver:
        agent = await build_common_agent(saver)
        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "这是完整验收测试：第一，必须调用 search_knowledge_base 检索"
                            "共享内核的 RAG 验证暗号；第二，必须调用 MCP 工具 add_numbers "
                            "计算 37+58。最后同时给出暗号、证据 source 和计算结果。"
                        ),
                    }
                ]
            },
            config=config,
        )

        tool_names = [
            message.name
            for message in result["messages"]
            if isinstance(message, ToolMessage)
        ]
        assert "search_knowledge_base" in tool_names, "模型没有调用真实 RAG 检索"
        assert "add_numbers" in tool_names, "模型没有调用真实 MCP 工具"

        final_text = message_content_to_text(result["messages"][-1].content)
        assert "ORANGE-RIVER-926" in final_text, "最终回答没有使用 RAG 证据"
        assert "knowledge/common_agent_facts.md" in final_text, "最终回答没有附证据来源"
        assert "95" in final_text, "最终回答没有使用 MCP 计算结果"

    # 上面的 with 已关闭第一条数据库连接。这里重新打开 Saver 和 Agent，
    # 再按相同 thread_id 读取，模拟“程序退出后重新启动”，验证不是内存假持久化。
    async with AsyncSqliteSaver.from_conn_string(str(DATABASE_PATH)) as reopened_saver:
        reopened_agent = await build_common_agent(reopened_saver)
        reopened_state = await reopened_agent.aget_state(config)
        saved_messages = reopened_state.values.get("messages", [])
        assert len(saved_messages) >= 5, "重新连接 SQLite 后没有恢复完整工具循环"

    print("common_agent 完整真实验收通过：")
    print(f"- RAG 建库：{index_result}")
    print(f"- 模型实际调用工具：{tool_names}")
    print(f"- SQLite 已保存消息数：{len(saved_messages)}")
    print(f"- 最终回答：{final_text}")


def main() -> None:
    asyncio.run(run_test())


if __name__ == "__main__":
    main()
