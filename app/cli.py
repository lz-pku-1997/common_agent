r"""common_agent 的真实命令行界面。

运行方式：在项目根目录执行  .\.venv\Scripts\python.exe -m app.cli
也可以直接在 PyCharm 中运行根目录的 run_cli.py。
"""

import re
import asyncio

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from app.agent import build_common_agent
from app.manual_loop import MAX_TOOL_ROUNDS
from app.config import (
    DATABASE_PATH,
    load_model_settings,
    prepare_runtime_directories,
)
from app.display import print_new_execution_trace


THREAD_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def normalize_thread_id(raw_thread_id: str) -> str:
    """检查会话编号，避免空格或特殊字符让 SQLite 里的会话难以管理。"""

    thread_id = raw_thread_id.strip() or "common-demo"
    if not THREAD_ID_PATTERN.fullmatch(thread_id):
        raise ValueError("会话编号只能包含字母、数字、下划线和短横线，最长 64 位。")
    return thread_id


def print_help() -> None:
    print(
        """
可用命令：
  /help              查看帮助
  /thread            查看当前会话编号
  /thread 新编号     切换到另一个会话；原会话仍保存在 SQLite
  /exit              退出程序

可以这样试：
  请列出工作区文件。
  请在 notes 目录新建 first_note.md，内容是“我们完成了真实工具调用”。
  请为 knowledge 建立索引，再检索 RAG 验证暗号并附来源。
  请使用 MCP 工具计算 37+58，并读取当前时间。
""".strip()
    )


async def invoke_with_human_approval(agent, user_input: str, config: dict) -> dict:
    """运行一轮 Agent，并在 LangGraph interrupt 时等待用户确认。

    普通调用返回最终 State；如果工具策略是 ``ask``，第一次调用会暂停并返回
    ``__interrupt__``。这里把待确认信息展示给用户，再用同一个 thread_id 发送
    ``Command(resume=...)``，LangGraph 就会从原来的工具节点继续，而不是重新
    开启一轮无关的对话。
    """

    result = await agent.ainvoke({"messages": [{"role": "user", "content": user_input}]}, config=config)

    while result.get("__interrupt__"):
        interruption = result["__interrupt__"][0]
        request = getattr(interruption, "value", {})
        print("\n[需要人工确认]")
        print(request.get("message", "Agent 请求执行受保护工具。"))
        for item in request.get("tools", []):
            print(f"  工具：{item['name']}")
            print(f"  参数：{item.get('args', {})}")

        answer = input("是否批准执行？输入 y 批准，其他任何内容都拒绝：").strip().lower()
        approved = answer in {"y", "yes", "是", "同意"}

        # resume 的值会作为 interrupt(...) 的返回值交回节点；节点随后才会
        # 进入 ask 工具的 ainvoke。拒绝时也要 resume，不能让图永远停在断点。
        result = await agent.ainvoke(
            Command(resume={"approved": approved}),
            config=config,
        )

    return result


async def main_async() -> None:
    prepare_runtime_directories()
    settings = load_model_settings()

    print("=" * 62)
    print("common_agent v1.0：真实模型 + 文件工具 + 向量 RAG + MCP + SQLite")
    print(f"当前模型：{settings['model']}")
    print("安全边界：工具只能访问本项目的 workspace，且不能覆盖已有文件。")
    print("=" * 62)

    raw_thread_id = input("会话编号（直接回车使用 common-demo）：")
    try:
        thread_id = normalize_thread_id(raw_thread_id)
    except ValueError as error:
        print(f"会话编号不合法：{error}")
        return

    # MCP 工具使用异步通信，所以这里使用 async with 和异步版 SQLite Saver。
    # Agent 使用期间连接始终打开，代码块结束后连接会自动关闭。
    async with AsyncSqliteSaver.from_conn_string(str(DATABASE_PATH)) as saver:
        agent = await build_common_agent(saver)
        print(f"已进入会话：{thread_id}。输入 /help 查看命令。")

        while True:
            try:
                user_text = input("\n你：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n已退出。对话仍保存在 data/agent.sqlite。")
                break

            if not user_text:
                continue
            if user_text == "/exit":
                print("已退出。对话仍保存在 data/agent.sqlite。")
                break
            if user_text == "/help":
                print_help()
                continue
            if user_text == "/thread":
                print(f"当前会话：{thread_id}")
                continue
            if user_text.startswith("/thread "):
                try:
                    thread_id = normalize_thread_id(user_text.removeprefix("/thread "))
                    print(f"已切换到会话：{thread_id}")
                except ValueError as error:
                    print(f"切换失败：{error}")
                continue

            config = {"configurable": {"thread_id": thread_id}}

            # 调用前读取旧消息数量，之后就能只展示“本轮新增”的执行轨迹。
            old_state = await agent.aget_state(config)
            old_messages = old_state.values.get("messages", []) if old_state.values else []

            try:
                result = await invoke_with_human_approval(agent, user_text, config)
                print_new_execution_trace(
                    result["messages"],
                    len(old_messages),
                    stop_reason=result.get("stop_reason"),
                    tool_rounds=result.get("tool_rounds", 0),
                    max_tool_rounds=MAX_TOOL_ROUNDS,
                )
            except Exception as error:
                # 终端应用不能把密钥或完整内部对象打印出来，只给用户错误类型与说明。
                print(f"\n[本轮失败] {type(error).__name__}: {error}")
                print("可以检查网络、.env 配置或模型免费额度后重试；程序不会伪装成功。")


def main() -> None:
    """同步启动壳：帮普通 Python 文件启动并管理 asyncio 事件循环。"""

    asyncio.run(main_async())


if __name__ == "__main__":
    main()
