r"""common_agent 的真实命令行界面。

运行方式：在项目根目录执行  .\.venv\Scripts\python.exe -m app.cli
也可以直接在 PyCharm 中运行根目录的 run_cli.py。
"""

import re
import asyncio

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.agent import build_common_agent
from app.config import (
    DATABASE_PATH,
    load_agent_engine,
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
  请列出工作区文件，并读取 welcome.md，告诉我项目代号。
  请在 notes 目录新建 first_note.md，内容是“我们完成了真实工具调用”。
  请为 knowledge 建立索引，再检索 RAG 验证暗号并附来源。
  请使用 MCP 工具计算 37+58，并读取当前时间。
""".strip()
    )


async def main_async() -> None:
    prepare_runtime_directories()
    settings = load_model_settings()

    engine = load_agent_engine()
    engine_label = "自己画的图（manual_loop.py）" if engine == "manual" else "框架生成（create_agent）"

    print("=" * 62)
    print("common_agent v1.0：真实模型 + 文件工具 + 向量 RAG + MCP + SQLite")
    print(f"当前模型：{settings['model']}")
    print(f"当前引擎：{engine_label}")
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
                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": user_text}]},
                    config=config,
                )
                print_new_execution_trace(result["messages"], len(old_messages))
            except Exception as error:
                # 终端应用不能把密钥或完整内部对象打印出来，只给用户错误类型与说明。
                print(f"\n[本轮失败] {type(error).__name__}: {error}")
                print("可以检查网络、.env 配置或模型免费额度后重试；程序不会伪装成功。")


def main() -> None:
    """同步启动壳：帮普通 Python 文件启动并管理 asyncio 事件循环。"""

    asyncio.run(main_async())


if __name__ == "__main__":
    main()
