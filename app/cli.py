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
    WORKSPACE_ROOT,
)
from app.display import print_answer_chunk, print_model_update, print_progress_event, print_turn_end
from app.sessions import completed_turns, fork_turn, print_turns  # CLI 的 /time 与 /fork 命令复用这里的历史查询和分叉函数。


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
  /time              列出当前会话已完成的历史轮次
  /time N 新编号     从第 N 个已完成轮次开新会话并切换
  /fork 新编号       从最新已完成轮次开新会话并切换
  /exit              退出程序

可以这样试：
  请列出工作区文件。
  请在 notes 目录新建 first_note.md，内容是“我们完成了真实工具调用”。
  请为 knowledge 建立索引，再检索 RAG 验证暗号并附来源。
  请使用高德 MCP 查询杭州天气。
""".strip()
    )


async def invoke_with_human_approval(agent, user_input: str, config: dict) -> dict:
    """运行一轮 Agent，并在 LangGraph interrupt 时等待用户确认。

    普通调用返回最终 State；如果工具策略是 ``ask``，第一次调用会暂停并返回
    ``__interrupt__``。这里把待确认信息展示给用户，再用同一个 thread_id 发送
    ``Command(resume=...)``，LangGraph 就会从原来的工具节点继续，而不是重新
    开启一轮无关的对话。
    """

    previous = await agent.aget_state(config)
    old_message_count = len(previous.values.get("messages", []))  # 停止时不能把上一轮答案当作本轮回答。
    graph_input = {"messages": [{"role": "user", "content": user_input}]}
    final_streamed = False
    while True:  # 普通输入与审批 resume 共用同一个流式入口，避免两套界面逻辑。
        interruptions = ()
        printed_text = False
        async for mode, event in agent.astream(  # 每收到一个 LangGraph 事件，就进入循环体处理一次。
            graph_input, config=config, stream_mode=["messages", "updates", "custom"],  # 同时订阅模型片段、节点更新和程序主动发送的事件。
        ):
            if mode == "messages":  # 当前事件携带模型生成的消息片段。
                chunk, metadata = event  # 分开读取内容片段和来源等元数据。
                # context 节点也调用模型做摘要，但摘要不是用户回答，不能打印。
                if metadata.get("langgraph_node") != "model":  # 摘要节点也调用模型，但它的输出不是给用户看的回答。
                    continue
                printed_text = print_answer_chunk(chunk.content, printed_text)  # 逐段打印，只在第一段前显示回答标签。
            elif mode == "custom":  # 当前事件由 Agent 主动发出，用于界面进度提示。
                print_progress_event(event)  # 进度提示和工具结果都从 custom 流显示，不再从 updates 重复打印。
            else:  # messages 和 custom 已在上面处理；这里处理第三种流 updates。
                if "model" in event:  # model 节点结束时，updates 携带该节点新增的完整消息。
                    message = event["model"]["messages"][-1]  # 取 model 节点这次产出的最后一条消息。
                    final_streamed = print_model_update(message, printed_text)  # 模型完成时显示工具请求，记录最终答案是否已打印。
                    printed_text = False  # 当前 model 节点已处理完，清除本节点的输出标记。
                interruptions = event.get("__interrupt__", interruptions)
        if printed_text:
            print()  # 流中途结束时也保留终端换行。
        if not interruptions:
            break
        interruption = interruptions[0]
        request = getattr(interruption, "value", {})
        print("\n[需要人工确认]")
        print(request.get("message", "Agent 请求执行受保护工具。"))
        for item in request.get("tools", []):
            print(f"  工具：{item['name']}")
            arguments = item.get("args", {})
            if item["name"] == "execute_command":
                print(f"  准备做什么（模型说明）：{arguments.get('description', '')}")
                print(f"  执行目录：{WORKSPACE_ROOT}")
                print(f"  原始命令：{arguments.get('command', '')}")  # 不能仅显示模型解释，用户应能核对真实命令。
            else:
                print(f"  参数：{arguments}")

        answer = input("是否批准执行？输入 y 批准，其他任何内容都拒绝：").strip().lower()
        approved = answer in {"y", "yes", "是", "同意"}

        # resume 的值会作为 interrupt(...) 的返回值交回节点；节点随后才会
        # 进入 ask 工具的 ainvoke。拒绝时也要 resume，不能让图永远停在断点。
        graph_input = Command(resume={"approved": approved})  # 下一次循环流式恢复同一个审批断点。

    result = dict((await agent.aget_state(config)).values)  # 从 checkpoint 取完整 State；流片段只用于显示，不是完整状态。
    print_turn_end(result, old_message_count, final_streamed, MAX_TOOL_ROUNDS)  # 只补显示尚未输出的最终答案或停止原因。
    return result


async def main_async() -> None:
    prepare_runtime_directories()
    fast_settings = load_model_settings("fast")
    strong_settings = load_model_settings("strong")

    print("=" * 62)
    print("common_agent v1.0：真实模型 + 文件工具 + 向量 RAG + MCP + SQLite")
    print(f"模型：默认 {fast_settings['model']}，必要时单向升级为 {strong_settings['model']}")
    print("权限：固定 allow / ask，未定级调用由安全子 Agent 审查；人工确认仍在主 Agent。")
    print("Shell 没有沙箱：子 Agent 的判断不是强隔离，仅供可信用户本机使用。")
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
            parts = user_text.split()
            if parts[0] in {"/time", "/fork"}:
                try:
                    if parts == ["/time"]:
                        print_turns(await completed_turns(agent, config))
                    else:
                        if parts[0] == "/fork" and len(parts) == 2:
                            number = None  # 不指定轮次，由分叉函数选择最新已完成回合。
                        elif parts[0] == "/time" and len(parts) == 3:
                            number = int(parts[1])  # 历史轮次用 /time 列出的编号定位。
                        else:
                            raise ValueError("用法：/time 查看历史；/time N 新编号 或 /fork 新编号 开新会话。")
                        name = normalize_thread_id(parts[-1])  # 两种命令都在最后指定新会话名。
                        await fork_turn(agent, config, name, number)  # 共用复制流程，只改变所选快照。
                        thread_id = name  # 成功保存新分支后才切换，失败不影响旧会话。
                        print(f"已进入分支：{thread_id}。只复制对话状态，不撤销已经发生的操作。")
                except ValueError as error:
                    print(f"历史操作失败：{error}")
                continue

            try:
                await invoke_with_human_approval(agent, user_text, config)  # 流式入口负责展示过程和收尾，不重复打印答案。
            except Exception as error:
                # 终端应用不能把密钥或完整内部对象打印出来，只给用户错误类型与说明。
                print(f"\n[本轮失败] {type(error).__name__}: {error}")
                print("可以检查网络、.env 配置或模型免费额度后重试；程序不会伪装成功。")


def main() -> None:
    """同步启动壳：帮普通 Python 文件启动并管理 asyncio 事件循环。"""

    asyncio.run(main_async())


if __name__ == "__main__":
    main()
