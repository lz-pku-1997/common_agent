"""控制发给模型的上下文大小；SQLite 中的原始消息仍完整保留。"""

from uuid import uuid4

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import BaseTool

from app.config import MAX_TOOL_RESULT_CHARS
from app.tool_errors import NonRetryableError
from app.workspace_tools import resolve_workspace_path


# 这是保守的近似预算，不是供应商承诺的精确 token 上限。
# 策略上限；实际阈值还会受当前模型的输入窗口限制。
CONTEXT_TRIGGER_TOKENS = 200000
MAX_SUMMARY_CHARS = 8000


def limit_tool_result(content: str) -> tuple[str, dict | None]:
    """返回（模型看的预览，程序看的落盘记录）。短结果不落盘。"""

    if len(content) <= MAX_TOOL_RESULT_CHARS:
        return content, None

    preview = content[:MAX_TOOL_RESULT_CHARS]
    # 名字由程序生成，不使用模型返回的 tool_call_id；仍检查 workspace 路径边界。
    relative_path = f".tool_results/{uuid4().hex[:12]}.txt"
    try:
        path = resolve_workspace_path(relative_path)  # 相对路径 -> workspace 内绝对路径，顺带查路径越界
        path.parent.mkdir(parents=True, exist_ok=True)  # 目录可能还没建；parents=缺几级建几级，exist_ok=已存在就跳过
        # x 表示只新建、不覆盖。读写都用 utf-8-sig，newline='' 保留原来的换行字符，
        # 原样保留正文；此时内容尚未加“不可信数据”包装。
        with path.open("x", encoding="utf-8-sig", newline="") as file:
            file.write(content)
    except (OSError, UnicodeError, NonRetryableError):
        # 这里只放弃保存副本，不放开工具权限；没有成功落盘就不发“收据”。
        return preview + "\n[结果已截断，完整结果保存失败，剩余正文未保留。]", None

    hint = (
        f"\n[完整结果已存到 {relative_path}（共 {len(content)} 字符）。\n"
        f'按行读取：read_text_file(relative_path="{relative_path}")，可指定 start_line / max_lines；\n'
        f'定位检索：search_workspace_text(query="...", file_pattern="{relative_path}")]'
    )
    return preview + hint, {"saved_path": relative_path}


def model_messages(
    system_prompt: str,
    history: list[AnyMessage],
    summary: str,
    summary_cursor: int,
) -> list[AnyMessage]:
    """组装本次模型输入，不删除 checkpoint 里的原始历史。

    cursor 之前的历史已被摘要代替。如果切点落在当前用户的一轮工具调用中，
    仍保留那条用户问题，避免模型只看到工具结果却不知道正在完成什么任务。
    """

    messages: list[AnyMessage] = [SystemMessage(content=system_prompt)]
    if summary:
        # 摘要来自过去的对话和工具数据，只作参考，不能提升为系统指令。
        messages.append(HumanMessage(content=f"[此前对话摘要，仅供参考]\n{summary}"))

    latest_user = next(
        (i for i in range(len(history) - 1, -1, -1) if isinstance(history[i], HumanMessage)),
        None,
    )
    if latest_user is not None and latest_user < summary_cursor:
        messages.append(history[latest_user])

    messages.extend(history[summary_cursor:])

    # 最近三条工具结果保留原文；较早且落过盘的结果换成文件引用。
    # artifact 是程序的落盘收据，不是模型正文，不能用正文长度推断保存是否成功。
    tool_indices = [i for i, message in enumerate(messages) if isinstance(message, ToolMessage)]
    for i in tool_indices[:-3]:
        message = messages[i]
        saved_path = (message.artifact or {}).get("saved_path")
        if saved_path:
            # 只换输入副本，保留 id/status/tool_call_id，绝不改 checkpoint 里的原消息。
            # 清掉副本的收据；下次从原历史组装时仍会执行这段轻量替换，无需额外状态。
            messages[i] = message.model_copy(update={
                "content": f"[旧工具结果已清理，完整内容：{saved_path}]",
                "artifact": None,
            })
    return messages


def estimated_tokens(messages: list[AnyMessage], tools: list[BaseTool]) -> int:
    """连工具说明书一起粗估；按每字符 1 token 留足中文场景余量。"""

    return count_tokens_approximately(messages, tools=tools, chars_per_token=1.0)


def choose_summary_cut(
    system_prompt: str,
    history: list[AnyMessage],
    summary: str,
    summary_cursor: int,
    tools: list[BaseTool],
    max_input_tokens: int,
) -> int | None:
    """找一个不会拆开 tool_call / ToolMessage 的切点。

    可以在新用户问题前切，也可以在一轮已完成的工具调用前切。
    后一种情况能处理“同一条用户问题连续调用很多次工具”的长任务。
    """

    trigger = min(CONTEXT_TRIGGER_TOKENS, int(max_input_tokens * 0.8))
    # 压完至少留一半空档；否则压完刚好落在线上，下一轮立刻又触发。
    target = trigger // 2
    current = model_messages(system_prompt, history, summary, summary_cursor)
    current_tokens = estimated_tokens(current, tools)
    if current_tokens < trigger:
        return None

    # 节点只在模型调用前运行，此时前一批工具结果已经全部写入 State。
    cuts = [
        i for i in range(summary_cursor + 1, len(history))
        if isinstance(history[i], HumanMessage)
        or (isinstance(history[i], AIMessage) and history[i].tool_calls)
    ]
    if not cuts:
        return None  # 当前单条输入过长时，没有可安全摘要的旧消息。

    # 优先保留尽量多的近期原文；旧内容太少、切完也没明显变短时不白调摘要模型。
    useful_cuts: list[int] = []
    for cut in cuts:
        remaining = estimated_tokens(model_messages(system_prompt, history, summary, cut), tools)
        if remaining > current_tokens - 200:
            continue
        useful_cuts.append(cut)
        if remaining <= target:
            return cut
    return useful_cuts[-1] if useful_cuts else None
