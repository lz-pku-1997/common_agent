"""把 LangChain 消息转换成适合终端阅读的文字和执行轨迹。"""

import json

from langchain_core.messages import AIMessage, ToolMessage


def message_content_to_text(content) -> str:
    """兼容模型返回纯字符串或多个 content block 的情况。"""

    if isinstance(content, str):
        return content

    text_parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
    return "\n".join(part for part in text_parts if part).strip()


def print_new_execution_trace(
    messages: list,
    old_message_count: int,
    stop_reason: str | None = None,
    tool_rounds: int = 0,
    max_tool_rounds: int | None = None,
) -> None:
    """只打印本轮新增的工具轨迹，最后再打印 Agent 的最终回答。

    没有可显示的最终回答时，通常说明本轮被保险丝（重复请求 / 轮数上限）
    收口了。这时用 stop_reason 和 tool_rounds 补一句具体原因，避免用户
    把"没有输出"误读成"没有发生任何事"。
    """

    new_messages = messages[old_message_count:]
    final_answer = ""

    for message in new_messages:
        if isinstance(message, AIMessage):
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    arguments = json.dumps(tool_call.get("args", {}), ensure_ascii=False)
                    print(f"\n[Agent 决定调用工具] {tool_call.get('name')} {arguments}")
            else:
                possible_answer = message_content_to_text(message.content)
                if possible_answer:
                    final_answer = possible_answer

        elif isinstance(message, ToolMessage):
            status = getattr(message, "status", "success")
            tool_result = message_content_to_text(message.content)
            if len(tool_result) > 500:
                tool_result = tool_result[:500] + "……（终端仅预览前 500 字）"
            print(f"[工具返回：{message.name}，状态={status}] {tool_result}")

    if final_answer:
        print(f"\ncommon_agent：{final_answer}")
    elif stop_reason == "repeated_tool_call":
        print("\ncommon_agent：因检测到重复工具请求而中止，本轮任务未完成。")
    elif stop_reason == "retry_limit":
        print("\ncommon_agent：参数修正次数已用完，本轮任务未完成。")
    elif stop_reason == "permission_denied":
        print("\ncommon_agent：工具调用被权限策略拒绝，本轮任务未完成。")
    elif stop_reason == "approval_denied":
        print("\ncommon_agent：你未批准工具操作，因此没有执行，本轮任务未完成。")
    elif stop_reason == "non_retryable_error":
        print("\ncommon_agent：工具发生不可重试错误，本轮任务已安全停止。")
    elif max_tool_rounds is not None and tool_rounds >= max_tool_rounds:
        print(f"\ncommon_agent：已达到 {max_tool_rounds} 轮工具上限，本轮任务未确认完成。")
    else:
        print("\ncommon_agent：本轮没有得到可显示的最终回答。")
