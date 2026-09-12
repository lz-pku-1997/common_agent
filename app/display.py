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


def print_new_execution_trace(messages: list, old_message_count: int) -> None:
    """只打印本轮新增的工具轨迹，最后再打印 Agent 的最终回答。"""

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
    else:
        print("\ncommon_agent：本轮没有得到可显示的最终回答。")
