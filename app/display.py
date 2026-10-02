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


def print_answer_chunk(content, started: bool) -> bool:
    """显示一段模型文字；返回这次回答是否已经开始打印。"""
    text = content
    if isinstance(content, list):  # 模型也可能返回多个内容块，只提取其中的文字。
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        text = "".join(parts)  # 片段边界上的空格和换行必须保留。
    if text:
        if not started:
            print("\ncommon_agent：", end="", flush=True)  # 一次模型回答只打印一次前缀。
        print(text, end="", flush=True)  # 收到片段就显示，后续文字接在后面。
        return True
    return started


def print_model_update(message: AIMessage, text_printed: bool) -> bool:
    """模型节点完成时换行并显示工具请求；返回最终回答是否已流式显示。"""
    if text_printed:
        print()  # 结束刚才逐段打印的那一行。
    for call in message.tool_calls:  # 普通回答没有 tool_calls，循环自然跳过。
        arguments = json.dumps(call.get("args", {}), ensure_ascii=False)
        print(f"\n[Agent 决定调用工具] {call.get('name')} {arguments}")
    return text_printed and not message.tool_calls  # 带工具请求的消息还不是最终回答。


def print_tool_result(message: ToolMessage) -> None:
    """只显示一条工具结果，不负责模型回答或收尾。"""
    text = message_content_to_text(message.content)
    if len(text) > 500:
        text = text[:500] + "……（终端仅预览前 500 字）"  # 屏幕预览不影响模型收到的正文。
    print(f"[工具返回：{message.name}，状态={message.status}] {text}")


def print_progress_event(event: dict) -> None:
    """显示 Agent 主动发送的进度事件。所有工具结果只从这个入口打印。"""
    kind = event.get("type")
    if kind == "model_upgrade":
        print(f"[模型升级] 后续使用 {event['model']}，本会话不再降档。", flush=True)
    elif kind == "tool_start":
        print(f"[执行工具] {event['name']}", flush=True)
    elif kind == "retry":
        print(f"[服务补试] {event['name']}：第 {event['attempt']} 次，等待 {event['delay']} 秒", flush=True)
    elif kind == "tool_result":
        print_tool_result(event["message"])  # 成功、失败和被拒绝的工具结果都走这里。


def print_turn_end(result: dict, old_message_count: int, answer_streamed: bool, max_tool_rounds: int) -> None:
    """整轮结束后补显示答案或停止原因；已经流式显示的最终回答不重复输出。"""
    if answer_streamed:
        return
    new_messages = result.get("messages", [])[old_message_count:]  # 只查本轮，避免显示上一轮答案。
    for message in reversed(new_messages):  # 从后往前找最后一条非空普通 AI 回答。
        if isinstance(message, AIMessage) and not message.tool_calls:
            text = message_content_to_text(message.content)
            if text:
                print(f"\ncommon_agent：{text}")
                return
    reasons = {  # 停止原因和提示文字一一对应，无需一长串 if/elif。
        "repeated_tool_call": "因检测到重复工具请求而中止，本轮任务未完成。",
        "retry_limit": "参数修正次数已用完，本轮任务未完成。",
        "permission_denied": "工具调用被权限策略拒绝，本轮任务未完成。",
        "approval_denied": "你未批准工具操作，因此没有执行，本轮任务未完成。",
        "retry_exhausted": "工具服务补试三次仍未成功，本轮任务未完成。",
        "non_retryable_error": "工具发生不可重试错误，本轮任务已安全停止。",
    }
    text = reasons.get(result.get("stop_reason"))
    if text is None:
        if result.get("tool_rounds", 0) >= max_tool_rounds:
            text = f"已达到 {max_tool_rounds} 轮工具上限，本轮任务未确认完成。"
        else:
            text = "本轮没有得到可显示的最终回答。"
    print(f"\ncommon_agent：{text}")
