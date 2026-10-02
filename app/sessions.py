"""只分叉已完成的对话轮次；不是文件备份，也不重放历史工具。"""

from copy import deepcopy

from app.display import message_content_to_text


async def completed_turns(agent, config: dict) -> list:
    """返回由旧到新的已完成回合；执行中或等待审批的快照不能成为分叉入口。"""
    snapshots = [state async for state in agent.aget_state_history(config)]  # SQLite 历史默认是新到旧。
    return [
        state for state in reversed(snapshots)
        if state.values.get("messages") and not state.next  # 有消息且本轮已结束；messages 只排除空初始状态，不区分轮次。
        and not any(task.interrupts or task.error for task in state.tasks)  # 排除中断和失败中的任务。
    ]


def print_turns(turns: list) -> None:
    if not turns:
        print("当前会话还没有已完成的轮次。")
    for number, state in enumerate(turns, start=1):
        last = state.values["messages"][-1]
        text = message_content_to_text(last.content).replace("\n", " ")[:100]
        print(f"{number}. {state.created_at} | {last.type} | {text}")  # 编号只对这次查询的会话历史有效。


async def fork_turn(agent, config: dict, new_thread: str, number: int | None = None) -> dict:
    """从指定完成轮次开新会话；不传轮次时取最新完成轮次。"""
    target = {"configurable": {"thread_id": new_thread}}
    existing = await agent.aget_state(target)
    if existing.values or existing.created_at:
        raise ValueError("目标会话已存在，请换一个新编号，不能覆盖已有历史。")
    turns = await completed_turns(agent, config)
    if not turns:
        raise ValueError("当前会话还没有已完成的轮次，不能分叉。")
    if number is not None and not 1 <= number <= len(turns):
        raise ValueError(f"轮次必须在 1～{len(turns)} 之间，请先用 /time 查看。")
    snapshot = turns[-1] if number is None else turns[number - 1]  # /fork 取最新完成回合；/time 的编号从 1 开始。
    values = deepcopy(snapshot.values)  # 摘要和游标必须一起复制，不能只复制聊天文字。
    values.update(tool_rounds=0, retry_count=0, last_tool_call_signature=None, stop_reason=None)
    # 让图把复制状态视为 model 已完成；已完成轮次的最后消息没有待执行 tool_calls，
    # 因此只保存新 checkpoint，不调用 invoke(None)，也就不会重跑历史工具。
    await agent.aupdate_state(target, values, as_node="model")
    return target
