r"""手写的 Agent 主循环：把「模型 -> 工具 -> 再问模型」一步一步显式画成图。

为什么要有这个文件？
------------------------------------------------------------------
`create_agent`（LangChain 1.x 的高层入口）内部确实跑在 LangGraph 上，
但它的节点、边和停止条件都由框架生成。一旦被问到：

    「这个循环什么时候停？」
    「工具报错之后走哪条路？」
    「能不能不调模型就往前走一步？」

就只能回答「框架处理的」。这在面试里是最虚的一种答案。

这个文件把同一件事自己写一遍：**每一条边都出现在代码里**，
所以上面三个问题都可以指着某一行回答。

图长这样
------------------------------------------------------------------

    START ──> [model] ──有 tool_calls──> [tools] ──没转够──> [model]
                 │                          │
                 │ 没有 tool_calls          │ 转够 MAX_TOOL_ROUNDS
                 v                          v
                END                        END

两条结束路径的含义完全不同，这是本文件最关键的一点：

- **model -> END**：模型自己说完了。这是**正常出口**，唯一的正常出口。
- **tools -> END**：转够轮数被强制掐断。这是**保险丝**，不是正常结束。

为什么必须有第二条？
因为第一条只在模型「自己愿意停」时才停。模型可以一直申请工具，
把轮数耗完 —— 光靠模型自觉防不住转圈。
"""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


# 最多允许完成多少轮“模型申请工具 → 执行工具 → 回到模型”。
# 达到这个数后，工具节点仍会把本轮结果写回 State，但不会再回到模型节点。
# 这是防止模型因反复申请工具而无限循环、持续消耗时间和额度的保险丝。
MAX_TOOL_ROUNDS = 10


class AgentState(TypedDict):
    """图里唯一的共享白板（State）。

    这里刻意放了两种不同合并规则的字段，正好演示 reducer 的两种典型用法：

    - `messages`：用 `add_messages`，新消息**追加**到旧列表后面。
      对话历史必须追加，否则每一轮都会把上一轮冲掉。
    - `tool_rounds`：普通 int，后写的**覆盖**先写的。
      它是个计数器，我们要的是「当前是第几轮」，不是累加。
    """

    messages: Annotated[list[AnyMessage], add_messages]
    tool_rounds: int


def build_agent_graph(model, tools: list[BaseTool], system_prompt: str, checkpointer=None):
    """把上面的图画出来并编译。

    这个函数只负责三件事：

    1. 把工具对象整理成“工具名 -> 工具对象”的本地查找表；
    2. 定义两个节点：`model` 负责决策，`tools` 负责执行；
    3. 用两条条件边把循环和两个出口连起来。

    返回的是 LangGraph 的 CompiledStateGraph，上层用 `ainvoke` / `aget_state` 驱动。

    参数里的 checkpointer 由外层传入：SQLite 连接必须在使用期间保持打开。
    """

    # 这是 build_agent_graph 内部的快速查找表，不是项目级的工具 Registry。
    # 模型返回的 tool_call 只有一个字符串名字，例如 "read_file"；
    # 执行节点用这个名字在表里找到真正的 BaseTool 对象，再调用它的 ainvoke。
    tools_by_name: dict[str, BaseTool] = {tool.name: tool for tool in tools}

    # bind_tools 把工具的 JSON Schema 交给模型，但不执行任何东西。
    # 模型拿到的只是“说明书”；真正动手的永远是下面的 execute_tools_node。
    model_with_tools = model.bind_tools(tools)

    async def call_model_node(state: AgentState) -> dict[str, Any]:
        """模型节点：把到目前为止的对话交给模型，让它决定「直接回答」还是「要调工具」。"""

        # system_prompt 不写回 State，而是在每次真正调用模型前临时放到请求最前面。
        # 必须每轮都加：Chat API 是无状态的，历史消息里不会自己带着系统提示。
        messages = [SystemMessage(content=system_prompt), *state["messages"]]

        response = await model_with_tools.ainvoke(messages)
        # 节点只返回这一步新增的消息；完整 State 由 LangGraph 按 reducer 合并。
        return {"messages": [response]}

    async def execute_tools_node(state: AgentState) -> dict[str, Any]:
        """工具节点：真的执行上一轮模型申请的工具，把结果回填成 ToolMessage。

        这里有一条重要原则：**工具报错不往外抛，而是变成一条「失败的结果」还给模型。**
        如果在这里 raise，整个图就崩了；把错误交给模型，它才能自己决定
        是换个参数重试、换个工具，还是如实告诉用户做不到。

        这也是 Agent 和普通脚本的本质区别：出错之后由模型决定下一步，不是由代码写死。
        """

        # route_after_model 只有在最后一条消息带有 tool_calls 时才会把流程送到这里。
        # 这里仍然按“可能没有、可能是空”取值，让节点不依赖上游一定传对：
        # 属性缺失或值为 None 都按“本轮没有工具调用”处理，而不是抛 TypeError。
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", None) or []
        results: list[ToolMessage] = []

        for call in tool_calls:
            tool_name = call["name"]
            tool = tools_by_name.get(tool_name)

            if tool is None:
                # 模型调了一个不存在的工具。不要崩，让它知道自己错了，给它改正的机会。
                available = ", ".join(sorted(tools_by_name))
                content = f"工具不存在：{tool_name}。可用的工具有：{available}。"
            else:
                try:
                    # ainvoke 对同步工具和异步（MCP）工具都能用。
                    content = str(await tool.ainvoke(call["args"]))
                except Exception as error:
                    # 不把 Python 堆栈交给模型，只把可读的错误类型和消息变成 ToolMessage。
                    # 模型随后可以基于这条失败结果决定重试、换工具或如实说明失败。
                    content = f"工具执行失败：{type(error).__name__}: {error}"

            results.append(ToolMessage(content=content, tool_call_id=call["id"]))

        # tool_rounds 是覆盖式字段，所以这里要自己 +1。
        return {
            "messages": results,
            "tool_rounds": state.get("tool_rounds", 0) + 1,
        }

    def route_after_model(state: AgentState) -> str:
        """条件边之一：模型要工具就去执行，不要工具就结束。

        这是整个循环**唯一的正常出口**。模型不再申请工具时，这一轮才算真正完成。
        """

        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    def route_after_tools(state: AgentState) -> str:
        """条件边之二：工具执行完，决定是回去继续问模型，还是强制收手。

        这就是保险丝。它防的是「模型一直申请工具、把轮数耗完」这种情况。
        注意它跟 route_after_model 的区别：那一条是模型自愿停，这一条是我们掐断。
        """

        if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS:
            return END
        return "model"

    graph = StateGraph(AgentState)
    graph.add_node("model", call_model_node)
    graph.add_node("tools", execute_tools_node)

    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route_after_model, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", route_after_tools, {"model": "model", END: END})

    return graph.compile(checkpointer=checkpointer, name="common_agent")
