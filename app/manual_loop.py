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

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


# 最多转多少轮工具。超过就强制结束，避免模型一直要工具把钱烧光。
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

    返回的同样是 LangGraph 的 CompiledStateGraph，跟 `create_agent` 的产物
    有一样的 `ainvoke` / `aget_state`，所以上层的 cli.py 不需要改。

    参数里的 checkpointer 由外层传入：SQLite 连接必须在使用期间保持打开。
    """

    # 工具注册表：把「模型给的工具名」映射到「真正能执行的对象」。
    # 模型只会给出一个字符串名字，能不能执行、执行什么，完全由这张表决定。
    tool_registry: dict[str, BaseTool] = {tool.name: tool for tool in tools}

    # bind_tools 把工具的 JSON Schema 交给模型，但不执行任何东西。
    # 模型拿到的只是「说明书」，真正动手的永远是 run_tools。
    model_with_tools = model.bind_tools(tools)

    async def call_model(state: AgentState) -> dict:
        """模型节点：把到目前为止的对话交给模型，让它决定「直接回答」还是「要调工具」。"""

        messages = list(state["messages"])

        # SystemMessage 只在第一轮补上。之后它已经留在 messages 里，
        # 每一轮都重新塞一遍只会白白多花 token。
        if not any(isinstance(message, SystemMessage) for message in messages):
            messages = [SystemMessage(content=system_prompt), *messages]

        response = await model_with_tools.ainvoke(messages)
        # 只返回「这一刀新增了什么」，不返回整个 State —— 合并由框架按 reducer 做。
        return {"messages": [response]}

    async def run_tools(state: AgentState) -> dict:
        """工具节点：真的执行上一轮模型申请的工具，把结果回填成 ToolMessage。

        这里有一条重要原则：**工具报错不往外抛，而是变成一条「失败的结果」还给模型。**
        如果在这里 raise，整个图就崩了；把错误交给模型，它才能自己决定
        是换个参数重试、换个工具，还是如实告诉用户做不到。

        这也是 Agent 和普通脚本的本质区别：出错之后由模型决定下一步，不是由代码写死。
        """

        tool_calls = state["messages"][-1].tool_calls
        results: list[ToolMessage] = []

        for call in tool_calls:
            tool = tool_registry.get(call["name"])

            if tool is None:
                # 模型调了一个不存在的工具。不要崩，让它知道自己错了，给它改正的机会。
                available = ", ".join(sorted(tool_registry))
                content = f"工具不存在：{call['name']}。可用的工具有：{available}。"
            else:
                try:
                    # ainvoke 对同步工具和异步（MCP）工具都能用。
                    content = str(await tool.ainvoke(call["args"]))
                except Exception as error:
                    # 只把错误类型和说明交给模型，不把堆栈和内部对象泄漏出去。
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
    graph.add_node("model", call_model)
    graph.add_node("tools", run_tools)

    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route_after_model, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", route_after_tools, {"model": "model", END: END})

    return graph.compile(checkpointer=checkpointer, name="common_agent")
