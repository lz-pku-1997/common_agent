"""common_agent 的工具契约与权限登记表。

LangChain 已经负责把 Python 函数包装成 Tool、生成参数 Schema，并在模型请求中
暴露工具说明。这里不重复实现这些能力，只补项目自己的治理信息：

* 工具从哪里来；
* 工具默认采用 allow、ask 还是 deny；

三档策略已经接入执行链：allow 直接执行，ask 触发 LangGraph interrupt，
deny 直接拒绝。参数校验由各工具 handler 负责，不在这里重复登记一个布尔字段。
"""

from dataclasses import dataclass

from langchain_core.tools import BaseTool


PermissionMode = str  # 约定只使用："allow"、"ask"、"deny"


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    """一个工具在 common_agent 中的治理策略。

    ``permission`` 是三档决策：
    - allow：可以自动执行；
    - ask：执行前通过 LangGraph interrupt 请求人工确认；
    - deny：不应交给当前 Agent 执行。
    """

    source: str
    permission: PermissionMode


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    """把 LangChain Tool 和项目策略放在同一条登记记录里。"""

    tool: BaseTool
    policy: ToolPolicy


class ToolRegistry:
    """工具总表：按工具名登记 Tool，并提供统一查找入口。

    这里的 Registry 是项目层治理表，不是 LangChain 内部的 Tool 注册机制。
    LangChain 仍然负责 Schema 和 invoke/ainvoke；本类只负责“这个工具能不能
    进入当前 Agent，以及它应该按什么安全策略被看待”。
    """

    def __init__(self) -> None:
        self._entries: dict[str, RegisteredTool] = {}

    def register(self, tool: BaseTool, policy: ToolPolicy) -> None:
        """登记一个工具，并拒绝重名，避免模型名字被静默覆盖。"""

        if tool.name in self._entries:
            raise ValueError(f"工具名称重复，拒绝覆盖已有登记：{tool.name}")
        if policy.permission not in {"allow", "ask", "deny"}:
            raise ValueError(f"不支持的权限模式：{policy.permission}")
        self._entries[tool.name] = RegisteredTool(tool=tool, policy=policy)

    def register_many(
        self,
        tools: list[BaseTool],
        *,
        source: str,
        permission: PermissionMode,
    ) -> None:
        """用同一组默认策略登记一批同来源工具。"""

        for tool in tools:
            self.register(
                tool,
                ToolPolicy(
                    source=source,
                    permission=permission,
                ),
            )

    def tools_for_model(self) -> list[BaseTool]:
        """返回允许暴露给模型的工具。

        deny 工具不会进入模型说明书；ask 工具仍会进入说明书，执行节点
        再根据策略触发人工确认。这样“模型知道工具存在”和“工具可以直接执行”
        是两个明确的层次。
        """

        return [
            entry.tool
            for entry in self._entries.values()
            if entry.policy.permission != "deny"
        ]

    def as_tool_map(self) -> dict[str, BaseTool]:
        """生成执行节点需要的“工具名 -> Tool”查找表。"""

        return {tool.name: tool for tool in self.tools_for_model()}

    def policy_for(self, tool_name: str) -> ToolPolicy:
        """按模型返回的工具名查策略；不存在的名字直接报错。"""

        try:
            return self._entries[tool_name].policy
        except KeyError as error:
            raise KeyError(f"工具未登记：{tool_name}") from error

    def entries(self) -> tuple[RegisteredTool, ...]:
        """返回只读快照，便于日志、调试和后续审批节点查看。"""

        return tuple(self._entries.values())


def build_tool_registry(
    workspace_tools: list[BaseTool],
    rag_tools: list[BaseTool],
    mcp_tools: list[BaseTool],
) -> ToolRegistry:
    """按工具来源建立 common_agent 的第一版治理登记表。"""

    registry = ToolRegistry()

    # workspace 读工具可以自动执行，但路径和文件类型仍由工具函数校验。
    workspace_read_tools = [
        tool for tool in workspace_tools if tool.name != "save_new_text_file"
    ]
    registry.register_many(
        workspace_read_tools,
        source="workspace",
        permission="allow",
    )

    # 写文件会产生持久副作用，因此登记为 ask；执行前会触发 HITL 确认。
    workspace_write_tools = [
        tool for tool in workspace_tools if tool.name == "save_new_text_file"
    ]
    registry.register_many(
        workspace_write_tools,
        source="workspace",
        permission="ask",
    )

    # 建索引会写 SQLite 并消耗 Embedding 额度，因此属于 ask；纯检索是 allow。
    for tool in rag_tools:
        is_indexing = tool.name == "index_knowledge_base"
        registry.register(
            tool,
            ToolPolicy(
                source="rag",
                permission="ask" if is_indexing else "allow",
            ),
        )

    # MCP 工具来自外部 Server，先统一标记来源和外部风险；具体工具的细分策略
    # 后续可以在发现 Schema 后按工具名覆盖，而不需要改变 MCP 桥接代码。
    registry.register_many(
        mcp_tools,
        source="mcp",
        permission="allow",
    )

    return registry
