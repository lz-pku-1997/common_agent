"""common_agent 的工具契约与权限登记表。

LangChain 已经负责把 Python 函数包装成 Tool、生成参数 Schema，并在模型请求中
暴露工具说明。这里不重复实现这些能力，只补项目自己的治理信息：

* 工具从哪里来；
* 工具固定 allow、固定 ask，还是交给安全子 Agent 逐次审查（None）；

子 Agent 返回 allow / ask / deny；人工确认仍只在主图触发 interrupt。
参数校验由各工具 handler 负责，不因为模型允许就跳过路径等硬限制。
"""

from dataclasses import dataclass

from langchain_core.tools import BaseTool


PermissionMode = str | None  # None 表示未定级，不等于拒绝或允许。


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    """一个工具在 common_agent 中的治理策略。

    ``permission`` 是初始策略：
    - allow：可以自动执行；
    - ask：执行前通过 LangGraph interrupt 请求人工确认；
    - None：每次调用交给安全子 Agent 决定，不修改登记表。
    """

    source: str
    permission: PermissionMode = None


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
        if policy.permission not in {"allow", "ask", None}:
            raise ValueError(f"不支持的权限模式：{policy.permission}")
        self._entries[tool.name] = RegisteredTool(tool=tool, policy=policy)

    def tools_for_model(self) -> list[BaseTool]:
        """登记的工具都可见；不能提供的能力不登记，逐次审查可拒绝某次调用。"""
        return [entry.tool for entry in self._entries.values()]

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
    skill_tools: list[BaseTool] | None = None,  # 没有此类工具时省略，下面统一按空清单处理。
    memory_tools: list[BaseTool] | None = None,
    web_tools: list[BaseTool] | None = None,
    routing_tools: list[BaseTool] | None = None,
    shell_tools: list[BaseTool] | None = None,
) -> ToolRegistry:
    """只固定明确的策略；没有配置的工具交给安全子 Agent，不默认放行。"""

    registry = ToolRegistry()
    # 只给明确列出的只读工具自动执行权限。按来源分表，避免外部 MCP 工具
    # 仅凭与本地工具同名就拿到本地权限；路径和参数校验仍由 handler 负责。
    permissions_by_source = {
        "workspace": {
            "list_workspace_files": "allow",
            "read_text_file": "allow",
            "search_workspace_text": "allow",
            "delete_file": "ask",  # 删除整个文件不进入回收站，必须获得批准。
        },
        "rag": {
            "search_knowledge_base": "allow",
        },
        "mcp": {},  # 远程工具均未定级，仍经过安全子 Agent 审查。
        "skills": {
            "skill_view": "allow",  # 只读登记过的指南；allow 不会赋予指南里的操作额外权限。
        },
        "memory": {
            "memory_read": "allow",
        },
        "web": {},  # 外部查询交给子 Agent 审查；本轮不增加敏感信息检测。
        "routing": {"upgrade_to_strong": "allow"},  # 只改变模型档位，不改变其他工具权限。
        "shell": {},  # 不再接受主模型自报免审批，统一交给独立子 Agent 审查。
    }

    for source, tools in (
        ("workspace", workspace_tools),
        ("rag", rag_tools),
        ("mcp", mcp_tools),
        ("skills", skill_tools or []),  # None 转为空列表，统一走下面同一套登记循环。
        ("memory", memory_tools or []),
        ("web", web_tools or []),
        ("routing", routing_tools or []),
        ("shell", shell_tools or []),
    ):
        for tool in tools:
            permission = permissions_by_source[source].get(tool.name)  # 未登记策略为 None，执行前必须审查。
            registry.register(tool, ToolPolicy(source=source, permission=permission))

    return registry
