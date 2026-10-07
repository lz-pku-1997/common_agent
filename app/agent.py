"""创建 common_agent 的模型与 Agent 图。

只有一套引擎：`app/manual_loop.py` 里自己画的图。
每一步都在明面上，被问「循环什么时候停」可以指着某一行代码回答。
"""

from langchain_openai import ChatOpenAI

from app.config import AGENT_RULES_PATH, load_model_settings
from app.manual_loop import build_agent_graph, upgrade_to_strong
from app.mcp_client import load_mcp_tools
from app.memory import MEMORY_TOOLS, load_memory_context
from app.rag_tools import RAG_TOOLS, search_knowledge_base
from app.skills import build_skill_tools, load_skill_catalog
from app.tool_registry import build_tool_registry
from app.workspace_tools import WORKSPACE_TOOLS, read_text_file
from app.web_tools import WEB_TOOLS
from app.shell_tools import SHELL_TOOLS
from app.safety_agent import build_safety_reviewer


def create_chat_model(mode: str) -> ChatOpenAI:
    """按照 .env 创建真实聊天模型。

    ChatOpenAI 在这里是“OpenAI 兼容协议客户端”，不代表只能调用 OpenAI。
    当前 .env 指向 DashScope，所以请求会真正发给千问。
    """

    settings = load_model_settings(mode)  # 模型名与输入窗口成对切换；地址和密钥共用。
    return ChatOpenAI(
        model=settings["model"],
        api_key=settings["api_key"],
        base_url=settings["base_url"],
        temperature=0,
        streaming=True,
        timeout=60,
        max_retries=2,
        profile={"max_input_tokens": settings["max_input_tokens"]},
    )


async def build_common_agent(checkpointer):
    """组装模型、工具循环、SQLite 会话与文件长期记忆，返回可运行的 Agent。

    checkpointer 由外层传入，因为 SQLite 连接必须在使用期间保持打开。
    """

    # 运行规则每次启动从文件读取，维护者改 Markdown 后重启即可生效。
    system_prompt = AGENT_RULES_PATH.read_text(encoding="utf-8-sig").strip()  # 启动时读取固定规则；每次模型请求复用它。
    if not system_prompt:
        raise ValueError("prompts/AGENTS.md 不能为空。")
    catalog = load_skill_catalog()  # 只读取每个 SKILL.md 的 name/description，不读取正文。
    if catalog:
        # 模型只看到每个 Skill 的名称和用途；全文留给 skill_view 按需读取。
        descriptions = [f"- {name}：{description}" for name, (description, _) in catalog.items()]  # 每个技能转成一行清单；_ 表示此处不用路径。
        system_prompt += "\n\n可用 Skill（名称和用途）：\n" + "\n".join(descriptions)  # 只把目录注入提示词，省下未选中指南的上下文。

    # MCP 工具不是写死在 Agent 里的：启动时先向 Server 请求工具清单和 JSON Schema。
    mcp_tools = await load_mcp_tools()
    evidence_tools = [read_text_file, search_knowledge_base]  # 本地文件和 RAG 都可用于取证。
    for tool in mcp_tools:
        if tool.name in {"context7__resolve-library-id", "context7__query-docs"}:
            evidence_tools.append(tool)  # 复用已加载的 Context7 工具，查询公开库文档。
    # 先登记来源和权限，再把允许暴露的工具交给主循环。
    # LangChain 仍负责 Tool/Schema；Registry 只负责项目自己的治理元数据。
    tool_registry = build_tool_registry(  # 把 Skill 工具和其他来源放进同一权限登记表。
        WORKSPACE_TOOLS, RAG_TOOLS, mcp_tools, build_skill_tools(catalog), MEMORY_TOOLS, WEB_TOOLS,
        [upgrade_to_strong],  # 升级是内置控制工具，也经过同一权限登记与执行出口。
        shell_tools=SHELL_TOOLS,
    )
    models = {mode: create_chat_model(mode) for mode in ("fast", "strong")}  # 共用一个图；State 档位决定本次调用哪个客户端。

    return build_agent_graph(
        models=models,
        tools=tool_registry.tools_for_model(),  # 工具可见不等于可以执行，调用时才按参数审查权限。
        tool_registry=tool_registry,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        memory_context_loader=load_memory_context,  # 每次请求重读文件，不把长期记忆固化进图或 checkpoint。
        safety_reviewer=build_safety_reviewer(models["fast"], evidence_tools),  # 子 Agent 只拿取证工具，不拿父 Agent 的完整工具清单。
    )
