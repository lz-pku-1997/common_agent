# common_agent v1.0：两条未来路线共用的完整 Agent 内核

这是一个小而完整的 Agent 应用，不是单文件演示。当前阶段只做好共同底座：

- 真实模型：通过 OpenAI 兼容协议调用 `.env` 中配置的模型；当前是千问。
- 真实工具循环：模型自己选择工具，工具真的读取或新建本地文件，再把结果交还模型。
- 真实向量 RAG：文档切块，调用百炼 `text-embedding-v4`，向量存入 SQLite，语义检索返回原文与 source。
- 真实 MCP：官方 MCP Python SDK 2.2.0；Client 通过 stdio 启动独立 Server，动态发现 Schema 并调用工具。
- 真实持久化：LangGraph checkpoint 写入 SQLite，同一 `thread_id` 重启后仍能续聊。
- 真实安全边界：工具只能访问本项目 `workspace`，不能用 `../` 逃出去，不能覆盖文件。
- 真实交互入口：既能在 PyCharm 运行，也能在 PowerShell 连续聊天。
- 真实验收：本地单元测试验证工具；联网 smoke test 验证“模型 → 工具 → 模型”和 SQLite。

它仍然不包含数据分析、多智能体、复杂规划或自主执行；这些是两台电脑分叉后的业务能力。
但 RAG 和 MCP 是两条方向都会复用的标准接入能力，因此已经作为真实、最小、完整的参考实现纳入共同基线。

## 1. 整体执行流程

```text
用户在 CLI 输入问题
        │
        ▼
LangChain create_agent（内部由 LangGraph 驱动循环）
        │
        ├─ 模型能直接回答 ──────────────────────┐
        │                                       │
        └─ 模型申请一种或多种工具                │
                 │                              │
                 ├─ workspace 受限文件工具       │
                 ├─ RAG 向量知识检索             │
                 └─ MCP stdio 外部工具           │
                         │                       │
                         └─ 真实结果回到模型 ─────┤
                                                ▼
                                          最终回答

每一步消息和工具结果 ──> SQLite checkpoint（按 thread_id 隔离）
```

这里最关键的不是“调用了一次大模型”，而是形成了闭环：模型能观察工具结果，再决定继续调用工具还是回答。

## 2. 为什么使用 `create_agent`

LangGraph 1.x 已经把推荐的高层 Agent 入口放到 `langchain.agents.create_agent`。
我们在 01–19 章手写过 State、节点、边、工具循环和 checkpoint；作品阶段可以使用官方高层入口，
把精力放在工具质量、安全边界和产品行为上。它底层仍然运行在 LangGraph 上，并没有绕开所学知识。

官方资料：

- [LangChain Agents](https://docs.langchain.com/oss/python/langchain/agents)
- [LangGraph v1 migration guide](https://docs.langchain.com/oss/python/migrate/langgraph-v1)
- [Short-term memory / checkpointer](https://docs.langchain.com/oss/python/langchain/short-term-memory)
- [LangChain MCP](https://docs.langchain.com/oss/python/langchain/mcp)
- [官方 MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [百炼文本向量模型](https://help.aliyun.com/zh/model-studio/embedding)

## 3. 文件结构与阅读顺序

```text
common_agent/
├─ run_cli.py                 # PyCharm 从这里启动
├─ app/
│  ├─ config.py              # 路径、.env、模型配置
│  ├─ workspace_tools.py     # 四个真实工具与安全边界
│  ├─ rag_tools.py           # 切块、真实 Embedding、SQLite 向量检索
│  ├─ mcp_bridge.py          # MCP 动态发现到 LangChain 工具的桥
│  ├─ agent.py               # 模型 + 三类工具 + Agent 图
│  ├─ display.py             # 把执行轨迹显示给人
│  ├─ cli.py                 # 异步多轮命令行产品入口
│  ├─ rag_smoke_test.py      # RAG 独立真实验收
│  ├─ mcp_smoke_test.py      # MCP 独立真实验收
│  └─ smoke_test.py          # Agent + RAG + MCP 联合验收
├─ mcp_servers/
│  └─ common_tools_server.py # 真正独立的 MCP 2.x stdio Server
├─ workspace/
│  ├─ welcome.md             # Agent 可操作的真实工作区
│  └─ knowledge/             # RAG 文档目录
├─ data/
│  ├─ agent.sqlite           # 对话 checkpoint
│  └─ knowledge.sqlite       # RAG 文本块与向量
├─ tests/test_local.py       # 8 项不联网测试
├─ .env                      # 本机真实密钥，已被 gitignore
├─ .env.example              # 可分享的配置模板，无密钥
└─ requirements.txt          # 锁定依赖版本
```

推荐按这个顺序学习：

1. `config.py` → `workspace_tools.py`：先复习配置和普通工具。
2. `rag_tools.py` → `rag_smoke_test.py`：看清完整 RAG 数据链路。
3. `common_tools_server.py` → `mcp_bridge.py` → `mcp_smoke_test.py`：看清 MCP 的两个进程。
4. `agent.py` → `cli.py`：理解三类工具如何汇入同一个 Agent。
5. `smoke_test.py`：看端到端验收如何防止“看起来能跑”。

## 4. 第一次安装

在 PowerShell 进入本目录后执行：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果已经由 GPT 配好环境，不需要重复安装。

### PyCharm 解释器

打开本项目后，把解释器选为：

```text
C:\Users\75662\Desktop\agent_projects\common_agent\.venv\Scripts\python.exe
```

然后右键运行 `run_cli.py`。

## 5. 运行

PyCharm：右键 `run_cli.py` → Run。

PowerShell：

```powershell
.\.venv\Scripts\python.exe run_cli.py
```

第一次可以输入：

```text
请列出工作区文件，并读取 welcome.md，告诉我项目代号。
```

终端会把模型的工具决定、工具真实返回和最终答案分开显示。透明轨迹非常重要：
否则模型即使没有查文件，我们也可能误以为它查过了。

继续测试写工具：

```text
请在 notes 目录新建 first_note.md，内容是“我们完成了真实工具调用”。
```

再让它列目录或读取该文件，就能看到真实落盘结果。再次用同名文件写入会被拒绝，不会悄悄覆盖。

测试 RAG：

```text
请先为 knowledge 目录建立知识索引，再用知识库回答：共享内核的 RAG 验证暗号是什么？请附来源。
```

第一次建库会真实调用 Embedding API；文件没有变化时再次建库会根据 SHA-256 摘要跳过，避免重复消耗额度。

测试 MCP：

```text
请使用 MCP 工具准确计算 37+58，并读取 MCP Server 所在电脑的当前时间。
```

你会在终端轨迹中看到 `add_numbers` 与 `get_current_time`。它们来自独立 Server 进程，不是 Agent 内部假装调用。

## 6. 会话与记忆

启动时的“会话编号”就是 LangGraph 的 `thread_id`：

- 同一个编号：读取同一段历史，可在程序重启后续聊。
- 不同编号：历史隔离，互不污染。
- `/thread work-1`：运行中切换会话。
- `/thread`：查看当前编号。
- `/exit`：退出；SQLite 数据仍保留。

这里的 SQLite 是对话短期记忆/运行状态，不是公共记忆知识库。两者目的不同，不能混为一谈。

## 7. 测试与验收

不联网测试：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_local -v
```

RAG 独立验收（真实消耗少量 Embedding 额度）：

```powershell
.\.venv\Scripts\python.exe -m app.rag_smoke_test
```

MCP 独立验收（本机跨进程，不调用聊天模型）：

```powershell
.\.venv\Scripts\python.exe -m app.mcp_smoke_test
```

完整联网验收（真实消耗聊天模型和 Embedding 额度）：

```powershell
.\.venv\Scripts\python.exe -m app.smoke_test
```

联网验收不是只检查“返回了一段文字”。它会断言：

1. RAG 真实建立或复用 `text-embedding-v4` 向量索引；
2. 模型实际调用了 `search_knowledge_base`；
3. 模型实际通过 MCP 调用了 `add_numbers`；
4. 最终回答包含检索暗号、文件 source 和 MCP 计算结果 95；
5. 关闭并重开 SQLite 后，完整消息与工具调用链仍可恢复。

任一条件不成立，程序会直接报错，不会把失败包装成成功。

## 8. 当前安全边界

- 只允许 `.md`、`.txt`、`.json`、`.csv`、`.py` 文本文件。
- 单个读取文件最大 1 MB，单次写入最大 10 万字符。
- 搜索最多返回 50 条，列目录最多展示 200 项，防止上下文无限膨胀。
- 只允许 workspace 内路径。
- 只能新建，不能覆盖、删除或运行 shell 命令。
- RAG 索引只读取 workspace 中大小合规的 md/txt/json/csv；检索结果被当作不可信证据，不当作系统指令。
- MCP Server 不接收 `.env` 或 API Key，只暴露显式注册的两个工具；stdio 生命周期由 Client 管理。
- `.env` 和 SQLite 已加入 `.gitignore`。

这是内核的能力边界，不是缺陷：Agent 的工具权限必须按真实需求逐项开放，不能一开始就把整台电脑交给模型。

## 9. 两台电脑分叉时复制什么

本项目整体就是共同基线。确认学懂和验收通过后，两台电脑分别复制/克隆同一个版本，再各自加业务工具：

- 数据分析方向以后增加表格读取、数据检查、Python 沙箱、图表和分析评测，复用当前 RAG/MCP 接口。
- 通用深度方向以后增加网络搜索、任务规划、HITL、子 Agent 和更完整评测，复用当前 RAG/MCP 接口。

当前阶段不实现这些分支。共同内核的完成标准，就是本 README 第 7 节全部真实验收通过。

## 10. 一个重要的版本事实

当前官方 `mcp` Python SDK 已是 2.2.0，而 `langchain-mcp-adapters` 0.3.2 仍声明依赖 `mcp>=1.24,<2`。
本项目没有偷偷降级到 1.x，而是直接使用官方 2.2 `MCPServer` 和 `Client`，自己实现约 100 行透明桥接：

1. Client 通过 stdio 启动 Server；
2. `list_tools()` 动态获取名称、说明和 JSON Schema；
3. Schema 原样变成 LangChain `StructuredTool`；
4. Agent 异步调用时，再由 Client 真正执行 MCP 工具。

等官方适配器支持 MCP 2.x 后，可以替换 `mcp_bridge.py`，其他模块和 MCP Server 不必重写。

## 11. RAG 的真实边界

本项目是完整的最小向量 RAG，但还不是企业搜索平台：

- 已有：切块、重叠、内容哈希增量索引、真实 Embedding、SQLite 向量保存、余弦召回、来源引用。
- 未有：PDF/Word 解析、混合检索、rerank、权限过滤、大规模 ANN 向量数据库和 RAG 评测集。

这些未有能力不是共同内核必须项，应在具体业务分支出现真实需求时增加。
