# common_agent v1.0：两条未来路线共用的完整 Agent 内核

这是一个小而完整的 Agent 应用，不是单文件演示。当前阶段只做好共同底座：

- 真实模型：通过 OpenAI 兼容协议调用 `.env` 中配置的模型；当前是千问。
- 真实工具循环：模型自己选择工具，工具真的读取或新建本地文件，再把结果交还模型。
- 工具治理登记：在 LangChain Tool 之上记录工具来源、allow/ask/deny 权限、风险和可用范围。
- 真实向量 RAG：文档切块，调用百炼 `text-embedding-v4`，向量存入 SQLite，语义检索返回原文与 source。
- 真实 MCP：官方 MCP Python SDK 2.2.0；Client 通过 stdio 启动独立 Server，动态发现 Schema 并调用工具。
- 真实持久化：LangGraph checkpoint 写入 SQLite，同一 `thread_id` 重启后仍能续聊。
- 真实安全边界：工具只能访问本项目 `workspace`，不能用 `../` 逃出去，不能覆盖文件。
- 真实交互入口：既能在 PyCharm 运行，也能在 PowerShell 连续聊天。
- 真实交互链路：可以直接体验“模型 → 工具 → 模型”和 SQLite 会话续聊。

它仍然不包含数据分析、多智能体、复杂规划或自主执行；这些是两台电脑分叉后的业务能力。
但 RAG 和 MCP 是两条方向都会复用的标准接入能力，因此已经作为真实、最小、完整的参考实现纳入共同基线。

## 1. 整体执行流程

```text
用户在 CLI 输入问题
        │
        ▼
app/manual_loop.py 手写的图（唯一引擎）
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

工具治理采用三档 `allow / ask / deny`，并已经接入工具节点的真实分发：`allow` 才进入 `tool.ainvoke`，`ask` 通过 LangGraph `interrupt` 暂停，CLI 用 `Command(resume=...)` 恢复后才执行，`deny` 直接拒绝。后续 M7 只继续补取消、幂等和更复杂的审批恢复，不重复实现这条最小 HITL 链路。

## 2. 为什么手写主循环

`app/manual_loop.py` 里的图只有四个零件：两个节点、两条条件边。

```text
START ──> [model] ──有 tool_calls──> [tools] ──没转够──> [model]
             │                          │
             │ 没有 tool_calls          │ 转够轮数或发现连续重复
             v                          v
            END                        END
```

**两条结束路径的含义完全不同，这是整个项目最值得讲的一点：**

| 结束路径 | 谁决定的 | 含义 |
| --- | --- | --- |
| `model -> END` | 模型 | **正常出口**。模型不再申请工具，说明它认为信息够了 |
| `tools -> END` | 代码 | **保险丝**。转够 10 轮被强制掐断，不是正常结束 |

为什么必须有第二条？因为第一条只在模型「自己愿意停」时才生效。
模型可以一直申请工具把轮数耗完 —— 光靠模型自觉防不住转圈。

代价是：工具并行调用、错误分类这些细节都得自己补 —— 这正是后面几项要做的事。

`execute_tools_node` **不往外抛异常**，
而是把错误变成一条 `ToolMessage` 还给模型 —— 由模型决定是换参数重试、
换工具，还是如实告诉用户。这是 Agent 和普通脚本的本质区别。

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
│  ├─ tool_registry.py       # 工具契约、来源、权限和风险登记表
│  ├─ manual_loop.py         # 手写主循环：两个节点 + 两条条件边（唯一引擎）
│  ├─ agent.py               # 模型 + 三类工具 + 组装图
│  ├─ display.py             # 把执行轨迹显示给人
│  └─ cli.py                 # 异步多轮命令行产品入口
├─ mcp_servers/
│  └─ common_tools_server.py # 真正独立的 MCP 2.x stdio Server
├─ workspace/
│  ├─ notes/                 # Agent 运行中自己生成的文件（不提交）
│  └─ knowledge/             # RAG 文档目录
├─ data/
│  ├─ agent.sqlite           # 对话 checkpoint
│  └─ knowledge.sqlite       # RAG 文本块与向量
├─ .env                      # 本机真实密钥，已被 gitignore
├─ .env.example              # 可分享的配置模板，无密钥
└─ requirements.txt          # 锁定依赖版本
```

推荐按这个顺序学习：

1. `config.py` → `workspace_tools.py`：先复习配置和普通工具。
2. `rag_tools.py`：看清完整 RAG 数据链路。
3. `common_tools_server.py` → `mcp_bridge.py`：看清 MCP 的两个进程。
4. `tool_registry.py` → `agent.py`：理解工具如何登记、筛选后汇入同一个 Agent。
5. `cli.py`：理解外层如何启动和持续运行会话。

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
请列出工作区文件，并在 notes 下新建一个 hello.md 写入一句问候。
```

终端会把模型的工具决定、工具真实返回和最终答案分开显示。透明轨迹非常重要：
否则模型即使没有查文件，我们也可能误以为它查过了。

继续体验写工具：

```text
请在 notes 目录新建 first_note.md，内容是“我们完成了真实工具调用”。
```

再让它列目录或读取该文件，就能看到真实落盘结果。再次用同名文件写入会被拒绝，不会悄悄覆盖。

体验 RAG：

```text
请先为 knowledge 目录建立知识索引，再用知识库回答：共享内核的 RAG 验证暗号是什么？请附来源。
```

第一次建库会真实调用 Embedding API；文件没有变化时再次建库会根据 SHA-256 摘要跳过，避免重复消耗额度。

体验 MCP：

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

## 7. 当前学习主线

当前只沿着产品主线阅读和开发：配置 → 工作区工具 → RAG → MCP → Agent 组装 → CLI 会话。
当前主线不包含评测/eval 目录和验收脚本；等核心能力完成后另行设计，不要在现在的阅读过程中寻找它们。

主循环现在会记录本轮上一次“工具名 + 参数”的调用签名。如果模型连续发出完全相同的工具请求，
系统会把“已重复、停止继续调用”作为工具结果写回 State，并安全结束本轮，避免无意义地消耗额度或重复产生副作用。
用户发来新问题时，本轮的工具轮数、重复签名和停止原因都会清零；它们不会污染同一 `thread_id` 的下一次回答。

项目总路线请看：`00_项目完善路线图_融合最终版.md`。

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

本项目整体就是共同基线。确认主线能力学懂并能运行后，两台电脑分别复制/克隆同一个版本，再各自加业务工具：

- 数据分析方向以后增加表格读取、数据检查、Python 沙箱和图表，复用当前 RAG/MCP 接口。
- 通用深度方向以后增加网络搜索、任务规划、HITL 和子 Agent，复用当前 RAG/MCP 接口。

当前阶段不实现这些分支。共同内核的完成标准，是第 1～6 节的主线能力能够被读懂、运行并逐步接入手写循环。

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
- 未有：PDF/Word 解析、混合检索、rerank、权限过滤和大规模 ANN 向量数据库。

这些未有能力不是共同内核必须项，应在具体业务分支出现真实需求时增加。

## 12. TODO：升级为真正的向量数据库检索

- [ ] 用主流向量检索方案替换当前“从 SQLite `fetchall()` 全部向量，再在 Python 内逐条计算余弦相似度”的教学实现。
- [ ] 优先评估 PostgreSQL + pgvector；如果项目更适合独立向量数据库，再评估 Qdrant。
- [ ] 让数据库或向量索引直接完成 ANN / Top-K 候选召回，查询时只把少量相关文本块返回 Python，避免知识库扩大后占满内存。
- [ ] 保留文档路径、分块编号、Embedding 模型等元数据，并支持按业务字段过滤。
- [ ] 增加增量索引、空知识库和较大数据量下的工程测试；混合检索与 rerank 暂留到向量库改造之后再决定。

这项改造暂不在当前源码阅读阶段实施；先把共同内核现有链路读完，再单独设计和迁移。
