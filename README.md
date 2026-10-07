# common_agent v1.0：通用 Agent 应用内核

这是一个小而完整的 Agent 应用，不是单文件演示。当前项目作为一套独立的通用 Agent 内核维护：

- 真实模型：通过 OpenAI 兼容协议调用 `.env` 中配置的模型；当前是千问。
- 真实工具循环：模型自己选择工具，工具真的读取或新建本地文件，再把结果交还模型。
- 工具治理登记：固定 allow / ask；未定级为 None，交给隔离的安全子 Agent 逐次决定 allow / ask / deny。服务不可用时退回 ask，路径等代码校验不变。
- 工具数据隔离：workspace、RAG、MCP 的真实返回统一标记为不可信数据，不能获得指令权限。
- 文件化运行规则：启动时读取 `prompts/AGENTS.md`，编辑规则无需修改 Python，重启后生效。
- Skill 渐进式加载：系统提示只带名称和用途，`skill_view(name)` 按需读取项目维护的任务指南，仍受用户要求和工具权限约束。
- 分层长期记忆：`user.md` 画像与 `memory.md` 索引常驻；动态主题按需读取，新建和更新经过人工批准。
- 真实 RAG：文档切块与向量入库；SQLite 关键词检索和向量检索双路召回，RRF 融合后由百炼 `qwen3-rerank` 重排，返回原文与 source。
- 真实 MCP：官方 Python SDK 远程 Streamable HTTP 客户端；配置驱动工具发现、筛选和调用，不自建服务器。
- 真实持久化：LangGraph checkpoint 写入 SQLite，同一 `thread_id` 重启后仍能续聊。
- 上下文预算：工具单条结果限长；历史接近预算时滚动摘要旧消息，原始 checkpoint 仍保留。
- 真实安全边界：普通文件工具限定 `workspace`；Skill 只读已登记指南；长期记忆工具限定 `memory/`，新建和更新都需批准。
- 文件增删改查：新增 `delete_file`，只删除 workspace 单个文件、必须审批、不进入回收站。
- PowerShell 能力兜底：`execute_command(command, description)`，独立安全子 Agent 审查权限；需要审批时展示中文说明和原始命令。仅适合可信用户本机使用，没有沙箱隔离。
- 真实交互入口：既能在 PyCharm 运行，也能在 PowerShell 连续聊天。
- 真实交互链路：可以直接体验“模型 → 工具 → 模型”和 SQLite 会话续聊。
- 联网搜索：`web_search` 调用千问 Responses API，搜索与网页提取对外合成一个工具；查询需人工确认，结果带来源并按外部数据隔离。
- 单向模型交接：会话默认 Flash，棘手任务通过 upgrade_to_strong 升级 Max；沿用原循环和历史，此后不降档。
- 会话分叉：`/time` 查看已完成轮次；从历史轮次开新分支，只复制对话状态，不回滚现实操作。
- CLI 流式：`ChatOpenAI(streaming=True)` 让模型正文分块到达 `messages` 流并即时显示；`custom` 流实时报告工具开始和服务补试，审批与最终 checkpoint 保持原机制。

当前内核暂不包含数据分析、多智能体、复杂规划或自主执行；是否增加这些能力，后续按真实需求逐项决定。
Shell 的免审批判断依赖模型，可能出错；关键词名单也不完整，不能保证拦截所有危险命令。默认 cwd 为 workspace 不限制绝对路径、网络或系统访问。
RAG 和 MCP 已作为当前内核的标准接入能力，提供真实、最小、完整的参考实现。

## 1. 整体执行流程

### 本轮新增命令

| 命令 | 作用 |
|---|---|
| `/time` | 按旧到新列出当前会话已完成的轮次 |
| `/time N 新编号` | 从编号 N 的完成轮次开指定新会话，并切换过去 |
| `/fork 新编号` | 从最新完成轮次开指定新会话，并切换过去；已有编号拒绝覆盖 |

快/强模型共享 `LLM_API_KEY` 与 `LLM_BASE_URL`，分别配置
`LLM_FAST_MODEL` / `LLM_FAST_MAX_INPUT_TOKENS` 与
`LLM_STRONG_MODEL` / `LLM_STRONG_MAX_INPUT_TOKENS`。四项均须填写，聊天不再使用手动切换。
模型名称必须在当前端点可用；窗口按官方文档填写，不能只改名称而沿用另一模型的窗口。
没有分类模型请求或 route 节点，也没有手动模型切换命令。新会话默认使用 Flash；
Flash 判断当前任务难以可靠完成时，调用 upgrade_to_strong(reason)。只有 Flash 的说明书包含该工具。
工具节点先生成与 tool_call_id 对应的交接回执，并返回 model_mode="strong"，由 LangGraph 写回 State；
随后同一个 context→model→tools 循环改用 Max，消息、摘要和已有工具结果不清空、不重跑。
同一 thread_id 的后续问题、审批恢复、退出重启和分叉均沿用档位；另开新会话才默认回到 Flash。
摘要模型及输入窗口跟随档位，其他工具的权限和人工审批不变。CLI 显示 `[模型升级]`。
升级由模型判断，不保证判断准确；只允许 fast→strong，可减少反复切换，但不保证缓存命中或省钱。
应用层复用 State 不等于供应商能复用不同模型的缓存；效果留到评测。
联网搜索固定复用 `LLM_STRONG_MODEL`，不随会话档位切换，无需另配模型名称。

联网请求发送到 `{LLM_BASE_URL}/responses`，不是 Chat Completions；端点必须支持内置搜索。
公开查询示例：`请使用 web_search 查询 Python pathlib 官方文档的用途，并提供来源链接。`
批准后才发出请求；搜索可能计费，服务补试也可能增加费用。网页正文仍是不可信数据。
搜索请求显式采用低推理强度，避免默认长推理拖慢查询；返回中必须存在真实 web_search_call，
没有搜索调用记录的纯模型回答不能冒充联网成功。2026-10-02 已用公开问题验证真实搜索与审批后流式回答。

`/time` 和 `/fork` 共用 `app/sessions.py`：新分支复制 messages、摘要和摘要游标，
清零本轮计数和停止原因。原会话不变，已经保存的文件不会消失；不开放工具执行中间点重放。
分叉保存后不执行历史节点；下一次用户输入才启动新的模型请求。运行规则、Skill 和长期记忆
仍读取当前项目文件，不回滚到历史版本。只有单进程 CLI，不承诺跨进程同时分叉同一编号。

流式入口使用 `messages`（回答片段）、`updates`（工具结果／审批断点）和 `custom`
（模型升级／工具开始／服务补试）三种 LangGraph 流。只打印 model 节点的正文，不展示 context 摘要或思考块。
审批恢复仍传 `Command(resume={"approved": bool})`，不产生第二次写入；最终 State 从 SQLite 读取。
停止时使用本轮专属说明，不把上一轮答案冒充本轮成功。Tracing 与评测本轮没有开发。

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
                 └─ MCP 远程外部工具             │
                         │                       │
                         └─ 真实结果回到模型 ─────┤
                                                ▼
                                          最终回答

每一步消息和工具结果 ──> SQLite checkpoint（按 thread_id 隔离）
```

这里最关键的不是“调用了一次大模型”，而是形成了闭环：模型能观察工具结果，再决定继续调用工具还是回答。

登记表只固定 `allow / ask`，其余工具策略为 `None`，先进入安全子 Agent；最终执行决策是 `allow / ask / deny`。`ask` 仍由主 Agent 的 `interrupt` 暂停，CLI 用 `Command(resume=...)` 恢复；子 Agent 不审批、不执行待审工具。审查节点先保存当前批次决定，审批恢复不会重复取证；任何一次拒绝都会使同批工具不执行。

## 2. 为什么手写主循环

`app/manual_loop.py` 有四个职责清楚的节点：上下文、模型、审查、工具。模型升级复用工具节点，不新建第二套循环。

```text
START ──> [context] ──> [model] ──有 tool_calls──> [review] ──> [tools]
              ▲              │                         │
              │              │ 无 tool_calls            │ 可继续
              │              ▼                         │
              │             END                        │
              └────────────────────────────────────────┘
工具侧触发保险丝或不可重试错误 ──> END
```

`context` 只在估算输入接近预算时调用模型更新旧摘要；正常短对话不额外调用模型。
摘要调用失败或返回空内容时保留旧摘要，继续本轮回答。
工具结果超过 20000 字符时，先把未包装的完整正文存入 `workspace/.tool_results/`，
再把预览及续读提示写入 `ToolMessage`。写盘失败只返回截断内容和失败提示，不中断工具流程。
SQLite checkpoint 保留预览与落盘记录；模型输入保留最近三条工具结果，较早且保存成功的结果
换成带文件路径的占位提示。这只修改输入副本，不回写 checkpoint。
上下文预算在 `app/context.py`，工具字符预算在 `app/config.py`：策略上限为约 200000 token 触发、压到约 100000 token；
实际触发值取策略上限与本轮所选档位 `LLM_FAST_MAX_INPUT_TOKENS` / `LLM_STRONG_MAX_INPUT_TOKENS` 的 80% 中较小者；
目标值不超过实际触发值的一半。换模型时须同步修改窗口配置。
token 计数是保守估算，工具预览的 20000 则是字符数，两者不是同一单位。
摘要使用目标、约束、关键事实、用户消息要点、已完成、待办六个小节；提示词要求约 2000 字符，
8000 字符的硬截断只作失控兜底，不代表无损摘要。

读取使用 `read_text_file(relative_path=".tool_results/文件名.txt", start_line=1, max_lines=2000)`。
行号从 1 开始；不填 `max_lines` 时最多读 2000 行，页尾给出下一页行号或明确提示文件结束。
单行正文最多 5000 字符，超长补 `...`；整页含行号和提示最多 20000 字符，装不下的行留给下一页。
超长行被省略的尾部不会在下一页续读。搜索沿用同一套行号，每条命中的正文预览最多 500 字符。
搜索多个文件用 `search_workspace_text(query="关键词", file_pattern="*.txt")`；只搜某份落盘结果，就把同一个参数换成 `file_pattern=".tool_results/文件名.txt"`，避免历史结果占满 50 条上限。参数可填具体文件或通配表达式；候选文件在打开前必须位于 workspace 内。
大于 1 MB 的文件也能分段读、逐行搜索；原有路径边界不变，RAG 索引的文件大小限制不变。
`artifact` 是给程序看的保存记录，不进入模型请求；清理副本时置空，checkpoint 原记录保留。
落盘文件当前不会自动过期或删除，不要手动删掉仍需续读的文件；这些运行产物已被 `.gitignore` 排除。

**两条结束路径的含义完全不同，这是整个项目最值得讲的一点：**

| 结束路径 | 谁决定的 | 含义 |
| --- | --- | --- |
| `model -> END` | 模型 | **正常出口**。模型不再申请工具，说明它认为信息够了 |
| `tools -> END` | 代码 | **保险丝**。转够 25 轮被强制掐断，不是正常结束 |

为什么必须有第二条？因为第一条只在模型「自己愿意停」时才生效。
模型可以一直申请工具把轮数耗完 —— 光靠模型自觉防不住转圈。

代价是：工具并行调用等细节需要自己决定和实现。

`execute_tools_node` 把工具失败写成 `status="error"` 的 `ToolMessage`。
工具明确抛出 `FixableError`，或入参不符合工具声明的参数结构（由框架校验层拦下）时，
模型才有最多两次修改参数的机会；`NonRetryableError` 和未知异常立即收口。
`RetryableError` 表示可安全重复的临时服务故障：代码原地等待 1、4、9 秒补试三次，
连首次调用最多四次，不重复请求模型或人工审批；耗尽后安全停止。
权限拒绝独立处理；用户拒绝人工确认后，
模型会得到一次不带工具的回答机会，解释操作没有执行。
MCP Server 正常返回的工具错误属于有限改参重试。连接关闭或超时只有在请求尚未发出时
才原地补试；请求发出后结果不明则安全停止，不擅自重复远端操作。

官方资料：

- [LangChain Agents](https://docs.langchain.com/oss/python/langchain/agents)
- [LangGraph v1 migration guide](https://docs.langchain.com/oss/python/migrate/langgraph-v1)
- [Short-term memory / checkpointer](https://docs.langchain.com/oss/python/langchain/short-term-memory)
- [LangChain MCP](https://docs.langchain.com/oss/python/langchain/mcp)
- [官方 MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [百炼文本向量模型](https://help.aliyun.com/zh/model-studio/embedding)

### 2.1 运行规则与 Skill

`prompts/AGENTS.md` 是 common_agent 运行时的行为规则，由 `app/agent.py` 在启动时读取。
它不属于仓库根目录给开发助手看的 `AGENTS.md`，也不保存用户长期记忆。

Skill 是一个任务操作指南，放在 `skills/<名称>/SKILL.md`。启动时读取的只有文件头：

```markdown
---
name: workspace-note
description: 用户要求根据 workspace 中的文件整理摘要、复习笔记或新建笔记时使用
---

这里才是详细步骤，模型调用 skill_view 后才会看到。
```

当前格式只支持 `name`、`description` 两个单行无引号字段，不支持多行 YAML、嵌套数据或其他字段。
`name` 与父目录同名，使用小写字母、数字和单个短横线，最长 64 字符；用途最长 1024 字符。
完整 Skill 文件不超过 20000 字符，确保指南一次读完整。超出时应由维护者缩短。
没有 Skill 目录或目录为空时，正常运行，只是不提供 `skill_view` 工具。

```text
启动：运行规则 + 所有 Skill 的名称和用途 → 模型
任务相关：模型 → skill_view(name) → 完整步骤 → 现有工具 → 回答
```

清单通过普通字符串拼接加入系统提示；正文由只读工具返回，随后留在现有消息历史中，
继续使用现有 checkpoint 和摘要机制。新增正文仍会占上下文，节省的是未选中 Skill 的正文。
这版不自动执行 Skill 中的脚本，也不加载附属资源目录；步骤必须依靠当前已有工具完成。

项目 Skill 由维护者编辑，Agent 的 workspace 写工具不能修改它。`skill_view` 只接收清单中的名称，
按登记路径读取，拒绝目录外的符号链接。它按 `skills` 来源登记为 `allow`；同名外部 MCP 工具不会继承权限。
普通工具结果仍标为不可信数据；Skill 返回明确标为操作指南，不能覆盖用户要求、运行规则或实际权限。
Skill 即使要求写文件，也必须经过原有 HITL 审批。请只放入自己审阅过的项目指南。
新增或修改元数据、运行规则后需要重启；Skill 正文在调用时读取。

可以这样体验（先用实际文件名替换示例路径）：

```text
请按 workspace-note 技能，把 knowledge/某份资料.md 整理成复习笔记，先只给我看，不保存。
```

应先看到 `skill_view`，再看到资料读取工具。明确要求保存时，还应看到人工确认；拒绝后不会写入。
另一份示例是 `knowledge-answer`，用于根据知识库资料回答并保留来源。

## 3. 文件结构与阅读顺序

```text
common_agent/
├─ run_cli.py                 # PyCharm 从这里启动
├─ prompts/AGENTS.md          # common_agent 的运行规则，启动时加载
├─ skills/
│  ├─ workspace-note/SKILL.md # 工作区资料整理为笔记
│  └─ knowledge-answer/SKILL.md # 知识库检索与来源引用
├─ app/
│  ├─ config.py              # 路径、.env、模型配置
│  ├─ workspace_tools.py     # 列目录、读文件、搜索、新建和编辑，统一工作区边界
│  ├─ rag_tools.py           # 切块、真实 Embedding、SQLite 向量检索
│  ├─ mcp_client.py          # 远程 MCP 连接、工具筛选与 LangChain 接线
│  ├─ tool_registry.py       # 工具契约、来源和权限登记表
│  ├─ tool_errors.py         # 可重试/不可重试的工具失败约定
│  ├─ skills.py              # Skill 目录发现、元数据与按名称读取
│  ├─ context.py             # token 粗估、摘要切点和单条工具结果限长
│  ├─ manual_loop.py         # 手写上下文/模型/工具循环（唯一引擎）
│  ├─ agent.py               # 运行规则 + Skill 清单 + 模型/工具组装
│  ├─ display.py             # 把执行轨迹显示给人
│  └─ cli.py                 # 异步多轮命令行产品入口
├─ mcp.json                  # 远程服务地址、可选 headers 与工具名称列表
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
3. `mcp.json` → `mcp_client.py`：看清远程工具发现与调用。
4. `skills.py` → `tool_registry.py` → `agent.py`：理解规则、Skill 清单和工具如何汇入同一个 Agent。
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

修改已有文件使用 `edit_text_file(relative_path, old_text, new_text)`：先读原文，旧文本必须精确且唯一匹配；匹配不到或匹配多处时拒绝，模型应补充上下文再提交。`new_text=""` 可删除选中的那一段。编辑先经过独立权限审查，需要人工确认时批准后才写盘；先写同目录临时文件，再原子替换，保留未修改的 UTF-8 BOM 与换行格式。限制文件不超过 1 MB、修改后不超过 10 万字符，不做模糊替换或全局替换。

删除整个文件使用 `delete_file(relative_path)`：必须人工确认，允许工作区内任意后缀的单个文件，不递归删除目录，删除不进入回收站。

没有专用工具覆盖时使用 `execute_command`。固定调用 Windows 自带 `powershell.exe`，不加载用户 profile，关闭交互输入；每次独立进程，从 workspace 启动。命令等待上限 100 秒，超时/取消尝试 taskkill 清理进程树，外层仍有 120 秒总兜底。stdout/stderr 写临时文件，有限读取后走通用结果截断和不可信数据包装；命令失败不自动补试，退出码非零或超时应先检查是否有部分副作用。没有持久终端、沙箱或完整命令解析器。

Shell 不再接收主模型自报的 `needs_approval`，也不再维护危险词表；未定级调用由安全子 Agent 看真实契约和参数。批准界面仍显示模型中文说明、真实命令和工作目录；说明不保证准确。独立模型审查是风险判断，不是操作系统级安全隔离。

### 安全子 Agent：只读取证与独立决策

2026-10-05 顺序调整：Jev 先判断，uncertain 才取证。新顺序的 9 项本地分支检查通过；下述真实 API 样例验证的是此前先取证版本，新顺序尚未重新跑真实端到端验收。

实现位于 `app/safety_agent.py`。主图固定 `allow` 的读工具直接执行，固定 `ask` 的 `delete_file` 必须人工批准。文件写入、记忆更新、索引构建、Shell、远端 MCP 和联网工具未定级，逐次审查，不永久修改登记表。

```text
主模型申请一批工具
  → 固定 allow / ask 直接采用；None 调用各自启动安全子 Agent
  → asyncio.gather 并发收集审查决定 → review 节点保存 checkpoint
  → 主 Agent 统一人工确认（仅 ask）→ 按原顺序执行真实工具

独立审批偏好 + 当前用户诉求 + 待审工具契约和参数
  → Jev Choice 先判断（第一次证据列表为空）
      allow 且概率 > 0.85 → allow；低概率 allow → ask
      ask / deny → 返回决定
      uncertain → 千问检索内部工具知识、按需读文件 → 再交给 Jev
                  uncertain 时可继续只读取证，整次审查最多 120 秒；未尝试检索则 ask
  → 最多十次 Jev；仍不确定、服务出错或总等待超 120 秒 → ask
```

每个子任务有独立消息与证据列表，不读取父对话历史，不继承主 Agent 的长期记忆，只读取维护者单独编辑的 `memory/safety/user.md`；文件不存在时按无偏好处理。普通记忆工具不能修改这个路径。取证工具仅为 `read_text_file` 和 `search_knowledge_base`，不联网、不写文件、不审批。中间证据不回灌父消息，只保存本批次调用 ID 对应的最终决定；CLI 仅显示开始与决定，不把取证输出当用户回答。

在 `.env` 中填写 `OPENROUTER_API_KEY`，模型默认使用 `typesafe/jev-1.13`，也可通过 `JEV_MODEL` 指定其他 Jev 型号。Jev 经 OpenRouter 的 [Decisions API](https://openrouter.ai/api/alpha/decisions) 调用，不是 Chat Completions。只有选中 `allow` 且 `probabilities.allow > 0.85` 才自动放行；低概率或缺失/非法概率直接 `ask`，不继续取证。这里检查选项概率，不是总体 `confidence`；门槛不代表安全正确率。千问使用正常工具选择，由提示词要求检索，不强制设置 `tool_choice`，保留思考模式；本轮未尝试检索时退回人工确认。未填密钥直接退回人工确认，不空跑千问。**配置前需确认待审工具参数、当前用户诉求、权限偏好和只读取证结果可发送给 OpenRouter。** 本地桩模型与真实 SQLite 验证通过；2026-10-05 本机真实只读取证 → Jev 决策链路通过（两次 RAG、一次读文件，allow 概率 0.98），未执行待审 Shell；安全质量和复杂样本仍留到评测。

`memory/` 不进入 Git，新电脑需自行创建审批偏好文件。可写：只读且不涉及敏感数据的操作可自动运行；删除、覆盖、安装、未知脚本及 workspace 外操作需要确认；资料不足先取证、仍不确定请人判断。偏好不替代代码路径校验；Shell 仍无沙箱。

```text
请把 notes/first_note.md 中的“我们完成了真实工具调用”改成“我们完成了真实工具调用与文件编辑”。
```

体验 RAG：

```text
请先为 knowledge 目录建立知识索引，再用知识库回答：共享内核的 RAG 验证暗号是什么？请附来源。
```

第一次建库会真实调用 Embedding API；文件内容和模型没有变化时再次建库会跳过，避免重复消耗额度。本项目的知识库是测试数据：切块或分词规则改变后，删除 `data/knowledge.sqlite`，再运行建库工具即可重新生成，不维护旧库迁移代码。重新索引某目录时，还会清理该目录中已经删除的文件记录，不影响其他目录。
检索还会调用百炼重排接口：请先在 `.env` 填好 `RERANK_API_URL`（格式见 `.env.example`）。未配置时会明确报错；接口暂时不可用时会标明“仅按 RRF 排序”，不会假装已完成重排。

对话和向量可以不共用端点：`.env` 里配了 `EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY` / `RERANK_API_KEY` 就走这一套，没配则回落到对话的 `LLM_BASE_URL` / `LLM_API_KEY`。需要拆分的典型场景是——对话用套餐专属端点（套餐 key 打通用地址会 401，且套餐不提供向量与重排），向量和重排仍走通用端点。

体验 MCP：

```text
请使用高德 MCP 查询杭州天气。
```

在高德开放平台创建 Web 服务 Key，写入本机 `.env` 的 `AMAP_API_KEY`，不要提交真实密钥。`mcp.json` 使用 `${AMAP_API_KEY}` 引用它，启动时在内存替换；可选 `headers` 也支持此形式。

`mcpServers` 按服务名配置远程地址。`tools` 是允许接入的原始工具名称列表：不写则加载全部，空列表跳过服务，名称拼错则启动报错。仅开放 6 个工具：地点搜索（关键词/名称＋城市）、天气，以及驾车、步行、骑行、公交路线规划。改配置后重启，不做热加载、stdio 或 OAuth。`mcpServers` 设为空对象可以禁用 MCP。

模型工具名统一为 `服务名__工具名`（例如 `amap__maps_weather`），避免不同服务重名；远端调用使用原始名称。清单只是能力筛选，不是执行授权：全部 MCP 工具仍未定级，经过现有安全子 Agent 与必要的人工确认，成功结果仍套不可信来源包装。

发现和调用共用 URL/鉴权配置；每次结束关闭 Client/HTTP 连接，不关闭第三方服务器。请求发出后连接故障不自动补试，以免重复远端操作。2026-10-05 重构时，高德真实握手、15 个工具发现及原清单四种工具调用通过；另通过真实主模型 → 人工审批恢复 → 天气工具 → 最终回答的 CLI 链路（此项将安全决策替换为 ask，不代表 Jev 验收）。随后按用户要求改为上述 6 个工具，本地筛选验证通过，新增的步行、骑行、公交路线尚未真实调用。本地 demo Server 已删除，不再展示双传输能力。

## 6. 会话与记忆

启动时的“会话编号”就是 LangGraph 的 `thread_id`：

- 同一个编号：读取同一段历史，可在程序重启后续聊。
- 不同编号：历史隔离，互不污染。
- `/thread work-1`：运行中切换会话。
- `/thread`：查看当前编号。
- `/exit`：退出；SQLite 数据仍保留。

这里的 SQLite 保存会话历史与运行状态；下面的 Markdown 保存跨会话长期记忆，两者目的不同。

### 6.1 产品自己的长期记忆

```text
memory/                   # 本机私人数据，整个目录不提交 Git
├── user.md               # 用户画像和稳定偏好；不存在时正常运行
├── memory.md             # 程序生成的纯索引，不存记忆正文
└── topics/
    └── 任意动态主题.md     # 实际文件名使用英文短横线，如 project-stack.md
```

每次模型请求都重读画像并从主题文件重建索引，启动或索引丢失也会自动恢复。
索引每行形如 `- [项目选型](topics/project-stack.md) — 框架与存储的已确认决定`；
修改主题头部的标题或说明后，索引会跟着更新。项目事实与决定放主题正文，用户画像放 user.md。
画像最多 4000 字符，索引最多 8000 字符，单主题最多 16000 字符；超预算明确报错，不静默截断。

工具与权限：

| 工具 | 作用 | 权限 |
|---|---|---|
| memory_read(relative_path) | 完整读取画像、索引或某个主题 | allow |
| create_memory(relative_path, content) | 新建画像或动态主题，不覆盖 | 子 Agent 逐次审查 |
| update_memory(relative_path, old_text, new_text) | 精确替换唯一旧文本，保留其余内容 | 子 Agent 逐次审查 |

若审查结果为 ask，写入前审批展示路径及内容；更新展示 old_text 和 new_text。
更新前先读取文件；旧文本缺失或重复就要求重新读取，避免盲目覆盖。
主题格式为 `# 标题`、空行、单行说明、空行、正文；标题最多 80 字符，说明最多 160 字符。
例如 `topics/project-stack.md`：

```markdown
# 项目技术选型

记录服务入口与存储的已确认决定。

- HTTP 服务采用 Flask。
```

记忆工具限定在 memory/ 内；普通 workspace 工具不能写入这个目录。
每个文件先写同目录临时文件再替换，避免写到一半损坏正文。
正文与索引不是跨文件事务：正文保存后索引写失败会如实提示，下一次加载从正文重建索引。
当前按单进程个人 CLI 使用，不提供多个进程同时修改记忆的锁或合并协议。

用户要求记住或模型识别出明确的长期事实时，可以提出记忆写入，批准后才保存。
外部材料必须经用户确认，再用自己的话总结；改写本身不会使材料变可信。
记忆可能过时，与眼前核实事实冲突时应提出更新；它不能提升工具权限。
模型根据常驻索引选择主题，再通过 memory_read 按需读取正文；主题正文不再同步进知识库 RAG。
这一目录属于 common_agent 产品，与桌面的 common_memory 助手协作记忆无关。

## 7. 当前学习主线

当前只沿着产品主线阅读和开发：配置 → 工作区工具 → RAG → MCP → Agent 组装 → CLI 会话。
当前主线不包含评测/eval 目录和验收脚本；等核心能力完成后另行设计，不要在现在的阅读过程中寻找它们。

主循环现在会记录本轮上一次“工具名 + 参数”的调用签名。如果模型连续发出完全相同的工具请求，
系统会把“已重复、停止继续调用”作为工具结果写回 State，并安全结束本轮，避免无意义地消耗额度或重复产生副作用。
用户发来新问题时，本轮的工具轮数、重复签名和停止原因都会清零；它们不会污染同一 `thread_id` 的下一次回答。

项目总路线请看：`00_项目完善路线图_融合最终版.md`。

## 8. 当前安全边界

- 只允许 `.md`、`.txt`、`.json`、`.csv`、`.py` 文本文件。
- 普通文件读取按页限长，单次写入最大 10 万字符；完整 Skill 文件最多 20000 字符。
- 搜索最多返回 50 条，列目录最多展示 200 项，防止上下文无限膨胀。
- 普通文件工具只允许 workspace 内路径；skill_view 只能读取已登记、解析后仍在 skills 目录内的 SKILL.md。
- 新建拒绝覆盖；编辑只替换唯一匹配的文本；删除单个文件固定审批。写入和 Shell 经独立子 Agent 审查，需要确认时由主 Agent 统一询问。
- 安全子 Agent 的模型判断不是沙箱；Shell 可访问当前 Windows 用户有权限访问的资源，仅用于可信用户本机。普通工具的路径、类型、大小校验不因模型允许而跳过。
- RAG 索引只读取 workspace 中大小合规的 md/txt/json/csv；检索结果被当作不可信证据，不当作系统指令。
- MCP 连接目标由维护者在 mcp.json 配置，不接受模型传入连接地址；密钥留在 .env，工具元数据不保存含密钥的 URL。第三方会接收工具参数，执行前仍进行权限审查。
- `.env` 和 SQLite 已加入 `.gitignore`。

这是内核的能力边界，不是缺陷：Agent 的工具权限必须按真实需求逐项开放，不能一开始就把整台电脑交给模型。

## 9. 当前内核边界与后续扩展

项目由一台专用电脑独立维护。后续在同一个项目中，按真实场景和求职价值逐项决定是否增加相关能力：

- 可选能力包括表格与数据处理、网络搜索、任务规划、增强版 HITL、子 Agent、Python 沙箱或图表；这里只列候选项，不代表当前承诺全部实现。

当前阶段的完成标准，是第 1～6 节的核心能力能够被读懂、运行，并逐步接入手写循环。

## 10. 一个重要的版本事实

当前官方 `mcp` Python SDK 已是 2.2.0，而 `langchain-mcp-adapters` 0.3.2 仍声明依赖 `mcp>=1.24,<2`。
本项目直接使用官方 2.2 `Client` 和 Streamable HTTP 传输，保留一层透明的客户端适配：

1. 从 mcp.json 读取远程服务地址、鉴权头与工具列表；
2. `list_tools()` 动态获取名称、说明和 JSON Schema；
3. 筛选并加服务名前缀，Schema 原样变成 LangChain `StructuredTool`；
4. Agent 异步调用时，再由 Client 真正执行 MCP 工具。

实现集中在 `mcp_client.py`，不自建 MCP Server，不新增 Manager 或插件工厂。

## 11. RAG 的真实边界

本项目已有真实双路检索，但还不是企业搜索平台：

- 已有：Markdown 标题分节与递归切块（512 字符正文、64 字符重叠）、文件内容哈希跳过未变文件、真实 Embedding、SQLite 向量保存；中英文经 jieba 精确模式分词后由 SQLite FTS5 的 BM25 做关键词召回与排序，证据仍返回原文。关键词与向量两路按 `(路径, 块号)` 去重、RRF 融合，再交给真实重排模型；结果保留来源与块号。重排接口失败时明确退回 RRF。再次索引会清理该目录已删除文件的记录；切块或分词规则变化则删除测试库后重建。
- 未有：PDF/Word 解析、文档删除同步、权限过滤和大规模 ANN 向量数据库。当前向量召回仍会把全部向量读入内存，适合学习和小知识库，不适合大规模数据。

这些未有能力不是共同内核必须项，应在具体业务分支出现真实需求时增加。

## 12. TODO：升级为真正的向量数据库检索

- [ ] 用主流向量检索方案替换当前“从 SQLite `fetchall()` 全部向量，再在 Python 内逐条计算余弦相似度”的教学实现。
- [ ] 优先评估 PostgreSQL + pgvector；如果项目更适合独立向量数据库，再评估 Qdrant。
- [ ] 让数据库或向量索引直接完成 ANN / Top-K 候选召回，查询时只把少量相关文本块返回 Python，避免知识库扩大后占满内存。
- [ ] 保留文档路径、分块编号、Embedding 模型等元数据，并支持按业务字段过滤。
- [ ] 增加删除同步、空知识库和较大数据量下的工程测试；双路召回与重排已接入，不依赖先换向量数据库。

这项改造暂不在当前源码阅读阶段实施；先把共同内核现有链路读完，再单独设计和迁移。
