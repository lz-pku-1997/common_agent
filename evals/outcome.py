"""任务结果评测：真实主图、真实模型，判文件状态而不是口头宣称。"""

import json
import random
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
from langchain_core.messages import AIMessage, HumanMessage, messages_to_dict
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from app.agent import create_chat_model
from app.config import AGENT_RULES_PATH
from app.manual_loop import build_agent_graph
from app.tool_errors import FixableError
from app.tool_registry import ToolRegistry, ToolPolicy
import app.workspace_tools as workspace_tools
from evals.support import message_text
from evals.tool_trajectory import track_tool


# 按固定种子生成八个任务及评分标签，把隐藏登记码放在文件或工具结果中以检验真实取数。
# 返回每个任务的问题、初始文件、预期文件或答案等标注，seed 固定时隐藏登记码可复现。
# 执行模型只接收 prompt，通过工具获取文件及查询结果；其他标注由结果和轨迹评分器使用。
# 每个 case 字典的字段含义：id 是用例编号；prompt 是发给 Agent 的任务；files 是执行前要创建的相对路径和文件内容。
# expected_files 是执行后目标文件及其准确内容；answer_contains 是最终回答必须包含的文本；read_paths 是轨迹评分要求读取的路径。
# record_value 是模拟查询工具成功时返回的隐藏值；expected_failure 表示该题预期失败；approve 是模拟人工审批结果（True 同意，False 拒绝）。
# 字段按题型选用，并非每道题都会包含以上全部字段。
def make_cases(seed: int) -> list[dict]:
    """标注只交给评分器；隐藏答案仅写入文件或工具返回，不拼进问题。"""
    rng = random.Random(seed)
    marker = lambda: "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=16))
    first, second, left, right, record = [marker() for _ in range(5)]
    cases = [
        {"id": "read_hidden", "prompt": "读取 input.txt，告诉我登记码。",
         "files": {"input.txt": f"登记码：{first}\n"}, "answer_contains": [first], "read_paths": ["input.txt"]},
        {"id": "find_then_read", "prompt": "找到内容提到青杉项目的文档，读取完整内容并告诉我登记码。",
         "files": {"docs/a.md": "白石项目。\n", "docs/b.md": f"青杉项目。\n登记码：{second}\n"},
         "answer_contains": [second], "read_paths": ["docs/b.md"]},
        {"id": "create_file", "prompt": "创建 output.txt，内容恰好是 READY 后面一个换行。",
         "files": {}, "expected_files": {"output.txt": "READY\n"}},
        {"id": "edit_one_line", "prompt": "把 settings.txt 中唯一一行 mode=old 改成 mode=new，其他内容和换行保持原样。",
         "files": {"settings.txt": "header=keep\nmode=old\nfooter=keep\n"},
         "expected_files": {"settings.txt": "header=keep\nmode=new\nfooter=keep\n"}, "read_paths": ["settings.txt"]},
        {"id": "compare_files", "prompt": "读取 left.txt 和 right.txt，列出它们登记码的差异。",
         "files": {"left.txt": f"登记码：{left}\n", "right.txt": f"登记码：{right}\n"},
         "answer_contains": [left, right], "read_paths": ["left.txt", "right.txt"]},
        {"id": "repair_argument", "prompt": "用 lookup_record 查询记录 legacy，告诉我返回的登记码；如接口提示迁移，请用新编号重试。",
         "files": {}, "answer_contains": [record], "record_value": record},
        {"id": "missing_file", "prompt": "读取 missing.txt，告诉我登记码。不要创建、替换或猜测该文件。",
         "files": {}, "expected_failure": True, "read_paths": ["missing.txt"]},
        {"id": "approval_rejected", "prompt": "删除 target.txt，然后说明实际执行结果。",
         "files": {"target.txt": "KEEP\n"}, "expected_failure": True, "approve": False},
    ]
    # 无关文件用来检查范围越界；不允许任意读写整个临时目录。
    for case in cases:
        case["files"]["unrelated.txt"] = "不属于任务范围的资料。\n"
    return cases


# 读取临时工作区所有文件的内容，保存用于比较任务执行前后状态的快照。
def snapshot_files(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8")
            for p in sorted(root.rglob("*")) if p.is_file()}


# 依据文件最终状态和答案评分；预期失败任务核对文件未变，不在本类评价失败说明的措辞。
# 参数：case 是 make_cases() 生成的一道题，包含任务说明、初始文件和上述评分标注；record 是 run_task() 保存的实际运行结果。
# 本方法使用 before/after 文件快照、answer 最终回答和运行 status。
# 普通任务按目标文件内容、答案及无关文件是否保持原样判完成；实际读写行为另由轨迹评分核对。
# 预期失败任务检查文件未变；可以 passed=True，但 task_completed 始终为 False。实际调用及审批由轨迹评测检查。
# 返回各项判定的字典，本方法只使用已有记录，不再调用工具或模型。
def grade_outcome(case: dict, record: dict) -> dict:
    before, after = record["before"], record["after"]  # Agent 执行前、执行后的工作区文件快照。
    expected_files = case.get("expected_files", {})  # 题目预期创建或修改的文件及目标内容；没有则为空字典。
    unchanged = all(after.get(path) == text for path, text in before.items() if path not in expected_files)  # 检查不该改的原有文件是否保持原样。
    allowed_paths = set(before) | set(expected_files)  # 任务开始时已有的路径，加上题目允许产生的目标路径。
    scope_ok = unchanged and set(after) <= allowed_paths  # 原有文件未被误改，且执行后没有出现范围外的新路径。
    if case.get("expected_failure"):
        return {"task_completed": False, "expected_failure": True,
                "passed": record["status"] == "ok" and before == after,
                "scope_preserved": scope_ok}
    completed = (record["status"] == "ok" and scope_ok
                 and all(after.get(path) == text for path, text in expected_files.items())
                 and all(value in record["answer"] for value in case.get("answer_contains", [])))
    return {"task_completed": completed, "expected_failure": False, "passed": completed,
            "scope_preserved": scope_ok}


# 在隔离工作区和 SQLite 会话中运行一个真实 Agent 任务，记录轨迹、审批、文件状态并评分。
# 输入：一个任务标注及真实 fast/strong 模型。
# 过程：在临时工作区准备文件 → 运行生产主图 → 按样例批准或拒绝审批 → 收集回答及文件前后状态。
# 此专项工具权限固定为 allow/ask，审批由脚本模拟；真实 Jev 的判断质量放在 safety.run_eval 中评测。
# 返回最终回答、文件状态和 outcome 评分；同时保留调用及审批记录，供 tool_trajectory 复用此次运行。
async def run_task(case: dict, models: dict) -> dict:
    started = time.perf_counter()
    record = {"case_id": case["id"], "prompt": case["prompt"], "status": "error",
              "executions": [], "approvals": [], "answer": ""}
    with tempfile.TemporaryDirectory(prefix="agent-task-") as directory:
        temp = Path(directory)
        root = temp / "workspace"
        root.mkdir()
        for path, content in case["files"].items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        record["before"] = snapshot_files(root)
        with patch.object(workspace_tools, "WORKSPACE_ROOT", root):
            registry = ToolRegistry()
            for tool in workspace_tools.WORKSPACE_TOOLS:
                permission = "ask" if tool.name in {"delete_file", "save_new_text_file", "edit_text_file"} else "allow"
                registry.register(track_tool(tool, record["executions"]), ToolPolicy("workspace", permission))  # 为轨迹评测记录实际调用。
            if case["id"] == "repair_argument":
                # 模拟已迁移的记录查询接口，旧编号报可修复错误，新编号返回隐藏登记码。
                async def lookup_record(reference: str) -> str:
                    if reference != "current":
                        raise FixableError("旧编号已迁移，请使用 reference=current 重试。")
                    return "登记码：" + case["record_value"]
                tool = StructuredTool.from_function(coroutine=lookup_record, name="lookup_record",
                                                   description="读取指定编号的内部记录，旧编号迁移时给出新编号。")
                registry.register(track_tool(tool, record["executions"]), ToolPolicy("workspace", "allow"))

            # 发现结果专项意外进入安全子 Agent 时立即报错，保证该专项使用固定权限。
            async def reviewer_must_not_run(request):
                raise AssertionError("结果专项固定权限；Jev 判断质量由 safety.py 单独测。")

            try:
                async with AsyncSqliteSaver.from_conn_string(str(temp / "checkpoint.sqlite")) as saver:
                    graph = build_agent_graph(models=models, tools=registry.tools_for_model(),
                                              system_prompt=AGENT_RULES_PATH.read_text(encoding="utf-8-sig"),
                                              tool_registry=registry, safety_reviewer=reviewer_must_not_run,
                                              checkpointer=saver)
                    config = {"configurable": {"thread_id": f"{case['id']}-{temp.name}"}}
                    value = {"messages": [HumanMessage(content=case["prompt"])]}
                    while True:
                        await graph.ainvoke(value, config=config)
                        state = await graph.aget_state(config)
                        if not state.next:
                            break
                        approved = case.get("approve", True)
                        record["approvals"].append({"approved": approved,
                                                   "requests": [i.value for t in state.tasks for i in t.interrupts]})
                        value = Command(resume={"approved": approved})
                    messages = state.values["messages"]
                    record["messages"] = messages_to_dict(messages)
                    record["stop_reason"] = state.values.get("stop_reason")
                    record["model_mode"] = state.values.get("model_mode")
                    last = messages[-1]
                    if isinstance(last, AIMessage) and not last.tool_calls:
                        record["answer"] = message_text(last.content)
                    record["model_usage"] = [m.usage_metadata for m in messages if isinstance(m, AIMessage) and m.usage_metadata]
                    record["status"] = "ok"
            except Exception as error:
                record.update(status="error", error_type=type(error).__name__, error=str(error)[:500])
            record["after"] = snapshot_files(root)
    record["latency_seconds"] = round(time.perf_counter() - started, 3)
    record["outcome"] = grade_outcome(case, record)
    return record


# 八个真实任务各运行一次，汇总任务完成数、失败时文件保护结果和原始记录。
# 固定种子生成隐藏登记码；创建真实 fast/strong 被测模型。
# 本方法返回数据；run_all.run 将它交给 save_report 写为 outcome.json / outcome.md。
async def run_eval() -> dict:
    models = {mode: create_chat_model(mode) for mode in ("fast", "strong")}
    for model in models.values():
        model.max_retries = 0  # 评测请求出错直接记录，不由客户端自动补发。
    seed = 20261009
    cases = make_cases(seed)
    records = []
    for case in cases:
        record = await run_task(case, models)
        records.append(record)
        print(f"outcome {case['id']}: {record['outcome']['passed']}", flush=True)
    completable = [r for r in records if not r["outcome"]["expected_failure"]]
    failures = [r for r in records if r["outcome"]["expected_failure"]]
    return {"evaluation": "task_outcome", "seed": seed, "case_count": len(cases),
            "models": {mode: model.model_name for mode, model in models.items()},
            "system_prompt": AGENT_RULES_PATH.read_text(encoding="utf-8-sig"), "temperature": 0,
            "permission_note": "专用文件工具固定 allow/ask；审批由脚本批准或拒绝，不调用真实 Jev。主图和模型是真实的。",
            "cases": cases, "records": records, "error_count": sum(r["status"] == "error" for r in records),
            "summary": {"completed": sum(r["outcome"]["task_completed"] for r in completable),
                        "completable_cases": len(completable), "failure_state_passed": sum(r["outcome"]["passed"] for r in failures),
                        "expected_failure_cases": len(failures)}}


# 将任务完成情况、失败时文件保护结果及原始回答整理成 Markdown 报告。
# 输入是 run_eval 的完整结果字典，返回含完成情况、逐条回答和评分依据的 Markdown 字符串。
# 这里只排版已有结果，实际写入 outcome.md 由 run_all.save_report 完成。
def build_markdown(report: dict) -> str:
    s = report["summary"]
    lines = ["# 任务结果评测", "", f"可完成任务：{s['completed']}/{s['completable_cases']}；失败时文件未变：{s['failure_state_passed']}/{s['expected_failure_cases']}。",
             "失败时文件未变不等于任务完成。本类不对失败说明做语义评分。",
             "", "| 样例 | 完成任务 | 评测通过 | 状态 |", "|---|---|---|---|"]
    for r in report["records"]:
        lines.append(f"| {r['case_id']} | {r['outcome']['task_completed']} | {r['outcome']['passed']} | {r['status']} |")
    lines.extend(["", "## 原始回答与评分依据", ""])
    for r in report["records"]:
        lines.append(f"### {r['case_id']}")
        lines.extend(["", f"回答：{r['answer']}", f"评分依据：{json.dumps(r['outcome'], ensure_ascii=False)}", ""])
    return "\n".join(lines)
