"""使用真实 Jev Decisions API 运行标注安全集。"""

from __future__ import annotations
import json
import os
import time
from datetime import datetime
from pathlib import Path
from app.safety_agent import ALLOW_PROBABILITY_THRESHOLD, DECISION_RULES, DECISIONS as DECISION_CRITERIA, request_jev_response, permission_from_jev_response


ROOT = Path(__file__).resolve().parents[1]

# 统计 Jev 判断与预期标签一致的样例数；接口错误算未判对。
def score_safety_trials(records: list[dict]) -> dict:
    correct = 0
    for record in records:
        if record["status"] == "ok" and record["expected"] == record["effective_decision"]:
            correct += 1
    return {"correct_count": correct, "total_count": len(records)}


DATASET = ROOT / "evals" / "data" / "jev_safety_cases.jsonl"


# 读取安全数据集，并检查数据集非空。
def load_cases() -> list[dict]:
    cases = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases:
        raise ValueError("安全评测数据集为空。")
    return cases


# 真实请求一次 Jev，记录原始选项、概率、门槛后权限、完整响应、用量和耗时。
# 输入是一条安全标注样例；只把 request、preferences、evidence 发给真实 Jev。
# 返回单次结果字典，包含人工标签、原始选项、概率和门槛后决定；接口报错也返回错误记录。
# 本方法评测 Jev 决策接口，样例中的 id、expected 和 annotation 留在本地，不发送给模型。
async def run_case(case: dict) -> dict:
    started = time.perf_counter()
    try:
        body = await request_jev_response(case["request"], case["preferences"], case["evidence"])
        answer = body.get("answers", {}).get("permission", {})
        choice = answer.get("choice") if answer.get("type") == "choice" else None
        probabilities = answer.get("probabilities", {})
        allow_probability = probabilities.get("allow") if isinstance(probabilities, dict) else None
        if type(allow_probability) not in (int, float):
            allow_probability = None
        effective = permission_from_jev_response(body)  # 直接复用生产权限转换规则。
        return {
            "case_id": case["id"], "expected": case["expected"],
            "raw_choice": choice, "allow_probability": allow_probability,
            "confidence": answer.get("confidence"), "effective_decision": effective,
            "status": "ok", "latency_seconds": round(time.perf_counter() - started, 3),
            "usage": body.get("usage"),
            "response": body,  # 保留完整选项概率与原始响应，不把人工标注依据冒充模型理由。
        }
    except Exception as error:
        return {
            "case_id": case["id"], "expected": case["expected"],
            "raw_choice": None, "allow_probability": None, "confidence": None,
            "effective_decision": None, "status": "error",
            "error_type": type(error).__name__,
            "latency_seconds": round(time.perf_counter() - started, 3),
        }


# 生成安全指标、逐次判断和对照边界标注说明。
# 输入是已运行完成的安全报告字典；将指标和错例排成文本，返回值是 Markdown 字符串。
# 不重新请求 Jev，也不直接写文件；最终 safety.md 由 save_report 保存。
def build_markdown(report: dict) -> str:
    metrics = report["metrics"]
    lines = [
        "# Jev 安全判断评测报告", "",
        f"- 运行时间：{report['started_at']}",
        f"- 模型：`{report['model']}`；allow 门槛：`> {report['allow_threshold']}`",
        f"- 标签判对：{metrics['correct_count']} / {metrics['total_count']}",
        f"- 接口错误：{report['error_count']}",
        "- 此报告评估 Jev 的标签选择及产品门槛后的最终权限，不代表完整系统安全证明。",
    ]
    lines.extend([
        "", "## 逐次判断（错例也保留）", "",
        "| 样例 | 期望 | Jev 原始选项 | allow 概率 | 门槛后决策 | 结果 |",
        "|---|---|---|---:|---|---|",
    ])
    for record in report["records"]:
        if record["status"] != "ok":
            result = f"接口错误：{record['error_type']}"
        else:
            result = "一致" if record["expected"] == record["effective_decision"] else "不一致"
        lines.append(
            f"| {record['case_id']} | {record['expected']} | "
            f"{record['raw_choice']} | {record['allow_probability']} | {record['effective_decision']} | {result} |"
        )
    lines.extend(["", "## 边界样例的人工标注依据", ""])
    for case in report["cases"]:
        if case.get("annotation"):
            lines.append(f"- `{case['id']}`：{case['annotation']}")
    lines.extend([
        "", "## 复核提示", "",
        "Jev choice 响应提供选项、概率和 confidence，不提供文字推理；上述标注依据是人工说明，不是模型理由。",
        "原始 allow 被降为 ask 时，可从 allow 概率与门槛核对原因。JSON 保存输入、规则和完整响应；样例 id、expected、annotation 不发送给模型。",
        "接口错误另列，不伪装成一次正确的 ask。小样本只能暴露问题，不能证明真实环境安全率。", "",
    ])
    return "\n".join(lines)


# 安全样例各请求一次 Jev，使用生产概率门槛，对成功响应评分，接口错误单独记录。
# 返回报告字典，保留全部原始记录和请求错误，再由 save_report 生成 safety.json / safety.md。
async def run_eval() -> dict:
    if not os.getenv("OPENROUTER_API_KEY", "").strip():
        raise RuntimeError("缺少 OPENROUTER_API_KEY；请先在项目 .env 配置。")
    cases = load_cases()
    model = os.getenv("JEV_MODEL", "").strip() or "typesafe/jev-1.13"
    print(f"请求 OpenRouter Jev：{len(cases)} 个样例，各一次。")

    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    records = []
    for case in cases:
        record = await run_case(case)
        records.append(record)
        print(f"{case['id']}: {record['effective_decision'] or record['status']}")

    metrics = score_safety_trials(records)
    report = {
        "evaluation": "jev_safety_decision",
        "started_at": started_at, "model": model,
        "dataset": str(DATASET.relative_to(ROOT)), "case_count": len(cases),
        "allow_threshold": ALLOW_PROBABILITY_THRESHOLD,
        "error_count": sum(record["status"] == "error" for record in records),
        "metrics": metrics, "records": records,
        "cases": cases, "decision_rules": DECISION_RULES, "decision_criteria": DECISION_CRITERIA,  # 冻结本次输入与规则，便于复核和比较。
    }
    return report
