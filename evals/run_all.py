"""统一运行四类真实评测，共 20 个独立样例，每个样例跑一次，报告保存在同一目录。"""

import argparse
import asyncio
import json
from pathlib import Path
import secrets
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals import outcome, tool_trajectory, safety, answer_quality

CATEGORIES = {"outcome": "任务结果评测", "tool_trajectory": "工具调用与轨迹评测",
              "safety": "权限安全评测", "answer_quality": "回答质量 / LLM-as-judge"}
MODULES = {"outcome": outcome, "tool_trajectory": tool_trajectory, "safety": safety,
           "answer_quality": answer_quality}


# 将一类评测结果保存成 JSON 和 Markdown；运行失败时也保留错误报告。
# 输入：directory 是本次运行目录，name 是类别名（如 outcome），report 是该类 run_eval 已返回的结果字典。
# 过程：保存完整 JSON，再调用对应模块的 build_markdown 排版；顶层运行错误则生成简短错误说明。
# 输出文件：例如 outcome.json 与 outcome.md；本方法负责写文件，四类汇总 summary 由 run() 另外生成。
def save_report(directory: Path, name: str, report: dict):
    (directory / f"{name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    markdown = (MODULES[name].build_markdown(report) if report.get("status") != "error"
                else f"# {CATEGORIES[name]}\n\n运行错误：{report['error_type']}：{report['error']}\n")
    (directory / f"{name}.md").write_text(markdown, encoding="utf-8")


# 按评测类别提取关键指标，生成汇总表中展示的一句话结论。
# 输入是已经生成的单类结果字典；读取其中的汇总数字，返回供 summary.md 展示的文字。
# 例如任务结果返回“完成 6/6；失败时文件未变 2/2；运行错误 0”，此处不重新执行任务或评分。
def conclusion(name: str, report: dict) -> str:
    if report.get("status") == "error":
        return "运行失败，见详细报告"
    if name == "outcome":
        s = report["summary"]
        return f"完成 {s['completed']}/{s['completable_cases']}；失败时文件未变 {s['failure_state_passed']}/{s['expected_failure_cases']}；运行错误 {report['error_count']}"
    if name == "tool_trajectory":
        s = report["summary"]
        return f"轨迹通过 {s['passed']}/{s['case_count']}（共用结果任务）"
    if name == "safety":
        m = report["metrics"]
        return f"标签判对 {m['correct_count']}/{m['total_count']}"
    scores = ["—" if score is None else str(score) for score in report["summary"]["scores"]]
    return f"逐题得分 {'、'.join(scores)}（每题满分6；运行错误 {report['error_count']}）"


# 运行所选评测并保存报告；结果和轨迹共用一次任务执行，回答质量另跑四道独立任务。
# 单类报错时保留错误报告并继续其他类别；最后生成 summary.json 和 summary.md。
# 退出码 2 表示运行报错，0 表示运行完成；各项质量是否达标看报告。
async def run(args) -> int:
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
    directory = ROOT / "evals" / "reports" / stamp
    directory.mkdir(parents=True)
    print(f"报告目录：{directory}", flush=True)
    reports = {}
    selected = set(args.categories or CATEGORIES)
    for name in CATEGORIES:
        if name not in selected:
            continue
        print(f"开始 {CATEGORIES[name]}", flush=True)
        try:
            if name == "outcome":
                report = await outcome.run_eval()
            elif name == "tool_trajectory":
                if "outcome" not in reports:
                    reports["outcome"] = await outcome.run_eval()
                    save_report(directory, "outcome", reports["outcome"])
                if reports["outcome"].get("status") == "error":
                    report = reports["outcome"]
                else:
                    report = tool_trajectory.run_eval(reports["outcome"])
            elif name == "safety":
                report = await safety.run_eval()
            else:
                report = await answer_quality.run_eval()
        except Exception as error:
            report = {"status": "error", "error_type": type(error).__name__, "error": str(error)[:500]}
        reports[name] = report
        save_report(directory, name, report)
    rows = []
    for name, title in CATEGORIES.items():
        report = reports.get(name)
        rows.append({"category": name, "title": title, "case_count": report.get("case_count") if report else None,
                     "conclusion": conclusion(name, report) if report else "未运行"})
    summary = {"run_id": stamp, "rows": rows,
               "note": "完整评测集共 20 个独立样例：八个结果任务、八个安全场景、四个回答质量任务。轨迹复用八个结果任务，不重复计数；不设跨类别综合总分，保留错例及运行错误。"}
    (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 四类评测汇总", "", summary["note"], "",
             "| 类别 | 样例数 | 评测结论 |", "|---|---:|---|"]
    for row in rows:
        lines.append(f"| {row['title']} | {row['case_count'] or '—'} | {row['conclusion']} |")
    lines.extend(["", "原文输入、输出、实际调用、文件状态和各项评分见同目录 JSON。"])
    (directory / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(r["title"] + "：" + r["conclusion"] for r in rows), flush=True)
    print(f"\n汇总：{directory / 'summary.md'}", flush=True)
    return 2 if any(r.get("status") == "error" or r.get("error_count", 0)
                   for r in reports.values()) else 0


# 解析可选的评测类别，并启动统一入口；不指定类别时运行全部四类。
def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, help="只跑指定评测类别")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
