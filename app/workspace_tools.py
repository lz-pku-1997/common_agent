"""common_agent 的真实文件工具。

这里没有模拟数据：模型调用工具后，工具会真的读取或新建 workspace 里的文件。
同时它也不是一个“能碰整台电脑”的文件助手。所有路径都必须留在 workspace 内，
这是 Agent 工具最重要的安全边界之一。
"""

from pathlib import Path

from langchain.tools import tool

from app.config import WORKSPACE_ROOT


ALLOWED_TEXT_SUFFIXES = {".md", ".txt", ".json", ".csv", ".py"}
MAX_READ_BYTES = 1_000_000
MAX_WRITE_CHARACTERS = 100_000
MAX_SEARCH_RESULTS = 50


def resolve_workspace_path(relative_path: str) -> Path:
    """把相对路径转换成绝对路径，并拦截逃出 workspace 的行为。

    例如 workspace 在 D:/common_agent/workspace：
    - notes/today.md 会落在 workspace 里面，允许。
    - ../../secret.txt 想退到项目外面，拒绝。

    为什么必须先 resolve 再判断？因为仅仅检查字符串里有没有 ".." 不可靠；
    resolve() 会先把所有 ../ 和绝对路径关系算清楚，我们再判断最终位置。
    """

    cleaned_path = relative_path.strip()
    if not cleaned_path:
        cleaned_path = "."

    candidate = (WORKSPACE_ROOT / cleaned_path).resolve()
    if not candidate.is_relative_to(WORKSPACE_ROOT):
        raise ValueError("拒绝访问：路径必须位于项目的 workspace 目录内。")
    return candidate


def ensure_supported_text_file(path: Path) -> None:
    """只允许处理本项目明确支持的文本文件类型。"""

    if path.suffix.lower() not in ALLOWED_TEXT_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_TEXT_SUFFIXES))
        raise ValueError(f"不支持 {path.suffix or '无后缀'} 文件；允许类型：{allowed}")


@tool
def list_workspace_files(relative_directory: str = ".") -> str:
    """列出 workspace 某个目录中的文件和子目录。路径必须相对 workspace，例如 '.' 或 'notes'。"""

    directory = resolve_workspace_path(relative_directory)
    if not directory.exists():
        raise ValueError(f"目录不存在：{relative_directory}")
    if not directory.is_dir():
        raise ValueError(f"这不是目录：{relative_directory}")

    entries = sorted(directory.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
    if not entries:
        return "这个目录目前是空的。"

    lines: list[str] = []
    for entry in entries[:200]:
        # relative_to 把绝对路径重新变成对用户更清楚的 workspace 相对路径。
        relative_name = entry.relative_to(WORKSPACE_ROOT).as_posix()
        kind = "目录" if entry.is_dir() else "文件"
        lines.append(f"[{kind}] {relative_name}")

    if len(entries) > 200:
        lines.append(f"……另有 {len(entries) - 200} 项未展示。")
    return "\n".join(lines)


@tool
def read_text_file(relative_path: str) -> str:
    """读取 workspace 中一个 UTF-8 文本文件。支持 md、txt、json、csv 和 py，最大 1 MB。"""

    path = resolve_workspace_path(relative_path)
    if not path.exists():
        raise ValueError(f"文件不存在：{relative_path}")
    if not path.is_file():
        raise ValueError(f"这不是文件：{relative_path}")
    ensure_supported_text_file(path)
    if path.stat().st_size > MAX_READ_BYTES:
        raise ValueError("文件超过 1 MB。内核版拒绝一次性读取，以免撑爆模型上下文。")

    # utf-8-sig 既能读取普通 UTF-8，也能自动去掉某些 Windows 文件开头的 BOM 标记。
    return path.read_text(encoding="utf-8-sig")


@tool
def search_workspace_text(query: str, file_pattern: str = "*.md") -> str:
    """在 workspace 的文本文件中搜索文字。query 是关键词，file_pattern 例如 '*.md' 或 '*.py'。"""

    keyword = query.strip()
    if not keyword:
        raise ValueError("搜索词不能为空。")

    # 只接受“*.扩展名”，不让模型借 pattern 拼出 workspace 外的路径。
    if not file_pattern.startswith("*.") or "/" in file_pattern or "\\" in file_pattern:
        raise ValueError("file_pattern 只能写成 '*.md'、'*.txt' 这一类形式。")

    suffix = file_pattern[1:].lower()
    if suffix not in ALLOWED_TEXT_SUFFIXES:
        raise ValueError("这个文件类型不在允许搜索的范围内。")

    matches: list[str] = []
    for path in sorted(WORKSPACE_ROOT.rglob(file_pattern)):
        if not path.is_file() or path.stat().st_size > MAX_READ_BYTES:
            continue

        text = path.read_text(encoding="utf-8-sig", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if keyword.casefold() in line.casefold():
                relative_name = path.relative_to(WORKSPACE_ROOT).as_posix()
                matches.append(f"{relative_name}:{line_number}: {line.strip()}")
                if len(matches) >= MAX_SEARCH_RESULTS:
                    return "\n".join(matches) + "\n……结果较多，已停止在前 50 条。"

    if not matches:
        return f"没有在 {file_pattern} 文件中找到：{keyword}"
    return "\n".join(matches)


@tool
def save_new_text_file(relative_path: str, content: str) -> str:
    """在 workspace 新建文本文件。为防止误覆盖，目标文件已经存在时会拒绝写入。"""

    path = resolve_workspace_path(relative_path)
    ensure_supported_text_file(path)

    if path.exists():
        raise ValueError(f"文件已存在，出于安全考虑不覆盖：{relative_path}")
    if len(content) > MAX_WRITE_CHARACTERS:
        raise ValueError("内容超过 10 万字符，内核版拒绝一次性写入。")

    # parents=True 会连同 notes/2026 这样的父目录一起建立。
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    relative_name = path.relative_to(WORKSPACE_ROOT).as_posix()
    return f"已真实新建文件：{relative_name}（{len(content)} 个字符）"


# Agent 只会看见这个列表里的工具。以后增加能力时，在这里显式注册，便于审查边界。
WORKSPACE_TOOLS = [
    list_workspace_files,
    read_text_file,
    search_workspace_text,
    save_new_text_file,
]
