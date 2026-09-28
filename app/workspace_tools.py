"""common_agent 的真实文件工具。

这里没有模拟数据：模型调用工具后，工具会真的读取或新建 workspace 里的文件。
同时它也不是一个“能碰整台电脑”的文件助手。所有路径都必须留在 workspace 内，
这是 Agent 工具最重要的安全边界之一。
"""

from pathlib import Path

from langchain.tools import tool

from app.config import MAX_TOOL_RESULT_CHARS, WORKSPACE_ROOT
from app.tool_errors import NonRetryableError, RetryableError


ALLOWED_TEXT_SUFFIXES = {".md", ".txt", ".json", ".csv", ".py"}
MAX_READ_BYTES = 1_000_000
MAX_WRITE_CHARACTERS = 100_000
MAX_SEARCH_RESULTS = 50
MAX_READ_LINE_CHARS = 5000
MAX_SEARCH_LINE_CHARS = 500


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
        raise NonRetryableError("拒绝访问：路径必须位于项目的 workspace 目录内。")
    return candidate


def ensure_supported_text_file(path: Path) -> None:
    """只允许处理本项目明确支持的文本文件类型。"""

    if path.suffix.lower() not in ALLOWED_TEXT_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_TEXT_SUFFIXES))
        raise RetryableError(f"不支持 {path.suffix or '无后缀'} 文件；允许类型：{allowed}")


@tool
def list_workspace_files(relative_directory: str = ".") -> str:
    """列出 workspace 某个目录中的文件和子目录。路径必须相对 workspace，例如 '.' 或 'notes'。"""

    directory = resolve_workspace_path(relative_directory)
    if not directory.exists():
        raise RetryableError(f"目录不存在：{relative_directory}")
    if not directory.is_dir():
        raise RetryableError(f"这不是目录：{relative_directory}")

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
def read_text_file(relative_path: str, start_line: int = 1, max_lines: int | None = None) -> str:
    """按行读取 workspace 的 UTF-8 文本，支持 md/txt/json/csv/py。

    start_line 从 1 开始，max_lines 不填时最多读 2000 行。
    单行正文最多展示 5000 字符，超长补 ...；整页含行号和提示不超过 20000 字符。
    返回下一页起点或文件结束提示；超长行被省略的部分不会在下一页续读。
    """

    if start_line < 1:
        raise RetryableError("start_line 必须从 1 开始。")
    if max_lines is not None and max_lines < 1:
        raise RetryableError("max_lines 必须大于 0。")
    page_size = 2000 if max_lines is None else max_lines

    path = resolve_workspace_path(relative_path)
    if not path.exists():
        raise RetryableError(f"文件不存在：{relative_path}")
    if not path.is_file():
        raise RetryableError(f"这不是文件：{relative_path}")
    ensure_supported_text_file(path)
    # 先流式数行，再流式取本页。多扫一遍是为了准确报告总行数，不把整份文件装进列表。
    with path.open(encoding="utf-8-sig", newline="") as file:
        total_lines = sum(1 for _ in file)
        if total_lines == 0:
            return "[文件结束，共 0 行。]"
        if start_line > total_lines:
            raise RetryableError(f"start_line 超出文件末尾，文件共 {total_lines} 行。")
        file.seek(0)  # 上面数行数已把文件读到末尾，游标拨回开头才能重新逐行取本页
        lines: list[str] = []
        used_chars = 0
        footer = ""
        for line_number, line in enumerate(file, start=1):
            if line_number < start_line:
                continue
            if len(lines) >= page_size:
                break
            text = line.rstrip("\r\n")
            if len(text) > MAX_READ_LINE_CHARS:
                text = text[:MAX_READ_LINE_CHARS] + "..."
            rendered = f"{line_number}: {text}"
            next_footer = f"[已显示第 {start_line}–{line_number} 行 / 共 {total_lines} 行；"
            if line_number == total_lines:
                next_footer += f"文件结束，共 {total_lines} 行。]"
            else:
                next_footer += f"下次 start_line={line_number + 1}。]"
            # 行号、换行和页尾提示也算进预算；装不下的整行留给下一页。
            if used_chars + len(rendered) + 1 + len(next_footer) > MAX_TOOL_RESULT_CHARS:
                break
            lines.append(rendered)
            used_chars += len(rendered) + 1
            footer = next_footer
    return "\n".join([*lines, footer])


@tool
def search_workspace_text(query: str, file_pattern: str = "*.md") -> str:
    """在 workspace 中搜索；支持单个文件路径或通配表达式。"""

    keyword = query.strip()
    if not keyword:
        raise RetryableError("搜索词不能为空。")

    # 具体路径只查一次；通配表达式沿用递归搜索，*.txt 仍能匹配子目录。
    try:
        paths = (WORKSPACE_ROOT.rglob(file_pattern) if any(char in file_pattern for char in "*?[")
                 else [WORKSPACE_ROOT / file_pattern])
    except (NotImplementedError, ValueError, OSError) as error:
        raise RetryableError("不支持这种路径写法，请写成 '*.md'、'notes/*.md' 或某个文件路径。") from error

    matches: list[str] = []
    for path in paths:
        # 批量扫描中也可能遇到指向 workspace 外的符号链接。
        path = path.resolve()
        if not path.is_relative_to(WORKSPACE_ROOT) or not path.is_file():
            continue

        # 逐行扫描，大文件也能搜索；命中行只显示前 500 字符。
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as file:
            for line_number, line in enumerate(file, start=1):
                if keyword.casefold() not in line.casefold():
                    continue
                relative_name = path.relative_to(WORKSPACE_ROOT).as_posix()
                snippet = line.strip()
                if len(snippet) > MAX_SEARCH_LINE_CHARS:
                    snippet = snippet[:MAX_SEARCH_LINE_CHARS] + "..."
                matches.append(f"{relative_name}:{line_number}: {snippet}")
                if len(matches) >= MAX_SEARCH_RESULTS:
                    return "\n".join(matches) + "\n……结果较多，已停止在前 50 条。"

    if not matches:
        return f"没有在 {file_pattern} 中找到：{keyword}"
    return "\n".join(matches)


@tool
def save_new_text_file(relative_path: str, content: str) -> str:
    """在 workspace 新建文本文件。为防止误覆盖，目标文件已经存在时会拒绝写入。"""

    path = resolve_workspace_path(relative_path)
    ensure_supported_text_file(path)

    if path.exists():
        raise RetryableError(f"文件已存在，出于安全考虑不覆盖：{relative_path}")
    if len(content) > MAX_WRITE_CHARACTERS:
        raise RetryableError("内容超过 10 万字符，内核版拒绝一次性写入。")

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
