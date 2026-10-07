"""common_agent 的真实文件工具。

这里没有模拟数据：模型调用工具后，工具会真的读取、新建或修改 workspace 里的文件。
同时它也不是一个“能碰整台电脑”的文件助手。所有路径都必须留在 workspace 内，
这是 Agent 工具最重要的安全边界之一。
"""

import os
import tempfile
from pathlib import Path

from langchain.tools import tool

from app.config import MAX_TOOL_RESULT_CHARS, WORKSPACE_ROOT
from app.tool_errors import NonRetryableError, FixableError


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
        raise FixableError(f"不支持 {path.suffix or '无后缀'} 文件；允许类型：{allowed}")


@tool
def list_workspace_files(relative_directory: str = ".") -> str:
    """列出 workspace 某个目录中的文件和子目录。路径必须相对 workspace，例如 '.' 或 'notes'。"""

    directory = resolve_workspace_path(relative_directory)
    if not directory.exists():
        raise FixableError(f"目录不存在：{relative_directory}")
    if not directory.is_dir():
        raise FixableError(f"这不是目录：{relative_directory}")

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
        raise FixableError("start_line 必须从 1 开始。")
    if max_lines is not None and max_lines < 1:
        raise FixableError("max_lines 必须大于 0。")
    page_size = 2000 if max_lines is None else max_lines

    path = resolve_workspace_path(relative_path)
    if not path.exists():
        raise FixableError(f"文件不存在：{relative_path}")
    if not path.is_file():
        raise FixableError(f"这不是文件：{relative_path}")
    ensure_supported_text_file(path)
    # 先流式数行，再流式取本页。多扫一遍是为了准确报告总行数，不把整份文件装进列表。
    with path.open(encoding="utf-8-sig", newline="") as file:
        total_lines = sum(1 for _ in file)
        if total_lines == 0:
            return "[文件结束，共 0 行。]"
        if start_line > total_lines:
            raise FixableError(f"start_line 超出文件末尾，文件共 {total_lines} 行。")
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
        raise FixableError("搜索词不能为空。")

    # 具体路径只查一次；通配表达式沿用递归搜索，*.txt 仍能匹配子目录。
    try:
        paths = (WORKSPACE_ROOT.rglob(file_pattern) if any(char in file_pattern for char in "*?[")
                 else [WORKSPACE_ROOT / file_pattern])
    except (NotImplementedError, ValueError, OSError) as error:
        raise FixableError("不支持这种路径写法，请写成 '*.md'、'notes/*.md' 或某个文件路径。") from error

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

    if len(content) > MAX_WRITE_CHARACTERS:
        raise FixableError("内容超过 10 万字符，内核版拒绝一次性写入。")

    # parents=True 会连同 notes/2026 这样的父目录一起建立。
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as file:  # x 只允许新建；目标已存在时拒绝打开，不会覆盖。
            file.write(content)
    except FileExistsError as error:
        raise FixableError(f"文件已存在，出于安全考虑不覆盖：{relative_path}") from error
    relative_name = path.relative_to(WORKSPACE_ROOT).as_posix()
    return f"已真实新建文件：{relative_name}（{len(content)} 个字符）"


@tool
def edit_text_file(relative_path: str, old_text: str, new_text: str) -> str:
    """修改 workspace 已有 UTF-8 文本文件：将唯一匹配的 old_text 替换为 new_text。

    先读取原文件，old_text 必须与原文精确一致，不能为空。匹配多处时补充上下文，
    不会全部替换。new_text 为空表示删除这段文字；本工具不创建新文件。
    """
    path = resolve_workspace_path(relative_path)  # 编辑与读取共用工作区边界，禁止越界路径和符号链接。
    ensure_supported_text_file(path)
    if not path.is_file():
        raise FixableError(f"文件不存在或不是文件：{relative_path}")
    if not old_text:
        raise FixableError("old_text 不能为空，请先读取文件并提供要替换的原文。")
    if path.stat().st_size > MAX_READ_BYTES:
        raise FixableError("文件超过 1 MB，不支持整份读取后编辑。")
    original = path.read_bytes()  # 使用字节读取，避免 Windows 自动改变 CRLF 换行或去掉 BOM。
    try:
        content = original.decode("utf-8")
    except UnicodeDecodeError as error:
        raise FixableError("编辑工具只支持 UTF-8 文本文件。") from error
    matches = content.count(old_text)  # 只允许一个精确匹配，不猜测模型想改哪一处。
    if matches != 1:
        raise FixableError(f"old_text 匹配 {matches} 处，必须恰好一处；请重新读取并补充定位上下文。")
    updated = content.replace(old_text, new_text, 1)
    if len(updated) > MAX_WRITE_CHARACTERS:
        raise FixableError("修改后超过 10 万字符，拒绝写入。")
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:  # 同目录临时文件，先写完再切换。
            temporary_path = Path(file.name)
            file.write(updated.encode("utf-8"))  # 原文未替换的换行、BOM 和文字保持原样。
        if path.read_bytes() != original:
            raise FixableError("文件在编辑期间已变化，请重新读取后再修改。")
        os.replace(temporary_path, path)  # 原子替换，写临时文件失败不会破坏原文件。
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)  # 成功后临时文件已移走；失败时清理残留。
    return f"已真实修改文件：{path.relative_to(WORKSPACE_ROOT).as_posix()}（替换 1 处）"


@tool
def delete_file(relative_path: str) -> str:
    """删除 workspace 内的单个文件，执行前必须人工批准；不删除目录，不进入回收站。"""
    path = resolve_workspace_path(relative_path)  # 与其他文件工具共用路径边界，越界目标不能删除。
    if not path.is_file():
        raise FixableError(f"文件不存在或不是文件：{relative_path}")
    path.unlink()  # 只删除一个文件；不使用递归删除，不限制文件后缀。
    return f"已删除文件：{path.relative_to(WORKSPACE_ROOT).as_posix()}（未进入回收站）"


# Agent 只会看见这个列表里的工具。以后增加能力时，在这里显式注册，便于审查边界。
WORKSPACE_TOOLS = [
    list_workspace_files,
    read_text_file,
    search_workspace_text,
    save_new_text_file,
    edit_text_file,
    delete_file,
]
