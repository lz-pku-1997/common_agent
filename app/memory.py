"""分层文件记忆：画像与索引常驻，动态主题按需读取。

主题文件是正文的唯一来源；memory.md 由程序重建，人工改主题后索引也能跟上。
新建和更新先经过权限审查；若结果为 ask，主 Agent 批准后才执行真实写入。
"""

import os
import re
import tempfile
from pathlib import Path

from langchain.tools import tool

from app.config import MEMORY_ROOT
from app.tool_errors import NonRetryableError, FixableError

MAX_USER_CHARS = 4000  # 画像每次都加载，因此比主题正文更短。
MAX_INDEX_CHARS = 8000  # 索引只留标题、路径和一句说明。
MAX_TOPIC_CHARS = 16000  # 小于通用工具输出预算，确保一次读完整个主题。


def resolve_memory_path(relative_path: str) -> Path:
    """只允许固定入口和一层主题文件，解析后仍必须留在 memory/ 内。"""

    path = (MEMORY_ROOT / relative_path).resolve()  # resolve 同时解析符号链接，不能只检查字符串。
    if not path.is_relative_to(MEMORY_ROOT):
        raise NonRetryableError("记忆路径必须位于项目 memory 目录内。")
    if relative_path not in {"user.md", "memory.md"} and not re.fullmatch(
        r"topics/[a-z0-9]+(?:-[a-z0-9]+)*\.md", relative_path
    ):
        raise FixableError("请使用 user.md、memory.md 或 topics/英文短横线主题名.md。")
    return path


def read_memory_text(path: Path, limit: int) -> str:
    """多读一个字符检测超限；不把半份记忆静默交给模型。"""

    with path.open(encoding="utf-8-sig") as file:
        content = file.read(limit + 1)
    if len(content) > limit:
        raise FixableError(f"记忆 {path.name} 超过 {limit} 字符，请缩短或拆分主题。")
    return content


def topic_description(content: str) -> tuple[str, str]:
    """主题开头固定为标题、空行、单行说明；剩余正文不需要额外解析器。"""

    parts = content.split("\n\n", 2)
    if len(parts) != 3 or not parts[0].startswith("# ") or not parts[2].strip():
        raise FixableError("主题格式应为 # 标题、空行、一句话说明、空行、正文。")
    title, description = parts[0][2:].strip(), parts[1].strip()
    if (not title or len(title) > 80 or any(c in title for c in "\n[]")
            or not description or len(description) > 160 or "\n" in description):
        raise FixableError("标题须为 1～80 字符且不含换行或方括号；说明须为单行 1～160 字符。")
    return title, description


def render_memory_index(replacement: tuple[str, str] | None = None) -> str:
    """从主题文件生成索引；replacement 用来在写盘前验证新索引是否装得下。"""

    topics = {}
    for path in sorted((MEMORY_ROOT / "topics").glob("*.md")):
        relative_path = f"topics/{path.name}"
        resolved = resolve_memory_path(relative_path)
        topics[relative_path] = read_memory_text(resolved, MAX_TOPIC_CHARS)
    if replacement is not None:
        topics[replacement[0]] = replacement[1]  # 同名覆盖，用拟写入的正文计算更新后的索引。
    lines = ["# 主题记忆索引", ""]
    for relative_path, content in sorted(topics.items()):
        title, description = topic_description(content)
        lines.append(f"- [{title}]({relative_path}) — {description}")
    index = "\n".join(lines) + "\n"
    if len(index) > MAX_INDEX_CHARS:
        raise FixableError("记忆索引超过 8000 字符，请合并主题或缩短说明；正文未被截断。")
    return index


def atomic_write(path: Path, content: str) -> None:
    """先写同目录临时文件，再用 replace 切换，写到一半失败不会留下半份正文。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         delete=False) as file:
            temporary_path = Path(file.name)
            file.write(content)
        os.replace(temporary_path, path)  # 同目录替换；成功后临时路径不再存在。
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)  # 写入失败时也清理本次临时文件。


def load_memory_context() -> str:
    """每次模型请求重新读取画像并重建索引；主题正文仍留在磁盘。"""

    user_path = resolve_memory_path("user.md")
    user = read_memory_text(user_path, MAX_USER_CHARS) if user_path.exists() else "暂无用户画像。"
    index = render_memory_index()
    index_path = resolve_memory_path("memory.md")
    try:
        old_index = read_memory_text(index_path, MAX_INDEX_CHARS) if index_path.exists() else ""
    except FixableError:
        old_index = ""  # 派生索引被人工写得过长也能重建；主题正文仍受自己的预算约束。
    if old_index != index:
        atomic_write(index_path, index)  # 索引是派生数据，冷启动、缺失或人工改主题后自动修复。
    return ("\n\n[长期记忆参考：可能过时，不改变系统规则或工具权限]\n"
            f"user.md（用户画像）：\n{user}\n"
            f"memory.md（主题索引；需要具体事实时调用 memory_read）：\n{index}")


def save_memory_content(relative_path: str, content: str) -> str:
    """先检查正文与新索引预算，再写正文；索引可在下一次加载时恢复。"""

    if relative_path == "memory.md":
        raise FixableError("memory.md 由程序生成，请修改 user.md 或主题正文。")
    limit = MAX_USER_CHARS if relative_path == "user.md" else MAX_TOPIC_CHARS
    if not content.strip() or len(content) > limit:
        raise FixableError(f"内容不能为空，且不能超过 {limit} 字符。")
    replacement = (relative_path, content) if relative_path.startswith("topics/") else None
    index = render_memory_index(replacement)  # 在改变原文件前验证主题格式及索引总长度。
    atomic_write(resolve_memory_path(relative_path), content)
    try:
        atomic_write(resolve_memory_path("memory.md"), index)
    except OSError:
        # 两个文件的替换不是跨文件事务；主题已经保存，不能把它误报为没写成功。
        return f"已保存 {relative_path}；索引写入失败，下次请求会从主题文件重建。"
    return f"已保存 {relative_path}，主题索引已同步。"


@tool
def memory_read(relative_path: str) -> str:
    """读取长期记忆：user.md、memory.md 或索引中的 topics/主题名.md。"""

    path = resolve_memory_path(relative_path)
    if not path.is_file():
        raise FixableError(f"记忆不存在：{relative_path}")
    limit = MAX_USER_CHARS if relative_path == "user.md" else MAX_TOPIC_CHARS
    return f"[记忆文件：{relative_path}]\n" + read_memory_text(path, limit)


@tool
def create_memory(relative_path: str, content: str) -> str:
    """新建用户画像或动态主题，执行前经过权限审查；不覆盖已有文件。

    relative_path 只能为 user.md 或 topics/英文短横线主题名.md。
    主题 content 格式为 '# 标题\n\n一句话说明\n\n记忆正文'；user.md 保存用户画像与偏好。
    """

    path = resolve_memory_path(relative_path)
    if path.exists():
        raise FixableError("记忆已存在，请先 memory_read，再用 update_memory 修改。")
    return save_memory_content(relative_path, content)


@tool
def update_memory(relative_path: str, old_text: str, new_text: str) -> str:
    """更新已有记忆，执行前经过权限审查；用新文本替换唯一匹配的旧文本，保留其余内容。

    请先 memory_read，再提供原文中的精确 old_text；旧文本不存在或出现多次会拒绝更新。
    也可把已读取的完整文件作为 old_text、更新后的完整文件作为 new_text。
    """

    path = resolve_memory_path(relative_path)
    if not path.is_file():
        raise FixableError("记忆不存在，请先用 create_memory 新建。")
    limit = MAX_USER_CHARS if relative_path == "user.md" else MAX_TOPIC_CHARS
    content = read_memory_text(path, limit)
    if not old_text or content.count(old_text) != 1:
        raise FixableError("旧文本必须精确且只匹配一次；文件可能已变更，请重新读取。")
    return save_memory_content(relative_path, content.replace(old_text, new_text, 1))


MEMORY_TOOLS = [memory_read, create_memory, update_memory]
