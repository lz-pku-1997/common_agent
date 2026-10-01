"""Skill 渐进式加载：启动只给模型名称和用途，选中后才读取正文。

这些文件由项目维护者编辑，位于 workspace 外，Agent 的写工具无法修改。
只支持两个单行、无引号的元数据字段；这是简单格式约定，不是完整 YAML 解析器。
"""

import re
from pathlib import Path

from langchain.tools import tool
from langchain_core.tools import BaseTool

from app.config import MAX_TOOL_RESULT_CHARS, SKILLS_ROOT
from app.tool_errors import NonRetryableError, RetryableError


def load_skill_catalog() -> dict[str, tuple[str, Path]]:  # 返回“技能名 -> (用途, 文件路径)”的字典。
    """建立 name -> (用途, 文件路径) 清单；这里只读取文件头，不读取正文。"""

    catalog = {}  # 这里只存目录信息，不把所有 Skill 正文提前塞进模型上下文。
    for path in sorted(SKILLS_ROOT.glob("*/SKILL.md")):  # * 只匹配一层子目录；不是递归搜索任意深度。
        resolved = path.resolve()  # 把相对路径、.. 和符号链接解析成实际的绝对路径。
        if not resolved.is_relative_to(SKILLS_ROOT):  # 检查解析后的真实位置仍在 skills 根目录内。
            raise NonRetryableError("Skill 文件必须位于项目 skills 目录内。")

        metadata = {}  # 暂存文件头的 name 和 description。
        with resolved.open(encoding="utf-8-sig") as file:  # with 结束时自动关闭文件；utf-8-sig 兼容带 BOM 的 UTF-8。
            if file.readline().strip() != "---":  # 读第一行并去空白；必须用 --- 开始元数据区。
                raise ValueError(f"{path} 必须以 --- 元数据块开始。")
            for line in file:  # 逐行读文件头，不一次性读取 Skill 正文。
                line = line.strip()  # 去掉换行和两端空格，后面校验更简单。
                if line == "---":  # 第二个 --- 表示元数据结束，后面才是 Skill 正文。
                    break  # break 会跳出 for，因此下面 for-else 不会执行。
                if not line:  # 允许元数据区里有空行，但不把空行当字段解析。
                    continue
                key, separator, value = line.partition(":")  # 只在第一个冒号处分成“字段名、分隔符、字段值”。
                key, value = key.strip(), value.strip()  # 清理冒号左右可能出现的空格。
                if (not separator or key not in {"name", "description"}
                        or key in metadata or not value
                        or value.startswith(('"', "'", "|", ">"))
                        or ": " in value or " #" in value):
                    raise ValueError(f"{path} 只支持 name、description 两个单行无引号字段。")
                metadata[key] = value  # 通过格式检查后，才写进元数据字典。
            else:  # Python 的 for-else：循环没遇到 break、读到文件末尾时才执行。
                raise ValueError(f"{path} 的元数据块缺少结尾 ---。")

        name = metadata.get("name", "")  # get 在字段缺失时返回空字符串，交给下面统一报错。
        description = metadata.get("description", "")  # Skill 用途会显示给模型，供它决定是否读取正文。
        if (not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name)
                or len(name) > 64 or name != path.parent.name
                or not description or len(description) > 1024):
            raise ValueError(f"{path} 的 name 必须与目录同名；description 须为 1～1024 字符。")
        # 值用二元组保存；第一项是用途，第二项是已经核对过的文件路径。
        catalog[name] = (description, path)  # 记下用途和文件位置；全文仍未加载。
    return catalog  # 没有 Skill 时返回空字典，调用方据此不注册 skill_view。


def build_skill_tools(catalog: dict[str, tuple[str, Path]]) -> list[BaseTool]:  # 把清单转成模型可调用的 LangChain 工具。
    """把清单留在工具闭包中，模型只能按名称读取已登记的 Skill。"""

    if not catalog:  # 没有指南就不暴露一个注定无法成功的工具。
        return []

    @tool  # LangChain 装饰器：把普通 Python 函数包装成带参数说明的 Tool。
    def skill_view(name: str) -> str:
        """读取一个已登记 Skill 的完整操作指南。name 必须来自系统提示中的 Skill 清单。"""

        if name not in catalog:  # 只接受启动扫描时登记的名字，不接受模型传入任意文件路径。
            raise RetryableError(f"Skill 不存在：{name}。可用名称：{', '.join(catalog)}")
        path = catalog[name][1].resolve()  # 清单里的第 2 项是路径；调用时重新解析以发现路径变化。
        # 调用时再核对一次，防止启动后有人把目录改成指向外面的符号链接。
        if not path.is_relative_to(SKILLS_ROOT):  # 再做一次边界检查，防止启动后符号链接被替换。
            raise NonRetryableError("Skill 文件必须位于项目 skills 目录内。")
        with path.open(encoding="utf-8-sig") as file:  # 只有模型选中该 Skill 后才打开正文。
            content = file.read(MAX_TOOL_RESULT_CHARS + 1)  # 多读 1 个字符，用来判断是否超过上限。
        if len(content) > MAX_TOOL_RESULT_CHARS:  # 超限就明确失败，避免静默返回半份操作指南。
            raise NonRetryableError("Skill 正文过长，请维护者缩短后重启。")
        # 限制全文长度，确保指南不会被通用工具入口截成半份或落盘到 workspace。
        return content  # 完整正文作为 ToolMessage 回给模型；frontmatter 也保留名称和用途。

    return [skill_view]  # Registry 接收的是工具列表，因此这里用单元素列表返回。
