"""可信用户本机使用的 PowerShell 工具；审批不是沙箱隔离。"""

import asyncio
import tempfile

from langchain_core.tools import tool

from app.config import MAX_TOOL_RESULT_CHARS, WORKSPACE_ROOT
from app.tool_errors import FixableError, NonRetryableError


async def terminate_command(process) -> None:
    """超时或取消时，尽力结束 PowerShell 和它的子进程，不撤销已发生的操作。"""
    if process.returncode is not None:
        return
    killer = await asyncio.create_subprocess_exec(
        "taskkill.exe", "/PID", str(process.pid), "/T", "/F",  # Windows 按 PID 清理进程树，不把模型命令传给 taskkill。
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        await asyncio.wait_for(killer.wait(), timeout=10)
    except TimeoutError:
        killer.kill()
        await killer.wait()
    if process.returncode is None:
        try:
            process.kill()  # taskkill 失败时至少结束直接启动的 PowerShell。
        except ProcessLookupError:
            pass
    await process.wait()


@tool
async def execute_command(command: str, description: str) -> str:
    """在 Windows PowerShell 执行命令，默认工作目录为 workspace。

    优先使用专用工具。description 用中文说明动作及删除、覆盖、安装、联网等影响；
    权限由独立安全子 Agent 判断，不接受调用模型自行免除审批。
    此工具没有沙箱，可访问当前用户有权限访问的资源；不要绕过其他工具的拒绝。
    """
    if not command.strip() or not description.strip():
        raise FixableError("command 和 description 都不能为空。")
    # 输出先写临时文件，避免子进程大量输出撑爆内存；返回后仍走统一截断和不可信包装。
    with tempfile.TemporaryFile() as output:
        process = await asyncio.create_subprocess_exec(
            "powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command",
            "$ErrorActionPreference='Stop'; [Console]::OutputEncoding=[System.Text.UTF8Encoding]::new(); " + command,
            cwd=str(WORKSPACE_ROOT), stdin=asyncio.subprocess.DEVNULL,
            stdout=output, stderr=asyncio.subprocess.STDOUT,  # 合并正常输出和错误，保留它们写入的顺序。
        )
        try:
            await asyncio.wait_for(process.wait(), timeout=100)  # 留出清理时间，仍受外层 120 秒总兜底约束。
        except (TimeoutError, asyncio.CancelledError) as error:
            await terminate_command(process)
            if isinstance(error, asyncio.CancelledError):
                raise  # 保留上层取消语义，不伪装成普通工具结果。
            raise NonRetryableError("命令执行超过 100 秒，已尝试终止进程树；已发生的修改不会撤销，请先核实结果") from error
        output.seek(0)
        content = output.read(MAX_TOOL_RESULT_CHARS * 4).decode("utf-8", errors="replace")  # 最多读取有限字节，不整份装入内存。
        if output.read(1):
            content += "\n[命令输出过长，后续输出已省略]"
    if process.returncode != 0:
        raise NonRetryableError(f"命令退出码 {process.returncode}，可能已有部分副作用，不自动重试。输出：\n{content}")
    return f"命令退出码：0\n{content or '（没有输出）'}"


SHELL_TOOLS = [execute_command]
