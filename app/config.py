"""集中管理项目配置。

为什么不在每个文件里各读一次 .env？
因为模型名、工作区、数据库路径都属于“整个应用的共同事实”。集中定义后，
以后切换模型或迁移电脑只改一个地方，其他模块不会各自形成不同配置。
"""

import os
from pathlib import Path

from dotenv import load_dotenv


# __file__ 是“当前这个 config.py 文件的路径”。
# .resolve() 把它变成绝对路径；.parent 连续两次就回到项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 明确指定读取项目自己的 .env，避免从错误的运行目录读到别的项目配置。
load_dotenv(PROJECT_ROOT / ".env")

WORKSPACE_ROOT = (PROJECT_ROOT / "workspace").resolve()
AGENT_RULES_PATH = PROJECT_ROOT / "prompts" / "AGENTS.md"  # common_agent 启动时读取的运行规则文件。
SKILLS_ROOT = (PROJECT_ROOT / "skills").resolve()  # Skill 的安全根目录；读取指南时必须留在这里面。
MEMORY_ROOT = (PROJECT_ROOT / "memory").resolve()  # 产品自己的跨会话记忆，与助手协作的 common_memory 无关。
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_PATH = DATA_DIR / "agent.sqlite"
KNOWLEDGE_DATABASE_PATH = DATA_DIR / "knowledge.sqlite"

# 文件读取和工具入口共用这条字符预算，避免一页读完又被入口截断。
MAX_TOOL_RESULT_CHARS = 20000


def require_environment_variable(name: str) -> str:
    """读取一个必填环境变量；没有配置就给出能看懂的错误。"""

    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"缺少配置 {name}。请复制 .env.example 为 .env，并填写真实值。"
        )
    return value


def resolve_credential(name: str, fallback: str) -> str:
    """读取一个专用配置；没有配置时回落到通用配置。

    为什么要回落：对话可以走套餐专用端点，但套餐不提供 Embedding 与重排，
    这两项仍然走通用端点，而两套端点的地址和密钥不同。
    """

    return os.getenv(name, "").strip() or require_environment_variable(fallback)


def load_model_settings(mode: str) -> dict[str, str | int]:
    """返回创建模型所需的配置，但不打印密钥。"""

    if mode not in {"fast", "strong"}:
        raise ValueError("模型模式只能是 fast 或 strong。")
    prefix = f"LLM_{mode.upper()}"  # 聊天模型必须明确选择 fast 或 strong，不再隐式回落到通用配置。
    window = int(require_environment_variable(f"{prefix}_MAX_INPUT_TOKENS"))
    if window <= 0:
        raise ValueError(f"{prefix}_MAX_INPUT_TOKENS 必须大于 0。")
    return {
        "api_key": require_environment_variable("LLM_API_KEY"),
        "base_url": require_environment_variable("LLM_BASE_URL"),
        "model": require_environment_variable(f"{prefix}_MODEL"),
        "max_input_tokens": window,
    }


def load_embedding_settings() -> dict[str, str | int]:
    """返回真实向量模型配置。

    Embedding 与聊天模型用途不同：聊天模型生成文字，Embedding 模型把文字变成向量。
    没有在 .env 中显式配置时，使用百炼当前推荐的 text-embedding-v4 与 1024 维。

    地址与密钥优先读 EMBEDDING_BASE_URL / EMBEDDING_API_KEY，没配才回落到对话那套，
    因为套餐端点只提供对话模型，向量必须走通用端点。
    """

    dimensions_text = os.getenv("EMBEDDING_DIMENSIONS", "1024").strip()
    try:
        dimensions = int(dimensions_text)
    except ValueError as error:
        raise RuntimeError("EMBEDDING_DIMENSIONS 必须是整数。") from error

    return {
        "api_key": resolve_credential("EMBEDDING_API_KEY", "LLM_API_KEY"),
        "base_url": resolve_credential("EMBEDDING_BASE_URL", "LLM_BASE_URL"),
        "model": os.getenv("EMBEDDING_MODEL", "text-embedding-v4").strip(),
        "dimensions": dimensions,
    }


def prepare_runtime_directories() -> None:
    """确保运行时目录存在。

    exist_ok=True 的意思是：目录已经存在也不报错。因此每次启动都可以安全调用。
    """

    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
