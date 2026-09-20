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
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_PATH = DATA_DIR / "agent.sqlite"
KNOWLEDGE_DATABASE_PATH = DATA_DIR / "knowledge.sqlite"


def require_environment_variable(name: str) -> str:
    """读取一个必填环境变量；没有配置就给出能看懂的错误。"""

    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"缺少配置 {name}。请复制 .env.example 为 .env，并填写真实值。"
        )
    return value


def load_model_settings() -> dict[str, str]:
    """返回创建模型所需的三项配置，但不打印密钥。"""

    return {
        "api_key": require_environment_variable("LLM_API_KEY"),
        "base_url": require_environment_variable("LLM_BASE_URL"),
        "model": require_environment_variable("LLM_MODEL"),
    }


def load_embedding_settings() -> dict[str, str | int]:
    """返回真实向量模型配置。

    Embedding 与聊天模型用途不同：聊天模型生成文字，Embedding 模型把文字变成向量。
    没有在 .env 中显式配置时，使用百炼当前推荐的 text-embedding-v4 与 1024 维。
    """

    dimensions_text = os.getenv("EMBEDDING_DIMENSIONS", "1024").strip()
    try:
        dimensions = int(dimensions_text)
    except ValueError as error:
        raise RuntimeError("EMBEDDING_DIMENSIONS 必须是整数。") from error

    return {
        "api_key": require_environment_variable("LLM_API_KEY"),
        "base_url": require_environment_variable("LLM_BASE_URL"),
        "model": os.getenv("EMBEDDING_MODEL", "text-embedding-v4").strip(),
        "dimensions": dimensions,
    }


def load_agent_engine() -> str:
    """选择用哪套 Agent 引擎：`manual`（自己画的图）或 `framework`（create_agent）。

    两套都保留是为了对照。默认走 manual，因为它每一步都能解释。
    """

    engine = os.getenv("COMMON_AGENT_ENGINE", "manual").strip().lower()
    return engine if engine in {"manual", "framework"} else "manual"


def prepare_runtime_directories() -> None:
    """确保运行时目录存在。

    exist_ok=True 的意思是：目录已经存在也不报错。因此每次启动都可以安全调用。
    """

    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
