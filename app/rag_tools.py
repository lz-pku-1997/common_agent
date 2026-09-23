"""真实向量 RAG：文档切块、Embedding、SQLite 存储、余弦相似度检索。

RAG = Retrieval-Augmented Generation，中文常译“检索增强生成”。
它不是一个单独的大模型，而是一条链路：

文档 -> 切块 -> Embedding 向量 -> 保存
问题 -> Embedding 向量 -> 找相似块 -> 把证据交给聊天模型回答

为了让学习者看清原理，本项目没有把核心步骤藏进大型向量数据库。向量由百炼真实
text-embedding-v4 生成，文本和向量真实保存在 SQLite，检索由这里明确计算。
"""

import hashlib
import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path

from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings

from app.config import KNOWLEDGE_DATABASE_PATH, WORKSPACE_ROOT, load_embedding_settings
from app.tool_errors import RetryableError
from app.workspace_tools import MAX_READ_BYTES, resolve_workspace_path


RAG_FILE_SUFFIXES = {".md", ".txt", ".json", ".csv"}
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def split_text(text: str) -> list[str]:
    """把长文切成约 800 字、相邻重复约 120 字的小块。

    为什么要 overlap（重叠）？如果一句话恰好横跨两个块，没有重叠就可能两边都不完整。
    重复一点边界文字，可以提高检索到完整语义的概率。
    """

    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(cleaned_text):
        end = min(start + CHUNK_SIZE, len(cleaned_text))

        # 如果还没到文章结尾，优先在后半段的换行处切开，少切断一个自然段。
        if end < len(cleaned_text):
            newline_position = cleaned_text.rfind("\n", start + CHUNK_SIZE // 2, end)
            if newline_position != -1:
                end = newline_position

        chunk = cleaned_text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(cleaned_text):
            break

        next_start = end - CHUNK_OVERLAP
        # 保险丝：无论文本长什么样，start 都必须向前移动，避免 while 死循环。
        start = next_start if next_start > start else end

    return chunks


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """计算两个向量方向有多接近，1 最相似，0 表示大致无关。"""

    if len(left) != len(right) or not left:
        raise ValueError("两个向量必须非空且维度相同。")

    dot_product = sum(a * b for a, b in zip(left, right))
    left_length = math.sqrt(sum(value * value for value in left))
    right_length = math.sqrt(sum(value * value for value in right))
    if left_length == 0 or right_length == 0:
        return 0.0
    return dot_product / (left_length * right_length)


def create_embeddings() -> OpenAIEmbeddings:
    """创建百炼 OpenAI 兼容 Embedding 客户端。"""

    settings = load_embedding_settings()
    return OpenAIEmbeddings(
        model=str(settings["model"]),
        api_key=str(settings["api_key"]),
        base_url=str(settings["base_url"]),
        dimensions=int(settings["dimensions"]),
        chunk_size=10,
        # 这是一个真实兼容性边界：LangChain 默认可能先把文本变成 OpenAI token ID；
        # 百炼 embeddings 接口要求字符串。关闭预分词后，原始字符串会直接发送给百炼。
        check_embedding_ctx_length=False,
    )


def prepare_knowledge_database(connection: sqlite3.Connection) -> None:
    """创建 RAG 所需的两张表；已有表不会被清空。"""

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS rag_documents (
            path TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            embedding_model TEXT NOT NULL,
            indexed_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS rag_chunks (
            path TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding_json TEXT NOT NULL,
            PRIMARY KEY (path, chunk_index)
        )
        """
    )


def find_indexable_files(directory: Path) -> list[Path]:
    """找到目录下所有大小合规的 RAG 文本文件。"""

    files: list[Path] = []
    for path in directory.rglob("*"):
        # 逐个校验真实路径，避免目录内的符号链接把外部文件送去建立索引。
        path = path.resolve()
        if (
            path.is_relative_to(WORKSPACE_ROOT)
            and path.is_file()
            and path.suffix.lower() in RAG_FILE_SUFFIXES
            and path.stat().st_size <= MAX_READ_BYTES
        ):
            files.append(path)
    return sorted(files)


@tool
def index_knowledge_base(relative_directory: str = ".") -> str:
    """把 workspace 某目录的 md/txt/json/csv 文档切块并建立真实向量索引；默认索引整个 workspace。"""

    directory = resolve_workspace_path(relative_directory)
    if not directory.exists() or not directory.is_dir():
        raise RetryableError(f"知识目录不存在或不是目录：{relative_directory}")

    files = find_indexable_files(directory)
    if not files:
        return "没有找到可以建立知识索引的文本文件。"

    embedding_settings = load_embedding_settings()
    embedding_model = str(embedding_settings["model"])
    embeddings = create_embeddings()
    KNOWLEDGE_DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    indexed_files = 0
    indexed_chunks = 0
    unchanged_files = 0

    with sqlite3.connect(KNOWLEDGE_DATABASE_PATH) as connection:
        prepare_knowledge_database(connection)

        for path in files:
            content = path.read_text(encoding="utf-8-sig", errors="replace")
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            relative_path = path.relative_to(WORKSPACE_ROOT).as_posix()

            old_row = connection.execute(
                "SELECT content_hash, embedding_model FROM rag_documents WHERE path = ?",
                (relative_path,),
            ).fetchone()
            if old_row == (content_hash, embedding_model):
                unchanged_files += 1
                continue

            chunks = split_text(content)
            if not chunks:
                continue

            # 这一步会真实调用 Embedding API；必须先成功拿到全部向量，才更新数据库。
            vectors = embeddings.embed_documents(chunks)

            connection.execute("DELETE FROM rag_chunks WHERE path = ?", (relative_path,))
            for chunk_index, (chunk, vector) in enumerate(zip(chunks, vectors)):
                connection.execute(
                    """
                    INSERT INTO rag_chunks(path, chunk_index, content, embedding_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (relative_path, chunk_index, chunk, json.dumps(vector)),
                )

            connection.execute(
                """
                INSERT INTO rag_documents(path, content_hash, embedding_model, indexed_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    embedding_model = excluded.embedding_model,
                    indexed_at = excluded.indexed_at
                """,
                (
                    relative_path,
                    content_hash,
                    embedding_model,
                    datetime.now().astimezone().isoformat(timespec="seconds"),
                ),
            )
            indexed_files += 1
            indexed_chunks += len(chunks)

    return (
        f"知识索引完成：新增或更新 {indexed_files} 个文件、{indexed_chunks} 个文本块；"
        f"跳过 {unchanged_files} 个未变化文件。向量模型：{embedding_model}。"
    )


@tool
def search_knowledge_base(query: str, top_k: int = 4) -> str:
    """对已建立索引的 workspace 知识做语义检索，返回最相关原文、来源和相似度；top_k 为 1 到 8。"""

    question = query.strip()
    if not question:
        raise RetryableError("RAG 查询不能为空。")
    if top_k < 1 or top_k > 8:
        raise RetryableError("top_k 必须在 1 到 8 之间。")
    if not KNOWLEDGE_DATABASE_PATH.exists():
        return "知识库尚未建立。请先调用 index_knowledge_base。"

    embedding_model = str(load_embedding_settings()["model"])
    with sqlite3.connect(KNOWLEDGE_DATABASE_PATH) as connection:
        prepare_knowledge_database(connection)
        rows = connection.execute(
            """
            SELECT c.path, c.chunk_index, c.content, c.embedding_json
            FROM rag_chunks AS c
            JOIN rag_documents AS d ON d.path = c.path
            WHERE d.embedding_model = ?
            """,
            (embedding_model,),
        ).fetchall()

    if not rows:
        return "当前向量模型下没有索引内容。请先调用 index_knowledge_base。"

    query_vector = create_embeddings().embed_query(question)
    scored_chunks: list[tuple[float, str, int, str]] = []
    for path, chunk_index, content, embedding_json in rows:
        stored_vector = json.loads(embedding_json)
        score = cosine_similarity(query_vector, stored_vector)
        scored_chunks.append((score, path, chunk_index, content))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)

    evidence_blocks: list[str] = []
    for rank, (score, path, chunk_index, content) in enumerate(
        scored_chunks[:top_k], start=1
    ):
        evidence_blocks.append(
            f"[证据 {rank}] source={path}#chunk-{chunk_index} score={score:.4f}\n{content}"
        )
    return "\n\n".join(evidence_blocks)


RAG_TOOLS = [index_knowledge_base, search_knowledge_base]
