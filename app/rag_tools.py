"""真实 RAG：向量与关键词双路召回，融合后交给重排模型。

RAG = Retrieval-Augmented Generation，中文常译“检索增强生成”。
它不是一个单独的大模型，而是一条链路：

文档 -> 切块 -> Embedding 向量 -> 保存
问题 -> 关键词/向量各找候选 -> 合并排名 -> 重排 -> 交给聊天模型回答

为了让学习者看清原理，本项目没有把核心步骤藏进大型向量数据库。向量由百炼真实
text-embedding-v4 生成，文本和向量真实保存在 SQLite，检索由这里明确计算。
"""

import hashlib
import json
import math
import re
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import httpx
import jieba
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.config import (
    KNOWLEDGE_DATABASE_PATH,
    WORKSPACE_ROOT,
    load_embedding_settings,
    require_environment_variable,
    resolve_credential,
)
from app.tool_errors import RetryableError
from app.workspace_tools import MAX_READ_BYTES, resolve_workspace_path


RAG_FILE_SUFFIXES = {".md", ".txt", ".json", ".csv"}
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
RRF_K = 60


def search_terms(text: str) -> list[str]:
    """中英混排先分词；标点不是词，避免把 FTS 查询语法当作用户输入。"""

    return [part for word in jieba.cut(text) for part in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", word)]


def split_text(text: str, is_markdown: bool = True) -> list[str]:
    """Markdown 先按标题分节，再按段落、句子递归切块；普通文本只做递归切块。"""

    if not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        keep_separator="end",
    )
    if is_markdown:
        headers = [("#" * level, f"h{level}") for level in range(1, 7)]
        sections = MarkdownHeaderTextSplitter(headers_to_split_on=headers).split_text(text)
    else:
        sections = [Document(page_content=text)]

    chunks: list[str] = []
    for section in sections:
        # 标题路径是现成的上下文，让独立的块也知道自己属于哪一节。
        heading = " > ".join(section.metadata.values())
        for body in splitter.split_text(section.page_content):
            chunks.append(f"{heading}\n\n{body}" if heading else body)

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
    """为测试知识库创建 RAG 表；切块或分词规则变化时直接重建测试库。

    注意：内容哈希只取决于文件原文，因此改了切块规则（split_text）或分词规则
    （search_terms）之后，索引不会自动失效，必须删掉 data/knowledge.sqlite 重建，
    否则会看到"跳过 N 个未变化文件"而实际一条都没更新。
    """

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
    connection.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS rag_chunks_fts USING fts5(path UNINDEXED, chunk_index UNINDEXED, content)"
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
    if not files and not KNOWLEDGE_DATABASE_PATH.exists():
        return "没有找到可以建立知识索引的文本文件。"

    embedding_model = str(load_embedding_settings()["model"]) if files else None
    embeddings = create_embeddings() if files else None
    KNOWLEDGE_DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    indexed_files = 0
    indexed_chunks = 0
    unchanged_files = 0
    removed_files = 0

    # sqlite3 的 with 只负责提交/回滚；closing 另外负责关闭连接，避免 Windows 锁住数据库。
    with closing(sqlite3.connect(KNOWLEDGE_DATABASE_PATH)) as connection, connection:
        prepare_knowledge_database(connection)

        # 只同步本次索引的目录；索引 knowledge/ 不会清掉 notes/ 的记录。
        scope = directory.relative_to(WORKSPACE_ROOT).as_posix()
        current_paths = {path.relative_to(WORKSPACE_ROOT).as_posix() for path in files}
        stored_paths = connection.execute(
            "SELECT path FROM rag_documents UNION SELECT path FROM rag_chunks UNION SELECT path FROM rag_chunks_fts"
        ).fetchall()
        for (stored_path,) in stored_paths:
            if (scope == "." or stored_path.startswith(scope + "/")) and stored_path not in current_paths:
                connection.execute("DELETE FROM rag_documents WHERE path = ?", (stored_path,))
                connection.execute("DELETE FROM rag_chunks WHERE path = ?", (stored_path,))
                connection.execute("DELETE FROM rag_chunks_fts WHERE path = ?", (stored_path,))
                removed_files += 1

        if not files:
            return f"目录里没有可索引文件；清理 {removed_files} 个已删除文件。"

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

            chunks = split_text(content, is_markdown=path.suffix.lower() == ".md")
            if not chunks:
                continue

            # 这一步会真实调用 Embedding API；必须先成功拿到全部向量，才更新数据库。
            vectors = embeddings.embed_documents(chunks)

            connection.execute("DELETE FROM rag_chunks WHERE path = ?", (relative_path,))
            connection.execute("DELETE FROM rag_chunks_fts WHERE path = ?", (relative_path,))
            for chunk_index, (chunk, vector) in enumerate(zip(chunks, vectors)):
                connection.execute(
                    """
                    INSERT INTO rag_chunks(path, chunk_index, content, embedding_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (relative_path, chunk_index, chunk, json.dumps(vector)),
                )
                connection.execute(
                    "INSERT INTO rag_chunks_fts(path, chunk_index, content) VALUES (?, ?, ?)",
                    (relative_path, chunk_index, " ".join(search_terms(chunk))),
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
        f"跳过 {unchanged_files} 个未变化文件；清理 {removed_files} 个已删除文件。"
        f"向量模型：{embedding_model}。"
    )


@tool
def search_knowledge_base(query: str, top_k: int = 4) -> str:
    """关键词与向量各召回候选，RRF 融合，再用真实重排模型选证据；top_k 为 1 到 8。"""

    question = query.strip()
    if not question:
        raise RetryableError("RAG 查询不能为空。")
    if top_k < 1 or top_k > 8:
        raise RetryableError("top_k 必须在 1 到 8 之间。")
    if not KNOWLEDGE_DATABASE_PATH.exists():
        return "知识库尚未建立。请先调用 index_knowledge_base。"

    embedding_model = str(load_embedding_settings()["model"])
    candidate_limit = max(top_k * 4, 8)
    with closing(sqlite3.connect(KNOWLEDGE_DATABASE_PATH)) as connection, connection:
        prepare_knowledge_database(connection)
        # 入库和查询用同一套中文分词；FTS5 的 bm25() 统一给中英文关键词排序。
        keyword_query = " OR ".join(f'"{word}"' for word in dict.fromkeys(search_terms(question)))
        keyword_rows = connection.execute(
            """
            SELECT c.path, c.chunk_index, c.content
            FROM rag_chunks_fts AS f
            JOIN rag_chunks AS c ON c.path = f.path AND c.chunk_index = f.chunk_index
            JOIN rag_documents AS d ON d.path = f.path
            WHERE rag_chunks_fts MATCH ? AND d.embedding_model = ?
            ORDER BY bm25(rag_chunks_fts)
            LIMIT ?
            """,
            (keyword_query, embedding_model, candidate_limit),
        ).fetchall() if keyword_query else []
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

    # 同一 chunk 可能被两路都找出；按 (路径, 块号) 去重，两个名次共同加分。
    fused: dict[tuple[str, int], dict] = {}
    for rank, (path, chunk_index, content) in enumerate(keyword_rows, start=1):
        fused[(path, chunk_index)] = {
            "path": path, "chunk_index": chunk_index,
            "content": content, "rrf": 1 / (RRF_K + rank),
        }
    for rank, (_, path, chunk_index, content) in enumerate(scored_chunks[:candidate_limit], start=1):
        key = (path, chunk_index)
        if key in fused:
            fused[key]["rrf"] += 1 / (RRF_K + rank)
        else:
            fused[key] = {
                "path": path, "chunk_index": chunk_index,
                "content": content, "rrf": 1 / (RRF_K + rank),
            }
    candidates = sorted(fused.values(), key=lambda item: item["rrf"], reverse=True)[:candidate_limit]

    # 重排会产生额外调用；接口地址从配置读取，不在代码中写死。
    try:
        response = httpx.post(
            require_environment_variable("RERANK_API_URL"),
            headers={"Authorization": f"Bearer {resolve_credential('RERANK_API_KEY', 'LLM_API_KEY')}"},
            json={"model": "qwen3-rerank", "query": question,
                  "documents": [item["content"] for item in candidates], "top_n": top_k},
            timeout=30,
        )
        response.raise_for_status()
    except httpx.HTTPError:
        # 网络或接口故障时证据仍可用，但必须明说这次只按 RRF 排序。
        return "\n\n".join(
            f"[证据 {rank}] source={item['path']}#chunk-{item['chunk_index']} "
            f"ranking=RRF_only（重排失败）\n{item['content']}"
            for rank, item in enumerate(candidates[:top_k], start=1)
        )
    results = response.json().get("results")
    if not isinstance(results, list) or not results:
        raise RuntimeError("重排接口未返回有效结果；本次检索没有冒充已完成重排。")

    evidence_blocks: list[str] = []
    for rank, result in enumerate(results[:top_k], start=1):
        index = result.get("index")
        if not isinstance(index, int) or not 0 <= index < len(candidates):
            raise RuntimeError("重排接口返回了无效候选编号。")
        item = candidates[index]
        evidence_blocks.append(
            f"[证据 {rank}] source={item['path']}#chunk-{item['chunk_index']} "
            f"rerank_score={result['relevance_score']:.4f}\n{item['content']}"
        )
    return "\n\n".join(evidence_blocks)


RAG_TOOLS = [index_knowledge_base, search_knowledge_base]
