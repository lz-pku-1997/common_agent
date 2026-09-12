"""不依赖聊天模型，直接验收“切块 -> 向量 -> SQLite -> 语义检索”。"""

from app.rag_tools import index_knowledge_base, search_knowledge_base


def main() -> None:
    index_result = index_knowledge_base.invoke({"relative_directory": "knowledge"})
    search_result = search_knowledge_base.invoke(
        {"query": "共享基础版本用来验收检索的秘密代号是什么？", "top_k": 2}
    )

    assert "ORANGE-RIVER-926" in search_result
    assert "source=knowledge/common_agent_facts.md#chunk-" in search_result

    print("RAG 真实验收通过：")
    print(f"- {index_result}")
    print("- 语义查询召回了 ORANGE-RIVER-926，并包含文件级来源。")


if __name__ == "__main__":
    main()
