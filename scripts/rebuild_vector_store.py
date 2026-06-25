"""重建向量数据库脚本（父子索引版）

流程：
    1. 加载医疗文档
    2. Markdown 标题切分 → Parent 文档（完整章节，~400 字符）
    3. Parent → Child 切分（按行分组，~150 字符）
    4. Child chunks 写入 Chroma 向量库
    5. Parent 文档存入 ParentChildManager（InMemoryStore + 磁盘持久化）
    6. BM25 索引用 child chunks 重建
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.rag import load_medical_documents, split_documents, get_vector_store
from app.rag.parent_child_store import (
    get_parent_child_manager,
    reset_parent_child_manager,
)


def rebuild():
    print("=" * 60)
    print("开始重建向量数据库（父子索引版）")
    print("=" * 60)

    # 1. 加载文档
    print("\n[1/5] 加载医疗文档...")
    docs = load_medical_documents()
    print(f"  加载了 {len(docs)} 个文档")

    # 2. Markdown 标题切分 → Parent 文档
    print("\n[2/5] Markdown 标题切分（Parent 文档）...")
    parent_docs = split_documents(docs)
    print(f"  切分了 {len(parent_docs)} 个 Parent 文档")

    # 3. 构建父子索引：Parent → Child 切分
    print("\n[3/5] 构建父子索引...")
    reset_parent_child_manager()
    parent_manager = get_parent_child_manager()
    child_chunks = parent_manager.build_index(parent_docs, child_chunk_size=150)
    print(f"  生成 {len(child_chunks)} 个 Child chunk")
    print(f"  Parent 文档数：{parent_manager.parent_count()}")

    # 4. Child chunks 写入 Chroma 向量库
    print("\n[4/5] Child chunks 写入向量库...")
    vector_store = get_vector_store(child_chunks, force_rebuild=True)
    print(f"  向量库创建完成！")

    # 5. Parent store 持久化到磁盘
    print("\n[5/5] Parent store 持久化...")
    parent_manager.save_to_disk()
    print(f"  Parent store 已保存")

    # 6. 测试检索
    print("\n" + "=" * 60)
    print("测试检索...")
    from app.rag.hybrid_retriever import get_hybrid_retriever
    retriever = get_hybrid_retriever(k=3, rerank_top_k=5)

    test_queries = ["布洛芬的用法用量", "感冒怎么办", "高血压的注意事项"]
    for q in test_queries:
        print(f"\n查询: {q}")
        results = retriever.invoke(q)
        print(f"  返回 {len(results)} 个文档:")
        for i, doc in enumerate(results, 1):
            preview = doc.page_content[:80].replace("\n", " ")
            parent_id = doc.metadata.get("parent_id", "N/A")
            print(f"  [{i}] (parent_id={parent_id[:20]}...) {preview}...")

    print("\n" + "=" * 60)
    print("重建完成！")
    print("=" * 60)


if __name__ == "__main__":
    rebuild()
