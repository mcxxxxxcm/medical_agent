import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.rag.loader import load_medical_documents, split_documents
from app.rag.vector_store import (
    get_vector_store,
    get_retriever,
    clear_vector_store,
    add_documents_to_store
)


def test_vector_store():
    print("=" * 60)
    print("开始测试向量库")
    print("=" * 60)

    # 1. 加载文档
    print("\n[1/6] 加载医疗文档...")
    documents = load_medical_documents()
    print(f"✓ 加载了 {len(documents)} 个文档")

    # 2. 切分文档
    print("\n[2/6] 切分文档...")
    chunks = split_documents(documents, chunk_size=500, chunk_overlap=50)
    print(f"✓ 切分后得到 {len(chunks)} 个文档块")

    # 3. 清除旧向量库
    print("\n[3/6] 清除旧向量库...")
    clear_vector_store()

    # 4. 创建向量库
    print("\n[4/6] 创建向量库...")
    vector_store = get_vector_store(
        documents=chunks,
        force_rebuild=True
    )
    print("✓ 向量库创建成功")

    # 5. 获取检索器
    print("\n[5/6] 获取检索器...")
    retriever = get_retriever(vector_store=vector_store, k=3)
    print("✓ 检索器获取成功")

    # 6. 测试检索
    print("\n[6/6] 测试检索...")
    test_queries = [
        "糖尿病有哪些常见症状？",
        "高血压患者应该注意什么？"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n--- 查询 {i} ---")
        print(f"问题: {query}")

        docs = retriever.invoke(query)
        print(f"检索到 {len(docs)} 个相关文档")

        for j, doc in enumerate(docs, 1):
            print(f"\n  文档 {j}:")
            print(f"  来源: {doc.metadata.get('source', '未知')}")
            print(f"  内容: {doc.page_content[:100]}...")

    print("\n" + "=" * 60)
    print("✓ 向量库测试完成")
    print("=" * 60)


def test_load_vector_store():
    print("\n[1/3] 正在加载现有的向量库...")
    vector_store = get_vector_store()

    print("\n[2/3] 获取检索器...")
    retriever = get_retriever(vector_store=vector_store, k=3)

    print("\n[3/3] 测试检索...")
    docs = retriever.invoke("感冒的症状？")
    print(f'检索到{len(docs)}个文档')
    for j, doc in enumerate(docs, 1):
        print(f"\n  文档 {j}:")
        print(f"  来源: {doc.metadata.get('source', '未知')}")
        print(f"  内容: {doc.page_content[:100]}...")


if __name__ == "__main__":
    test_vector_store()
