"""重建向量数据库脚本
功能描述：
    加载医疗文档，切分文档，并重建向量数据库
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.rag import load_medical_documents, split_documents, get_vector_store

if __name__ == "__main__":
    print("开始加载医疗文档...")
    docs = load_medical_documents()
    print(f"加载了 {len(docs)} 个文档")

    print("\n开始切分文档...")
    chunks = split_documents(docs)
    print(f"切分了 {len(chunks)} 个文档块")

    print("\n开始创建向量数据库...")
    vector_store = get_vector_store(chunks, force_rebuild=True)
    print("向量数据库创建完成！")

    print("\n测试检索...")
    retriever = vector_store.as_retriever(k=3)
    test_docs = retriever.invoke("感冒")
    print(f"检索到 {len(test_docs)} 个相关文档")
