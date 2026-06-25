from app.rag.loader import (
    load_medical_documents,
    split_documents,
    print_docs,
)

from app.rag.vector_store import (
    get_vector_store,
    get_retriever,
    add_documents_to_store,
    clear_vector_store,
    VectorStoreManager,
    get_vector_store_manager
)

from app.rag.qa_chain import (
    QAChain,
    get_qa_chain
)

# ✅ 新增：导入混合检索器
from app.rag.hybrid_retriever import (
    HybridRetriever,
    get_hybrid_retriever
)

# ✅ 新增：导入父子索引管理器
from app.rag.parent_child_store import (
    ParentChildManager,
    get_parent_child_manager,
)

__all__ = [
    # Loader
    "load_medical_documents",
    "split_documents",
    "print_docs",

    # Vector Store
    "get_vector_store",
    "get_retriever",
    "add_documents_to_store",
    "clear_vector_store",
    "VectorStoreManager",
    "get_vector_store_manager",

    # QA Chain
    "QAChain",
    "get_qa_chain",

    # Hybrid Retriever
    "HybridRetriever",
    "get_hybrid_retriever",

    # Parent-Child Index
    "ParentChildManager",
    "get_parent_child_manager",
]
