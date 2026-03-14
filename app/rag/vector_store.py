from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.core.embeddings import get_embeddings
from app.core.config import get_config

config = get_config()


class VectorStoreManager:
    """向量库管理器"""

    def __init__(self, persist_directory: str = None):
        """初始化向量库管理器
        Args:
            persist_directory：向量库持久化目录
        """
        self.persist_directory = Path(persist_directory or config.PERSIST_DIRECTORY)
        self.embeddings = get_embeddings()
        self.vector_store = None

    def create_vector_store(self, documents: List[Document], force_rebuild: bool = False) -> Chroma:
        """创建或者加载向量数据库
        Args:
            documents: 要添加的文档列表
            force_rebuild: 是否强制重建数据库

        Returns:
            Chroma: 向量库实例
        """
        if force_rebuild or not self.persist_directory.exists():
            abs_path = self.persist_directory.resolve()
            print(f"向量库位于：{abs_path}")
            # 如果要求强制重建向量数据库或者当前不存在向量库
            print(f"正在创建向量库，文档数量：{len(documents)}")
            # print(f"正在创建向量库，文档数量：???")
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(self.persist_directory),
            )
            print(f"向量数据库已保存到：{self.persist_directory}")
        else:
            abs_path = self.persist_directory.resolve()
            print(f"向量库位于：{abs_path}")
            print(f"从{self.persist_directory}加载现有向量库。")
            self.vector_store = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings,
            )
        return self.vector_store

    def get_retriever(self, k: int = None, search_type: str = None) -> BaseRetriever:
        """获取检索器
        Args:
            k: 返回的最相关文档数量
            search_type: 检索类型（mmr/similarity）

        Returns:
            BaseRetriever: 检索器实例
        """
        # 检测向量数据库是否存在
        if self.vector_store is None:
            raise ValueError(f"向量数据库未初始化，请先调用create_vector_store")
        # 获取检索器函数，返回的检索器的检索方式是向量检索，是在EmbeddingConfig里配置好的参数。
        return self.vector_store.as_retriever(
            search_type=search_type or config.DEFAULT_SEARCH_TYPE,
            search_kwargs={"k": k or config.DEFAULT_K},
        )

    def add_documents(self, documents: List[Document]) -> None:
        """对向量数据库增加新文档
        Args:
            documents: 新增加的文档列表
        """
        # ??
        if self.vector_store is None:
            self.vector_store = Chroma(
                persist_directory=str(self.persist_directory),
                embedding=self.embeddings,
            )
        self.vector_store.add_documents(documents)
        print(f'已经添加{len(documents)}个文档到向量库中。')

    def delete_collection(self) -> None:
        """删除向量库集合"""
        if self.persist_directory.exists():
            import shutil
            shutil.rmtree(self.persist_directory)
            print(f'已删除向量库：{self.persist_directory}')
        else:
            print(f'向量库不存在：{self.persist_directory}')

    def get_collection_info(self) -> dict:
        """获取向量库集合信息
        Returns:
            dict: 集合信息字典
        """
        if self.vector_store is None:
            raise ValueError(f"向量数据库未初始化，请先调用create_vector_store")
        try:
            # ???
            collection_info = self.vector_store._collection
            return {
                "name": collection_info.name,
                "count": collection_info.count,
                "persist_directory": str(self.persist_directory),
            }
        except Exception as e:
            return {"error": str(e)}


# 全局向量库管理器实例
_vector_store_manager = None


def get_vector_store_manager(persist_directory: str = None) -> VectorStoreManager:
    """获取向量库管理器实例（单例模式？？？）
    Args:
        persist_directory: 向量库持久化目录
    Returns:
        VectorStoreManager: 向量库管理器实例
    """
    global _vector_store_manager
    if _vector_store_manager is None:
        _vector_store_manager = VectorStoreManager(persist_directory)
    return _vector_store_manager


def get_vector_store(
        documents: Optional[List[Document]] = None,
        persist_directory: str = None,
        force_rebuild: bool = False) -> Chroma:
    """获取或创建向量库（便携函数？？？）
    Args:
        documents: 要添加的文档
        persist_directory: 向量库持久化目录
        force_rebuild: 是否强制重建向量库

    Returns:
        Chroma: 向量库实例
    """
    manager = get_vector_store_manager(persist_directory=persist_directory)
    return manager.create_vector_store(documents=documents, force_rebuild=force_rebuild)


def get_retriever(
        vector_store: Optional[Chroma] = None,
        k: int = None,
        search_type: str = None,
) -> BaseRetriever:
    """获取检索器（便携函数）
    Args:
        vector_store: 向量库实例
        k: 召回的文档数量
        search_type: 检索类型

    Returns:
        BaseRetriever: 检索器实例
    """
    if vector_store is None:
        vector_store = get_vector_store()
    return vector_store.as_retriever(
        search_type=search_type or config.DEFAULT_SEARCH_TYPE,
        search_kwargs={"k": k or config.DEFAULT_K},
    )


def add_documents_to_store(
        documents: List[Document],
        persist_directory: str = None,
) -> None:
    """向现有向量库增加新文档
    Args:
        documents: <UNK>
        persist_directory: <UNK>
    """
    manager = get_vector_store_manager(persist_directory=persist_directory)
    manager.add_documents(documents)


def clear_vector_store(persist_directory: str = None) -> None:
    """清空向量库
    Args:
        persist_directory: 向量库持久化目录
    """
    manager = get_vector_store_manager(persist_directory=persist_directory)
    manager.delete_collection()
