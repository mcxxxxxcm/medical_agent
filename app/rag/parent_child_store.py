"""父子索引管理器

架构：
    Parent 文档：完整章节（如"布洛芬"完整条目，~400 字符），存入 DocStore
    Child 文档：从 Parent 按行切分的小块（~100-150 字符），存入向量库 + BM25

检索流程：
    Dense/BM25 检索 child → RRF 融合 → Reranker 重排 child → 取回 parent → 送入 LLM

优势：
    1. 检索精度：child 小块与查询向量匹配更精准
    2. 上下文完整：parent 提供完整上下文，无需截断
    3. Reranker 加速：child 序列更短，ONNX 推理更快
"""
import os
import pickle
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore

from app.core.app_logging import get_logger
from app.core.config import get_config

logger = get_logger(__name__)
config = get_config()

# 父子索引持久化路径
PARENT_STORE_CACHE_PATH = str(
    Path(getattr(config, "BM25_CACHE_PATH", "data/bm25_index.pkl")).parent / "parent_store.pkl"
)


class ParentChildManager:
    """父子索引管理器

    职责：
        1. 入库阶段：将 parent 文档切分为 child chunks，parent 存入 InMemoryStore
        2. 检索阶段：根据 child 的 parent_id 取回完整 parent 文档
        3. 持久化：InMemoryStore 支持 pickle 序列化到磁盘
    """

    def __init__(self):
        self.store: InMemoryStore = InMemoryStore()
        self._initialized = False

    def build_index(self, parent_documents: List[Document], child_chunk_size: int = 150) -> List[Document]:
        """将 parent 文档切分为 child chunks，并将 parent 存入 store

        Args:
            parent_documents: 父文档列表（通常是 Markdown 标题切分后的 chunks）
            child_chunk_size: child chunk 的最大字符数，默认 150

        Returns:
            child_chunks: 子文档列表，每个子文档的 metadata 中包含 parent_id
        """
        child_chunks: List[Document] = []

        for parent_doc in parent_documents:
            # 生成 parent_id
            parent_id = f"p_{uuid.uuid4().hex[:12]}"

            # 存入 parent 原文到 store
            self.store.mset([(parent_id, parent_doc)])

            # 切分 parent 为 child chunks
            children = self._split_to_children(parent_doc, parent_id, child_chunk_size)
            child_chunks.extend(children)

        self._initialized = True
        logger.info(
            f"父子索引构建完成：{len(parent_documents)} 个父文档 → {len(child_chunks)} 个子文档"
        )
        return child_chunks

    def _split_to_children(
        self, parent_doc: Document, parent_id: str, max_chars: int = 150
    ) -> List[Document]:
        """将父文档按行分组切分为子文档

        策略：
            1. 提取标题行（### 布洛芬）作为每个 child 的上下文前缀
            2. 按行分组，每组不超过 max_chars
            3. 每个 child 的 metadata 中写入 parent_id
        """
        lines = parent_doc.page_content.split("\n")

        # 提取最近的标题行作为上下文
        header = ""
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                header = stripped

        children: List[Document] = []
        current_group: List[str] = []
        current_len = 0

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # 计算带 header 前缀的长度
            line_len = len(stripped)
            prefix_len = len(header) + 1 if header else 0

            # 当前组加上这行会超限 → 先保存当前组
            if current_group and current_len + line_len + prefix_len > max_chars:
                child_content = self._build_child_content(header, current_group)
                child = Document(
                    page_content=child_content,
                    metadata={**parent_doc.metadata, "parent_id": parent_id},
                )
                children.append(child)
                current_group = []
                current_len = 0

            current_group.append(stripped)
            current_len += line_len

        # 最后一组
        if current_group:
            child_content = self._build_child_content(header, current_group)
            child = Document(
                page_content=child_content,
                metadata={**parent_doc.metadata, "parent_id": parent_id},
            )
            children.append(child)

        return children

    def _build_child_content(self, header: str, lines: List[str]) -> str:
        """构建 child 内容：header 前缀 + 内容行"""
        if header:
            return f"{header}\n" + "\n".join(lines)
        return "\n".join(lines)

    def get_parents(self, child_docs: List[Document]) -> List[Document]:
        """根据 child 文档列表，取回去重后的 parent 文档

        Args:
            child_docs: Reranker 重排后的 child 文档列表

        Returns:
            去重后的 parent 文档列表，保持被检索到的优先级顺序
        """
        if not child_docs:
            return []

        seen_parent_ids: set = set()
        parents: List[Document] = []

        for child in child_docs:
            parent_id = child.metadata.get("parent_id")
            if not parent_id or parent_id in seen_parent_ids:
                continue

            parent_doc = self.store.mget([parent_id])
            if parent_doc and parent_doc[0]:
                # 从 child 的 rerank_score 传递到 parent
                parent = parent_doc[0]
                rerank_score = child.metadata.get("rerank_score")
                if rerank_score is not None:
                    parent.metadata["rerank_score"] = rerank_score
                parents.append(parent)
                seen_parent_ids.add(parent_id)

        logger.info(
            f"父子映射：{len(child_docs)} 个 child → {len(parents)} 个 parent（去重 {len(child_docs) - len(parents)} 个）"
        )
        return parents

    def save_to_disk(self, path: str = None) -> None:
        """持久化 parent store 到磁盘"""
        path = path or PARENT_STORE_CACHE_PATH
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # InMemoryStore 内部是 dict: {key: value}
            store_data = dict(self.store.store) if hasattr(self.store, "store") else {}
            with open(path, "wb") as f:
                pickle.dump(store_data, f)
            logger.info(f"Parent store 已缓存到：{path}（{len(store_data)} 条）")
        except Exception as e:
            logger.warning(f"Parent store 缓存失败：{e}")

    def load_from_disk(self, path: str = None) -> bool:
        """从磁盘加载 parent store

        Returns:
            True 表示加载成功，False 表示无缓存或加载失败
        """
        path = path or PARENT_STORE_CACHE_PATH
        if not os.path.exists(path):
            return False

        try:
            with open(path, "rb") as f:
                store_data: Dict = pickle.load(f)
            # 重建 InMemoryStore
            self.store = InMemoryStore()
            if store_data:
                items = list(store_data.items())
                self.store.mset(items)
            self._initialized = True
            logger.info(f"Parent store 从磁盘加载成功：{path}（{len(store_data)} 条）")
            return True
        except Exception as e:
            logger.warning(f"Parent store 加载失败：{e}")
            return False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def parent_count(self) -> int:
        """返回 parent 文档数量"""
        try:
            return len(self.store.store) if hasattr(self.store, "store") else 0
        except Exception:
            return 0


# 全局单例
_parent_child_manager: Optional[ParentChildManager] = None


def get_parent_child_manager() -> ParentChildManager:
    """获取 ParentChildManager 单例"""
    global _parent_child_manager
    if _parent_child_manager is None:
        _parent_child_manager = ParentChildManager()
        # 尝试从磁盘加载
        _parent_child_manager.load_from_disk()
    return _parent_child_manager


def reset_parent_child_manager() -> None:
    """重置全局单例（用于测试或重建索引）"""
    global _parent_child_manager
    _parent_child_manager = None
