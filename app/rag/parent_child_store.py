"""父子索引管理器

架构：
    Parent 文档：完整章节（如"布洛芬"完整条目，~400 字符），存入 DocStore
    Child 文档：从 Parent 按行切分的小块（~100-150 字符），存入向量库 + BM25

检索流程：
    Dense/BM25 检索 child → RRF 融合 → Reranker 重排 child → 取回 parent → 邻域扩展 → 送入 LLM

优势：
    1. 检索精度：child 小块与查询向量匹配更精准
    2. 上下文完整：parent 提供完整上下文，无需截断
    3. Reranker 加速：child 序列更短，ONNX 推理更快
    4. 邻域扩展：同一文档中相邻章节自动补全，解决跨章节信息缺失

v8.3 邻域扩展：
    问题：用户问"头痛怎么办"，答案分布在"危险信号"+"药物选择"+"非药物治疗"等多个 Parent 中，
         但 Reranker 只返回了"危险信号"1 个 Parent → LLM 编造药物名 → 幻觉
    方案：Parent 邻域扩展（Sibling Expansion），检索命中后自动拉取同一文档中的相邻章节
    实现：每个 Parent 写入 doc_id（所属原始文档）和 section_index（章节序号），
         检索后按 section_index ± window 拉取兄弟 Parent
"""
import os
import pickle
import uuid
from collections import defaultdict
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
        3. 邻域扩展：检索命中的 parent 自动拉取同一文档中的相邻章节
        4. 持久化：InMemoryStore 支持 pickle 序列化到磁盘
    """

    def __init__(self):
        self.store: InMemoryStore = InMemoryStore()
        self._initialized = False
        # 邻域扩展索引：doc_id → [(section_index, parent_id), ...]，按 section_index 排序
        self._doc_sections: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        # parent_id → (doc_id, section_index) 反向索引
        self._parent_location: Dict[str, Tuple[str, int]] = {}

    def build_index(self, parent_documents: List[Document], child_chunk_size: int = 150) -> List[Document]:
        """将 parent 文档切分为 child chunks，并将 parent 存入 store

        入库阶段自动为每个 Parent 写入 doc_id 和 section_index 元数据：
            - doc_id：所属原始文档标识（取自 metadata["source"]）
            - section_index：在该文档中的章节序号（从 0 开始，按 Markdown 标题切分顺序）

        Args:
            parent_documents: 父文档列表（通常是 Markdown 标题切分后的 chunks）
            child_chunk_size: child chunk 的最大字符数，默认 150

        Returns:
            child_chunks: 子文档列表，每个子文档的 metadata 中包含 parent_id
        """
        child_chunks: List[Document] = []

        # 清空旧数据（重建索引时确保干净状态）
        self.store = InMemoryStore()
        self._doc_sections = defaultdict(list)
        self._parent_location = {}

        # 按 source 分组，统计每个 source 下的 section 序号
        source_counters: Dict[str, int] = defaultdict(int)

        for parent_doc in parent_documents:
            # 生成 parent_id
            parent_id = f"p_{uuid.uuid4().hex[:12]}"

            # 写入 doc_id 和 section_index 元数据
            doc_id = parent_doc.metadata.get("source", "unknown")
            section_index = source_counters[doc_id]
            source_counters[doc_id] += 1

            # 更新 Parent 文档的元数据
            parent_doc.metadata["doc_id"] = doc_id
            parent_doc.metadata["section_index"] = section_index

            # 存入 parent 原文到 store
            self.store.mset([(parent_id, parent_doc)])

            # 构建邻域扩展索引
            self._doc_sections[doc_id].append((section_index, parent_id))
            self._parent_location[parent_id] = (doc_id, section_index)

            # 切分 parent 为 child chunks
            children = self._split_to_children(parent_doc, parent_id, child_chunk_size)
            child_chunks.extend(children)

        # 对每个 doc 的 sections 按 section_index 排序
        for doc_id in self._doc_sections:
            self._doc_sections[doc_id].sort(key=lambda x: x[0])

        self._initialized = True
        logger.info(
            f"父子索引构建完成：{len(parent_documents)} 个父文档 → {len(child_chunks)} 个子文档，"
            f"覆盖 {len(self._doc_sections)} 个原始文档"
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

    def expand_with_siblings(
        self,
        parents: List[Document],
        sibling_window: int = 1,
        max_total_chars: int = 2000,
    ) -> List[Document]:
        """邻域扩展：对已检索的 Parent，自动拉取同一文档中的相邻章节

        核心动机：
            用户问"头痛怎么办"，答案分布在"危险信号"+"药物选择"+"非药物治疗"等多个 Parent 中。
            Embedding 对"怎么办"和"药物选择"的语义映射不够（Dense 排名 #50+），
            但"危险信号"已被正确召回（同一文档的相邻章节）。
            邻域扩展利用文档结构关系补足 Embedding 的语义盲区。

        工作流程：
            1. 从已检索的 Parent 中读取 doc_id 和 section_index
            2. 对每个 Parent，拉取 section_index ± window 范围内的兄弟 Parent
            3. 合并去重，保持检索优先级（原始 Parent 在前，扩展 Parent 在后）
            4. 如果总字符数超过 max_total_chars，优先保留原始 Parent，截断扩展 Parent

        Args:
            parents: 已检索的 Parent 文档列表（来自 get_parents()）
            sibling_window: 邻域窗口大小，默认 1（取前后各 1 个章节）
            max_total_chars: 最大总字符数，防止注入过多内容撑爆 LLM 上下文

        Returns:
            扩展后的 Parent 文档列表（原始在前，扩展在后）
        """
        if not parents or sibling_window <= 0:
            return parents

        # 检查是否支持邻域扩展（需要 _doc_sections 索引）
        if not self._doc_sections:
            # 尝试从已有 parent 元数据中重建索引
            self._rebuild_section_index(parents)
            if not self._doc_sections:
                logger.info("邻域扩展：无章节索引，跳过（旧版 parent_store 无 doc_id/section_index）")
                return parents

        original_ids: set = set()
        for p in parents:
            pid = p.metadata.get("parent_id") or self._find_parent_id(p)
            if pid:
                original_ids.add(pid)

        expanded: List[Document] = list(parents)  # 原始 Parent 在前
        expanded_ids: set = set(original_ids)
        expanded_chars = sum(len(p.page_content) for p in expanded)

        sibling_count = 0
        for parent in parents:
            doc_id = parent.metadata.get("doc_id")
            section_index = parent.metadata.get("section_index")

            if doc_id is None or section_index is None:
                continue

            # 获取该文档的所有章节
            sections = self._doc_sections.get(doc_id, [])
            if not sections:
                continue

            # 查找当前 section 在列表中的位置
            current_pos = None
            for i, (sidx, _) in enumerate(sections):
                if sidx == section_index:
                    current_pos = i
                    break

            if current_pos is None:
                continue

            # 扩展窗口 [current_pos - window, current_pos + window]
            start = max(0, current_pos - sibling_window)
            end = min(len(sections), current_pos + sibling_window + 1)

            for i in range(start, end):
                _, sibling_id = sections[i]

                if sibling_id in expanded_ids:
                    continue  # 已包含（原始或已扩展）

                # 取回兄弟 Parent
                sibling_doc = self.store.mget([sibling_id])
                if sibling_doc and sibling_doc[0]:
                    sib = sibling_doc[0]
                    sib_chars = len(sib.page_content)

                    # 检查总字符数限制
                    if expanded_chars + sib_chars > max_total_chars:
                        logger.info(
                            f"邻域扩展：达到字符上限 {max_total_chars}，"
                            f"跳过后续扩展（已扩展 {sibling_count} 个兄弟章节）"
                        )
                        return expanded

                    # 标记为扩展文档（非原始检索结果）
                    sib.metadata["sibling_expanded"] = True
                    expanded.append(sib)
                    expanded_ids.add(sibling_id)
                    expanded_chars += sib_chars
                    sibling_count += 1

        if sibling_count > 0:
            logger.info(
                f"邻域扩展：{len(parents)} 个原始 Parent → {len(expanded)} 个（+{sibling_count} 个兄弟章节），"
                f"总字符数={expanded_chars}"
            )
        return expanded

    def _find_parent_id(self, parent: Document) -> Optional[str]:
        """根据 Document 对象反查 parent_id（用于旧版 parent_store 兼容）"""
        doc_id = parent.metadata.get("doc_id")
        section_index = parent.metadata.get("section_index")
        if doc_id is not None and section_index is not None:
            sections = self._doc_sections.get(doc_id, [])
            for sidx, pid in sections:
                if sidx == section_index:
                    return pid
        return None

    def _rebuild_section_index(self, parents: List[Document]) -> None:
        """从已有 Parent 文档的元数据中重建章节索引（兼容旧版 parent_store）"""
        for parent in parents:
            doc_id = parent.metadata.get("doc_id")
            section_index = parent.metadata.get("section_index")
            parent_id = parent.metadata.get("parent_id") or self._find_parent_id(parent)

            if doc_id is not None and section_index is not None:
                if parent_id:
                    self._doc_sections[doc_id].append((section_index, parent_id))
                    self._parent_location[parent_id] = (doc_id, section_index)
                else:
                    # 无 parent_id，用 doc_id+section_index 做临时 key
                    temp_id = f"p_{doc_id}_{section_index}"
                    self._doc_sections[doc_id].append((section_index, temp_id))
                    self._parent_location[temp_id] = (doc_id, section_index)

        # 排序
        for doc_id in self._doc_sections:
            self._doc_sections[doc_id].sort(key=lambda x: x[0])

    def save_to_disk(self, path: str = None) -> None:
        """持久化 parent store 到磁盘（含邻域扩展索引）"""
        path = path or PARENT_STORE_CACHE_PATH
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # InMemoryStore 内部是 dict: {key: value}
            store_data = dict(self.store.store) if hasattr(self.store, "store") else {}
            # 同时保存邻域扩展索引
            index_data = {
                "doc_sections": dict(self._doc_sections),
                "parent_location": dict(self._parent_location),
            }
            with open(path, "wb") as f:
                pickle.dump({"store": store_data, "index": index_data}, f)
            logger.info(f"Parent store 已缓存到：{path}（{len(store_data)} 条，{len(self._doc_sections)} 个文档索引）")
        except Exception as e:
            logger.warning(f"Parent store 缓存失败：{e}")

    def load_from_disk(self, path: str = None) -> bool:
        """从磁盘加载 parent store（含邻域扩展索引）

        Returns:
            True 表示加载成功，False 表示无缓存或加载失败
        """
        path = path or PARENT_STORE_CACHE_PATH
        if not os.path.exists(path):
            return False

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            # 兼容新旧两种格式
            if isinstance(data, dict) and "store" in data:
                # 新格式（v8.3+）：含 store + index
                store_data = data["store"]
                index_data = data.get("index", {})
                self._doc_sections = defaultdict(list, index_data.get("doc_sections", {}))
                self._parent_location = index_data.get("parent_location", {})
            else:
                # 旧格式：仅 store dict
                store_data = data
                self._doc_sections = defaultdict(list)
                self._parent_location = {}

            # 重建 InMemoryStore
            self.store = InMemoryStore()
            if store_data:
                items = list(store_data.items())
                self.store.mset(items)

                # 旧格式：尝试从 parent 元数据中重建索引
                if not self._doc_sections:
                    self._rebuild_index_from_store()

            self._initialized = True
            logger.info(f"Parent store 从磁盘加载成功：{path}（{len(store_data) if isinstance(store_data, dict) else 0} 条）")
            if self._doc_sections:
                logger.info(f"邻域扩展索引已加载：{len(self._doc_sections)} 个文档，{len(self._parent_location)} 个章节")
            else:
                logger.info("邻域扩展索引为空（旧版 parent_store），需重建索引以启用邻域扩展")
            return True
        except Exception as e:
            logger.warning(f"Parent store 加载失败：{e}")
            return False

    def _rebuild_index_from_store(self) -> None:
        """从 InMemoryStore 中的所有 Parent 文档重建章节索引"""
        if not hasattr(self.store, "store"):
            return

        for parent_id, parent_doc in self.store.store.items():
            if isinstance(parent_doc, Document):
                doc_id = parent_doc.metadata.get("doc_id")
                section_index = parent_doc.metadata.get("section_index")
                if doc_id is not None and section_index is not None:
                    self._doc_sections[doc_id].append((section_index, parent_id))
                    self._parent_location[parent_id] = (doc_id, section_index)

        # 排序
        for doc_id in self._doc_sections:
            self._doc_sections[doc_id].sort(key=lambda x: x[0])

        if self._doc_sections:
            logger.info(f"从 Parent 元数据重建章节索引：{len(self._doc_sections)} 个文档")

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
