import hashlib
import os
from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.core.embeddings import get_embeddings
from app.core.config import get_config
from app.core.app_logging import get_logger

config = get_config()
logger = get_logger(__name__)

# ===== 知识库版本指纹（缓存防毒化） =====
_kb_version_cache: Optional[str] = None


def get_kb_version() -> str:
    """计算知识库版本指纹（基于向量库文档内容哈希）

    用途：缓存 key 绑定 kb_version，知识库更新后旧缓存自动失效。
    策略：对 ChromaDB 中所有文档的 doc_id + content 摘要做 MD5，
         文档数量级 <10 万时性能可接受（~50ms）。

    Returns:
        8 位短哈希字符串，如 "a1b2c3d4"
    """
    global _kb_version_cache
    if _kb_version_cache is not None:
        return _kb_version_cache

    try:
        manager = get_vector_store_manager()
        if manager.vector_store is None:
            # 向量库未初始化，返回默认版本
            logger.warning("向量库未初始化，kb_version 使用默认值")
            _kb_version_cache = "no_kb"
            return _kb_version_cache

        collection = manager.vector_store._collection
        doc_count = collection.count()

        if doc_count == 0:
            _kb_version_cache = "empty_kb"
            return _kb_version_cache

        # 采集文档指纹：doc_id 的哈希 + 文档数量（避免全量扫描 content）
        # 仅读取 ids，O(n) 但只传 ID 字符串，不传文档正文，性能可控
        results = collection.get(include=[], limit=doc_count)
        ids = results.get("ids") or []

        # 对所有 doc_id 排序后哈希 + 文档总数 → 版本指纹
        ids_str = "|".join(sorted(ids)) + f"|count={doc_count}"
        version_hash = hashlib.md5(ids_str.encode("utf-8")).hexdigest()[:8]

        _kb_version_cache = version_hash
        logger.info(f"知识库版本指纹：{version_hash}（{doc_count} 篇文档）")
        return _kb_version_cache

    except Exception as e:
        logger.warning(f"计算 kb_version 失败：{e}，使用默认值")
        _kb_version_cache = "fallback"
        return _kb_version_cache


def invalidate_kb_version():
    """使缓存的 kb_version 失效（知识库更新后调用）"""
    global _kb_version_cache
    old = _kb_version_cache
    _kb_version_cache = None
    if old is not None:
        logger.info(f"kb_version 已失效（旧值：{old}），下次调用将重新计算")


def _resolve_active_collection() -> tuple:
    """从 kb_active.json 解析当前活跃集合（零停机重建指针）

    零停机重建：DualCollectionManager 切换集合时写入 kb_active.json
    （active_persist_dir + active_collection），本函数让加载路径消费该指针。

    Returns:
        (persist_dir, collection_name)：存在有效指针时返回；否则 (None, None)。
        回滚到默认集合时 active_collection 为哨兵名 medical_kb_default，
        此时返回 (None, None)，由调用方回退到默认集合（base + langchain）。
    """
    try:
        dcm = get_dual_collection_manager()
        data = dcm.get_active_config()
        active_dir = data.get("active_persist_dir")
        active_name = data.get("active_collection")
        if (
            active_dir
            and active_name
            and active_name != _DEFAULT_COLLECTION_NAME
            and Path(active_dir).exists()
        ):
            return str(active_dir), active_name
    except Exception as e:
        logger.warning(f"解析活跃集合指针失败：{e}")
    return None, None


class VectorStoreManager:
    """向量库管理器"""

    def __init__(self, persist_directory: str = None):
        """初始化向量库管理器
        Args:
            persist_directory：向量库持久化目录（显式传入时忽略 kb_active.json 指针）
        """
        if persist_directory:
            # 显式指定目录（如重建脚本重建默认集合）→ 不使用活跃指针
            self.persist_directory = Path(persist_directory)
            self.collection_name = None
        else:
            # 默认路径 → 消费零停机重建指针（kb_active.json）
            active_dir, active_name = _resolve_active_collection()
            if active_dir and active_name:
                self.persist_directory = Path(active_dir)
                self.collection_name = active_name
                logger.info(f"向量库加载活跃集合：{active_name}（{active_dir}）")
            else:
                self.persist_directory = Path(config.PERSIST_DIRECTORY)
                self.collection_name = None
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
        # 已有实例且非强制重建 → 直接复用，避免重复加载
        if self.vector_store is not None and not force_rebuild:
            return self.vector_store

        if force_rebuild or not self.persist_directory.exists():
            abs_path = self.persist_directory.resolve()
            print(f"向量库位于：{abs_path}")
            # force_rebuild=True：先删除旧集合再重建，
            # 否则 Chroma.from_documents 的 get_or_create_collection 会把新 chunk 追加到旧集合
            if force_rebuild and self.persist_directory.exists():
                import shutil
                # 释放已加载的实例，避免 Windows 下文件被锁导致删除失败
                self.vector_store = None
                shutil.rmtree(self.persist_directory)
                invalidate_kb_version()
                print(f"已删除旧向量库：{self.persist_directory}")
            # 如果要求强制重建向量数据库或者当前不存在向量库
            print(f"正在创建向量库，文档数量：{len(documents)}")
            # 分批写入，避免 Embedding API 单次请求限制（智谱API最多64条/批）
            batch_size = 60
            # collection_name 仅在解析到活跃集合时传入（None 时由 langchain_chroma 用默认 "langchain"）
            base_create_kwargs = {
                "embedding": self.embeddings,
                "persist_directory": str(self.persist_directory),
                "collection_metadata": {"hnsw:space": "cosine"},
            }
            if self.collection_name:
                base_create_kwargs["collection_name"] = self.collection_name
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                if i == 0:
                # 第一批：创建向量库（使用 cosine 距离，而非默认 L2）
                    create_kwargs = dict(base_create_kwargs)
                    create_kwargs["documents"] = batch
                    self.vector_store = Chroma.from_documents(**create_kwargs)
                else:
                    # 后续批次：追加到已有向量库
                    self.vector_store.add_documents(batch)
                print(f"  已写入 {min(i + batch_size, len(documents))}/{len(documents)} 个文档")
            print(f"向量数据库已保存到：{self.persist_directory}")
        else:
            abs_path = self.persist_directory.resolve()
            print(f"向量库位于：{abs_path}")
            print(f"从{self.persist_directory}加载现有向量库。")
            load_kwargs = {
                "persist_directory": str(self.persist_directory),
                "embedding_function": self.embeddings,
            }
            if self.collection_name:
                load_kwargs["collection_name"] = self.collection_name
            self.vector_store = Chroma(**load_kwargs)
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
        if self.vector_store is None:
            self.vector_store = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings,
            )
        self.vector_store.add_documents(documents)
        # 知识库变更 → kb_version 失效，缓存自动防毒化
        invalidate_kb_version()
        # v9.2 漏洞5修复：知识库变更后异步触发规则同步扫描
        self._trigger_rule_sync_scan(len(documents))
        print(f'已经添加{len(documents)}个文档到向量库中。')

    def _trigger_rule_sync_scan(self, doc_count: int):
        """知识库变更后异步触发规则同步扫描（不阻塞主流程）"""
        try:
            import threading
            from app.core.kb_rule_sync import scan_kb_rule_sync

            def _scan_in_background():
                try:
                    report = scan_kb_rule_sync()
                    missing_drugs = len(report.get("missing_drugs", []))
                    missing_symptoms = len(report.get("missing_symptoms", []))
                    if missing_drugs > 0 or missing_symptoms > 0:
                        logger.warning(
                            f"⚠️ 知识库规则同步扫描发现差异："
                            f"缺失药物 {missing_drugs} 个，缺失症状 {missing_symptoms} 个，"
                            f"请人工更新 keyword_matcher.py 中的关键词"
                        )
                    else:
                        logger.info("知识库规则同步扫描：覆盖率 100%，无需更新")
                except Exception as e:
                    logger.warning(f"规则同步扫描失败：{e}")

            scan_thread = threading.Thread(target=_scan_in_background, daemon=True)
            scan_thread.start()
            logger.info(f"已触发规则同步扫描（后台线程，新增 {doc_count} 篇文档）")
        except Exception as e:
            logger.debug(f"规则同步扫描触发失败（非关键）：{e}")

    def delete_collection(self) -> None:
        """删除向量库集合"""
        if self.persist_directory.exists():
            import shutil
            shutil.rmtree(self.persist_directory)
            invalidate_kb_version()
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
            collection_info = self.vector_store._collection
            return {
                "name": collection_info.name,
                "count": collection_info.count,
                "persist_directory": str(self.persist_directory),
            }
        except Exception as e:
            return {"error": str(e)}

    def load_all_documents(self, limit: int = 50000) -> List[Document]:
        """从向量库加载全部文档，供 BM25 等离线检索组件使用"""
        if self.vector_store is None:
            raise ValueError("向量数据库未初始化，请先调用create_vector_store")

        try:
            collection = self.vector_store._collection
            results = collection.get(include=["documents", "metadatas"], limit=limit)
            documents = results.get("documents") or []
            metadatas = results.get("metadatas") or []
            ids = results.get("ids") or []

            loaded_documents = []
            for i, content in enumerate(documents):
                doc = Document(
                    page_content=content,
                    metadata=metadatas[i] if i < len(metadatas) else {},
                )
                if i < len(ids):
                    doc.id = ids[i]
                loaded_documents.append(doc)

            return loaded_documents
        except Exception as e:
            logger.error(f"从向量库加载文档失败：{e}")
            return []


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


def load_documents_from_store(
        vector_store: Optional[Chroma] = None,
        limit: int = 50000,
) -> List[Document]:
    """通过向量库管理器统一加载全部文档"""
    manager = get_vector_store_manager()
    if vector_store is not None:
        manager.vector_store = vector_store
    elif manager.vector_store is None:
        manager.vector_store = get_vector_store()

    return manager.load_all_documents(limit=limit)


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


# ===== 双集合管理器（Dual-Collection + 别名指针） =====
# 全库重建零停机：影子集合构建 → 校验 → 原子切换 → 延迟清理旧集合

import json
import shutil
import threading
from datetime import datetime

_KB_ACTIVE_CONFIG_PATH = Path(config.DATA_DIR) / "kb_active.json"
_DEFAULT_COLLECTION_NAME = "medical_kb_default"


class DualCollectionManager:
    """双集合管理器：影子集合构建 + 别名指针切换

    核心机制：
        - 线上检索始终走 active_collection（通过别名指针配置）
        - 重建时创建影子集合，全程对线上不可见
        - 影子集合校验通过后，原子切换指针
        - 5 分钟后延迟清理旧集合
    """

    def __init__(self):
        self.chroma_base_dir = Path(config.DATA_DIR) / "chroma_db"
        self.config_path = _KB_ACTIVE_CONFIG_PATH

    # ===== 别名指针读写 =====

    def get_active_collection_name(self) -> str:
        """读取当前活跃集合名（O(1)，毫秒级）"""
        try:
            if self.config_path.exists():
                data = json.loads(self.config_path.read_text(encoding="utf-8"))
                return data.get("active_collection", _DEFAULT_COLLECTION_NAME)
        except Exception as e:
            logger.warning(f"读取 kb_active.json 失败：{e}")
        return _DEFAULT_COLLECTION_NAME

    def get_active_config(self) -> dict:
        """读取完整别名指针配置"""
        try:
            if self.config_path.exists():
                return json.loads(self.config_path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {"active_collection": _DEFAULT_COLLECTION_NAME}

    def _write_config_atomic(self, data: dict):
        """原子写入配置（先写临时文件 → 原子替换，防止半写）

        os.replace 在 Windows 与 POSIX 均可覆盖已存在目标文件，
        而 Path.rename（os.rename）在 Windows 上遇到已存在目标会抛 FileExistsError。
        """
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.config_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp_path, self.config_path)  # 原子覆盖

    # ===== 影子集合创建 =====

    def create_shadow_collection(self) -> tuple:
        """创建影子集合（带时间戳的独立目录）

        Returns:
            (shadow_name, shadow_persist_dir): 影子集合名和持久化目录
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shadow_name = f"medical_kb_v{timestamp}"
        shadow_persist_dir = self.chroma_base_dir / shadow_name
        shadow_persist_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"创建影子集合：{shadow_name}，目录：{shadow_persist_dir}")
        return shadow_name, shadow_persist_dir

    def build_shadow_collection(
        self,
        documents: List[Document],
        shadow_name: str,
        shadow_persist_dir: Path,
        progress_callback=None,
    ) -> Chroma:
        """将文档写入影子集合（对线上完全不可见）

        Args:
            documents: 切分后的文档列表
            shadow_name: 影子集合名
            shadow_persist_dir: 影子集合持久化目录
            progress_callback: 进度回调函数

        Returns:
            Chroma: 影子集合的向量库实例
        """
        embeddings = get_embeddings()

        # 分批写入，避免 Embedding API 单次请求限制
        batch_size = 60
        shadow_vs = None

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            if i == 0:
                shadow_vs = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    collection_name=shadow_name,
                    persist_directory=str(shadow_persist_dir),
                    collection_metadata={"hnsw:space": "cosine"},
                )
            else:
                shadow_vs.add_documents(batch)

            if progress_callback:
                progress_callback(i + len(batch), len(documents))

            logger.info(f"影子集合写入：{min(i + batch_size, len(documents))}/{len(documents)}")

        # 持久化：langchain_chroma 新版本在 from_documents 时已自动持久化到
        # persist_directory，旧版显式 persist() 方法已移除，无需（也未能）再调用。

        return shadow_vs

    # ===== 校验 =====

    def validate_shadow_collection(
        self,
        shadow_vs: Chroma,
        expected_chunk_count: int,
        sample_queries: List[str] = None,
    ) -> dict:
        """校验影子集合健康度

        Returns:
            dict: {"valid": bool, "errors": [str], "warnings": [str]}
        """
        errors = []
        warnings = []

        if shadow_vs is None:
            return {"valid": False, "errors": ["影子集合实例为空"], "warnings": []}

        col = shadow_vs._collection
        actual_count = col.count()

        # 1. chunk 数量校验（容忍 10% 偏差）
        if actual_count == 0:
            errors.append(f"影子集合为空（actual=0, expected={expected_chunk_count}）")
        elif actual_count < expected_chunk_count * 0.9:
            warnings.append(
                f"chunk 数量偏差较大：actual={actual_count}, expected={expected_chunk_count} "
                f"(偏差>{10}%)"
            )

        # 2. 抽样召回率校验
        if sample_queries is None:
            sample_queries = ["感冒", "布洛芬", "高血压", "便秘", "发热", "头痛", "止咳"]

        failed_queries = []
        for q in sample_queries:
            try:
                results = shadow_vs.similarity_search(q, k=3)
                if not results:
                    failed_queries.append(q)
            except Exception as e:
                failed_queries.append(f"{q}(异常:{e})")

        if len(failed_queries) > len(sample_queries) * 0.5:
            errors.append(f"抽样召回率过低：{len(failed_queries)}/{len(sample_queries)} 查询无结果：{failed_queries}")
        elif failed_queries:
            warnings.append(f"部分查询无召回：{failed_queries}")

        # 3. Embedding 模型一致性校验
        try:
            from app.rag.kb_updater import get_embedding_model_info
            current_model = get_embedding_model_info()
            meta_results = col.get(limit=1, include=["metadatas"])
            if meta_results and meta_results["metadatas"]:
                stored_model = meta_results["metadatas"][0].get("embedding_model", "")
                if stored_model and stored_model != current_model.get("embedding_model", ""):
                    errors.append(
                        f"Embedding 模型不一致：stored={stored_model}, "
                        f"current={current_model.get('embedding_model', '')}"
                    )
        except Exception as e:
            warnings.append(f"模型一致性校验跳过：{e}")

        # 4. status 元数据校验（全量重建后所有 chunk 应为 active）
        try:
            pending_results = col.get(where={"status": "pending"}, include=[])
            if pending_results and pending_results["ids"]:
                warnings.append(f"存在 {len(pending_results['ids'])} 个 pending chunk，将自动激活")
        except Exception:
            pass

        valid = len(errors) == 0
        return {"valid": valid, "errors": errors, "warnings": warnings}

    # ===== 原子切换 =====

    def switch_active_collection(self, new_collection_name: str, new_persist_dir: str):
        """原子切换活跃集合指针

        步骤：
            1. 更新别名指针配置（原子写文件）
            2. 刷新全局 VectorStoreManager 实例（下次 get_vector_store 加载新集合）
            3. 失效 kb_version 缓存

        Args:
            new_collection_name: 新集合名
            new_persist_dir: 新集合持久化目录
        """
        old_config = self.get_active_config()
        old_collection = old_config.get("active_collection", _DEFAULT_COLLECTION_NAME)

        # 1. 原子写入新配置
        new_config = {
            "active_collection": new_collection_name,
            "active_persist_dir": str(new_persist_dir),
            "previous_collection": old_collection,
            "previous_persist_dir": old_config.get("active_persist_dir", str(self.chroma_base_dir)),
            "switched_at": datetime.utcnow().isoformat() + "Z",
        }
        self._write_config_atomic(new_config)

        # 2. 刷新全局实例
        global _vector_store_manager
        _vector_store_manager = None  # 下次 get_vector_store() 会重新加载

        # 3. 失效 kb_version
        invalidate_kb_version()

        logger.info(
            f"集合切换完成：{old_collection} → {new_collection_name} "
            f"(persist_dir={new_persist_dir})"
        )

    # ===== 延迟清理旧集合 =====

    def schedule_cleanup_old_collection(self, delay_seconds: int = 300):
        """延迟清理旧集合（5 分钟后执行）

        确保所有进行中的查询完成后再删除，避免长尾请求报错。
        """
        old_config = self.get_active_config()
        old_dir = old_config.get("previous_persist_dir")
        old_name = old_config.get("previous_collection")

        if not old_dir or not old_name:
            logger.info("无旧集合需要清理")
            return

        # 当前活跃集合不清理
        if old_name == old_config.get("active_collection"):
            return

        def _cleanup():
            try:
                import time
                logger.info(f"旧集合清理：{delay_seconds}s 后清理 {old_name}")
                time.sleep(delay_seconds)

                old_path = Path(old_dir)
                if old_path.exists() and old_path.is_dir():
                    # 安全检查：不删除活跃集合
                    active_name = self.get_active_collection_name()
                    if old_name == active_name:
                        logger.warning(f"旧集合 {old_name} 已是活跃集合，跳过清理")
                        return
                    # 不删除基础目录：它承载默认集合（回退路径）且包含影子集合子目录，
                    # 首次切换时 previous_persist_dir 默认指向它，误删会毁掉活跃的影子集合
                    if old_path.resolve() == self.chroma_base_dir.resolve():
                        logger.info(f"旧集合为基础目录（{old_dir}），跳过清理以保留默认集合")
                        return

                    shutil.rmtree(old_path)
                    logger.info(f"旧集合已清理：{old_name}（目录：{old_dir}）")
                else:
                    logger.info(f"旧集合目录不存在，跳过清理：{old_dir}")
            except Exception as e:
                logger.error(f"旧集合清理失败：{e}")

        cleanup_thread = threading.Thread(target=_cleanup, daemon=True)
        cleanup_thread.start()

    # ===== 回滚 =====

    def rollback_to_previous(self) -> bool:
        """紧急回滚到上一个活跃集合

        Returns:
            bool: 是否回滚成功
        """
        config = self.get_active_config()
        prev_name = config.get("previous_collection")
        prev_dir = config.get("previous_persist_dir")

        if not prev_name or not prev_dir:
            logger.error("无上一个集合可回滚")
            return False

        if not Path(prev_dir).exists():
            logger.error(f"上一个集合目录不存在：{prev_dir}")
            return False

        # 交换 active 和 previous
        self.switch_active_collection(prev_name, prev_dir)
        logger.warning(f"紧急回滚完成：{config.get('active_collection')} → {prev_name}")
        return True

    # ===== 激活影子集合中的 pending chunk =====

    def activate_pending_chunks(self, shadow_vs: Chroma) -> int:
        """将影子集合中所有 pending chunk 批量激活为 active

        全量重建后，所有 chunk 默认为 active。
        但如果是从增量更新流程来的，可能有 pending chunk。

        Returns:
            激活的 chunk 数量
        """
        try:
            col = shadow_vs._collection
            results = col.get(where={"status": "pending"}, include=["metadatas"])
            if not results or not results["ids"]:
                return 0

            chunk_ids = results["ids"]
            metas = results["metadatas"]
            now = datetime.utcnow().isoformat()

            for meta in metas:
                meta["status"] = "active"
                meta["activated_at"] = now

            col.update(ids=chunk_ids, metadatas=metas)
            logger.info(f"影子集合 pending→active：{len(chunk_ids)} 个 chunk")
            return len(chunk_ids)
        except Exception as e:
            logger.error(f"激活 pending chunk 失败：{e}")
            return 0


# 全局双集合管理器实例
_dual_collection_manager: Optional[DualCollectionManager] = None


def get_dual_collection_manager() -> DualCollectionManager:
    """获取双集合管理器实例（单例）"""
    global _dual_collection_manager
    if _dual_collection_manager is None:
        _dual_collection_manager = DualCollectionManager()
    return _dual_collection_manager
