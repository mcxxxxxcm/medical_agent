"""知识库更新管理模块（v9.9）

参考日志1最佳实践，解决：
    1. 增量更新时重复 Embedding（content_hash 去重）
    2. 文档修改后旧向量残留（软删除 + 旧版本清理）
    3. Embedding 模型混用（模型版本校验）
    4. 删除后仍被召回（软删除 is_deleted 标记）
    5. 切分策略变更后历史数据不重建（chunk_strategy 版本化）
    6. 更新操作无审计记录（审计日志）

元数据体系（每个 chunk 必备）：
    - doc_id: 文档唯一标识（source 文件名）
    - chunk_id: chunk 唯一标识（doc_id + content_hash 前缀）
    - content_hash: SHA-256 内容哈希（增量去重核心）
    - version_id: 文档版本号（每次修改 +1）
    - is_deleted: 软删除标记（True=不参与检索）
    - embedding_model: Embedding 模型名
    - embedding_dimension: 向量维度
    - chunk_strategy: 切分策略标识
    - updated_at: 最后更新时间
"""
import hashlib
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.documents import Document

from app.core.app_logging import get_logger
from app.core.config import get_config

logger = get_logger(__name__)

config = get_config()

# ===== 常量 =====
# 审计日志数据库路径
_AUDIT_DB_PATH = Path(config.DATA_DIR) / "kb_audit.db"

# 切分策略版本标识（变更时需更新，触发全量重建检测）
CHUNK_STRATEGY_VERSION = "v9.9_row_level_table_aware"


# ===== content_hash 计算 =====

def compute_content_hash(content: str) -> str:
    """计算 chunk 内容的 SHA-256 哈希（前 16 位，用于增量去重）

    生产推荐 SHA-256（碰撞风险极低），MD5 有碰撞风险。
    取前 16 位（64 bit）足够在知识库规模（< 1M chunks）下去重。
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


# ===== Embedding 模型版本 =====

_embedding_info_cache: Dict[str, Any] = {"ts": 0.0, "data": None}
_EMBEDDING_INFO_TTL = 60  # 秒；避免高频轮询（如 /api/admin/kb/status）每次触发外部 embedding API


def get_embedding_model_info() -> Dict[str, Any]:
    """获取当前 Embedding 模型信息（用于元数据标注和一致性校验）

    带 TTL 缓存：embed_query("test") 会调用外部 embedding API（慢且不稳定，
    日志中曾出现 40s 的超时重试），不应在每次轮询时都触发。
    """
    now = time.time()
    if _embedding_info_cache["data"] is not None and now - _embedding_info_cache["ts"] < _EMBEDDING_INFO_TTL:
        return _embedding_info_cache["data"]

    info = _compute_embedding_model_info()
    _embedding_info_cache["ts"] = now
    _embedding_info_cache["data"] = info
    return info


def _compute_embedding_model_info() -> Dict[str, Any]:
    """实际计算 Embedding 模型信息（无缓存）"""
    try:
        from app.core.embeddings import get_embeddings
        embeddings = get_embeddings()
        model_name = getattr(embeddings, 'model', '') or getattr(embeddings, 'model_name', '') or "unknown"
        # 尝试获取维度
        dimension = 0
        try:
            test_vec = embeddings.embed_query("test")
            dimension = len(test_vec)
        except Exception:
            pass
        return {
            "embedding_model": model_name,
            "embedding_dimension": dimension,
        }
    except Exception:
        return {"embedding_model": "unknown", "embedding_dimension": 0}


def check_embedding_consistency(existing_metadata: Dict) -> bool:
    """校验已有 chunk 的 Embedding 模型与当前模型是否一致

    Returns:
        True = 一致，False = 不一致（需重建）
    """
    current = get_embedding_model_info()
    existing_model = existing_metadata.get("embedding_model", "")
    existing_dim = existing_metadata.get("embedding_dimension", 0)

    if not existing_model or existing_dim == 0:
        return True  # 旧数据无模型信息，放行

    if existing_model != current["embedding_model"]:
        logger.warning(
            f"Embedding 模型不一致：索引={existing_model}，当前={current['embedding_model']}，需全量重建"
        )
        return False

    if existing_dim != current["embedding_dimension"]:
        logger.warning(
            f"Embedding 维度不一致：索引={existing_dim}，当前={current['embedding_dimension']}，需全量重建"
        )
        return False

    return True


# ===== 元数据增强 =====

def enrich_chunk_metadata(
    chunks: List[Document],
    source: str,
    version_id: int = 1,
    status: str = "pending",
) -> List[Document]:
    """为 chunk 列表增强元数据（知识库更新必备字段）

    Args:
        chunks: 切分后的 chunk 列表
        source: 文档来源（文件名）
        version_id: 文档版本号
        status: 切片状态（"pending"=不可检索, "active"=可检索, "deprecated"=已废弃）

    Returns:
        增强后的 chunk 列表
    """
    model_info = get_embedding_model_info()
    now = datetime.utcnow().isoformat()

    for i, chunk in enumerate(chunks):
        # content_hash（增量去重核心）
        content_hash = compute_content_hash(chunk.page_content)

        # chunk_id（确定性，基于 doc_id + content_hash）
        chunk_id = f"{source}_{content_hash}"

        chunk.metadata.update({
            "doc_id": source,
            "chunk_id": chunk_id,
            "content_hash": content_hash,
            "version_id": version_id,
            "status": status,               # pending → active → deprecated
            "is_deleted": False,
            "embedding_model": model_info["embedding_model"],
            "embedding_dimension": model_info["embedding_dimension"],
            "chunk_strategy": CHUNK_STRATEGY_VERSION,
            "updated_at": now,
        })

    return chunks


# ===== 增量去重 =====

def filter_unchanged_chunks(
    new_chunks: List[Document],
    existing_hashes: set,
) -> Tuple[List[Document], List[Document]]:
    """过滤内容未变化的 chunk，跳过重复 Embedding

    Args:
        new_chunks: 新切分的 chunk 列表
        existing_hashes: 已有 chunk 的 content_hash 集合

    Returns:
        (changed_chunks, unchanged_chunks)
    """
    changed = []
    unchanged = []

    for chunk in new_chunks:
        content_hash = chunk.metadata.get("content_hash", compute_content_hash(chunk.page_content))
        if content_hash in existing_hashes:
            unchanged.append(chunk)
        else:
            changed.append(chunk)

    if unchanged:
        logger.info(f"增量去重：{len(new_chunks)} 个 chunk 中 {len(unchanged)} 个未变化，跳过 Embedding")

    return changed, unchanged


def get_existing_content_hashes(vector_store) -> set:
    """从 ChromaDB 获取已有 chunk 的 content_hash 集合

    用于增量更新时判断哪些 chunk 内容未变化，可以跳过 Embedding。
    """
    hashes = set()
    try:
        if hasattr(vector_store, '_collection') and vector_store._collection:
            results = vector_store._collection.get(
                include=["metadatas"],
                where={"is_deleted": False},  # 只查未删除的
            )
            if results and results["metadatas"]:
                for meta in results["metadatas"]:
                    h = meta.get("content_hash", "")
                    if h:
                        hashes.add(h)
    except Exception as e:
        logger.warning(f"获取已有 content_hash 失败：{e}")
    return hashes


# ===== 双缓冲：pending → active → deprecated 状态机 =====

def activate_document_version(vector_store, source: str, version_id: int) -> int:
    """校验并激活新版本 chunk：status=pending → active

    双缓冲核心步骤：
        1. 查找该文档新版本的所有 pending chunk
        2. 轻量校验：检查 chunk 数量 > 0、content_hash 无重复
        3. 批量更新 status=active（原子操作）
        4. 新版本激活后，再将旧版本的 chunk 标记为 deprecated

    Returns:
        激活的 chunk 数量（0 表示校验失败）
    """
    count = 0
    try:
        if not hasattr(vector_store, '_collection') or not vector_store._collection:
            return 0

        col = vector_store._collection

        # 1. 查找新版本的 pending chunk
        results = col.get(
            where={"source": source, "version_id": version_id, "status": "pending"},
            include=["metadatas", "documents"],
        )
        if not results or not results["ids"]:
            logger.warning(f"激活失败：{source} v{version_id} 无 pending chunk")
            return 0

        chunk_ids = results["ids"]
        metas = results["metadatas"]
        count = len(chunk_ids)

        # 2. 轻量校验
        # 2a. chunk 数量 > 0（已在上面确认）
        # 2b. content_hash 无重复
        hashes = [m.get("content_hash", "") for m in metas]
        if len(hashes) != len(set(hashes)):
            logger.error(f"激活失败：{source} v{version_id} 存在重复 content_hash，数据异常")
            return 0

        # 2c. 抽样查询验证（检查新 chunk 是否可被向量检索命中）
        # 取第一个 chunk 的 embedding 做相似度查询，确保向量已就绪
        try:
            sample_hash = hashes[0]
            sample_results = col.get(
                where={"content_hash": sample_hash, "status": "pending"},
                include=["embeddings"],
            )
            if not sample_results or not sample_results.get("embeddings") or not sample_results["embeddings"][0]:
                logger.warning(f"激活校验：{source} v{version_id} 向量未就绪，等待下次激活")
                return 0
        except Exception as e:
            logger.warning(f"激活校验向量就绪检查跳过：{e}")

        # 3. 批量更新 status=active（原子操作）
        now = datetime.utcnow().isoformat()
        updated_metas = []
        for meta in metas:
            meta["status"] = "active"
            meta["activated_at"] = now
            updated_metas.append(meta)
        col.update(ids=chunk_ids, metadatas=updated_metas)

        logger.info(f"激活新版本：{source} v{version_id}，{count} 个 chunk status=pending→active")

        # 4. 将旧版本标记为 deprecated（非删除，仍可被紧急回滚）
        deprecate_old_versions(vector_store, source, version_id)

    except Exception as e:
        logger.error(f"激活失败：{source} v{version_id} - {e}")
        return 0

    return count


def deprecate_old_versions(vector_store, source: str, current_version_id: int) -> int:
    """将旧版本的 active chunk 标记为 deprecated

    与 soft_delete 不同：
        - deprecated：旧版本但可能被紧急回滚引用，5 分钟后物理删除
        - is_deleted=True：管理员主动删除，30 天后物理删除

    Returns:
        废弃的 chunk 数量
    """
    count = 0
    try:
        if not hasattr(vector_store, '_collection') or not vector_store._collection:
            return 0

        col = vector_store._collection
        # 查找该文档所有旧版本的 active chunk
        results = col.get(
            where={"source": source, "status": "active"},
            include=["metadatas"],
        )
        if not results or not results["ids"]:
            return 0

        # 筛选 version_id < current_version_id 的 chunk
        old_ids = []
        old_metas = []
        now = datetime.utcnow().isoformat()
        for i, meta in enumerate(results["metadatas"]):
            if meta.get("version_id", 0) < current_version_id:
                old_ids.append(results["ids"][i])
                meta["status"] = "deprecated"
                meta["deprecated_at"] = now
                old_metas.append(meta)

        if old_ids:
            col.update(ids=old_ids, metadatas=old_metas)
            count = len(old_ids)
            logger.info(f"旧版本已废弃：{source}，{count} 个 chunk status=active→deprecated（v<{current_version_id}）")

    except Exception as e:
        logger.error(f"废弃旧版本失败：{source} - {e}")

    return count


def cleanup_deprecated_chunks(vector_store, delay_seconds: int = 300) -> int:
    """清理超过延迟时间的 deprecated chunk（物理删除）

    Args:
        delay_seconds: 延迟时间（默认 5 分钟 = 300 秒）
        处理切换瞬间可能存在的长尾请求

    Returns:
        物理删除的 chunk 数量
    """
    count = 0
    cutoff = datetime.utcnow().timestamp() - delay_seconds
    try:
        if not hasattr(vector_store, '_collection') or not vector_store._collection:
            return 0

        col = vector_store._collection
        results = col.get(
            where={"status": "deprecated"},
            include=["metadatas"],
        )
        if not results or not results["ids"]:
            return 0

        stale_ids = []
        for i, meta in enumerate(results["metadatas"]):
            deprecated_at = meta.get("deprecated_at", "")
            if deprecated_at:
                try:
                    ts = datetime.fromisoformat(deprecated_at).timestamp()
                    if ts < cutoff:
                        stale_ids.append(results["ids"][i])
                except Exception:
                    pass

        if stale_ids:
            col.delete(ids=stale_ids)
            count = len(stale_ids)
            logger.info(f"清理 deprecated chunk：{count} 个（>{delay_seconds}s）")

    except Exception as e:
        logger.error(f"清理 deprecated 失败：{e}")

    return count


# ===== 软删除 =====

def soft_delete_document(vector_store, source: str) -> int:
    """软删除文档：将 is_deleted 标记为 True（而非物理删除）

    优势（参考日志1）：
        1. 保留变更历史，支持误删恢复
        2. 同一文档再次上传时可判断是新文档还是重新上传
        3. 审计和回滚的基础

    Returns:
        软删除的 chunk 数量
    """
    count = 0
    try:
        if hasattr(vector_store, '_collection') and vector_store._collection:
            # 查找该文档的所有未删除 chunk
            results = vector_store._collection.get(
                where={"source": source, "is_deleted": False},
            )
            if results and results["ids"]:
                # 更新 is_deleted = True
                metas = []
                for meta in results["metadatas"]:
                    meta["is_deleted"] = True
                    meta["deleted_at"] = datetime.utcnow().isoformat()
                    metas.append(meta)
                vector_store._collection.update(
                    ids=results["ids"],
                    metadatas=metas,
                )
                count = len(results["ids"])
                logger.info(f"软删除文档：{source}，{count} 个 chunk 标记为 is_deleted=True")
    except Exception as e:
        logger.error(f"软删除失败：{source} - {e}")
    return count


def restore_deleted_document(vector_store, source: str) -> int:
    """恢复软删除的文档（误删恢复）

    Returns:
        恢复的 chunk 数量
    """
    count = 0
    try:
        if hasattr(vector_store, '_collection') and vector_store._collection:
            results = vector_store._collection.get(
                where={"source": source, "is_deleted": True},
            )
            if results and results["ids"]:
                metas = []
                for meta in results["metadatas"]:
                    meta["is_deleted"] = False
                    meta.pop("deleted_at", None)
                    meta["updated_at"] = datetime.utcnow().isoformat()
                    metas.append(meta)
                vector_store._collection.update(
                    ids=results["ids"],
                    metadatas=metas,
                )
                count = len(results["ids"])
                logger.info(f"恢复软删除文档：{source}，{count} 个 chunk")
    except Exception as e:
        logger.error(f"恢复软删除失败：{source} - {e}")
    return count


def physical_cleanup_stale_deletes(vector_store, days: int = 30) -> int:
    """物理删除超过 N 天的软删除记录（延迟物理删除）

    参考：软删除 + 延迟物理删除 + 删除审计日志
    """
    count = 0
    cutoff = datetime.utcnow().timestamp() - days * 86400
    try:
        if hasattr(vector_store, '_collection') and vector_store._collection:
            results = vector_store._collection.get(
                where={"is_deleted": True},
                include=["metadatas"],
            )
            if results and results["ids"]:
                stale_ids = []
                for i, meta in enumerate(results["metadatas"]):
                    deleted_at = meta.get("deleted_at", "")
                    if deleted_at:
                        try:
                            ts = datetime.fromisoformat(deleted_at).timestamp()
                            if ts < cutoff:
                                stale_ids.append(results["ids"][i])
                        except Exception:
                            pass
                if stale_ids:
                    vector_store._collection.delete(ids=stale_ids)
                    count = len(stale_ids)
                    logger.info(f"物理清理过期软删除：{count} 个 chunk（>{days}天）")
    except Exception as e:
        logger.error(f"物理清理失败：{e}")
    return count


# ===== 版本管理 =====

def get_document_version(vector_store, source: str) -> int:
    """获取文档当前版本号（用于修改时递增）"""
    try:
        if hasattr(vector_store, '_collection') and vector_store._collection:
            results = vector_store._collection.get(
                where={"source": source, "is_deleted": False},
                include=["metadatas"],
            )
            if results and results["metadatas"]:
                versions = [m.get("version_id", 0) for m in results["metadatas"]]
                return max(versions) if versions else 0
    except Exception:
        pass
    return 0


# ===== 审计日志 =====

def _ensure_audit_db():
    """确保审计日志数据库存在"""
    _AUDIT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_AUDIT_DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kb_audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            change_type TEXT NOT NULL,  -- 'add' / 'modify' / 'delete' / 'restore' / 'rebuild'
            chunk_count INTEGER DEFAULT 0,
            version_id INTEGER DEFAULT 0,
            operator TEXT DEFAULT 'auto',  -- 'auto' / 'manual'
            result TEXT DEFAULT 'success',  -- 'success' / 'failed' / 'partial'
            error_message TEXT DEFAULT '',
            content_hash_sample TEXT DEFAULT '',  -- 前 3 个 chunk 的 hash（采样）
            embedding_model TEXT DEFAULT '',
            chunk_strategy TEXT DEFAULT '',
            elapsed_ms INTEGER DEFAULT 0,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def log_kb_audit(
    doc_id: str,
    change_type: str,
    chunk_count: int = 0,
    version_id: int = 0,
    operator: str = "auto",
    result: str = "success",
    error_message: str = "",
    content_hash_sample: str = "",
    elapsed_ms: int = 0,
):
    """记录知识库更新审计日志

    审计字段（参考日志1）：doc_id, change_type, timestamp, operator, result, error_message
    """
    try:
        _ensure_audit_db()
        conn = sqlite3.connect(str(_AUDIT_DB_PATH))
        conn.execute("""
            INSERT INTO kb_audit_log
            (doc_id, change_type, chunk_count, version_id, operator, result,
             error_message, content_hash_sample, embedding_model, chunk_strategy,
             elapsed_ms, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            doc_id, change_type, chunk_count, version_id, operator, result,
            error_message, content_hash_sample,
            get_embedding_model_info().get("embedding_model", ""),
            CHUNK_STRATEGY_VERSION,
            elapsed_ms,
            datetime.utcnow().isoformat(),
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"审计日志写入失败：{e}")


def get_kb_audit_log(limit: int = 50) -> List[Dict]:
    """查询知识库审计日志"""
    try:
        _ensure_audit_db()
        conn = sqlite3.connect(str(_AUDIT_DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM kb_audit_log ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


# ===== 变更检测（轮询兜底） =====

def detect_stale_documents(docs_dir: Path, vector_store, max_staleness_hours: int = 24) -> List[Dict]:
    """检测陈旧文档：源系统已更新但索引未更新

    策略：对比磁盘文件的修改时间 vs 向量库中 updated_at
    超过 max_staleness_hours 的文档视为陈旧，需要重新索引。

    Returns:
        陈旧文档列表 [{"source": ..., "disk_mtime": ..., "index_updated_at": ..., "staleness_hours": ...}]
    """
    stale = []
    try:
        # 获取向量库中所有文档的最后更新时间
        index_times = {}
        if hasattr(vector_store, '_collection') and vector_store._collection:
            results = vector_store._collection.get(
                where={"is_deleted": False},
                include=["metadatas"],
            )
            if results and results["metadatas"]:
                for meta in results["metadatas"]:
                    source = meta.get("source", "")
                    updated_at = meta.get("updated_at", "")
                    if source and updated_at:
                        # 保留最新的 updated_at
                        if source not in index_times or updated_at > index_times[source]:
                            index_times[source] = updated_at

        # 对比磁盘文件修改时间
        if docs_dir.exists():
            for f in docs_dir.iterdir():
                if not f.is_file():
                    continue
                source = f.name
                disk_mtime = datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                index_updated = index_times.get(source, "")

                if not index_updated:
                    # 向量库中无此文档→新文档，不算陈旧
                    continue

                # 计算陈旧度
                try:
                    disk_ts = datetime.fromisoformat(disk_mtime)
                    index_ts = datetime.fromisoformat(index_updated)
                    staleness_hours = (disk_ts - index_ts).total_seconds() / 3600

                    if staleness_hours > max_staleness_hours:
                        stale.append({
                            "source": source,
                            "disk_mtime": disk_mtime,
                            "index_updated_at": index_updated,
                            "staleness_hours": round(staleness_hours, 1),
                        })
                except Exception:
                    pass

    except Exception as e:
        logger.warning(f"变更检测失败：{e}")

    return stale


# ===== 一致性校验（reconciliation） =====

def run_reconciliation(docs_dir: Path, vector_store) -> Dict[str, Any]:
    """一致性校验：对比源系统、向量库、BM25 缓存，修复差异

    参考：定期 reconciliation 对比源系统、元数据库、向量库、全文索引，
    修复漏删、漏写和乱序事件。

    Returns:
        校验报告 {"missing_in_index": [...], "missing_on_disk": [...], "soft_deleted_count": N, ...}
    """
    report = {
        "checked_at": datetime.utcnow().isoformat(),
        "missing_in_index": [],   # 磁盘有但索引无
        "missing_on_disk": [],    # 索引有但磁盘无
        "soft_deleted_count": 0,
        "embedding_mismatch": [],
        "chunk_strategy_mismatch": [],
    }

    try:
        from app.rag.loader import LOADERS

        # 1. 磁盘文件列表
        disk_files = set()
        if docs_dir.exists():
            for f in docs_dir.iterdir():
                if f.is_file() and f.suffix.lower() in LOADERS:
                    disk_files.add(f.name)

        # 2. 索引中文档列表
        index_sources = set()
        soft_deleted = 0
        embedding_mismatches = []
        strategy_mismatches = []

        if hasattr(vector_store, '_collection') and vector_store._collection:
            results = vector_store._collection.get(include=["metadatas"])
            if results and results["metadatas"]:
                current_model = get_embedding_model_info()
                for meta in results["metadatas"]:
                    source = meta.get("source", "")
                    if meta.get("is_deleted"):
                        soft_deleted += 1
                        continue
                    if source:
                        index_sources.add(source)

                    # Embedding 模型校验
                    if meta.get("embedding_model") and meta["embedding_model"] != current_model.get("embedding_model"):
                        if source not in [m["source"] for m in embedding_mismatches]:
                            embedding_mismatches.append({
                                "source": source,
                                "index_model": meta["embedding_model"],
                                "current_model": current_model.get("embedding_model"),
                            })

                    # 切分策略校验
                    if meta.get("chunk_strategy") and meta["chunk_strategy"] != CHUNK_STRATEGY_VERSION:
                        if source not in [m["source"] for m in strategy_mismatches]:
                            strategy_mismatches.append({
                                "source": source,
                                "index_strategy": meta["chunk_strategy"],
                                "current_strategy": CHUNK_STRATEGY_VERSION,
                            })

        # 3. 差异计算
        report["missing_in_index"] = list(disk_files - index_sources)  # 新文档未入库
        report["missing_on_disk"] = list(index_sources - disk_files)   # 磁盘已删除但索引残留
        report["soft_deleted_count"] = soft_deleted
        report["embedding_mismatch"] = embedding_mismatches
        report["chunk_strategy_mismatch"] = strategy_mismatches

    except Exception as e:
        report["error"] = str(e)

    return report
