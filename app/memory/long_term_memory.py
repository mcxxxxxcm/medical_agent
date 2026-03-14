"""长期记忆管理模块 (优化版)
修复点：
    1. 移除不必要的 async/await，改为同步操作 (匹配 PostgresStore 默认行为)
    2. 增加 store.setup() 初始化调用
    3. 优化 search 结果的排序逻辑
    4. 增加异常处理
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

from langgraph.store.postgres import PostgresStore
from langgraph.store.base import Item

# 假设你的日志模块
logger = logging.getLogger(__name__)

_long_term_memory: Optional["LongTermMemoryManager"] = None
_store_context = None


class LongTermMemoryManager:
    """长期记忆管理器 (同步版本)"""

    def __init__(self, store: PostgresStore):
        self.store = store
        # ✅ 移除这里的 setup()，在外部调用
        logger.info("长期记忆管理器已初始化")


    def get_symptom_history(
            self,
            user_id: str,
            limit: int = 10
    ) -> List[Dict[str, Any]]:
        """获取症状历史记录 (按时间倒序)"""
        items = self.store.search(
            namespace_prefix=("symptom_history", user_id)
        )

        # 🔴 关键：应用层排序
        # 提取 value 并过滤掉没有 timestamp 的脏数据
        records = []
        for item in items:
            if item.value and "timestamp" in item.value:
                records.append(item.value)

        # 按时间戳倒序排序
        records.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        return records[:limit]

    def save_query_record(
            self,
            user_id: str,
            query_id: str,
            query_data: Dict[str, Any]
    ) -> None:
        """保存查询记录"""
        query_data["timestamp"] = datetime.now().isoformat()

        self.store.put(
            namespace=("query_history", user_id),
            key=query_id,
            value=query_data
        )
        logger.info(f"已保存查询记录：user_id={user_id}, query_id={query_id}")

    def get_query_history(
            self,
            user_id: str,
            limit: int = 10
    ) -> List[Dict[str, Any]]:
        """获取查询历史记录 (按时间倒序)"""
        items = self.store.search(
            namespace_prefix=("query_history", user_id)
        )

        records = []
        for item in items:
            if item.value and "timestamp" in item.value:
                records.append(item.value)

        records.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return records[:limit]

    def save_document_cache(
            self,
            doc_id: str,
            content: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """缓存检索到的大文档"""
        self.store.put(
            namespace=("document_cache",),  # 全局缓存，不按用户隔离
            key=doc_id,
            value={"content": content, "metadata": metadata or {}, "cached_at": datetime.now().isoformat()}
        )
        logger.info(f"已缓存文档：doc_id={doc_id}")

    def get_document_cache(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """获取缓存的文档"""
        item = self.store.get(
            namespace=("document_cache",),
            key=doc_id
        )
        return item.value if item else None

    def close(self):
        """关闭连接池 (可选，通常在服务停止时调用)"""
        # PostgresStore 内部通常依赖 sqlalchemy 引擎，可能需要显式关闭
        if hasattr(self.store, 'engine'):
            self.store.engine.dispose()
        logger.info("长期记忆存储器连接已关闭")

    def save_user_profile(
            self,
            user_id: str,
            profile: Dict[str, Any]
    )-> None:
        """保存用户档案，覆盖更新"""
        profile["updated_at"] = datetime.now().isoformat()
        self.store.put(
            namespace=("user_profile", user_id),
            key="profile",
            value=profile
        )
        logger.info(f"已保存用户档案：user_id={user_id}")

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取用户档案"""
        item = self.store.get(
            namespace=("user_profile", user_id),
            key="profile"
        )
        return item.value if item else None

    def update_user_profile(
            self,
            user_id: str,
            updates: Dict[str, Any]
    )-> None:
        """更新用户档案（覆盖）"""
        existing=self.get_user_profile(user_id) or {}
        existing.update(updates)
        self.save_user_profile(user_id, existing)

# 🔴 修改为同步单例获取，或在 FastAPI 中使用 lifespan 管理
def get_long_term_memory() -> LongTermMemoryManager:
    """获取长期记忆管理器实例（单例）"""
    global _long_term_memory, _store_context
    if _long_term_memory is None:
        # 假设你的配置获取方式是同步的
        from app.core.config import get_config
        config = get_config()
        db_uri = config.DATABASE_URL

        _store_context = PostgresStore.from_conn_string(db_uri)
        store = _store_context.__enter__()

        store.setup()
        logger.info("长期记忆存储器表结构检查/创建完成")

        _long_term_memory = LongTermMemoryManager(store)
        logger.info("长期记忆管理器单例已创建")

    return _long_term_memory


def reset_long_term_memory() -> None:
    """重置单例 (用于测试)"""
    global _long_term_memory, _store_context
    if _store_context:
        try:
            _store_context.__exit__(None, None, None)
        except Exception:
            pass
    _long_term_memory = None
    _store_context = None
    logger.info("长期记忆管理器已重置")
