"""长期记忆管理模块 (优化版)
修复点：
    1. 移除不必要的 async/await，改为同步操作 (匹配 PostgresStore 默认行为)
    2. 增加 store.setup() 初始化调用
    3. 优化 search 结果的排序逻辑
    4. 增加异常处理
    5. 新增 symptom_events / medication_events 命名空间（Append-Only 事件流）
    6. v9.2 漏洞3修复：新增 prune 机制，防止无界膨胀
"""
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
import uuid
import logging

from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.base import Item

# 假设你的日志模块
logger = logging.getLogger(__name__)

_long_term_memory: Optional["LongTermMemoryManager"] = None
_store_context = None

# v9.2 漏洞3修复：各命名空间的默认保留天数和上限
_NAMESPACE_RETENTION = {
    "symptom_events": 90,      # 症状事件保留 90 天
    "medication_events": 90,   # 用药事件保留 90 天
    "bad_cases": 180,          # Bad Case 保留 180 天（回归测试需要）
    "query_history": 30,       # 查询历史保留 30 天
}
_NAMESPACE_MAX_ITEMS = {
    "symptom_events": 500,     # 每用户最多 500 条症状事件
    "medication_events": 300,  # 每用户最多 300 条用药事件
    "bad_cases": 500,          # 每用户最多 500 条 Bad Case
    "query_history": 200,      # 每用户最多 200 条查询历史
}


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
        items = self.store.search(("symptom_history", user_id))

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
        items = self.store.search(("query_history", user_id))

        records = []
        for item in items:
            if item.value and "timestamp" in item.value:
                records.append(item.value)
            # v9.2 漏洞3修复：提前截断
            if len(records) >= limit * 2:
                break

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

    # ===== v9.2 漏洞3修复：Prune 机制 =====

    def prune_namespace(
            self,
            namespace_name: str,
            user_id: str,
            max_age_days: Optional[int] = None,
            max_items: Optional[int] = None,
    ) -> int:
        """清理指定命名空间的过期/超量记录

        Args:
            namespace_name: 命名空间名（如 "symptom_events"）
            user_id: 用户ID
            max_age_days: 最大保留天数（默认从 _NAMESPACE_RETENTION 读取）
            max_items: 最大条目数（默认从 _NAMESPACE_MAX_ITEMS 读取）

        Returns:
            删除的条目数
        """
        max_age_days = max_age_days or _NAMESPACE_RETENTION.get(namespace_name, 90)
        max_items = max_items or _NAMESPACE_MAX_ITEMS.get(namespace_name, 500)

        cutoff_date = (datetime.now() - timedelta(days=max_age_days)).isoformat()
        namespace = (namespace_name, user_id)

        try:
            items = self.store.search(namespace)
            if not items:
                return 0

            # 收集需要删除的 key
            keys_to_delete = []
            timestamp_field = "onset_iso" if namespace_name == "symptom_events" else "created_at"

            # 1. 删除超过保留天数的过期记录
            for item in items:
                if not item.value:
                    keys_to_delete.append(item.key)
                    continue
                ts = item.value.get(timestamp_field) or item.value.get("cached_at") or ""
                if ts and ts < cutoff_date:
                    keys_to_delete.append(item.key)

            # 2. 如果仍超量，按时间排序删除最早的
            remaining = [item for item in items if item.key not in keys_to_delete]
            if len(remaining) > max_items:
                # 按时间排序（升序 = 最早的在前）
                remaining.sort(
                    key=lambda x: x.value.get(timestamp_field, "") if x.value else "",
                )
                excess = len(remaining) - max_items
                for item in remaining[:excess]:
                    keys_to_delete.append(item.key)

            # 执行删除
            deleted = 0
            for key in keys_to_delete:
                try:
                    self.store.delete(namespace, key=key)
                    deleted += 1
                except Exception as e:
                    logger.warning(f"删除 {namespace_name}/{key} 失败：{e}")

            if deleted > 0:
                logger.info(f"命名空间 {namespace_name}/{user_id} 清理完成：删除 {deleted} 条（保留天数={max_age_days}，上限={max_items}）")
            return deleted

        except Exception as e:
            logger.error(f"命名空间 {namespace_name}/{user_id} 清理失败：{e}")
            return 0

    def prune_all_namespaces(
            self,
            user_id: str,
            namespaces: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """批量清理所有可清理命名空间

        Args:
            user_id: 用户ID
            namespaces: 要清理的命名空间列表（默认清理全部四个）

        Returns:
            各命名空间的删除条目数
        """
        if namespaces is None:
            namespaces = ["symptom_events", "medication_events", "bad_cases", "query_history"]

        results = {}
        for ns in namespaces:
            results[ns] = self.prune_namespace(ns, user_id)

        total = sum(results.values())
        if total > 0:
            logger.info(f"用户 {user_id} 全量清理完成：{results}（共 {total} 条）")
        return results

    def prune_all_users(
            self,
            user_ids: Optional[List[str]] = None,
            namespaces: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, int]]:
        """清理多个用户的过期数据（管理员接口）

        Args:
            user_ids: 用户ID列表（None=遍历所有用户）
            namespaces: 要清理的命名空间列表

        Returns:
            {user_id: {namespace: deleted_count}}
        """
        if user_ids is None:
            # 尝试从各命名空间中收集所有 user_id
            user_ids = set()
            for ns in (namespaces or ["symptom_events", "medication_events", "bad_cases", "query_history"]):
                try:
                    # PostgresStore.list_namespaces 可能不支持，这里用 search 近似
                    # 遍历已知用户（从 query_history 中提取）
                    items = self.store.search(("query_history",))
                    for item in items:
                        if item.value and item.value.get("user_id"):
                            user_ids.add(item.value["user_id"])
                except Exception:
                    pass
            user_ids = list(user_ids)

        results = {}
        for uid in user_ids:
            results[uid] = self.prune_all_namespaces(uid, namespaces)
        return results

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

    # ===== 症状事件（Append-Only 事件流） =====

    def append_symptom_event(
            self,
            user_id: str,
            symptom_name: str,
            onset_iso: str,
            onset_ts: int,
            precision: str = "default",
            source_query: str = "",
    ) -> str:
        """追加一条症状报告事件到长期记忆

        Args:
            user_id: 用户ID
            symptom_name: 症状名称（如"头痛"）
            onset_iso: 症状首发绝对时间（ISO格式）
            onset_ts: 症状首发 Unix 时间戳
            precision: 时间精度（exact/approximate/vague/default）
            source_query: 触发此事件的原始用户问题

        Returns:
            事件ID
        """
        event_id = f"se_{uuid.uuid4().hex[:12]}"
        event = {
            "event_type": "symptom_report",
            "symptom": symptom_name,
            "onset_iso": onset_iso,
            "onset_ts": onset_ts,
            "precision": precision,
            "source_query": source_query,
            "created_at": datetime.now().isoformat(),
        }
        self.store.put(
            namespace=("symptom_events", user_id),
            key=event_id,
            value=event,
        )
        logger.info(f"症状事件已写入L1：user={user_id}, symptom={symptom_name}, onset={onset_iso}")
        return event_id

    def get_symptom_events(
            self,
            user_id: str,
            symptom_name: Optional[str] = None,
            limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """获取用户的症状事件列表

        Args:
            user_id: 用户ID
            symptom_name: 可选，按症状名过滤
            limit: 最大返回条数

        Returns:
            症状事件列表，按 onset_ts 倒序
        """
        items = self.store.search(("symptom_events", user_id))
        records = []
        for item in items:
            if not item.value or "onset_ts" not in item.value:
                continue
            if symptom_name and item.value.get("symptom") != symptom_name:
                continue
            records.append(item.value)
            # v9.2 漏洞3修复：提前截断，避免全量加载后排序
            # 最多读取 limit * 3 条（留余量给排序过滤），减少内存占用
            if len(records) >= limit * 3:
                break

        records.sort(key=lambda x: x.get("onset_ts", 0), reverse=True)
        return records[:limit]

    def get_latest_symptom_onset(
            self,
            user_id: str,
            symptom_name: str,
    ) -> Optional[Dict[str, Any]]:
        """获取某个症状的最早首发记录（用于计算持续时间）

        Args:
            user_id: 用户ID
            symptom_name: 症状名称

        Returns:
            最早的首发记录，如 {"iso": "...", "ts": 123, "precision": "exact"}
            未找到返回 None
        """
        events = self.get_symptom_events(user_id, symptom_name=symptom_name, limit=100)
        if not events:
            return None
        # 取最早的首发记录（onset_ts 最小的）
        earliest = min(events, key=lambda x: x.get("onset_ts", float("inf")))
        return {
            "iso": earliest.get("onset_iso", ""),
            "ts": earliest.get("onset_ts"),
            "precision": earliest.get("precision", "default"),
        }

    def get_all_symptom_onsets(
            self,
            user_id: str,
    ) -> Dict[str, Dict[str, Any]]:
        """获取用户所有症状的最早首发记录（用于新会话填充 L2）

        Returns:
            {"头痛": {"iso": "...", "ts": 123, "precision": "exact"}, ...}
        """
        events = self.get_symptom_events(user_id, limit=200)
        # 按症状名分组，取每组中 onset_ts 最小的
        symptom_map: Dict[str, Dict[str, Any]] = {}
        for event in events:
            name = event.get("symptom", "")
            if not name:
                continue
            existing = symptom_map.get(name)
            current_ts = event.get("onset_ts", float("inf"))
            if existing is None or current_ts < existing.get("ts", float("inf")):
                symptom_map[name] = {
                    "iso": event.get("onset_iso", ""),
                    "ts": current_ts,
                    "precision": event.get("precision", "default"),
                }
        return symptom_map

    # ===== Bad Case 采集（自包含性检测回归测试） =====

    def append_bad_case(
            self,
            case_type: str,
            original_query: str,
            rewritten_query: str,
            final_question: str,
            history_summary: str = "",
            route: str = "",
            top_doc_score: float = 0.0,
            grade_result: str = "",
            answer_preview: str = "",
            user_id: str = "system",
            thread_id: str = "",
            metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """追加一条 bad case 记录

        case_type 分类：
            - "rewrite_missed_anaphora": 指代词未重写（如"还有其他可以吃的吗"）
            - "rewrite_lost_entity": 重写丢失核心实体（守卫回退）
            - "rewrite_same_as_original": 重写结果与原问题一致（LLM未理解上下文）
            - "low_score_no_clarify": 检索低分但未触发澄清（幻觉出口）
            - "hallucination_suspected": 答案含检索文档中不存在的药物/实体（忠实度问题）
            - "retrieval_miss": 检索返回零文档（索引/查询词匹配问题）
            - "route_misclassification": 含症状词的问题被路由到 direct_answer
            - "user_negative_feedback": 用户显式标记答案不准确（前端反馈）
            - "manual_flag": 人工标注的 bad case

        Args:
            case_type: bad case 类型
            original_query: 用户原始问题
            rewritten_query: 重写后的检索查询
            final_question: 重写后的完整问题
            history_summary: 对话历史摘要
            route: 路由结果
            top_doc_score: 最高文档评分
            grade_result: 文档评分结果
            answer_preview: AI 回答预览（前200字）
            user_id: 用户ID
            thread_id: 会话线程ID
            metadata: 额外元数据

        Returns:
            case_id
        """
        case_id = f"bc_{uuid.uuid4().hex[:12]}"
        case = {
            "case_id": case_id,
            "case_type": case_type,
            "original_query": original_query,
            "rewritten_query": rewritten_query,
            "final_question": final_question,
            "history_summary": history_summary[:500] if history_summary else "",
            "route": route,
            "top_doc_score": top_doc_score,
            "grade_result": grade_result,
            "answer_preview": answer_preview[:200] if answer_preview else "",
            "thread_id": thread_id,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "reviewed": False,
            "expected_rewrite": "",  # 人工补填：期望的重写结果
            "is_self_contained": None,  # 人工标注：原问题是否自包含
        }
        self.store.put(
            namespace=("bad_cases", user_id),
            key=case_id,
            value=case,
        )
        logger.info(f"Bad case 已记录：type={case_type}, query={original_query[:30]}")
        return case_id

    def get_bad_cases(
            self,
            user_id: str = "system",
            case_type: Optional[str] = None,
            reviewed: Optional[bool] = None,
            limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """查询 bad case 列表

        Args:
            user_id: 用户ID（默认 "system" 查全局）
            case_type: 按类型过滤
            reviewed: 按审核状态过滤
            limit: 最大返回条数

        Returns:
            bad case 列表，按创建时间倒序
        """
        items = self.store.search(("bad_cases", user_id))
        records = []
        for item in items:
            if not item.value or "case_id" not in item.value:
                continue
            if case_type and item.value.get("case_type") != case_type:
                continue
            if reviewed is not None and item.value.get("reviewed") != reviewed:
                continue
            records.append(item.value)
            # v9.2 漏洞3修复：提前截断
            if len(records) >= limit * 2:
                break

        records.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return records[:limit]

    def update_bad_case_review(
            self,
            user_id: str,
            case_id: str,
            expected_rewrite: str = "",
            is_self_contained: Optional[bool] = None,
            reviewed: bool = True,
    ) -> None:
        """人工审核更新 bad case（补填期望重写结果和自包含标注）

        Args:
            user_id: 用户ID
            case_id: case ID
            expected_rewrite: 期望的重写结果
            is_self_contained: 原问题是否自包含
            reviewed: 是否已审核
        """
        item = self.store.get(
            namespace=("bad_cases", user_id),
            key=case_id,
        )
        if not item or not item.value:
            logger.warning(f"Bad case 不存在：{case_id}")
            return

        case = item.value
        if expected_rewrite:
            case["expected_rewrite"] = expected_rewrite
        if is_self_contained is not None:
            case["is_self_contained"] = is_self_contained
        case["reviewed"] = reviewed

        self.store.put(
            namespace=("bad_cases", user_id),
            key=case_id,
            value=case,
        )
        logger.info(f"Bad case 审核更新：{case_id}")

    # ===== 用药事件（Append-Only 事件流） =====

    def append_medication_event(
            self,
            user_id: str,
            drug: str,
            dosage: Optional[str] = None,
            effect: Optional[str] = None,
            source_query: str = "",
    ) -> str:
        """追加一条用药记录事件到长期记忆"""
        event_id = f"me_{uuid.uuid4().hex[:12]}"
        event = {
            "event_type": "medication_record",
            "drug": drug,
            "dosage": dosage,
            "effect": effect,
            "source_query": source_query,
            "created_at": datetime.now().isoformat(),
        }
        self.store.put(
            namespace=("medication_events", user_id),
            key=event_id,
            value=event,
        )
        logger.info(f"用药事件已写入L1：user={user_id}, drug={drug}")
        return event_id

    def get_medication_events(
            self,
            user_id: str,
            drug: Optional[str] = None,
            limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """获取用户的用药事件列表"""
        items = self.store.search(("medication_events", user_id))
        records = []
        for item in items:
            if not item.value or "drug" not in item.value:
                continue
            if drug and item.value.get("drug") != drug:
                continue
            records.append(item.value)
            # v9.2 漏洞3修复：提前截断
            if len(records) >= limit * 2:
                break

        records.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return records[:limit]

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
