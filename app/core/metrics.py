"""请求级可观测性模块（SQLite Metrics）

轻量级结构化指标采集，无需部署 Prometheus/Grafana。
包含三张表：
    1. node_metrics: 节点级耗时（P50/P95/P99 分析）
    2. token_usage: LLM Token 用量（成本估算 + 阈值告警）
    3. feedback: 用户反馈闭环（👍/👎 → 差评分析 → 黄金测试集候选）

用法：
    from app.core.metrics import get_metrics_collector

    # 记录节点耗时
    collector = get_metrics_collector()
    collector.record_node(request_id="req_123", node_name="知识检索",
                          duration_ms=1143.62, route_type="symptom",
                          cache_hit=False)

    # 记录 Token 用量
    collector.record_token_usage(request_id="req_123", model="glm-4-flash",
                                 prompt_tokens=800, completion_tokens=300)

    # 记录用户反馈
    collector.record_feedback(request_id="req_123", rating="down",
                              reason="answer_inaccurate", note="漏了禁忌症")

    # 查询统计数据
    stats = collector.get_node_stats(hours=24)
    token_stats = collector.get_token_stats(hours=24)
    feedback_stats = collector.get_feedback_stats(hours=24)

存储位置：data/metrics/metrics.db（约 1MB/万条记录）
自动清理：保留最近 30 天数据
"""
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.app_logging import get_logger

logger = get_logger(__name__)

_DEFAULT_DB_PATH = "data/metrics/metrics.db"
_RETENTION_DAYS = 30

# ===== 成本估算（元/千 tokens，基于智谱官方定价 2025） =====
_MODEL_PRICING = {
    "glm-4-flash": {"prompt": 0.0001, "completion": 0.0001},     # 极低
    "glm-4-plus": {"prompt": 0.05, "completion": 0.05},          # 中等
    "glm-4": {"prompt": 0.1, "completion": 0.1},                 # 标准
    "glm-4v-plus": {"prompt": 0.01, "completion": 0.01},         # 多模态
    "qwen2.5:1.5b": {"prompt": 0.0, "completion": 0.0},          # 本地免费
    "embedding-3": {"prompt": 0.0005, "completion": 0.0},        # Embedding
    "default": {"prompt": 0.0001, "completion": 0.0001},         # 降级
}


class MetricsCollector:
    """SQLite 的请求级指标采集器

    线程安全：使用 threading.Lock 保护写入
    轻量级：每条记录 ~100 bytes，30 天约 30MB
    """

    def __init__(self, db_path: str = _DEFAULT_DB_PATH):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                # 表1：节点耗时
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS node_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        request_id TEXT NOT NULL,
                        thread_id TEXT NOT NULL DEFAULT '',
                        node_name TEXT NOT NULL,
                        duration_ms REAL NOT NULL,
                        route_type TEXT NOT NULL DEFAULT '',
                        cache_hit INTEGER NOT NULL DEFAULT 0,
                        timestamp REAL NOT NULL,
                        metadata TEXT NOT NULL DEFAULT '{}'
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_node_timestamp ON node_metrics(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_node_name ON node_metrics(node_name)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_request_id ON node_metrics(request_id)")

                # 表2：Token 用量
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS token_usage (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        request_id TEXT NOT NULL,
                        thread_id TEXT NOT NULL DEFAULT '',
                        node_name TEXT NOT NULL DEFAULT '',
                        model TEXT NOT NULL DEFAULT '',
                        prompt_tokens INTEGER NOT NULL DEFAULT 0,
                        completion_tokens INTEGER NOT NULL DEFAULT 0,
                        total_tokens INTEGER NOT NULL DEFAULT 0,
                        estimated_cost REAL NOT NULL DEFAULT 0.0,
                        timestamp REAL NOT NULL
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_token_timestamp ON token_usage(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_token_model ON token_usage(model)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_token_request_id ON token_usage(request_id)")

                # 表3：用户反馈
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        request_id TEXT NOT NULL DEFAULT '',
                        thread_id TEXT NOT NULL DEFAULT '',
                        user_id TEXT NOT NULL DEFAULT '',
                        rating TEXT NOT NULL DEFAULT 'up',
                        reason TEXT NOT NULL DEFAULT '',
                        note TEXT NOT NULL DEFAULT '',
                        answer_preview TEXT NOT NULL DEFAULT '',
                        question TEXT NOT NULL DEFAULT '',
                        timestamp REAL NOT NULL,
                        converted_to_badcase INTEGER NOT NULL DEFAULT 0
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_request_id ON feedback(request_id)")

                conn.commit()
            finally:
                conn.close()

    # ===================================================================
    # 节点耗时
    # ===================================================================

    def record_node(
        self,
        request_id: str,
        node_name: str,
        duration_ms: float,
        route_type: str = "",
        cache_hit: bool = False,
        thread_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """记录一次节点执行耗时"""
        with self._lock:
            try:
                conn = sqlite3.connect(str(self._db_path))
                try:
                    conn.execute(
                        """
                        INSERT INTO node_metrics
                        (request_id, thread_id, node_name, duration_ms,
                         route_type, cache_hit, timestamp, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            request_id,
                            thread_id,
                            node_name,
                            duration_ms,
                            route_type,
                            1 if cache_hit else 0,
                            time.time(),
                            json.dumps(metadata or {}, ensure_ascii=False),
                        ),
                    )
                    conn.commit()
                finally:
                    conn.close()
            except Exception as e:
                logger.warning(f"Metrics 写入失败：{e}")

    def get_node_stats(self, hours: int = 24) -> List[Dict]:
        """查询各节点的 P50/P95/P99 耗时统计"""
        cutoff = time.time() - hours * 3600
        try:
            conn = sqlite3.connect(str(self._db_path))
            try:
                rows = conn.execute(
                    """
                    SELECT
                        node_name,
                        COUNT(*) as count,
                        AVG(duration_ms) as avg_ms,
                        MIN(duration_ms) as min_ms,
                        MAX(duration_ms) as max_ms,
                        SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) as cache_hits
                    FROM node_metrics
                    WHERE timestamp > ?
                    GROUP BY node_name
                    ORDER BY avg_ms DESC
                    """,
                    (cutoff,),
                ).fetchall()

                result = []
                for row in rows:
                    node_name, count, avg_ms, min_ms, max_ms, cache_hits = row
                    percentiles = conn.execute(
                        """
                        SELECT duration_ms FROM node_metrics
                        WHERE timestamp > ? AND node_name = ?
                        ORDER BY duration_ms
                        """,
                        (cutoff, node_name),
                    ).fetchall()

                    if percentiles:
                        p50_idx = int(len(percentiles) * 0.50)
                        p95_idx = int(len(percentiles) * 0.95)
                        p99_idx = min(int(len(percentiles) * 0.99), len(percentiles) - 1)
                        p50 = percentiles[p50_idx][0]
                        p95 = percentiles[p95_idx][0]
                        p99 = percentiles[p99_idx][0]
                    else:
                        p50 = p95 = p99 = 0

                    result.append({
                        "node_name": node_name,
                        "count": count,
                        "avg_ms": round(avg_ms, 2),
                        "min_ms": round(min_ms, 2),
                        "max_ms": round(max_ms, 2),
                        "p50_ms": round(p50, 2),
                        "p95_ms": round(p95, 2),
                        "p99_ms": round(p99, 2),
                        "cache_hit_rate": round(cache_hits / max(count, 1), 4),
                    })
                return result
            finally:
                conn.close()
        except Exception as e:
            logger.warning(f"Metrics 查询失败：{e}")
            return []

    def get_request_stats(self, hours: int = 24) -> List[Dict]:
        """查询请求级总耗时统计"""
        cutoff = time.time() - hours * 3600
        try:
            conn = sqlite3.connect(str(self._db_path))
            try:
                rows = conn.execute(
                    """
                    SELECT
                        request_id,
                        route_type,
                        SUM(duration_ms) as total_ms,
                        COUNT(*) as node_count,
                        MIN(timestamp) as start_time,
                        GROUP_CONCAT(node_name || ':' || CAST(duration_ms AS TEXT), '|') as breakdown
                    FROM node_metrics
                    WHERE timestamp > ?
                    GROUP BY request_id
                    ORDER BY total_ms DESC
                    LIMIT 100
                    """,
                    (cutoff,),
                ).fetchall()

                return [
                    {
                        "request_id": r[0],
                        "route_type": r[1],
                        "total_ms": round(r[2], 2),
                        "node_count": r[3],
                        "start_time": r[4],
                        "breakdown": r[5] or "",
                    }
                    for r in rows
                ]
            finally:
                conn.close()
        except Exception as e:
            logger.warning(f"Metrics 请求统计查询失败：{e}")
            return []

    # ===================================================================
    # Token 用量
    # ===================================================================

    def record_token_usage(
        self,
        request_id: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        node_name: str = "",
        thread_id: str = "",
    ):
        """记录一次 LLM 调用的 Token 用量"""
        total_tokens = prompt_tokens + completion_tokens
        pricing = _MODEL_PRICING.get(model, _MODEL_PRICING["default"])
        estimated_cost = (
            prompt_tokens / 1000 * pricing["prompt"]
            + completion_tokens / 1000 * pricing["completion"]
        )

        with self._lock:
            try:
                conn = sqlite3.connect(str(self._db_path))
                try:
                    conn.execute(
                        """
                        INSERT INTO token_usage
                        (request_id, thread_id, node_name, model,
                         prompt_tokens, completion_tokens, total_tokens,
                         estimated_cost, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            request_id,
                            thread_id,
                            node_name,
                            model,
                            prompt_tokens,
                            completion_tokens,
                            total_tokens,
                            round(estimated_cost, 6),
                            time.time(),
                        ),
                    )
                    conn.commit()
                finally:
                    conn.close()
            except Exception as e:
                logger.warning(f"Token usage 写入失败：{e}")

    def get_token_stats(self, hours: int = 24) -> Dict:
        """查询 Token 用量统计

        Returns:
            {
                "total_prompt_tokens": int,
                "total_completion_tokens": int,
                "total_tokens": int,
                "total_cost": float,
                "by_model": [{model, calls, prompt, completion, total, cost}],
                "by_node": [{node_name, calls, prompt, completion, total, cost}],
                "daily_trend": [{date, total_tokens, cost}],
            }
        """
        cutoff = time.time() - hours * 3600
        try:
            conn = sqlite3.connect(str(self._db_path))
            try:
                # 总计
                total_row = conn.execute(
                    """
                    SELECT
                        SUM(prompt_tokens),
                        SUM(completion_tokens),
                        SUM(total_tokens),
                        SUM(estimated_cost),
                        COUNT(*)
                    FROM token_usage WHERE timestamp > ?
                    """,
                    (cutoff,),
                ).fetchone()
                total_prompt = total_row[0] or 0
                total_completion = total_row[1] or 0
                total_tokens = total_row[2] or 0
                total_cost = total_row[3] or 0.0
                total_calls = total_row[4] or 0

                # 按模型分组
                model_rows = conn.execute(
                    """
                    SELECT model,
                           COUNT(*) as calls,
                           SUM(prompt_tokens),
                           SUM(completion_tokens),
                           SUM(total_tokens),
                           SUM(estimated_cost)
                    FROM token_usage WHERE timestamp > ?
                    GROUP BY model ORDER BY SUM(total_tokens) DESC
                    """,
                    (cutoff,),
                ).fetchall()
                by_model = [
                    {
                        "model": r[0],
                        "calls": r[1],
                        "prompt_tokens": r[2] or 0,
                        "completion_tokens": r[3] or 0,
                        "total_tokens": r[4] or 0,
                        "cost": round(r[5] or 0, 4),
                    }
                    for r in model_rows
                ]

                # 按节点分组
                node_rows = conn.execute(
                    """
                    SELECT node_name,
                           COUNT(*) as calls,
                           SUM(prompt_tokens),
                           SUM(completion_tokens),
                           SUM(total_tokens),
                           SUM(estimated_cost)
                    FROM token_usage WHERE timestamp > ?
                    GROUP BY node_name ORDER BY SUM(total_tokens) DESC
                    """,
                    (cutoff,),
                ).fetchall()
                by_node = [
                    {
                        "node_name": r[0],
                        "calls": r[1],
                        "prompt_tokens": r[2] or 0,
                        "completion_tokens": r[3] or 0,
                        "total_tokens": r[4] or 0,
                        "cost": round(r[5] or 0, 4),
                    }
                    for r in node_rows
                ]

                # 每日趋势
                daily_rows = conn.execute(
                    """
                    SELECT date(timestamp, 'unixepoch', 'localtime') as day,
                           SUM(total_tokens),
                           SUM(estimated_cost)
                    FROM token_usage WHERE timestamp > ?
                    GROUP BY day ORDER BY day
                    """,
                    (cutoff,),
                ).fetchall()
                daily_trend = [
                    {"date": r[0], "total_tokens": r[1] or 0, "cost": round(r[2] or 0, 4)}
                    for r in daily_rows
                ]

                return {
                    "total_prompt_tokens": total_prompt,
                    "total_completion_tokens": total_completion,
                    "total_tokens": total_tokens,
                    "total_cost": round(total_cost, 4),
                    "total_calls": total_calls,
                    "hours": hours,
                    "by_model": by_model,
                    "by_node": by_node,
                    "daily_trend": daily_trend,
                }
            finally:
                conn.close()
        except Exception as e:
            logger.warning(f"Token 统计查询失败：{e}")
            return {}

    # ===================================================================
    # 用户反馈
    # ===================================================================

    def record_feedback(
        self,
        request_id: str,
        rating: str,
        reason: str = "",
        note: str = "",
        answer_preview: str = "",
        question: str = "",
        user_id: str = "",
        thread_id: str = "",
    ) -> str:
        """记录用户反馈

        Args:
            request_id: 关联的请求 ID
            rating: "up" (👍) 或 "down" (👎)
            reason: 差评原因（answer_inaccurate/not_answering/missing_info/unsafe_content/other）
            note: 用户补充说明
            answer_preview: AI 回答预览
            question: 原始问题
            user_id: 用户 ID
            thread_id: 会话线程 ID

        Returns:
            feedback_id
        """
        import uuid
        feedback_id = f"fb_{uuid.uuid4().hex[:12]}"

        with self._lock:
            try:
                conn = sqlite3.connect(str(self._db_path))
                try:
                    conn.execute(
                        """
                        INSERT INTO feedback
                        (request_id, thread_id, user_id, rating, reason,
                         note, answer_preview, question, timestamp,
                         converted_to_badcase)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                        """,
                        (
                            request_id,
                            thread_id,
                            user_id,
                            rating,
                            reason,
                            note[:500] if note else "",
                            answer_preview[:500] if answer_preview else "",
                            question[:200] if question else "",
                            time.time(),
                        ),
                    )
                    conn.commit()
                finally:
                    conn.close()

                # 差评自动关联 Bad Case
                if rating == "down" and question:
                    self._auto_create_bad_case(
                        request_id, question, answer_preview, reason, note, user_id, thread_id
                    )
            except Exception as e:
                logger.warning(f"Feedback 写入失败：{e}")

            return feedback_id

    def _auto_create_bad_case(
        self,
        request_id: str,
        question: str,
        answer_preview: str,
        reason: str,
        note: str,
        user_id: str,
        thread_id: str,
    ):
        """差评自动创建 Bad Case 记录（写入 PostgresStore long_term_memory）"""
        try:
            from app.memory import get_long_term_memory
            memory = get_long_term_memory()
            memory.append_bad_case(
                case_type="user_negative_feedback",
                original_query=question,
                rewritten_query="",
                final_question=question,
                answer_preview=answer_preview[:200],
                user_id=user_id or "system",
                thread_id=thread_id,
                metadata={
                    "reason": reason,
                    "note": note[:200] if note else "",
                    "request_id": request_id,
                    "source": "feedback_api",
                },
            )
            logger.info(f"差评已自动创建 Bad Case：request_id={request_id}, reason={reason}")
        except Exception as e:
            logger.warning(f"差评自动创建 Bad Case 失败（不影响反馈记录）：{e}")

    def get_feedback_stats(self, hours: int = 24) -> Dict:
        """查询反馈统计

        Returns:
            {
                "total_feedback": int,
                "up_count": int,
                "down_count": int,
                "satisfaction_rate": float,  # 👍/(👍+👎)
                "by_reason": [{reason, count}],
                "recent_negative": [{question, reason, note, timestamp}],
                "daily_trend": [{date, up, down, rate}],
            }
        """
        cutoff = time.time() - hours * 3600
        try:
            conn = sqlite3.connect(str(self._db_path))
            try:
                # 总计
                total_row = conn.execute(
                    """
                    SELECT
                        COUNT(*),
                        SUM(CASE WHEN rating = 'up' THEN 1 ELSE 0 END),
                        SUM(CASE WHEN rating = 'down' THEN 1 ELSE 0 END)
                    FROM feedback WHERE timestamp > ?
                    """,
                    (cutoff,),
                ).fetchone()
                total = total_row[0] or 0
                up_count = total_row[1] or 0
                down_count = total_row[2] or 0
                satisfaction_rate = round(up_count / max(up_count + down_count, 1), 4)

                # 按原因分组
                reason_rows = conn.execute(
                    """
                    SELECT reason, COUNT(*) as count
                    FROM feedback
                    WHERE timestamp > ? AND rating = 'down' AND reason != ''
                    GROUP BY reason ORDER BY count DESC
                    """,
                    (cutoff,),
                ).fetchall()
                by_reason = [{"reason": r[0], "count": r[1]} for r in reason_rows]

                # 最近差评（前 20 条）
                negative_rows = conn.execute(
                    """
                    SELECT question, reason, note, timestamp
                    FROM feedback
                    WHERE timestamp > ? AND rating = 'down'
                    ORDER BY timestamp DESC LIMIT 20
                    """,
                    (cutoff,),
                ).fetchall()
                recent_negative = [
                    {
                        "question": r[0],
                        "reason": r[1],
                        "note": r[2],
                        "timestamp": r[3],
                    }
                    for r in negative_rows
                ]

                # 每日趋势
                daily_rows = conn.execute(
                    """
                    SELECT date(timestamp, 'unixepoch', 'localtime') as day,
                           SUM(CASE WHEN rating = 'up' THEN 1 ELSE 0 END),
                           SUM(CASE WHEN rating = 'down' THEN 1 ELSE 0 END)
                    FROM feedback WHERE timestamp > ?
                    GROUP BY day ORDER BY day
                    """,
                    (cutoff,),
                ).fetchall()
                daily_trend = []
                for r in daily_rows:
                    up = r[1] or 0
                    down = r[2] or 0
                    daily_trend.append({
                        "date": r[0],
                        "up": up,
                        "down": down,
                        "rate": round(up / max(up + down, 1), 4),
                    })

                return {
                    "total_feedback": total,
                    "up_count": up_count,
                    "down_count": down_count,
                    "satisfaction_rate": satisfaction_rate,
                    "hours": hours,
                    "by_reason": by_reason,
                    "recent_negative": recent_negative,
                    "daily_trend": daily_trend,
                }
            finally:
                conn.close()
        except Exception as e:
            logger.warning(f"Feedback 统计查询失败：{e}")
            return {}

    def get_feedback_candidates_for_golden_set(self, limit: int = 50) -> List[Dict]:
        """获取差评中适合转化为黄金测试集的候选

        条件：有 question + reason + 未转化过的差评
        """
        try:
            conn = sqlite3.connect(str(self._db_path))
            try:
                rows = conn.execute(
                    """
                    SELECT request_id, question, reason, note, answer_preview, timestamp
                    FROM feedback
                    WHERE rating = 'down'
                      AND question != ''
                      AND converted_to_badcase = 0
                    ORDER BY timestamp DESC LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
                return [
                    {
                        "request_id": r[0],
                        "question": r[1],
                        "reason": r[2],
                        "note": r[3],
                        "answer_preview": r[4],
                        "timestamp": r[5],
                    }
                    for r in rows
                ]
            finally:
                conn.close()
        except Exception as e:
            logger.warning(f"获取黄金测试集候选失败：{e}")
            return []

    # ===================================================================
    # 通用
    # ===================================================================

    def cleanup(self, days: int = _RETENTION_DAYS):
        """清理超过 N 天的旧数据"""
        cutoff = time.time() - days * 86400
        with self._lock:
            try:
                conn = sqlite3.connect(str(self._db_path))
                try:
                    for table in ("node_metrics", "token_usage", "feedback"):
                        deleted = conn.execute(
                            f"DELETE FROM {table} WHERE timestamp < ?", (cutoff,)
                        ).rowcount
                        if deleted > 0:
                            logger.info(f"Metrics 清理：{table} 删除 {deleted} 条超过 {days} 天的旧数据")
                    conn.commit()
                finally:
                    conn.close()
            except Exception as e:
                logger.warning(f"Metrics 清理失败：{e}")


# 单例
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
