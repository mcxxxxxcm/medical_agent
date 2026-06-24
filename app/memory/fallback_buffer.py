"""L1 写入失败本地缓冲队列

当 PostgresStore 暂时不可用时，将症状/用药事件写入本地 SQLite 文件，
待 L1 恢复后自动补写，防止数据永久丢失。

设计原则：
    1. 写入失败不阻塞主流程
    2. 服务启动时自动尝试 flush
    3. 定期后台 flush（每 5 分钟）
    4. 最多保留 7 天，过期自动清理
"""

import json
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.core.app_logging import get_logger

logger = get_logger(__name__)

_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "l1_fallback.db"
_FLUSH_INTERVAL_SECONDS = 300  # 5 分钟
_MAX_AGE_DAYS = 7


def _ensure_db():
    """确保数据库和表存在"""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pending_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            user_id TEXT NOT NULL,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL,
            retry_count INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


def enqueue_symptom_event(
    user_id: str,
    symptom_name: str,
    onset_iso: str,
    onset_ts: int,
    precision: str,
    source_query: str,
):
    """症状事件写入 L1 失败时，入队到本地缓冲"""
    try:
        _ensure_db()
        payload = json.dumps({
            "symptom_name": symptom_name,
            "onset_iso": onset_iso,
            "onset_ts": onset_ts,
            "precision": precision,
            "source_query": source_query[:100],
        }, ensure_ascii=False)
        conn = sqlite3.connect(str(_DB_PATH))
        conn.execute(
            "INSERT INTO pending_events (event_type, user_id, payload, created_at) VALUES (?, ?, ?, ?)",
            ("symptom", user_id, payload, datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()
        logger.info(f"症状事件已缓冲到本地：user={user_id}, symptom={symptom_name}")
    except Exception as e:
        logger.error(f"本地缓冲写入失败（事件将丢失）：{e}")


def enqueue_medication_event(
    user_id: str,
    drug: str,
    dosage: Optional[str],
    effect: Optional[str],
    source_query: str,
):
    """用药事件写入 L1 失败时，入队到本地缓冲"""
    try:
        _ensure_db()
        payload = json.dumps({
            "drug": drug,
            "dosage": dosage,
            "effect": effect,
            "source_query": source_query[:100],
        }, ensure_ascii=False)
        conn = sqlite3.connect(str(_DB_PATH))
        conn.execute(
            "INSERT INTO pending_events (event_type, user_id, payload, created_at) VALUES (?, ?, ?, ?)",
            ("medication", user_id, payload, datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()
        logger.info(f"用药事件已缓冲到本地：user={user_id}, drug={drug}")
    except Exception as e:
        logger.error(f"本地缓冲写入失败（事件将丢失）：{e}")


def get_pending_count() -> int:
    """获取待处理事件数量"""
    try:
        _ensure_db()
        conn = sqlite3.connect(str(_DB_PATH))
        count = conn.execute("SELECT COUNT(*) FROM pending_events").fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0


def flush() -> int:
    """尝试将所有缓冲事件重新写入 L1

    Returns:
        成功写入的事件数量
    """
    try:
        _ensure_db()
        conn = sqlite3.connect(str(_DB_PATH))
        rows = conn.execute(
            "SELECT id, event_type, user_id, payload FROM pending_events ORDER BY id"
        ).fetchall()
        if not rows:
            conn.close()
            return 0

        from app.memory import get_long_term_memory
        memory = get_long_term_memory()

        success_ids = []
        for row in rows:
            row_id, event_type, user_id, payload_str = row
            try:
                payload = json.loads(payload_str)
                if event_type == "symptom":
                    memory.append_symptom_event(
                        user_id=user_id,
                        symptom_name=payload["symptom_name"],
                        onset_iso=payload.get("onset_iso", ""),
                        onset_ts=payload["onset_ts"],
                        precision=payload.get("precision", "default"),
                        source_query=payload.get("source_query", ""),
                    )
                elif event_type == "medication":
                    memory.append_medication_event(
                        user_id=user_id,
                        drug=payload["drug"],
                        dosage=payload.get("dosage"),
                        effect=payload.get("effect"),
                        source_query=payload.get("source_query", ""),
                    )
                success_ids.append(row_id)
            except Exception as e:
                # 标记重试次数，超过 10 次则删除
                retry_count = conn.execute(
                    "SELECT retry_count FROM pending_events WHERE id=?", (row_id,)
                ).fetchone()
                current_retries = (retry_count[0] if retry_count else 0) + 1
                if current_retries >= 10:
                    conn.execute("DELETE FROM pending_events WHERE id=?", (row_id,))
                    logger.warning(f"事件 {row_id} 重试 {current_retries} 次仍失败，已丢弃")
                else:
                    conn.execute(
                        "UPDATE pending_events SET retry_count=? WHERE id=?",
                        (current_retries, row_id),
                    )

        # 删除成功写入的事件
        if success_ids:
            placeholders = ",".join("?" * len(success_ids))
            conn.execute(
                f"DELETE FROM pending_events WHERE id IN ({placeholders})",
                success_ids,
            )

        conn.commit()
        conn.close()

        if success_ids:
            logger.info(f"本地缓冲 flush 成功：{len(success_ids)} 条事件已写入 L1")
        return len(success_ids)
    except Exception as e:
        logger.warning(f"本地缓冲 flush 失败：{e}")
        return 0


def cleanup_expired():
    """清理超过最大保留时间的过期事件"""
    try:
        _ensure_db()
        cutoff = datetime.now().isoformat()
        conn = sqlite3.connect(str(_DB_PATH))
        deleted = conn.execute(
            "DELETE FROM pending_events WHERE created_at < ?",
            (cutoff,),
        ).rowcount
        conn.commit()
        conn.close()
        if deleted:
            logger.info(f"清理过期缓冲事件：{deleted} 条")
    except Exception as e:
        logger.warning(f"过期缓冲清理失败：{e}")


# ===== 后台 flush 调度 =====
_flush_timer: Optional[threading.Timer] = None


def _schedule_flush():
    """定期后台 flush"""
    global _flush_timer
    try:
        pending = get_pending_count()
        if pending > 0:
            flush()
        cleanup_expired()
    except Exception:
        pass
    _flush_timer = threading.Timer(_FLUSH_INTERVAL_SECONDS, _schedule_flush)
    _flush_timer.daemon = True
    _flush_timer.start()


def start_background_flush():
    """启动后台定期 flush"""
    count = get_pending_count()
    if count > 0:
        logger.info(f"检测到 {count} 条未同步的 L1 事件，立即尝试 flush")
        flushed = flush()
        logger.info(f"启动时 flush 完成：{flushed}/{count} 条")
    _schedule_flush()
    logger.info(f"L1 本地缓冲后台 flush 已启动（间隔 {_FLUSH_INTERVAL_SECONDS}s）")


def stop_background_flush():
    """停止后台 flush"""
    global _flush_timer
    if _flush_timer:
        _flush_timer.cancel()
        _flush_timer = None
