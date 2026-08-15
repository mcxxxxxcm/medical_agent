"""P0 修复验证：
1. 增量上传不丢内容：v1(A,B,C) → v2(改C→C') → v3(改回C)，全程 active 始终 3 块、无重复 hash
2. reset_hybrid_retriever 清 lru_cache（零停机重建生效）
3. fallback_buffer.cleanup_expired 只删超过 7 天的事件
"""
import random
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import chromadb
from langchain_core.documents import Document

from app.rag.kb_updater import (
    activate_document_version,
    compute_content_hash,
    filter_unchanged_chunks,
    get_existing_content_hashes,
)

failures = []
def check(label, ok):
    print(f"  {'✅' if ok else '❌'} {label}")
    if not ok:
        failures.append(label)

source = "测试药典.txt"
A = "布洛芬用法用量：成人每次200-400mg，每日3-4次，饭后服用。"
B = "对乙酰氨基酚用法用量：成人每次500mg，每日不超过4次。"
C = "发烧护理：减少衣物散热，温水擦浴，不要捂汗。"
C2 = "发烧护理：减少衣物散热，温水擦浴，多饮温水，不要捂汗。"
_DIM = 64


def _hash(text):
    return compute_content_hash(text)


def _chunk(text, version_id, status):
    h = _hash(text)
    return Document(page_content=text, metadata={
        "doc_id": source, "source": source, "chunk_id": f"{source}_{h}",
        "content_hash": h, "version_id": version_id, "status": status,
        "is_deleted": False,
    })


class _VS:
    def __init__(self, col):
        self._collection = col


def _add(col, chunks):
    # 用 upsert：确定性 chunk_id 写入同 id 时覆盖为新版本（chromadb add 对同 id 静默忽略，
    # "改回旧内容"时旧 id 会挡住新版本写入）
    col.upsert(
        ids=[c.metadata["chunk_id"] for c in chunks],
        metadatas=[c.metadata for c in chunks],
        documents=[c.page_content for c in chunks],
        embeddings=[[random.random() for _ in range(_DIM)] for _ in chunks],
    )


def _active_hashes(col):
    r = col.get(where={"status": "active"}, include=["metadatas"])
    return [m["content_hash"] for m in r["metadatas"]]


def test_incremental():
    print("\n[P0-1] 增量上传不丢内容")
    tmp = Path(tempfile.mkdtemp(prefix="p0_verify_"))
    try:
        client = chromadb.PersistentClient(path=str(tmp))
        col = client.get_or_create_collection("test", metadata={"hnsw:space": "cosine"})
        vs = _VS(col)

        # v1 全量 3 块
        _add(col, [_chunk(A, 1, "active"), _chunk(B, 1, "active"), _chunk(C, 1, "active")])
        check("v1 active 3 块", len(_active_hashes(col)) == 3)

        # v2：只改 C → C'，A B 未变
        new_v2 = [_chunk(A, 2, "pending"), _chunk(B, 2, "pending"), _chunk(C2, 2, "pending")]
        existing = get_existing_content_hashes(vs)
        changed, unchanged = filter_unchanged_chunks(new_v2, existing)
        check("v2 changed 仅 C'（1 块）", len(changed) == 1 and _hash(changed[0].page_content) == _hash(C2))
        check("v2 unchanged 为 A B（2 块）", len(unchanged) == 2)
        _add(col, changed)
        act = activate_document_version(vs, source, 2, keep_hashes={_hash(c.page_content) for c in new_v2})
        check("v2 激活 1 块", act == 1)
        hashes = _active_hashes(col)
        check("v2 后 active 仍 3 块（未变块保留）", len(hashes) == 3)
        check("v2 active 含 C'", _hash(C2) in hashes)
        check("v2 active 含 A", _hash(A) in hashes)
        check("v2 active 无重复", len(set(hashes)) == 3)
        dep = col.get(where={"status": "deprecated"}, include=["metadatas"])
        check("v2 后旧 C 已 deprecated（1 块）", len(dep["metadatas"]) == 1 and dep["metadatas"][0]["content_hash"] == _hash(C))

        # v3：把 C' 改回 C，应能重新激活
        new_v3 = [_chunk(A, 3, "pending"), _chunk(B, 3, "pending"), _chunk(C, 3, "pending")]
        existing = get_existing_content_hashes(vs)
        changed, _ = filter_unchanged_chunks(new_v3, existing)
        check("v3 C 可重新激活（deprecated 不再挡）", len(changed) == 1 and _hash(changed[0].page_content) == _hash(C))
        _add(col, changed)
        activate_document_version(vs, source, 3, keep_hashes={_hash(c.page_content) for c in new_v3})
        hashes = _active_hashes(col)
        check("v3 后 active 3 块（A B C 完整恢复）", len(hashes) == 3)
        check("v3 active 含 C（改回可恢复）", _hash(C) in hashes)
        check("v3 active 无重复", len(set(hashes)) == 3)
        dep = col.get(where={"status": "deprecated"}, include=["metadatas"])
        check("v3 后 C' 已 deprecated（1 块）", len(dep["metadatas"]) == 1 and dep["metadatas"][0]["content_hash"] == _hash(C2))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_reset():
    print("\n[P0-2] reset_hybrid_retriever 清 lru_cache")
    from app.rag.hybrid_retriever import get_hybrid_retriever, reset_hybrid_retriever
    get_hybrid_retriever.cache_clear()
    get_hybrid_retriever(k=5)
    get_hybrid_retriever(k=5)
    info = get_hybrid_retriever.cache_info()
    check("reset 前 lru 有命中", info.hits >= 1 and info.currsize >= 1)
    reset_hybrid_retriever()
    info2 = get_hybrid_retriever.cache_info()
    check("reset 后 lru 清空（currsize=0）", info2.currsize == 0)


def test_fallback():
    print("\n[P0-3] fallback_buffer.cleanup_expired 只删超 7 天")
    import app.memory.fallback_buffer as fb
    tmp = Path(tempfile.mkdtemp(prefix="p0_fb_"))
    old = fb._DB_PATH
    fb._DB_PATH = tmp / "t.db"
    try:
        fb._ensure_db()
        now = datetime.now()
        stale = (now - timedelta(days=8)).isoformat()   # 超过 7 天 → 应删
        fresh = (now - timedelta(hours=1)).isoformat()  # 1 小时前 → 应留
        conn = sqlite3.connect(str(fb._DB_PATH))
        conn.execute(
            "INSERT INTO pending_events (event_type,user_id,payload,created_at) VALUES ('symptom','u','{}',?)",
            (stale,),
        )
        conn.execute(
            "INSERT INTO pending_events (event_type,user_id,payload,created_at) VALUES ('symptom','u','{}',?)",
            (fresh,),
        )
        conn.commit()
        conn.close()
        fb.cleanup_expired()
        conn = sqlite3.connect(str(fb._DB_PATH))
        rows = conn.execute("SELECT created_at FROM pending_events").fetchall()
        conn.close()
        check("删除 8 天前事件（剩 1 条）", len(rows) == 1)
        check("保留 1 小时前事件", rows and datetime.fromisoformat(rows[0][0]) > (now - timedelta(hours=2)))
    finally:
        fb._DB_PATH = old
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    print("=" * 60)
    print("P0 修复验证")
    print("=" * 60)
    test_incremental()
    test_reset()
    test_fallback()
    print("\n" + "=" * 60)
    if failures:
        print(f"存在失败项: {failures}")
        return 1
    print("P0 修复验证全部通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
