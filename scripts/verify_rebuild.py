"""重建结果验证：全库 chunk 数、status/version_id 分布、各文档 chunk 数"""
import sys
from pathlib import Path
from collections import Counter

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.rag.vector_store import get_vector_store

vs = get_vector_store()
col = vs._collection
count = col.count()
print(f"全库 chunk 数: {count}")

results = col.get(include=["metadatas"], limit=100000)
metas = results.get("metadatas") or []
print(f"读取元数据: {len(metas)} 条")

status_dist = Counter(m.get("status", "?") for m in metas)
version_dist = Counter(m.get("version_id", "?") for m in metas)
deleted_dist = Counter(m.get("is_deleted", "?") for m in metas)
print(f"status 分布: {dict(status_dist)}")
print(f"version_id 分布: {dict(version_dist)}")
print(f"is_deleted 分布: {dict(deleted_dist)}")

# 各 source chunk 数 + 唯一内容 chunk 数
by_source = Counter()
unique_by_source = Counter()
seen = set()
for m in metas:
    src = m.get("source", "unknown")
    by_source[src] += 1
    cid = m.get("chunk_id") or m.get("content_hash") or m.get("id", "")
    key = (src, cid)
    if key not in seen:
        seen.add(key)
        unique_by_source[src] += 1

print("\n各文档 chunk 数:")
for src, cnt in by_source.most_common():
    print(f"  {cnt:>4} (唯一 {unique_by_source[src]:>4})  {src}")

# 检查是否有多个 chunk_id 相同的（重复版本）
id_counter = Counter()
for m in metas:
    cid = m.get("chunk_id") or "?"
    id_counter[cid] += 1
dups = {k: v for k, v in id_counter.items() if v > 1}
print(f"\n重复 chunk_id 数: {len(dups)}")
