"""v9.21: 清空 L2 语义缓存 + L0 answer 缓存，让新检索词/新阈值立即生效"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.cache.semantic_cache import get_semantic_cache
from app.cache.redis_cache import get_cache

sem = get_semantic_cache()
n_sem = sem.clear()
print(f"语义缓存已清空：{n_sem} 条")

cache = get_cache()
n_cache = cache.clear()
print(f"answer/检索缓存已清空：{n_cache} 条")

# 额外用 SCAN 兜底删除 semantic_cache:* 前缀残留（LRU 记录、旧哈希键等）
try:
    import redis as redis_lib
    from app.core.config import get_config
    config = get_config()
    r = redis_lib.Redis.from_url(config.REDIS_URL, decode_responses=True)
    cursor = 0
    deleted = 0
    while True:
        cursor, keys = r.scan(cursor, match="semantic_cache:*", count=200)
        if keys:
            r.delete(*keys)
            deleted += len(keys)
        if cursor == 0:
            break
    print(f"SCAN 兜底删除 semantic_cache:* 残留：{deleted} 条")
except Exception as e:
    print(f"SCAN 兜底跳过：{e}")
