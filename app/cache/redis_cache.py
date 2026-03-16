"""Redis 缓存模块

功能描述：
    基于 Redis 的检索结果缓存
    支持分布式部署、TTL 过期、LRU 淘汰

设计理念：
    1. 单例模式：全局共享 Redis 连接池
    2. 连接池：复用连接，提高性能
    3. 序列化：支持复杂对象的存储
    4. 降级处理：Redis 不可用时自动降级为内存缓存
"""
import json
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from langchain_core.documents import Document
from app.core.app_logging import get_logger
from app.core.config import get_config
import redis

logger = get_logger(__name__)


@dataclass
class CacheStats:
    """缓存统计信息"""
    hits: int = 0
    misses: int = 0
    errors: int = 0
    total_requests: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests * 100


class RedisCache:
    """Redis 缓存管理器"""

    def __init__(
            self,
            redis_url: str = "redis://localhost:6379/0",
            prefix: str = "medical_assistant:",
            default_ttl: int = 3600,  # 默认 1 小时过期
            enabled: bool = True
    ):
        self.redis_url = redis_url
        self.prefix = prefix
        self.default_ttl = default_ttl
        self.enabled = enabled

        self._redis = None
        self._available = False
        self._stats = CacheStats()

        # 降级用的内存缓存
        self._fallback_cache: Dict[str, Tuple[Any, float]] = {}
        self._fallback_max_size = 1000

        if self.enabled:
            self._connect()

    def _connect(self):
        """连接 Redis"""
        try:

            from redis.connection import ConnectionPool

            # 使用连接池
            pool = ConnectionPool.from_url(
                self.redis_url,
                max_connections=10,
                decode_responses=True
            )
            self._redis = redis.Redis(connection_pool=pool)

            # 测试连接
            self._redis.ping()
            self._available = True
            logger.info(f"✅ Redis 连接成功：{self.redis_url}")

        except ImportError:
            logger.warning("⚠️ redis 库未安装，使用内存缓存降级。请运行：pip install redis")
            self._available = False
        except Exception as e:
            logger.warning(f"⚠️ Redis 连接失败：{e}，使用内存缓存降级")
            self._available = False

    def _generate_key(self, query: str, **kwargs) -> str:
        """生成缓存键

        Args:
            query: 查询文本
            **kwargs: 其他参数

        Returns:
            带前缀的缓存键
        """
        key_parts = [query]
        for k, v in sorted(kwargs.items()):
            if v is not None:
                key_parts.append(f"{k}={v}")

        key_str = "|".join(key_parts)
        key_hash = hashlib.md5(key_str.encode('utf-8')).hexdigest()

        return f"{self.prefix}{key_hash}"

    def _serialize_documents(self, documents: List[Document]) -> str:
        """序列化文档列表

        Args:
            documents: 文档列表

        Returns:
            JSON 字符串
        """
        serializable = []
        for doc in documents:
            serializable.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })
        return json.dumps(serializable, ensure_ascii=False)

    def _deserialize_documents(self, data: str) -> List[Document]:
        """反序列化文档列表

        Args:
            data: JSON 字符串

        Returns:
            文档列表
        """
        items = json.loads(data)
        return [
            Document(page_content=item["page_content"], metadata=item["metadata"])
            for item in items
        ]

    def get(self, query: str, **kwargs) -> Optional[Tuple[List[Document], Dict]]:
        """获取缓存

        Args:
            query: 查询文本
            **kwargs: 其他参数

        Returns:
            (documents, metadata) 或 None
        """
        if not self.enabled:
            return None

        self._stats.total_requests += 1
        key = self._generate_key(query, **kwargs)

        try:
            if self._available:
                # Redis 获取
                data = self._redis.get(key)
                if data:
                    cached = json.loads(data)
                    documents = self._deserialize_documents(cached["documents"])
                    self._stats.hits += 1
                    logger.info(f"🎯 Redis 缓存命中：{query[:30]}... (命中率: {self._stats.hit_rate:.1f}%)")
                    return documents, cached.get("metadata", {})
            else:
                # 内存缓存降级
                if key in self._fallback_cache:
                    data, expires_at = self._fallback_cache[key]
                    if time.time() < expires_at:
                        self._stats.hits += 1
                        logger.info(f"💾 内存缓存命中：{query[:30]}...")
                        return data
                    else:
                        del self._fallback_cache[key]

            self._stats.misses += 1
            return None

        except Exception as e:
            self._stats.errors += 1
            logger.error(f"缓存读取失败：{e}")
            return None

    def set(
            self,
            query: str,
            documents: List[Document],
            metadata: Dict = None,
            ttl: int = None,
            **kwargs
    ) -> bool:
        """设置缓存

        Args:
            query: 查询文本
            documents: 文档列表
            metadata: 元数据
            ttl: 过期时间（秒）
            **kwargs: 其他参数

        Returns:
            是否成功
        """
        if not self.enabled:
            return False

        key = self._generate_key(query, **kwargs)
        ttl = ttl or self.default_ttl

        try:
            cached = {
                "documents": self._serialize_documents(documents),
                "metadata": metadata or {},
                "created_at": time.time()
            }
            data = json.dumps(cached, ensure_ascii=False)

            if self._available:
                # Redis 设置
                self._redis.setex(key, ttl, data)
                logger.debug(f"Redis 缓存写入：{query[:30]}... TTL={ttl}s")
            else:
                # 内存缓存降级
                # 检查容量
                if len(self._fallback_cache) >= self._fallback_max_size:
                    # 简单 LRU：删除最早的
                    oldest_key = next(iter(self._fallback_cache))
                    del self._fallback_cache[oldest_key]

                self._fallback_cache[key] = ((documents, metadata), time.time() + ttl)
                logger.debug(f"内存缓存写入：{query[:30]}... TTL={ttl}s")

            return True

        except Exception as e:
            logger.error(f"缓存写入失败：{e}")
            return False

    def delete(self, query: str, **kwargs) -> bool:
        """删除缓存

        Args:
            query: 查询文本
            **kwargs: 其他参数

        Returns:
            是否成功
        """
        key = self._generate_key(query, **kwargs)

        try:
            if self._available:
                self._redis.delete(key)
            else:
                self._fallback_cache.pop(key, None)

            return True
        except Exception as e:
            logger.error(f"缓存删除失败：{e}")
            return False

    def clear(self) -> int:
        """清空缓存

        Returns:
            清理的条目数
        """
        try:
            if self._available:
                # 删除所有匹配前缀的键
                keys = self._redis.keys(f"{self.prefix}*")
                if keys:
                    self._redis.delete(*keys)
                count = len(keys)
            else:
                count = len(self._fallback_cache)
                self._fallback_cache.clear()

            logger.info(f"清空缓存：{count} 条")
            return count

        except Exception as e:
            logger.error(f"清空缓存失败：{e}")
            return 0

    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = {
            "enabled": self.enabled,
            "available": self._available,
            "backend": "redis" if self._available else "memory",
            "hits": self._stats.hits,
            "misses": self._stats.misses,
            "errors": self._stats.errors,
            "hit_rate": f"{self._stats.hit_rate:.2f}%",
            "total_requests": self._stats.total_requests
        }

        if self._available:
            try:
                info = self._redis.info("memory")
                stats["redis_memory_used"] = info.get("used_memory_human", "unknown")
                stats["redis_keys"] = self._redis.dbsize()
            except:
                pass
        else:
            stats["memory_cache_size"] = len(self._fallback_cache)

        return stats

    def health_check(self) -> Dict:
        """健康检查"""
        result = {
            "status": "healthy" if self._available else "degraded",
            "backend": "redis" if self._available else "memory_fallback",
            "redis_url": self.redis_url if self._available else None
        }

        if self._available:
            try:
                latency = self._redis.ping()
                result["latency"] = f"{latency * 1000:.2f}ms"
            except Exception as e:
                result["status"] = "unhealthy"
                result["error"] = str(e)

        return result


# 全局单例
_cache_instance: Optional[RedisCache] = None


def get_cache() -> RedisCache:
    """获取缓存单例"""
    global _cache_instance

    if _cache_instance is None:
        config = get_config()
        _cache_instance = RedisCache(
            redis_url=getattr(config, 'REDIS_URL', 'redis://localhost:6379/0'),
            prefix="medical_assistant:retrieval:",
            default_ttl=getattr(config, 'CACHE_TTL_SECONDS', 3600),
            enabled=getattr(config, 'ENABLE_QUERY_CACHE', True)
        )

    return _cache_instance
