"""缓存模块

功能描述：
    提供检索结果的缓存功能
    支持 Redis 缓存、语义相似缓存
"""
from app.cache.redis_cache import get_cache, RedisCache
from app.cache.semantic_cache import get_semantic_cache, SemanticCache

__all__ = ["get_cache", "RedisCache", "get_semantic_cache", "SemanticCache"]