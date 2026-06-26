"""语义相似缓存模块
功能描述：
    基于向量相似度的查询缓存
    相似问题可以命中同一缓存，提高缓存命中率

设计理念：
    1、向量化：将查询转换为向量
    2、相似匹配：使用余弦相似度匹配相似查询
    3、阈值控制：只有相似度超过阈值时才命中缓存
    4、降级处理：embedding失败时自动降级为普通缓存

使用场景：
    ”感冒了怎么办“和”感冒怎么治疗“可以命中同一缓存
    大幅度提高缓存命中率（预计30-50%）
"""
from datetime import datetime
import hashlib
import time
import json
from typing import Optional, List, Tuple, Dict

import numpy as np
from langchain_core.documents import Document

from app.cache.redis_cache import get_cache
from app.core.config import get_config
from app.core.app_logging import get_logger

logger = get_logger(__name__)


class SemanticCache:
    """语义相似缓存管理器"""

    def __init__(
            self,
            similarity_threshold: float = 0.92,
            prefix: str = "semantic_cache:",
            ttl: int = 3600,
            enabled: bool = True,
            max_keys: int = 5000,  # 最大缓存条目数，超出时 LRU 淘汰
    ):
        """
        Args:
            similarity_threshold: 相似度阈值
            prefix: Redis 键前缀
            ttl: 缓存过期时间（秒）
            enabled: 是否启用
            max_keys: 最大缓存条目数，超出时淘汰最早的条目
        """
        self.similarity_threshold = similarity_threshold
        self.prefix = prefix
        self.ttl = ttl
        self.enabled = enabled
        self.max_keys = max_keys

        self._cache = get_cache()
        self._embeddings = None
        self._available = False

        # Redis Sorted Set 用于 LRU 淘汰（score = 最后访问时间戳）
        # 替代原来的 Redis Set（Set 无序，无法实现真正的 LRU）
        self._keys_zset = f"{prefix}lru_zset"
        # 保留旧 Set 的引用，用于兼容
        self._keys_set = f"{prefix}keys"

        # 本地 embedding 缓存，避免重复计算
        self._embedding_cache: Dict[str, List[float]] = {}

        self._stats = {
            "hits": 0,
            "misses": 0,
            "total_requests": 0,
            "similarity_matches": 0,
        }

        if self.enabled:
            self._init_embeddings()

    def _init_embeddings(self):
        """初始化embedding模型"""
        try:
            from app.core.embeddings import get_embeddings
            self._embeddings = get_embeddings()
            self._available = True
            logger.info(f"语义缓存初始化成功，相似度阈值：{self.similarity_threshold}")
        except Exception as e:
            logger.warning(f"Embedding初始化失败，语义缓存禁用：{e}")
            self._available = False

    def _get_embedding(self, query: str) -> Optional[List[float]]:
        """获取查询的向量表示
        Args：
            query：查询文本

        Returns：
            向量列表，失败则返回None
        """
        if not self._available or self._embeddings is None:
            return None

        try:
            # 使用embedding模型获取向量
            embedding = self._embeddings.embed_query(query)
            return embedding
        except Exception as e:
            logger.error(f"获取embedding失败：{e}")
            return None

    def get_embedding(self, query: str) -> Optional[List[float]]:
        """对外暴露 embedding 计算，便于上层复用同一 query 向量。"""
        return self._get_embedding(query)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度
        公式：cos(θ) = (A·B) / (|A| × |B|)

        Args：
            vec1：向量1
            vec2：向量2

        Returns：
            相似度值 [0, 1]
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _find_similar_query(
            self,
            query_embedding: List[float],
            top_k: int = 10
    ) -> Optional[Tuple[str, float, Dict]]:
        """查找相似查询（优化版：Redis Set 追踪键 + MGET 批量获取）

        性能优化（相比旧版 SCAN + N×GET）：
            - 用 SMEMBERS 替代 SCAN：O(n) 但在 Set 上比全库扫描更快
            - 用 MGET 批量获取：1 次 Redis 往返替代 N 次
            - 限制检查数量：前 top_k * 10 个键
            - 本地 embedding 缓存：避免重复计算
        """
        if not self._cache._available:
            return None

        try:
            all_keys = self._cache._redis.smembers(self._keys_set)
            if not all_keys:
                logger.info("L2 语义缓存为空")
                return None

            logger.info(f"L2 缓存键数量：{len(all_keys)}")

            # 限制检查数量，优先检查最近的条目
            check_keys = list(all_keys)[: top_k * 10]

            # 批量获取所有缓存值（单次 Redis 往返）
            raw_values = self._cache._redis.mget(check_keys)

            best_match = None
            best_similarity = 0.0
            query_np = np.array(query_embedding)

            for key_bytes, raw_value in zip(check_keys, raw_values):
                if not raw_value:
                    continue
                try:
                    data = json.loads(raw_value)
                    cached_embedding = data.get("embedding")
                    if not cached_embedding:
                        continue

                    # NumPy 批量计算余弦相似度
                    cached_np = np.array(cached_embedding)
                    dot_product = np.dot(query_np, cached_np)
                    norm1 = np.linalg.norm(query_np)
                    norm2 = np.linalg.norm(cached_np)
                    similarity = float(dot_product / (norm1 * norm2)) if norm1 and norm2 else 0.0

                    cached_query = data.get("query", "")[:20]
                    logger.debug(f"  对比：'{cached_query}...' 相似度={similarity:.2%}")

                    if similarity > best_similarity and similarity >= self.similarity_threshold:
                        best_similarity = similarity
                        best_match = (key_bytes.decode() if isinstance(key_bytes, bytes) else key_bytes,
                                      similarity, data)
                        if similarity >= 0.98:
                            break
                except Exception:
                    continue

            if best_match:
                logger.info(f"L2 找到相似查询，相似度：{best_similarity:.2%}")
            else:
                logger.info(f"L2 未找到相似查询，最高相似度：{best_similarity:.2%}（阈值：{self.similarity_threshold:.2%}）")

            return best_match

        except Exception as e:
            logger.error(f"查找相似查询失败：{e}")
            return None

    def get(self, query: str, query_embedding: List[float] = None) -> Optional[Tuple[List[Document], Dict]]:
        """获取语义相似度的缓存结果
        Args:
            query: 查询文本

        Returns：
            （documents， metadata）或None
        """
        if not self.enabled or not self._available:
            return None

        self._stats["total_requests"] += 1

        # 复用已经向量化好的结构，避免重复计算
        if query_embedding is None:
            query_embedding = self._get_embedding(query)


        if query_embedding is None:
            self._stats["misses"] += 1
            return None

        # 2、查找相似查询
        similar_result = self._find_similar_query(query_embedding)

        if similar_result:
            key, similarity, data = similar_result

            # 3、反序列化文档
            try:
                documents = self._cache._deserialize_documents(data["documents"])
                metadata = data.get("metadata", {})
                metadata["semantic_hit"] = True
                metadata["similarity"] = similarity
                metadata["original_query"] = data.get("query", "")

                self._stats["hits"] += 1
                self._stats["similarity_matches"] += 1

                # LRU 访问刷新：更新 ZSET 的 score 为当前时间戳
                try:
                    self._cache._redis.zadd(self._keys_zset, {key: time.time()})
                except Exception:
                    pass

                logger.info(
                    f"语义缓存命中：'{query[:20]}...' ≈ '{data.get('query', '')[:20]}...' "
                    f"(相似度: {similarity:.2%})"
                )

                return documents, metadata

            except Exception as e:
                logger.error(f"反序列化缓存文档失败：{e}")
                self._stats["misses"] += 1
                return None

        self._stats["misses"] += 1
        return None

    def set(
            self,
            query: str,
            documents: List[Document],
            metadata: Dict = None,
            query_embedding: List[float] = None,
    ) -> bool:
        """缓存查询结果（包含向量）"""
        if not self.enabled or not self._available:
            return False

        if not documents:
            return False

        if not self._cache._available:
            return False

        if query_embedding is None:
            query_embedding = self._get_embedding(query)

        if query_embedding is None:
            return False

        query_hash = hashlib.md5(query.encode("utf-8")).hexdigest()
        key = f"{self.prefix}{query_hash}"

        data = {
            "query": query,
            "embedding": query_embedding,
            "documents": self._cache._serialize_documents(documents),
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "doc_count": len(documents),
        }

        try:
            # LRU 淘汰：超过 max_keys 时删除最久未访问的条目
            current_count = self._cache._redis.zcard(self._keys_zset)
            if current_count >= self.max_keys:
                # ZRANGE 按 score 升序 → score 最小 = 最久未访问
                to_remove = self._cache._redis.zrange(self._keys_zset, 0, max(int(current_count * 0.2), 1) - 1)
                if to_remove:
                    # 删除缓存数据和 LRU 索引
                    self._cache._redis.delete(*to_remove)
                    for k in to_remove:
                        self._cache._redis.zrem(self._keys_zset, k)
                        self._cache._redis.srem(self._keys_set, k)  # 兼容清理
                    logger.info(f"语义缓存 LRU 淘汰：{len(to_remove)} 条（真 LRU，按访问时间排序）")

            # 写入缓存数据
            self._cache._redis.setex(key, self.ttl, json.dumps(data, ensure_ascii=False))
            # 维护 LRU Sorted Set（score = 当前时间戳，越大越新）
            now_ts = time.time()
            self._cache._redis.zadd(self._keys_zset, {key: now_ts})
            # 兼容：同时维护旧 Set
            self._cache._redis.sadd(self._keys_set, key)
            logger.info(f"语义缓存已写入：'{query[:20]}...' ({len(documents)} 个文档)")
            return True

        except Exception as e:
            logger.warning(f"语义缓存写入失败，跳过：{e}")
            self._cache._available = False
            return False

    def get_stats(self) -> Dict:
        """获取统计信息"""
        total = self._stats["total_requests"]
        hit_rate = (self._stats["hits"] / total * 100) if total > 0 else 0

        return {
            "enabled": self.enabled,
            "available": self._available,
            "similarity_threshold": self.similarity_threshold,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "similarity_matches": self._stats["similarity_matches"],
            "hit_rate": f"{hit_rate:.2f}%",
            "total_requests": total
        }

    def clear(self) -> int:
        """清空语义缓存"""
        try:
            keys = list(self._cache._redis.smembers(self._keys_set))
            if keys:
                self._cache._redis.delete(*keys)
                self._cache._redis.delete(self._keys_set)
                self._cache._redis.delete(self._keys_zset)
            logger.info(f"清空语义缓存：{len(keys)} 条")
            return len(keys)
        except Exception as e:
            logger.error(f"清空语义缓存失败：{e}")
            return 0


# 全局单例
_semantic_cache: Optional[SemanticCache] = None


def get_semantic_cache() -> SemanticCache:
    """获取语义缓存单例"""
    global _semantic_cache

    if _semantic_cache is None:
        config = get_config()
        _semantic_cache = SemanticCache(
            similarity_threshold=getattr(config, 'SEMANTIC_CACHE_THRESHOLD', 0.92),
            ttl=getattr(config, 'CACHE_TTL_SECONDS', 3600),
            enabled=getattr(config, 'ENABLE_SEMANTIC_CACHE', True)
        )

    return _semantic_cache
