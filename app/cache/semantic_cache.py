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
import json
from optparse import Option
from typing import Optional, List, Tuple, Dict

import numpy as np
from langchain_core.documents import Document
from sqlalchemy.testing.suite.test_reflection import metadata

from app.cache.redis_cache import get_cache
from app.core.config import get_config
from app.core.app_logging import get_logger

logger = get_logger(__name__)


class SemanticCache:
    """语义相似缓存管理器"""

    def __init__(
            self,
            similarity_threshold: float = 0.92,  # 相似度阈值（0.92表示非常相似）
            prefix: str = "semantic_cache:",
            ttl: int = 3600,  # 默认一个小时过期
            enabled: bool = True,
    ):
        """
        Args：
            similarity_threshold: 相似度阈值，超过此值认为查询相似
            prefix: Redis 键前缀
            ttl: 缓存过期时间（秒）
            enabled: 是否启用
        """
        self.similarity_threshold = similarity_threshold
        self.prefix = prefix
        self.ttl = ttl
        self.enabled = enabled

        self._cache = get_cache()
        self._embeddings = None
        self._available = False

        # 统计信息
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
        """查找相似查询
        Args：
            query_embedding：查询向量
            top_k：最多检查的缓存条目

        Returns：
            （缓存键，相似度，缓存数据）或None
        """
        if not self._cache._available:
            return None

        try:
            # 获取所有语义缓存键
            all_keys = self._cache._redis.keys(f"{self.prefix}*")

            # ✅ 添加调试日志
            logger.info(f"📊 L2缓存键数量：{len(all_keys)}")

            if not all_keys:
                logger.warning("⚠️ L2缓存为空，没有找到任何语义缓存键")
                return None

            best_match = None
            best_similarity = 0.0

            # 遍历缓存键，查找最相似的
            for key in all_keys[:top_k * 10]:  # 限制检查数量，避免性能问题
                try:
                    # 获取缓存的向量
                    cache_data = self._cache._redis.get(key)
                    if not cache_data:
                        continue

                    data = json.loads(cache_data)
                    cached_embedding = data["embedding"]

                    if not cached_embedding:
                        continue

                    # 计算相似度
                    similarity = self._cosine_similarity(query_embedding, cached_embedding)

                    # ✅ 添加调试日志
                    cached_query = data.get("query", "")[:20]
                    logger.debug(f"  对比：'{cached_query}...' 相似度={similarity:.2%}")

                    # 更新最佳匹配
                    if similarity > best_similarity and similarity >= self.similarity_threshold:
                        best_similarity = similarity
                        best_match = (key, similarity, data)

                        # 如果找到非常相似的，提前返回
                        if similarity >= 0.98:
                            break

                except Exception as e:
                    logger.debug(f"解析缓存键{key}失败：{e}")
                    continue

                # ✅ 添加结果日志
                if best_match:
                    logger.info(f"✅ L2找到相似查询，相似度：{best_similarity:.2%}")
                else:
                    logger.info(
                        f"❌ L2未找到相似查询，最高相似度：{best_similarity:.2%}（阈值：{self.similarity_threshold:.2%}）")

            return best_match

        except Exception as e:
            logger.error(f"查找相似查询失败：{e}")
            return None

    def get(self, query: str) -> Optional[Tuple[List[Document], Dict]]:
        """获取语义相似度的缓存结果
        Args:
            query: 查询文本

        Returns：
            （documents， metadata）或None
        """
        if not self.enabled or not self._available:
            return None

        self._stats["total_requests"] += 1

        # 1、获取查询向量
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

                logger.info(
                    f"🎯 语义缓存命中：'{query[:20]}...' ≈ '{data.get('query', '')[:20]}...' "
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

        # 获取向量
        if query_embedding is None:
            query_embedding = self._get_embedding(query)

        if query_embedding is None:
            return False

        # 生成缓存键
        query_hash = hashlib.md5(query.encode("utf-8")).hexdigest()
        key = f"{self.prefix}{query_hash}"

        # 构建缓存数据
        data = {
            "query": query,
            "embedding": query_embedding,
            "documents": self._cache._serialize_documents(documents),
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "doc_count": len(documents),
        }

        try:
            # 写入Redis
            self._cache._redis.setex(key, self.ttl, json.dumps(data, ensure_ascii=False))
            logger.info(f"语义缓存已写入：'{query[:20]}...' ({len(documents)}个文档)")
            return True

        except Exception as e:
            logger.error(f"语义缓存写入失败：{e}")
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
            keys = self._cache._redis.keys(f"{self.prefix}*")
            if keys:
                self._cache._redis.delete(*keys)
            logger.info(f"清空语义缓存：{len(keys)}条")
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
