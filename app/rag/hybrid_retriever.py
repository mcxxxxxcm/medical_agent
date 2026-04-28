"""混合检索模块 (LangChain EnsembleRetriever 版)
改进点：
    1. 使用 LangChain 官方 EnsembleRetriever 替代手动 RRF 融合
    2. 使用 BM25Retriever 替代手写 BM25Okapi 封装
    3. 保留缓存和 Reranker 逻辑
    4. 保留 BM25 索引持久化
"""
import os
import pickle
import time

import jieba
from functools import lru_cache
from typing import List, Dict, Any, Optional, Set

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.retrievers import BM25Retriever

try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    from langchain_classic.retrievers import EnsembleRetriever

from app.cache.semantic_cache import get_semantic_cache
from app.rag.reranker import get_reranker
from app.rag.vector_store import get_vector_store, load_documents_from_store
from app.core.app_logging import get_logger
from app.core.config import get_config

config = get_config()
logger = get_logger(__name__)

STOPWORDS: Set[str] = {
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人",
    "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
    "你", "会", "着", "没有", "看", "好", "自己", "这", "那", "他",
    "她", "它", "们", "吗", "呢", "吧", "啊", "呀", "哦", "嗯",
    "什么", "怎么", "为什么", "如何", "哪些", "哪里", "多少"
}

BM25_CACHE_PATH = "data/bm25_index.pkl"


def _tokenize(text: str) -> List[str]:
    """分词并过滤停用词"""
    words = jieba.lcut(text)
    return [w for w in words if w not in STOPWORDS and len(w.strip()) > 0]


def _load_documents_from_store(vector_store) -> List[Document]:
    """从向量库加载所有文档"""
    return load_documents_from_store(vector_store=vector_store, limit=50000)


class HybridRetriever(BaseRetriever):
    """基于 LangChain EnsembleRetriever 的混合检索器"""

    ensemble_retriever: Any = None
    vector_store: Any = None
    k: int = 5
    alpha: float = 0.5
    use_reranker: bool = True
    rerank_top_k: int = 20

    def __init__(
            self,
            vector_store=None,
            documents: List[Document] = None,
            k: int = 5,
            alpha: float = 0.5,
            use_cache: bool = True,
            use_reranker: bool = True,
            rerank_top_k: int = 20
    ):
        super().__init__()
        self.vector_store = vector_store or get_vector_store()
        self.k = k
        self.alpha = alpha
        self.use_reranker = use_reranker
        self.rerank_top_k = rerank_top_k

        # 初始化 EnsembleRetriever
        self._init_ensemble_retriever(documents, use_cache)

    def _init_ensemble_retriever(self, documents: List[Document] = None, use_cache: bool = True):
        """构建 EnsembleRetriever（向量检索 + BM25 检索）"""
        # 1. 向量检索器
        vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.k * 2}
        )

        # 2. 加载文档用于 BM25
        if documents is None:
            documents = self._load_bm25_documents(use_cache)

        if not documents:
            logger.warning("无可用文档，仅使用向量检索")
            self.ensemble_retriever = vector_retriever
            return

        # 3. 构建 BM25Retriever（使用 langchain_community 提供的实现）
        try:
            bm25_retriever = BM25Retriever.from_documents(
                documents,
                k=self.k * 2,
            )
            # 自定义分词函数
            bm25_retriever.k = self.k * 2
            bm25_retriever.vectorizer = _tokenize
        except Exception as e:
            logger.error(f"BM25Retriever 初始化失败：{e}，仅使用向量检索")
            self.ensemble_retriever = vector_retriever
            return

        # 4. 使用 EnsembleRetriever 进行 RRF 融合
        # weights=[alpha, 1-alpha] 对应 dense/sparse 的权重
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[self.alpha, 1 - self.alpha],
            c=60,  # RRF 常数，与原实现一致
        )
        logger.info(f"EnsembleRetriever 初始化完成 (alpha={self.alpha}, c=60)")

    def _load_bm25_documents(self, use_cache: bool = True) -> List[Document]:
        """加载 BM25 所需的文档（支持缓存）"""
        if use_cache and os.path.exists(BM25_CACHE_PATH):
            try:
                logger.info(f"从磁盘加载 BM25 缓存：{BM25_CACHE_PATH}")
                with open(BM25_CACHE_PATH, 'rb') as f:
                    data = pickle.load(f)
                    documents = data['documents']
                    logger.info(f"BM25 缓存加载成功，文档数：{len(documents)}")
                    return documents
            except Exception as e:
                logger.warning(f"加载 BM25 缓存失败，将重新构建：{e}")

        documents = _load_documents_from_store(self.vector_store)
        if not documents:
            logger.warning("向量库中未找到文档")
            return []

        # 保存缓存
        try:
            os.makedirs(os.path.dirname(BM25_CACHE_PATH), exist_ok=True)
            with open(BM25_CACHE_PATH, 'wb') as f:
                pickle.dump({'documents': documents}, f)
            logger.info(f"BM25 文档已缓存到：{BM25_CACHE_PATH}")
        except Exception as e:
            logger.warning(f"保存 BM25 缓存失败：{e}")

        return documents

    def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun,
            original_query: str = None
    ) -> List[Document]:
        """混合检索主流程"""
        cache_query = original_query if original_query else query

        # L1：Redis 查询缓存
        from app.cache.redis_cache import get_cache
        cache = get_cache()

        cached_result = cache.get(
            cache_query,
            k=self.k,
            use_reranker=self.use_reranker,
            rerank_top_k=self.rerank_top_k
        )

        if cached_result:
            documents, metadata = cached_result
            logger.info(f"L1缓存命中：{cache_query[:30]}... 使用缓存结果：{len(documents)} 个文档")
            return documents

        # L2：语义相似缓存
        query_embedding = None

        if getattr(config, 'ENABLE_SEMANTIC_CACHE', False):
            semantic_cache = get_semantic_cache()
            logger.info(f"L2语义缓存检查：{cache_query[:30]}...")

            query_embedding = semantic_cache._get_embedding(cache_query)
            semantic_result = semantic_cache.get(cache_query, query_embedding=query_embedding)

            if semantic_result:
                documents, metadata = semantic_result
                logger.info(
                    f"L2 语义缓存命中：{cache_query[:30]}... "
                    f"(相似度: {metadata.get('similarity', 0):.2%})"
                )
                return documents

        # 执行 EnsembleRetriever 检索
        start_time = time.time()

        try:
            candidates = self.ensemble_retriever.invoke(query)
        except Exception as e:
            logger.error(f"EnsembleRetriever 检索异常：{e}")
            candidates = []

        # Reranker 重排序
        if self.use_reranker and candidates:
            try:
                reranker = get_reranker()
                final_docs = reranker.rerank(
                    query=query,
                    documents=candidates[:self.rerank_top_k],
                    top_k=self.k,
                    score_threshold=config.RERANKER_THRESHOLD
                )
                logger.info(f"Reranker 重排序：{len(candidates)} -> {len(final_docs)}")
            except Exception as e:
                logger.warning(f"Reranker 失败，使用 EnsembleRetriever 结果：{e}")
                final_docs = candidates[:self.k]
        else:
            final_docs = candidates[:self.k]

        retrieval_time = (time.time() - start_time) * 1000
        logger.info(f"检索完成，耗时：{retrieval_time:.2f}ms")

        # 写入缓存
        if final_docs:
            cache.set(
                cache_query,
                final_docs,
                metadata={
                    "retrieval_time_ms": retrieval_time,
                    "sources": [d.metadata.get("source") for d in final_docs],
                    "original_query": original_query,
                    "search_query": query,
                },
                k=self.k,
                use_reranker=self.use_reranker,
                rerank_top_k=self.rerank_top_k
            )
            logger.info(f"L1缓存已写入：{cache_query[:30]}...")

            if getattr(config, 'ENABLE_SEMANTIC_CACHE', False):
                semantic_cache = get_semantic_cache()
                semantic_cache.set(cache_query, final_docs, query_embedding=query_embedding)
                logger.info(f"L2语义缓存已写入：{cache_query[:30]}...")

        return final_docs


@lru_cache(maxsize=8)
def get_hybrid_retriever(
        vector_store=None,
        k: int = 5,
        alpha: float = 0.5,
        use_cache: bool = True,
        use_reranker: bool = True,
        rerank_top_k: int = 20
) -> HybridRetriever:
    """工厂函数"""
    return HybridRetriever(
        vector_store=vector_store,
        k=k,
        alpha=alpha,
        use_cache=use_cache,
        use_reranker=use_reranker,
        rerank_top_k=rerank_top_k
    )
