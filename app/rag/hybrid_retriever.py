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
from app.core.embeddings import get_embeddings
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


def _looks_clear_medical_query(query: str) -> bool:
    """判断是否为语义明确、通常无需 rerank 的医疗问题。"""
    text = (query or "").strip().lower()
    if not text:
        return False

    clear_patterns = [
        "什么是", "是什么意思", "有哪些症状", "什么症状", "原因", "治疗", "如何治疗", "怎么治疗",
        "预防", "护理", "诊断", "检查", "高血压", "糖尿病", "感冒", "肺炎", "胃炎", "鼻炎",
    ]
    return any(pattern in text for pattern in clear_patterns)


class HybridRetriever(BaseRetriever):
    """基于 LangChain EnsembleRetriever 的混合检索器"""

    ensemble_retriever: Any = None
    vector_store: Any = None
    k: int = 5
    alpha: float = 0.5
    use_reranker: bool = True
    rerank_top_k: int = 5
    bm25_retriever: Any = None

    def __init__(
            self,
            vector_store=None,
            documents: List[Document] = None,
            k: int = 5,
            alpha: float = 0.5,
            use_cache: bool = True,
            use_reranker: bool = True,
            rerank_top_k: int = 5
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
            self.bm25_retriever = None
            return

        # 3. 构建 BM25Retriever（使用 langchain_community 提供的实现）
        try:
            bm25_retriever = BM25Retriever.from_documents(
                documents,
                k=self.k * 2,
                preprocess_func=_tokenize,
            )
            self.bm25_retriever = bm25_retriever
        except Exception as e:
            logger.error(f"BM25Retriever 初始化失败：{e}，仅使用向量检索")
            self.ensemble_retriever = vector_retriever
            self.bm25_retriever = None
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

    def _dense_search(self, query: str, query_embedding: Optional[List[float]] = None) -> List[Document]:
        """执行 dense 检索；若已提供 embedding，则直接复用。"""
        top_k = self.k * 2
        try:
            if query_embedding is not None and hasattr(self.vector_store, "similarity_search_by_vector"):
                return self.vector_store.similarity_search_by_vector(query_embedding, k=top_k)
            return self.vector_store.similarity_search(query, k=top_k)
        except Exception as e:
            logger.error(f"Dense 检索失败：{e}")
            return []

    def _sparse_search(self, query: str) -> List[Document]:
        """执行 BM25 稀疏检索。"""
        if self.bm25_retriever is None:
            return []
        try:
            return self.bm25_retriever.invoke(query)
        except Exception as e:
            logger.error(f"BM25 检索失败：{e}")
            return []

    def _reciprocal_rank_fusion(self, dense_docs: List[Document], sparse_docs: List[Document]) -> List[Document]:
        """对 dense / sparse 结果做轻量 RRF 融合。"""
        if not dense_docs and not sparse_docs:
            return []

        score_map: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        fusion_constant = 60
        weighted_results = [
            (dense_docs, self.alpha),
            (sparse_docs, 1 - self.alpha),
        ]

        for docs, weight in weighted_results:
            for rank, doc in enumerate(docs, start=1):
                doc_key = getattr(doc, "id", None) or f"{doc.metadata.get('source', '')}:{doc.metadata.get('file_path', '')}:{hash(doc.page_content)}"
                doc_map[doc_key] = doc
                score_map[doc_key] = score_map.get(doc_key, 0.0) + weight / (fusion_constant + rank)

        ranked_keys = sorted(score_map.keys(), key=lambda key: score_map[key], reverse=True)
        return [doc_map[key] for key in ranked_keys]

    def _should_skip_reranker(self, query: str, candidates: List[Document]) -> bool:
        """在候选很少或问题已足够明确时跳过 rerank，降低首 token 延迟。"""
        if not candidates:
            return True

        candidate_count = len(candidates)
        if candidate_count <= min(self.k, 3):
            return True

        if candidate_count <= self.k:
            return True

        if _looks_clear_medical_query(query) and candidate_count <= max(self.k, self.rerank_top_k):
            return True

        return False

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

        query_embedding = None
        semantic_cache = None
        semantic_lookup_ms = 0.0
        query_embedding_ms = 0.0

        # L2：语义相似缓存
        if getattr(config, 'ENABLE_SEMANTIC_CACHE', False):
            semantic_cache = get_semantic_cache()
            logger.info(f"L2语义缓存检查：{cache_query[:30]}...")

            embedding_start = time.time()
            query_embedding = semantic_cache.get_embedding(cache_query)
            query_embedding_ms = (time.time() - embedding_start) * 1000

            semantic_lookup_start = time.time()
            semantic_result = semantic_cache.get(cache_query, query_embedding=query_embedding)
            semantic_lookup_ms = (time.time() - semantic_lookup_start) * 1000

            logger.info(
                f"L2语义缓存耗时：embedding_ms={query_embedding_ms:.2f}, lookup_ms={semantic_lookup_ms:.2f}"
            )

            if semantic_result:
                documents, metadata = semantic_result
                logger.info(
                    f"L2 语义缓存命中：{cache_query[:30]}... "
                    f"(相似度: {metadata.get('similarity', 0):.2%})"
                )
                return documents

        elif query_embedding is None:
            embedding_start = time.time()
            try:
                query_embedding = get_embeddings().embed_query(query)
            except Exception as e:
                logger.warning(f"查询向量预计算失败，将回退到文本检索：{e}")
                query_embedding = None
            query_embedding_ms = (time.time() - embedding_start) * 1000

        retrieval_start = time.time()
        dense_start = time.time()
        dense_docs = self._dense_search(query, query_embedding=query_embedding)
        dense_ms = (time.time() - dense_start) * 1000

        sparse_start = time.time()
        sparse_docs = self._sparse_search(query)
        sparse_ms = (time.time() - sparse_start) * 1000

        fusion_start = time.time()
        candidates = self._reciprocal_rank_fusion(dense_docs, sparse_docs)
        fusion_ms = (time.time() - fusion_start) * 1000

        rerank_ms = 0.0
        should_skip_reranker = self._should_skip_reranker(query, candidates)

        # Reranker 重排序
        if self.use_reranker and candidates and not should_skip_reranker:
            try:
                reranker = get_reranker()
                rerank_start = time.time()
                final_docs = reranker.rerank(
                    query=query,
                    documents=candidates[:self.rerank_top_k],
                    top_k=self.k,
                    score_threshold=config.RERANKER_THRESHOLD
                )
                rerank_ms = (time.time() - rerank_start) * 1000
                logger.info(f"Reranker 重排序：{len(candidates)} -> {len(final_docs)}")
            except Exception as e:
                logger.warning(f"Reranker 失败，使用 EnsembleRetriever 结果：{e}")
                final_docs = candidates[:self.k]
        else:
            if self.use_reranker and candidates and should_skip_reranker:
                logger.info(
                    f"跳过 Reranker：candidate_count={len(candidates)}, k={self.k}, rerank_top_k={self.rerank_top_k}, query={query[:50]}"
                )
            final_docs = candidates[:self.k]

        retrieval_time = (time.time() - retrieval_start) * 1000
        logger.info(
            "检索完成，耗时：%.2fms（query_embedding=%.2fms, semantic_lookup=%.2fms, dense=%.2fms, sparse=%.2fms, fusion=%.2fms, rerank=%.2fms）",
            retrieval_time,
            query_embedding_ms,
            semantic_lookup_ms,
            dense_ms,
            sparse_ms,
            fusion_ms,
            rerank_ms,
        )

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

            if semantic_cache is not None:
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
        rerank_top_k: int = 5
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
