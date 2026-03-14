# app/rag/hybrid_retriever.py
"""混合检索模块 (生产优化版)
改进点：
    1. 安全的文档 ID 生成 (Hash 或 Metadata ID)
    2. 中文停用词过滤
    3. BM25 索引持久化 (可选，防止大内存占用)
    4. 完善的异常降级处理
"""
import os
import pickle
import hashlib
import time

import jieba
from typing import List, Dict, Any, Optional, Set
from rank_bm25 import BM25Okapi
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from app.cache.semantic_cache import get_semantic_cache
from app.rag.reranker import get_reranker
from app.rag.vector_store import get_vector_store
from app.core.app_logging import get_logger
from app.core.config import get_config

config = get_config()
logger = get_logger(__name__)

# === 配置常量 ===
# 简单的中文停用词表 (实际项目中建议加载外部停用词文件)
STOPWORDS: Set[str] = {
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人",
    "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
    "你", "会", "着", "没有", "看", "好", "自己", "这", "那", "他",
    "她", "它", "们", "吗", "呢", "吧", "啊", "呀", "哦", "嗯",
    "什么", "怎么", "为什么", "如何", "哪些", "哪里", "多少"
}

BM25_CACHE_PATH = "data/bm25_index.pkl"


class HybridRetriever(BaseRetriever):
    """生产级混合检索器"""

    vector_store: Any = None
    bm25: Optional[BM25Okapi] = None
    documents: List[Document] = []
    k: int = 5
    alpha: float = 0.5
    use_reranker: bool = True  # 是否引用Reranker
    rerank_top_k: int = 20  # RRF融合后送入Reranker的数量

    def __init__(
            self,
            vector_store=None,
            documents: List[Document] = None,
            k: int = 5,
            alpha: float = 0.5,
            use_cache: bool = True,
            use_reranker: bool = True,  # 新增参数
            rerank_top_k: int = 20  # 新增参数
    ):
        super().__init__()
        self.vector_store = vector_store or get_vector_store()
        self.k = k
        self.alpha = alpha

        # 初始化 BM25
        if documents:
            self._init_bm25(documents)
        else:
            self._init_bm25_from_store(use_cache=use_cache)

    def _tokenize(self, text: str) -> List[str]:
        """分词并过滤停用词"""
        words = jieba.lcut(text)
        return [w for w in words if w not in STOPWORDS and len(w.strip()) > 0]

    def _get_doc_id(self, doc: Document) -> str:
        """生成安全的文档唯一 ID"""
        # 优先使用 metadata 中的 ID (Chroma/Faiss 通常会有)
        if hasattr(doc, 'id') and doc.id:
            return str(doc.id)
        if doc.metadata.get('id'):
            return str(doc.metadata['id'])
        if doc.metadata.get('source') and doc.metadata.get('start_index'):
            return f"{doc.metadata['source']}:{doc.metadata['start_index']}"

        # 兜底：对全文做 MD5 哈希 (比截取前 100 字安全得多)
        content_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
        return content_hash

    def _init_bm25(self, documents: List[Document]):
        """初始化 BM25 索引"""
        self.documents = documents
        logger.info(f"开始构建 BM25 索引，文档数：{len(documents)}...")

        # 分词 (带停用词过滤)
        tokenized_docs = [self._tokenize(doc.page_content) for doc in documents]

        if not tokenized_docs or all(len(t) == 0 for t in tokenized_docs):
            logger.error("BM25 分词结果为空，请检查停用词表或文档内容")
            self.bm25 = None
            return

        self.bm25 = BM25Okapi(tokenized_docs)
        logger.info("BM25 索引构建完成")

    def _init_bm25_from_store(self, use_cache: bool = True):
        """从向量库加载文档并初始化 BM25 (支持缓存)"""
        # 1. 尝试加载缓存
        if use_cache and os.path.exists(BM25_CACHE_PATH):
            try:
                logger.info(f"从磁盘加载 BM25 缓存：{BM25_CACHE_PATH}")
                with open(BM25_CACHE_PATH, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.bm25 = data['bm25']
                    logger.info("BM25 缓存加载成功")
                    return
            except Exception as e:
                logger.warning(f"加载 BM25 缓存失败，将重新构建：{e}")

        # 2. 重新构建
        try:
            collection = self.vector_store._collection
            # 注意：如果文档量巨大，这里需要分页获取 (limit/offset)
            # ChromaDB 默认 limit 最大可能是几万，视配置而定
            results = collection.get(include=["documents", "metadatas"], limit=50000)

            if not results["documents"]:
                logger.warning("向量库中未找到文档，无法构建 BM25 索引")
                self.bm25 = None
                return

            documents = []
            for i, content in enumerate(results["documents"]):
                doc = Document(
                    page_content=content,
                    metadata=results["metadatas"][i] if results["metadatas"] else {}
                )
                # 如果有 ID，尽量保留
                if results.get("ids") and i < len(results["ids"]):
                    doc.id = results["ids"][i]
                documents.append(doc)

            self._init_bm25(documents)

            # 3. 保存缓存
            if self.bm25:
                try:
                    os.makedirs(os.path.dirname(BM25_CACHE_PATH), exist_ok=True)
                    with open(BM25_CACHE_PATH, 'wb') as f:
                        pickle.dump({'documents': self.documents, 'bm25': self.bm25}, f)
                    logger.info(f"BM25 索引已缓存到：{BM25_CACHE_PATH}")
                except Exception as e:
                    logger.warning(f"保存 BM25 缓存失败：{e}")

        except Exception as e:
            logger.error(f"从向量库初始化 BM25 严重失败：{e}")
            self.bm25 = None

    def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun,
            original_query: str = None  # 新增：原始查询参数，用于缓存
    ) -> List[Document]:
        """混合检索主流程
        三层缓存架构：
            L1：基于原始查询的查询结果缓存
            L2：语义相似缓存
            L3：热点访问文档缓存
        """
        # 确定缓存键，优先使用原始查询
        cache_query = original_query if original_query else query

        # L1：基于原始查询的查询结果缓存
        from app.cache.redis_cache import get_cache
        cache = get_cache()

        cached_result = cache.get(
            cache_query,  # 使用原始查询作为缓存键
            k=self.k,
            use_reranker=self.use_reranker,
            rerank_top_k=self.rerank_top_k
        )

        if cached_result:
            documents, metadata = cached_result
            logger.info(f"🎯 L1缓存命中：{cache_query[:30]}... 使用缓存结果：{len(documents)} 个文档")
            return documents

        # L2：语义相似缓存（相似匹配）
        if getattr(config, 'ENABLE_SEMANTIC_CACHE', False):
            semantic_cache = get_semantic_cache()

            logger.info(f"🔍 L2语义缓存检查：{cache_query[:30]}...")

            semantic_result = semantic_cache.get(cache_query)

            if semantic_result:
                documents, metadata = semantic_result
                logger.info(
                    f"L2 语义缓存命中：{cache_query[:30]}... "
                    f"(相似度: {metadata.get('similarity', 0):.2%})"
                )
                return documents

        # 原始的RAG文档检索
        start_time = time.time()
        # 1. Dense 检索
        dense_results = self._dense_search(query)

        # 2. Sparse 检索
        sparse_results = self._sparse_search(query)

        # 3. 降级处理：如果 BM25 不可用，直接返回向量结果
        if not sparse_results:
            logger.warning("BM25 不可用，降级为纯向量检索")
            candidates = [doc for doc, _ in dense_results][:self.rerank_top_k]
        else:
            # 4. RRF融合（取更多候选送入Reranker）
            candidates = self._rrf_fusion(
                dense_results,
                sparse_results,
                top_k=self.rerank_top_k
            )

        # 5. Reranker重排序
        if self.use_reranker and candidates:
            try:
                reranker = get_reranker()
                final_docs = reranker.rerank(
                    query=query,
                    documents=candidates,
                    top_k=self.k,
                    score_threshold=config.RERANKER_THRESHOLD  # 文档阈值，如果相关性小于0.3则不会返回文档，即RAG失效了
                )
                logger.info(f"Reranker 重排序：{len(candidates)} -> {len(final_docs)}")
                # return final_docs
            except Exception as e:
                logger.warning(f"Reranker 失败，使用 RRF 结果：{e}")
                final_docs = candidates[:self.k]

        else:
            final_docs = candidates[:self.k]

        retrieval_time = (time.time() - start_time) * 1000
        logger.info(f"检索完成，耗时：{retrieval_time:.2f}ms")

        # 3. 写入缓存
        if final_docs:
            # 写入L1查询结果缓存
            cache.set(
                cache_query,  # 使用原始查询作为缓存键
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
            logger.info(f"📝 L1缓存已写入：{cache_query[:30]}...)")

            # 写入L2语义缓存
            if getattr(config, 'ENABLE_SEMANTIC_CACHE', False):
                semantic_cache = get_semantic_cache()
                semantic_cache.set(cache_query, final_docs)
                logger.info(f"L2语义缓存已写入：{cache_query[:30]}...")

        return final_docs

    def _dense_search(self, query: str) -> List[tuple]:
        """向量检索"""
        try:
            docs = self.vector_store.similarity_search(query, k=self.k)
            # 确保 doc 有 id 属性 (LangChain 新版本可能需要在 metadata 里找)
            return [(doc, rank) for rank, doc in enumerate(docs)]
        except Exception as e:
            logger.error(f"Dense 检索异常：{e}")
            return []

    def _sparse_search(self, query: str) -> List[tuple]:
        """BM25 检索"""
        if self.bm25 is None or not self.documents:
            return []

        try:
            tokens = self._tokenize(query)
            if not tokens:
                return []

            scores = self.bm25.get_scores(tokens)

            # 获取 Top-K 索引
            # 注意：这里取 k*2 参与融合，给 RRF 更多选择空间
            top_k_count = min(self.k * 2, len(scores))
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k_count]

            results = []
            for rank, idx in enumerate(top_indices):
                if scores[idx] > 0:  # 只取有分数的
                    results.append((self.documents[idx], rank))

            return results
        except Exception as e:
            logger.error(f"Sparse 检索异常：{e}")
            return []

    def _rrf_fusion(
            self,
            dense_results: List[tuple],
            sparse_results: List[tuple],
            rrf_k: int = 60,
            top_k: int = None
    ) -> List[Document]:
        """RRF 融合算法"""
        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        # Dense 得分
        for doc, rank in dense_results:
            doc_id = self._get_doc_id(doc)
            score = self.alpha / (rrf_k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score
            doc_map[doc_id] = doc

        # Sparse 得分
        for doc, rank in sparse_results:
            doc_id = self._get_doc_id(doc)
            score = (1 - self.alpha) / (rrf_k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score
            doc_map[doc_id] = doc

        # 排序并截取
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[doc_id] for doc_id, _ in sorted_docs[:self.k]]


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
