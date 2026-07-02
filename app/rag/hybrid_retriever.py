"""混合检索模块 (LangChain EnsembleRetriever 版)
改进点：
    1. 使用 LangChain 官方 EnsembleRetriever 替代手动 RRF 融合
    2. 使用 BM25Retriever 替代手写 BM25Okapi 封装
    3. 保留缓存和 Reranker 逻辑
    4. 保留 BM25 索引持久化
"""
import os
import pickle
import re
import time

import jieba
from functools import lru_cache
from typing import List, Dict, Any, Optional, Set
from collections import OrderedDict

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
from app.rag.parent_child_store import get_parent_child_manager
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

BM25_CACHE_PATH = str(config.BM25_CACHE_PATH)


# ===== Embedding LRU 缓存 =====
# 相同查询复用 embedding 向量，避免重复 API 调用（节省 200~400ms）
class _EmbeddingLRUCache:
    """线程安全的 Embedding LRU 缓存

    - 缓存 query→embedding 映射，命中时 0ms（vs API 200~400ms）
    - 最多缓存 128 个查询（约 128 * 2048 * 4 bytes ≈ 1MB 内存）
    - TTL 30 分钟，避免过期 embedding 影响检索质量
    """
    _MAX_SIZE = 128
    _TTL_SECONDS = 1800  # 30 分钟

    def __init__(self):
        self._cache: OrderedDict[str, tuple] = OrderedDict()  # query -> (embedding, timestamp)

    def get(self, query: str) -> Optional[List[float]]:
        if query in self._cache:
            embedding, ts = self._cache[query]
            if time.time() - ts < self._TTL_SECONDS:
                # 命中，移到队尾（LRU）
                self._cache.move_to_end(query)
                return embedding
            else:
                # 过期，删除
                del self._cache[query]
        return None

    def put(self, query: str, embedding: List[float]):
        if query in self._cache:
            self._cache.move_to_end(query)
        self._cache[query] = (embedding, time.time())
        # 淘汰最老的
        while len(self._cache) > self._MAX_SIZE:
            self._cache.popitem(last=False)

    def clear(self):
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


_embedding_cache = _EmbeddingLRUCache()


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
        """加载 BM25 所需的文档（支持缓存）

        父子索引兼容：检测缓存中的文档是否含 parent_id，不含则视为旧缓存并重建
        """
        if use_cache and os.path.exists(BM25_CACHE_PATH):
            try:
                logger.info(f"从磁盘加载 BM25 缓存：{BM25_CACHE_PATH}")
                with open(BM25_CACHE_PATH, 'rb') as f:
                    data = pickle.load(f)
                    documents = data['documents']
                    # 父子索引兼容检测：旧缓存无 parent_id，需重建
                    if documents and not documents[0].metadata.get("parent_id"):
                        logger.info("BM25 缓存为旧版（无 parent_id），重新从向量库加载")
                    else:
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

    def _dense_search(self, query: str, query_embedding: Optional[List[float]] = None) -> tuple:
        """执行 dense 检索；若已提供 embedding，则直接复用。

        Returns:
            (documents, top1_score): 检索到的文档列表和 Top-1 距离得分
            ChromaDB cosine distance: 0.0=完全相同, <0.08 对应 cosine_similarity>0.92

        注意：langchain_chroma.Chroma 的方法名：
            - similarity_search_with_score → 文本查询 + ChromaDB distance
            - similarity_search_by_vector_with_relevance_scores → 向量查询 + 归一化分数
            不存在 similarity_search_by_vector_with_score（旧代码用了不存在的方法名，导致
            hasattr 永远为 False，top1_score 永远为 0.0，触发 High-Confidence Bypass 误跳过）
        """
        top_k = self.k * 2
        top1_score = 1.0  # 默认值改为 1.0（不可信），避免 0.0 触发 High-Confidence Bypass
        try:
            if query_embedding is not None:
                # 策略1：使用底层 Chroma collection 直接查询（最可靠，直接返回 cosine distance）
                if hasattr(self.vector_store, '_collection'):
                    try:
                        col = self.vector_store._collection
                        qr = col.query(
                            query_embeddings=[query_embedding],
                            n_results=top_k,
                            include=["documents", "metadatas", "distances"]
                        )
                        if qr and qr["ids"] and qr["ids"][0]:
                            docs = []
                            for i in range(len(qr["ids"][0])):
                                doc = Document(
                                    page_content=qr["documents"][0][i],
                                    metadata=qr["metadatas"][0][i]
                                )
                                docs.append(doc)
                            top1_score = float(qr["distances"][0][0])
                            logger.info(f"Dense 检索（底层 collection）：{len(docs)} 篇, top1_distance={top1_score:.4f}")
                            return docs, top1_score
                    except Exception as col_err:
                        logger.warning(f"底层 collection 查询失败，回退到 LangChain 接口：{col_err}")

                # 策略2：LangChain 接口（带分数的方法名在不同版本中不一致）
                # langchain_chroma 提供 similarity_search_by_vector_with_relevance_scores
                if hasattr(self.vector_store, "similarity_search_by_vector_with_relevance_scores"):
                    results = self.vector_store.similarity_search_by_vector_with_relevance_scores(
                        query_embedding, k=top_k
                    )
                    if results:
                        docs = [doc for doc, score in results]
                        # relevance_scores 是归一化分数 [0,1]，转换为 distance = 1 - score
                        top1_relevance = float(results[0][1])
                        top1_score = 1.0 - top1_relevance
                        return docs, top1_score

                # 策略3：无分数接口兜底
                docs = self.vector_store.similarity_search_by_vector(query_embedding, k=top_k)
                top1_score = 1.0  # 无分数时标记为不可信，不触发 High-Confidence Bypass
            else:
                if hasattr(self.vector_store, "similarity_search_with_score"):
                    results = self.vector_store.similarity_search_with_score(query, k=top_k)
                    if results:
                        docs = [doc for doc, score in results]
                        top1_score = float(results[0][1])
                        return docs, top1_score
                docs = self.vector_store.similarity_search(query, k=top_k)
                top1_score = 1.0  # 无分数时标记为不可信
            return docs, top1_score
        except Exception as e:
            logger.error(f"Dense 检索失败：{e}")
            return [], 1.0  # 异常时也返回不可信分数

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

    def _should_skip_reranker(self, query: str, candidates: List[Document],
                               top1_dense_score: float = 0.0) -> bool:
        """判断是否可以跳过 Reranker，降低首 token 延迟。

        跳过条件（基于置信度，而非数量）：
            1. 无候选文档 → 跳过
            2. 简单问候/寒暄 → 跳过
            3. Dense Top-1 置信度极高（> HIGH_CONFIDENCE_THRESHOLD）→ 跳过
               此时向量检索已找到近乎完美的匹配，Rerank 不会改变结果

        不跳过的理由（为什么不能按"数量少"跳过）：
            - Lost in the Middle 效应：LLM 对文档位置敏感，Rerank 确保最相关文档在第1位
            - 过滤噪声：召回少不代表质量高，可能是"矮子里拔将军"
            - 缩短 Prompt：Rerank 过滤不相关文档 → Prompt 更短 → TTFT 更快
        """
        if not candidates:
            return True

        # 简单问候/寒暄类查询直接跳过Reranker
        simple_patterns = ["你好", "您好", "hello", "hi", "hey", "谢谢", "再见", "拜拜"]
        query_lower = (query or "").strip().lower()
        query_no_punct = re.sub(r"[，。！？,.!?\s]", "", query_lower) if query_lower else ""
        if query_no_punct in simple_patterns or len(query_no_punct) <= 4:
            return True

        # 策略1：High-Confidence Bypass
        # Dense Top-1 相似度极高 → 向量检索已找到近乎完美匹配，Rerank 不会改变结果
        # 阈值说明：ChromaDB cosine distance，0.0=完全相同，<0.08 对应 cosine_similarity>0.92
        # 注意：top1_dense_score >= 0（distance=0.0 是完美匹配，必须触发跳过）
        HIGH_CONFIDENCE_THRESHOLD = 0.08
        if 0 <= top1_dense_score < HIGH_CONFIDENCE_THRESHOLD:
            logger.info(
                f"Dense Top-1 置信度极高（distance={top1_dense_score:.4f} < {HIGH_CONFIDENCE_THRESHOLD}），跳过重排"
            )
            return True

        return False

    def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun,
            original_query: str = None,
            hyde_answer: str = None
    ) -> List[Document]:
        """混合检索主流程

        Args:
            query: 检索查询（重写后的查询，用于 BM25 稀疏检索）
            original_query: 原始用户查询（用于缓存 key）
            hyde_answer: HyDE 假想答案（用于 Dense 向量检索，提升语义召回率）
        """
        cache_query = original_query if original_query else query

        # HyDE 策略：Dense 用假想答案，Sparse 用原始查询
        dense_query = hyde_answer if hyde_answer else query
        sparse_query = query  # BM25 用关键词查询，假想答案会引入噪声

        if hyde_answer:
            logger.info(f"HyDE 模式：Dense 用假想答案（{len(hyde_answer)}字），Sparse 用查询")

        query_embedding = None
        semantic_cache = None
        semantic_lookup_ms = 0.0
        query_embedding_ms = 0.0

        # L2：语义相似缓存（Redis不可用时跳过，避免超时阻塞）
        # 注：已移除 L1 精确匹配缓存，L2 语义缓存完全覆盖 L1 功能
        #     精确匹配时相似度=100%，必然命中 L2（阈值92%）
        from app.cache.redis_cache import get_cache
        cache = get_cache()

        if getattr(config, 'ENABLE_SEMANTIC_CACHE', False) and cache._available:
            semantic_cache = get_semantic_cache()
            logger.info(f"L2语义缓存检查：{cache_query[:30]}...")

            # 优化：先快速检查L2是否有数据，为空则跳过Embedding API（省600ms+）
            try:
                # 使用 SCAN 替代 KEYS，避免阻塞 Redis
                l2_keys = []
                cursor = 0
                while True:
                    cursor, batch = cache._redis.scan(cursor, match=f"{semantic_cache.prefix}*", count=100)
                    l2_keys.extend(batch)
                    if cursor == 0:
                        break
                if not l2_keys:
                    logger.info("L2语义缓存为空，跳过Embedding计算")
                else:
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
            except Exception as l2_err:
                logger.warning(f"L2缓存检查异常，跳过：{l2_err}")

        # L2 缓存为空或未开启时，仍需计算 query_embedding 供 Dense 检索复用
        if query_embedding is None:
            # 优先查 Embedding LRU 缓存（命中时 0ms vs API 200~400ms）
            cached_embedding = _embedding_cache.get(dense_query)
            if cached_embedding is not None:
                query_embedding = cached_embedding
                query_embedding_ms = 0.0
                logger.info(f"Embedding LRU 缓存命中：{dense_query[:30]}...")
            else:
                # v9.0: Circuit Breaker 保护 Embedding API
                from app.core.circuit_breaker import get_circuit_breaker
                emb_cb = get_circuit_breaker("embedding_api", failure_threshold=3, recovery_timeout=30)

                if emb_cb.is_open:
                    # 熔断中，降级为 BM25-only 检索
                    logger.warning(f"Embedding API 熔断中，降级为 BM25-only 检索")
                    query_embedding = None
                    query_embedding_ms = 0.0
                else:
                    embedding_start = time.time()
                    try:
                        # HyDE：用假想答案做 embedding（语义空间更接近文档）
                        query_embedding = get_embeddings().embed_query(dense_query)
                        # 写入 LRU 缓存
                        if query_embedding is not None:
                            _embedding_cache.put(dense_query, query_embedding)
                        emb_cb.record_success()
                    except Exception as e:
                        logger.warning(f"查询向量预计算失败，将回退到文本检索：{e}")
                        query_embedding = None
                        emb_cb.record_failure()
                    query_embedding_ms = (time.time() - embedding_start) * 1000

        retrieval_start = time.time()
        dense_start = time.time()
        dense_docs, top1_dense_score = self._dense_search(dense_query, query_embedding=query_embedding)
        dense_ms = (time.time() - dense_start) * 1000

        sparse_start = time.time()
        sparse_docs = self._sparse_search(sparse_query)
        sparse_ms = (time.time() - sparse_start) * 1000

        fusion_start = time.time()
        candidates = self._reciprocal_rank_fusion(dense_docs, sparse_docs)
        fusion_ms = (time.time() - fusion_start) * 1000

        rerank_ms = 0.0
        # Reranker 跳过判断：基于 Dense Top-1 置信度，而非候选数量
        should_skip_reranker = self._should_skip_reranker(query, candidates, top1_dense_score)

        # 三阶段检索优化：RRF 融合后先轻量截断 top 8，再进 Reranker 精排 top k
        # 减少 Reranker 入参数量，CPU 推理时间从 970ms 降至 ~300ms
        RERANKER_INPUT_CAP = 8
        reranker_input = candidates[:RERANKER_INPUT_CAP]

        # Reranker 重排序
        if self.use_reranker and candidates and not should_skip_reranker:
            try:
                reranker = get_reranker()
                rerank_start = time.time()
                final_docs = reranker.rerank(
                    query=query,
                    documents=reranker_input,
                    top_k=self.k,
                    score_threshold=config.RERANKER_THRESHOLD
                )
                rerank_ms = (time.time() - rerank_start) * 1000
                logger.info(f"Reranker 重排序：{len(reranker_input)} -> {len(final_docs)}")
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

        # 父子索引：child → parent 映射 + 邻域扩展
        # 检索/重排的是 child chunk，送入 LLM 的是完整 parent 文档
        parent_lookup_ms = 0.0
        parent_manager = get_parent_child_manager()
        if parent_manager.is_initialized and final_docs:
            parent_lookup_start = time.time()
            parent_docs = parent_manager.get_parents(final_docs)
            parent_lookup_ms = (time.time() - parent_lookup_start) * 1000
            if parent_docs:
                # 邻域扩展：同一文档中的相邻章节自动补全
                # 解决跨章节信息缺失（如"头痛怎么办"需要"危险信号"+"药物选择"两个章节）
                expand_start = time.time()
                parent_docs = parent_manager.expand_with_siblings(
                    parent_docs,
                    sibling_window=getattr(config, "SIBLING_WINDOW", 1),
                    max_total_chars=getattr(config, "MAX_SIBLING_CHARS", 2000),
                )
                expand_ms = (time.time() - expand_start) * 1000
                parent_lookup_ms += expand_ms

                final_docs = parent_docs
                logger.info(f"父子索引生效：返回 {len(final_docs)} 个完整父文档")
            else:
                logger.warning("父子映射未取回任何 parent，使用 child 文档兜底")
        # 未初始化时直接使用 child 文档（兼容旧索引）

        logger.info(
            "检索完成，耗时：%.2fms（query_embedding=%.2fms, semantic_lookup=%.2fms, dense=%.2fms, sparse=%.2fms, fusion=%.2fms, rerank=%.2fms, parent_lookup=%.2fms, top1_dense_dist=%.4f）",
            retrieval_time,
            query_embedding_ms,
            semantic_lookup_ms,
            dense_ms,
            sparse_ms,
            fusion_ms,
            rerank_ms,
            parent_lookup_ms,
            top1_dense_score,
        )

        # 写入缓存
        if final_docs:
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
        rerank_top_k: int = 10
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
