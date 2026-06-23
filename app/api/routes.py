"""FastAPI路由模块
功能描述：
    提供医疗助手的REST API接口
    支持同步和流式响应

设计理念：
    1、FastAPI：现代化异步框架，自动生成OpenAPI文档
    2、Pydantic：请求/响应模型参数
    3、SSE流式输出：支持Server-Sent Events实时推送
    4、错误处理：统一的异常处理中间件

API接口：
    POST /api/chat：同步聊天接口
    POST /api/chat/stream：流式聊天接口
    GET /api/health：健康检查
"""
from contextlib import asynccontextmanager
from typing import Optional, List, Dict
import json
import time
import uuid
import asyncio
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.app_logging import get_logger
from app.graph.graph import get_graph
from app.graph.nodes import (
    stream_answer_generation,
    stream_direct_answer,
    stream_vision_answer,
    memory_load_node,
    profile_extraction_node,
    router_node,
    symptom_analysis_node,
    query_rewrite_node,
    knowledge_retrieval_node,
    grade_documents_node,
)
from app.memory import get_checkpointer, get_long_term_memory
from app.memory.checkpointer import close_checkpointer
from app.rag.hybrid_retriever import get_hybrid_retriever
from app.rag.reranker import get_reranker
from app.cache.redis_cache import get_cache
from app.cache.semantic_cache import get_semantic_cache
from app.core.config import get_config
from pathlib import Path
from fastapi.staticfiles import StaticFiles
import jieba

logger = get_logger(__name__)

# Per-thread 快照更新锁，防止并发快照更新导致竞态
_snapshot_locks: Dict[str, asyncio.Lock] = {}


def _get_snapshot_lock(thread_id: str) -> asyncio.Lock:
    """获取指定线程的快照更新锁（懒创建）"""
    if thread_id not in _snapshot_locks:
        _snapshot_locks[thread_id] = asyncio.Lock()
    return _snapshot_locks[thread_id]


# Pydantic模型
class ChatRequest(BaseModel):
    """聊天请求模型"""
    question: str = Field(..., description="用户问题", min_length=1, max_length=1000)
    user_id: Optional[str] = Field(None, description="用户ID")
    thread_id: Optional[str] = Field(None, description="会话线程ID")
    image_base64: Optional[str] = Field(None, description="图片base64编码（多模态问诊）", max_length=10_000_000)


class SourceInfo(BaseModel):
    """来源信息模型"""
    source: str
    file_path: Optional[str] = None
    content: Optional[str] = None


class ChatResponse(BaseModel):
    """聊天响应模型"""
    answer: Optional[str] = Field(None, description="回答内容")
    sources: Optional[List[SourceInfo]] = Field(None, description="来源信息")
    warnings: List[str] = Field(default_factory=list, description="警告信息")


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str
    database: str
    vector_store: str
    cache: str
    reranker: str


# 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logger.info(f"应用启动中...")
    await get_checkpointer()
    logger.info(f"检查点保存器初始化完成")

    # 1. 预加载jieba分词器
    logger.info("预加载 jieba 分词器...")
    jieba.initialize()
    logger.info("jieba 分词器预加载完成")

    # 2. ✅ 新增：预加载 Reranker 模型
    logger.info("预加载 Reranker 模型...")
    get_reranker()  # 触发模型加载
    logger.info("Reranker 模型预加载完成")

    # 3. ✅ 可选：预热向量库和 BM25
    logger.info("预热混合检索器...")
    get_hybrid_retriever()
    logger.info("混合检索器预热完成")

    # 4. 预热Redis缓存和语义缓存（避免首次请求2s连接延迟）
    logger.info("预热缓存连接...")
    try:
        cache = get_cache()
        if cache._available:
            logger.info(f"Redis缓存预热成功：{cache.redis_url}")
        else:
            logger.info("Redis不可用，使用内存缓存降级")
    except Exception as e:
        logger.warning(f"Redis缓存预热失败：{e}")
    try:
        config = get_config()
        if getattr(config, 'ENABLE_SEMANTIC_CACHE', False):
            semantic_cache = get_semantic_cache()
            logger.info("语义缓存预热成功")
    except Exception as e:
        logger.warning(f"语义缓存预热失败：{e}")
    logger.info("缓存预热完成")

    # 预热本地模型（Ollama），避免首次请求冷启动
    try:
        config = get_config()
        if getattr(config, 'LOCAL_MODEL_ENABLED', False):
            logger.info("预热本地模型...")
            from app.core.llm import get_local_llm
            local_llm = get_local_llm()
            local_llm.invoke("你好")  # 简单请求触发模型加载
            logger.info("本地模型预热完成")
    except Exception as e:
        logger.warning(f"本地模型预热失败（不影响功能，首次请求会稍慢）：{e}")

    # 验证流式接口与 Graph 节点定义同步
    try:
        from app.graph.graph import validate_streaming_sync
        validate_streaming_sync()
    except Exception as e:
        logger.warning(f"流式接口同步验证失败：{e}")

    yield

    # 关闭时清理
    logger.info(f"应用关闭中")
    await close_checkpointer()
    # 关闭长期记忆存储器连接
    try:
        from app.memory.long_term_memory import reset_long_term_memory
        reset_long_term_memory()
        logger.info("长期记忆存储器连接已关闭")
    except Exception as e:
        logger.warning(f"关闭长期记忆存储器失败：{e}")
    logger.info(f"资源清理完成")


# FastAPI应用
app = FastAPI(
    title="Medical Assistant API",
    description="医疗助手智能问答系统 API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS中间件
# 安全：限制允许的来源，避免 CSRF 攻击
# 生产环境应通过 CORS_ORIGINS 环境变量指定具体域名
_config = get_config()
_cors_origins = getattr(_config, 'CORS_ORIGINS', '').split(',') if getattr(_config, 'CORS_ORIGINS', '') else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True if _cors_origins != ["*"] else False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== 速率限制中间件 =====
class RateLimitMiddleware(BaseHTTPMiddleware):
    """基于令牌桶算法的简易速率限制

    配置项（通过环境变量）：
        RATE_LIMIT_PER_MINUTE: 每分钟最大请求数，默认 20
    仅对 /api/chat 开头的接口生效，健康检查等不受限制。
    """

    def __init__(self, app, max_requests: int = 20, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # {client_ip: [timestamp1, timestamp2, ...]}
        self._requests: Dict[str, List[float]] = defaultdict(list)

    def _cleanup(self, ip: str, now: float):
        """清理过期的请求记录"""
        cutoff = now - self.window_seconds
        self._requests[ip] = [t for t in self._requests[ip] if t > cutoff]

    async def dispatch(self, request: Request, call_next):
        # 仅限制聊天接口
        if not request.url.path.startswith("/api/chat"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        self._cleanup(client_ip, now)

        if len(self._requests[client_ip]) >= self.max_requests:
            logger.warning(f"速率限制触发：{client_ip} 在 {self.window_seconds}s 内超过 {self.max_requests} 次请求")
            return JSONResponse(
                status_code=429,
                content={"detail": f"请求过于频繁，请 {self.window_seconds} 秒后重试"}
            )

        self._requests[client_ip].append(now)
        return await call_next(request)


_rate_limit = getattr(get_config(), 'RATE_LIMIT_PER_MINUTE', 20)
app.add_middleware(RateLimitMiddleware, max_requests=_rate_limit)


@app.middleware("http")
async def request_timing_middleware(request: Request, call_next):
    """记录请求级耗时与 request_id。"""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    request.state.request_start_time = time.time()

    response = await call_next(request)

    elapsed_ms = (time.time() - request.state.request_start_time) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-MS"] = f"{elapsed_ms:.2f}"
    logger.info(f"请求完成：request_id={request_id}, path={request.url.path}, status={response.status_code}, elapsed_ms={elapsed_ms:.2f}")
    return response


# 在 app 定义后添加
STATIC_DIR = Path(__file__).parent.parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
# 挂载静态文件
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# 添加首页路由
@app.get("/")
async def root():
    """首页重定向"""
    from fastapi.responses import FileResponse
    return FileResponse(str(STATIC_DIR / "index.html"))


# 异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理：生产环境不泄露内部信息"""
    logger.error(f"未处理的异常：{exc}", exc_info=True)
    config = get_config()
    if getattr(config, 'DEBUG', False):
        # 开发模式返回详细错误
        detail = f"服务器内部错误：{str(exc)}"
    else:
        # 生产环境返回通用消息
        detail = "服务器内部错误，请稍后重试"
    return JSONResponse(
        status_code=500,
        content={"detail": detail}
    )


# API路由
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """同步聊天窗口
    功能描述：
        接受用户问题，返回完整回答

    Args：
        request：聊天请求，包含用户和可选的用户id

    Returns：
        ChatResponse：包含回答、来源和警告信息
    """
    logger.info(f"收到聊天请求：user_id={request.user_id}, question={request.question[:50]}...")

    try:
        graph = await get_graph()
        checkpointer = await get_checkpointer()

        try:
            memory = get_long_term_memory()
            store = memory.store
        except Exception:
            store = None

        config = {
            "configurable": {
                "thread_id": request.thread_id or f"thread_{request.user_id or 'default'}",
                "user_id": request.user_id,
                "store": store,
            }
        }

        input_state = {
            "question": request.question,
            "user_id": request.user_id,
        }

        result = await graph.ainvoke(input_state, config)

        sources = None
        if result.get("sources"):
            sources = [
                SourceInfo(
                    source=s.get("source", "未知"),
                    file_path=s.get("file_path"),
                    content=s.get("content"),
                )
                for s in result.get("sources")
            ]

        return ChatResponse(
            answer=result.get("final_answer"),
            sources=sources,
            warnings=result.get("warnings", [])
        )

    except Exception as e:
        logger.error(f"聊天处理失败：{e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# L0 答案缓存TTL（秒），比文档缓存短
_ANSWER_CACHE_TTL = 1800  # 30分钟


def _save_answer_cache(question: str, answer: str):
    """将完整答案写入L0答案缓存（仅无用户档案时调用）"""
    try:
        cache = get_cache()
        if cache._available:
            answer_cache_key = cache.prefix + f"answer:{question}"
            cache._redis.setex(answer_cache_key, _ANSWER_CACHE_TTL, json.dumps(answer, ensure_ascii=False))
            logger.info(f"L0答案缓存已写入：{question[:30]}...")
    except Exception as e:
        logger.warning(f"L0答案缓存写入失败：{e}")


@app.post("/api/chat/stream")
async def stream(request: ChatRequest, http_request: Request):
    """流式聊天接口
    功能描述：
        接受用户问题，以SSE流式返回回答

    Args：
        request：聊天请求

    Returns：
        StreamResponse：SSE流式响应
    """
    request_id = getattr(http_request.state, "request_id", str(uuid.uuid4()))
    request_start_time = getattr(http_request.state, "request_start_time", time.time())
    thread_id = request.thread_id or f"thread_{request.user_id or 'default'}"

    logger.info(
        f"收到流式聊天请求：request_id={request_id}, thread_id={thread_id}, user_id={request.user_id}"
    )

    async def event_generator():
        first_token_sent = False
        full_answer = ""  # 初始化，确保保存对话历史时可访问
        current_state = {
            "question": request.question,
            "user_id": request.user_id,
            "image_base64": request.image_base64,
            "thread_id": thread_id,
            "request_id": request_id,
            "warnings": [],
            "sources": [],
            "retrieval_attempts": 0,
        }

        async def emit_data(payload):
            nonlocal first_token_sent
            if not first_token_sent:
                first_token_latency_ms = (time.time() - request_start_time) * 1000
                logger.info(
                    f"首个 token 已发送：request_id={request_id}, thread_id={thread_id}, latency_ms={first_token_latency_ms:.2f}"
                )
                first_token_sent = True
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        try:
            memory_state = memory_load_node(current_state)
            if memory_state:
                current_state.update(memory_state)

            # 从 LangGraph 加载对话历史（短期记忆），供 build_rag_prompt 注入
            try:
                graph = await get_graph()
                checkpoint_config = {"configurable": {"thread_id": thread_id}}
                state_snapshot = await graph.aget_state(checkpoint_config)
                if state_snapshot and state_snapshot.values:
                    if "messages" in state_snapshot.values:
                        current_state["messages"] = state_snapshot.values["messages"]
                        logger.info(f"从 checkpointer 加载对话历史：{len(state_snapshot.values['messages'])} 条消息")
                    if "clinical_checkpoint" in state_snapshot.values:
                        current_state["clinical_checkpoint"] = state_snapshot.values["clinical_checkpoint"]
            except Exception as e:
                logger.warning(f"加载对话历史失败（不影响当前请求）：{e}")

            # ===== 并行：缓存检查 + 路由同时执行 =====
            # 缓存命中则丢弃路由结果；缓存未命中则路由已完成，不浪费时间
            cached_docs = None
            cached_answer = None
            route_command = None
            has_profile = bool(current_state.get("user_profile"))

            if request.image_base64:
                # 图片问诊不走缓存，直接路由
                route_command = router_node(current_state)
            else:
                import asyncio
                cache_check_start = time.time()

                async def _check_cache():
                    """异步执行缓存检查（L0 → L2）"""
                    nonlocal cached_docs, cached_answer
                    try:
                        cache = get_cache()

                        # L0 答案缓存（仅无用户档案时）
                        if not has_profile:
                            answer_cache_key = f"answer:{request.question}"
                            try:
                                if cache._available:
                                    cached_raw = cache._redis.get(cache.prefix + answer_cache_key)
                                    if cached_raw:
                                        cached_answer = json.loads(cached_raw)
                                        logger.info(f"⚡ L0答案缓存命中，直接返回：{request.question[:30]}...")
                                        return
                                    else:
                                        logger.info(f"L0答案缓存未命中：{request.question[:30]}...")
                            except Exception:
                                pass
                        else:
                            logger.info(f"L0答案缓存跳过（用户有档案，答案个性化不可缓存）")

                        # L2 语义相似匹配（文档级）
                        # 注：已移除 L1 精确匹配缓存，L2 完全覆盖 L1 功能
                        config = get_config()
                        if getattr(config, 'ENABLE_SEMANTIC_CACHE', False) and cache._available:
                            semantic_cache = get_semantic_cache()
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
                                    query_embedding = semantic_cache.get_embedding(request.question)
                                    l2_result = semantic_cache.get(request.question, query_embedding=query_embedding)
                                    if l2_result:
                                        cached_docs, l2_meta = l2_result
                                        logger.info(
                                            f"⚡ 早期缓存L2命中(相似度{l2_meta.get('similarity', 0):.1%})，跳过路由+重写：{request.question[:30]}..."
                                        )
                            except Exception as l2_err:
                                logger.warning(f"L2缓存检查异常，跳过：{l2_err}")
                    except Exception as cache_err:
                        logger.warning(f"缓存检查异常：{cache_err}")

                async def _run_route():
                    """异步执行路由（在线程池中运行同步函数）"""
                    nonlocal route_command
                    loop = asyncio.get_event_loop()
                    route_command = await loop.run_in_executor(None, router_node, dict(current_state))

                # 并行执行缓存检查和路由
                await asyncio.gather(_check_cache(), _run_route())

                cache_check_ms = (time.time() - cache_check_start) * 1000
                logger.info(f"并行（缓存+路由）耗时：{cache_check_ms:.2f}ms")

            if cached_answer:
                # L0答案缓存命中：直接返回完整答案，跳过所有处理
                full_answer = cached_answer  # 保存到 full_answer，确保后续写入 checkpointer
                yield await emit_data(cached_answer)
            elif cached_docs:
                # L2文档缓存命中：跳过路由、症状解析、重写，直接用缓存文档生成答案
                current_state["retrieved_docs"] = cached_docs
                current_state["rewritten_query"] = request.question
                logger.info(f"缓存命中，直接进入答案生成：request_id={request_id}, docs={len(cached_docs)}")
                # 推送文档来源元数据
                sources = [
                    {
                        "source": doc.metadata.get("source", "未知来源"),
                        "file_path": doc.metadata.get("file_path", ""),
                    }
                    for doc in cached_docs
                ]
                if sources:
                    yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"
                full_answer = ""
                async for token in stream_answer_generation(current_state):
                    full_answer += token
                    yield await emit_data(token)
                if full_answer and not has_profile:
                    _save_answer_cache(request.question, full_answer)
            else:
                # 缓存未命中：路由已完成，直接使用路由结果
                # ⚠️ 同步提醒：以下节点编排顺序必须与 app/graph/graph.py 中的边定义一致
                # 修改 Graph 节点/边时，必须同步更新此处的编排逻辑
                # 启动时 validate_streaming_sync() 会自动检测不一致
                next_node = getattr(route_command, "goto", "direct_answer") if route_command else "direct_answer"
                logger.info(
                    f"流式请求路由完成：request_id={request_id}, thread_id={thread_id}, next_node={next_node}"
                )

                if next_node == "direct_answer":
                    async for token in stream_direct_answer(current_state):
                        yield await emit_data(token)

                elif next_node == "vision_analysis":
                    # 图片问诊分支（参考蚂蚁阿福的多模态问诊）
                    logger.info(f"开始流式图片问诊：request_id={request_id}, thread_id={thread_id}")
                    async for token in stream_vision_answer(current_state):
                        yield await emit_data(token)

                else:
                    if next_node == "symptom_analysis":
                        symptom_state = symptom_analysis_node(current_state)
                        if symptom_state:
                            current_state.update(symptom_state)

                    rewrite_state = query_rewrite_node(current_state)
                    if rewrite_state:
                        current_state.update(rewrite_state)

                    retrieval_state = knowledge_retrieval_node(current_state)
                    if retrieval_state:
                        current_state.update(retrieval_state)

                    grade_command = grade_documents_node(current_state)
                    grade_update = getattr(grade_command, "update", None) or {}
                    if grade_update:
                        current_state.update(grade_update)

                    grade_next_node = getattr(grade_command, "goto", "answer_generation")
                    logger.info(
                        f"文档评分完成：request_id={request_id}, thread_id={thread_id}, next_node={grade_next_node}, docs={len(current_state.get('retrieved_docs') or [])}"
                    )

                    if grade_next_node == "query_rewrite":
                        rewrite_state = query_rewrite_node(current_state)
                        if rewrite_state:
                            current_state.update(rewrite_state)

                        retrieval_state = knowledge_retrieval_node(current_state)
                        if retrieval_state:
                            current_state.update(retrieval_state)

                        grade_command = grade_documents_node(current_state)
                        grade_update = getattr(grade_command, "update", None) or {}
                        if grade_update:
                            current_state.update(grade_update)
                        grade_next_node = getattr(grade_command, "goto", "answer_generation")

                    if grade_next_node == "answer_generation" and current_state.get("final_answer"):
                        yield await emit_data(current_state["final_answer"])
                        if not has_profile:
                            _save_answer_cache(request.question, current_state["final_answer"])
                    elif current_state.get("retrieved_docs"):
                        logger.info(f"开始流式生成 RAG 答案：request_id={request_id}, thread_id={thread_id}")
                        # 推送文档来源元数据（独立于LLM回答，前端单独渲染）
                        sources = [
                            {
                                "source": doc.metadata.get("source", "未知来源"),
                                "file_path": doc.metadata.get("file_path", ""),
                            }
                            for doc in current_state.get("retrieved_docs", [])
                        ]
                        if sources:
                            yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"
                        full_answer = ""
                        async for token in stream_answer_generation(current_state):
                            full_answer += token
                            yield await emit_data(token)
                        if full_answer and not has_profile:
                            _save_answer_cache(request.question, full_answer)
                    else:
                        logger.info(f"无检索文档，降级为流式直接回答：request_id={request_id}, thread_id={thread_id}")
                        async for token in stream_direct_answer(current_state):
                            yield await emit_data(token)

            total_elapsed_ms = (time.time() - request_start_time) * 1000
            logger.info(
                f"流式请求完成：request_id={request_id}, thread_id={thread_id}, total_elapsed_ms={total_elapsed_ms:.2f}"
            )
            yield "data: [DONE]\n\n"

            # 保存对话历史到 checkpointer，确保下一轮能读取
            try:
                from langchain_core.messages import HumanMessage, AIMessage
                from app.graph.nodes import should_update_snapshot, update_clinical_snapshot_node
                graph = await get_graph()
                checkpoint_config = {"configurable": {"thread_id": thread_id}}

                # 追加用户消息
                await graph.aupdate_state(
                    checkpoint_config,
                    {"messages": [HumanMessage(content=request.question)]},
                    as_node="memory_load",
                )

                # 追加AI回答
                if full_answer:
                    await graph.aupdate_state(
                        checkpoint_config,
                        {"messages": [AIMessage(content=full_answer)]},
                        as_node="answer_generation",
                    )

                # 快照更新改为后台异步任务，不阻塞当前响应
                # 原因：快照更新需要调用LLM（3-5s），用户已看到回答无需等待
                # 使用 asyncio.create_task 让快照更新在后台执行
                # Per-thread 锁：防止同一会话的并发快照更新导致竞态
                async def _background_snapshot_update():
                    lock = _get_snapshot_lock(thread_id)
                    # 非阻塞获取锁：尝试在极短时间内获取，失败则跳过
                    try:
                        await asyncio.wait_for(lock.acquire(), timeout=0.01)
                    except asyncio.TimeoutError:
                        logger.info(f"快照更新已在进行中，跳过本次（thread_id={thread_id}）")
                        return
                    try:
                        # 重新读取最新状态（可能在等待锁期间已发生变化）
                        state_snapshot = await graph.aget_state(checkpoint_config)
                        if not state_snapshot or not state_snapshot.values:
                            return
                        bg_state = {
                            "messages": state_snapshot.values.get("messages", []),
                            "clinical_checkpoint": state_snapshot.values.get("clinical_checkpoint"),
                        }
                        # 再次检查是否仍需更新（可能之前的快照已经处理了）
                        snapshot_decision = should_update_snapshot(bg_state)
                        if snapshot_decision != "update_snapshot":
                            logger.info(f"快照已由前次任务处理，无需重复更新")
                            return
                        logger.info("后台触发临床状态快照更新")
                        snapshot_result = update_clinical_snapshot_node(bg_state)
                        if snapshot_result:
                            await graph.aupdate_state(
                                checkpoint_config,
                                snapshot_result,
                                as_node="update_snapshot",
                            )
                            logger.info(f"临床状态快照已更新，删除 {len(snapshot_result.get('messages', []))} 条早期消息")
                    except Exception as e:
                        logger.warning(f"后台快照更新失败（不影响后续对话）：{e}")
                    finally:
                        lock.release()

                asyncio.create_task(_background_snapshot_update())

                logger.info(f"对话历史已保存")
            except Exception as e:
                logger.warning(f"保存对话历史失败（不影响当前回答）：{e}")

            # 优化：档案提取移到回答完成之后，异步执行不阻塞用户
            try:
                profile_state = profile_extraction_node(current_state)
                if profile_state:
                    current_state.update(profile_state)
                    logger.info(f"档案提取完成（回答后）：request_id={request_id}")
            except Exception as profile_err:
                logger.warning(f"档案提取失败（不影响回答）：{profile_err}")

        except Exception as e:
            logger.error(f"流式处理失败：request_id={request_id}, error={e}", exc_info=True)
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Request-ID": request_id,
        }
    )


@app.get("/api/health", response_model=HealthResponse)
async def health():
    """健康检查接口
    功能描述：
        检查服务各组件状态

    Returns：
        HealthResponse：包含各组件状态
    """
    database_status = "healthy"
    vector_store_status = "healthy"
    cache_status = "healthy"
    reranker_status = "healthy"

    # 检查数据库连接
    try:
        memory = get_long_term_memory()
        if memory.store is None:
            database_status = "degraded"
    except Exception as e:
        database_status = f"unhealthy: {str(e)[:50]}"

    # 检查向量库
    try:
        from app.rag.vector_store import get_vector_store
        vs = get_vector_store()
        if vs is None:
            vector_store_status = "degraded"
    except Exception as e:
        vector_store_status = f"unhealthy: {str(e)[:50]}"

    # 检查缓存
    try:
        from app.cache.redis_cache import get_cache
        cache = get_cache()
        cache_health_result = cache.health_check()
        cache_status = cache_health_result.get("status", "unknown")
    except Exception as e:
        cache_status = f"unhealthy: {str(e)[:50]}"

    # 检查 reranker
    try:
        reranker = get_reranker()
        reranker_status = "healthy" if getattr(reranker, "_available", False) else "degraded"
    except Exception as e:
        reranker_status = f"unhealthy: {str(e)[:50]}"

    component_statuses = [database_status, vector_store_status, cache_status, reranker_status]
    if any(status.startswith("unhealthy") for status in component_statuses):
        overall_status = "unhealthy"
    elif any(status != "healthy" for status in component_statuses):
        overall_status = "degraded"
    else:
        overall_status = "healthy"

    return HealthResponse(
        status=overall_status,
        database=database_status,
        vector_store=vector_store_status,
        cache=cache_status,
        reranker=reranker_status,
    )


@app.get("/api/cache/stats")
async def cache_stats():
    """获取缓存统计信息"""
    from app.cache.redis_cache import get_cache
    cache = get_cache()
    return cache.get_stats()


@app.get("/api/cache/health")
async def cache_health():
    """缓存健康检查"""
    from app.cache.redis_cache import get_cache
    cache = get_cache()
    return cache.health_check()


def _verify_admin_key(request: Request) -> bool:
    """验证管理员 API Key（用于缓存管理等敏感接口）"""
    config = get_config()
    admin_key = getattr(config, 'ADMIN_API_KEY', '')
    request_key = request.headers.get("X-Admin-API-Key", "")
    if not admin_key or admin_key == "admin-api-key-change-in-production":
        # 未配置安全密钥时，仅允许本地访问
        return request.client.host in ("127.0.0.1", "::1", "localhost")
    return request_key == admin_key


@app.post("/api/cache/clear")
async def clear_cache(request: Request):
    """清空缓存（需管理员认证）"""
    if not _verify_admin_key(request):
        return JSONResponse(status_code=403, content={"detail": "无权访问，请提供有效的 X-Admin-API-Key"})
    from app.cache.redis_cache import get_cache
    cache = get_cache()
    count = cache.clear()
    return {"cleared": count, "message": f"已清空 {count} 条缓存"}


@app.delete("/api/cache/{query}")
async def delete_cache(query: str, request: Request):
    """删除指定查询的缓存（需管理员认证）"""
    if not _verify_admin_key(request):
        return JSONResponse(status_code=403, content={"detail": "无权访问，请提供有效的 X-Admin-API-Key"})
    from app.cache.redis_cache import get_cache
    cache = get_cache()
    success = cache.delete(query)
    return {"deleted": success}


# 启动入口
if __name__ == '__main__':
    import os
    import uvicorn

    # Ollama 优化：缩减上下文窗口加速本地模型推理（默认4096太慢）
    os.environ.setdefault("OLLAMA_NUM_CTX", "1024")
    uvicorn.run("app.api.routes:app", host="0.0.0.0", port=8000, reload=True)
