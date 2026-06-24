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
import asyncio
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, StreamingResponse
import jieba

from app.cache.redis_cache import get_cache
from app.cache.semantic_cache import get_semantic_cache
from app.core.app_logging import get_logger
from app.core.config import get_config
from app.graph.graph import get_graph
from app.graph.streaming import StreamingOrchestrator
from app.memory import get_checkpointer, get_long_term_memory
from app.memory.checkpointer import close_checkpointer
from app.rag.hybrid_retriever import get_hybrid_retriever
from app.rag.reranker import get_reranker

logger = get_logger(__name__)

# 快照更新锁，防止同一 thread 的并发快照更新导致竞态
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


class FeedbackRequest(BaseModel):
    """用户反馈请求模型"""
    user_id: str = Field("default", description="用户ID")
    thread_id: Optional[str] = Field(None, description="会话线程ID")
    reason: str = Field(..., description="反馈原因：answer_inaccurate/not_answering/missing_info/unsafe_content/other")
    note: Optional[str] = Field(None, description="补充说明", max_length=500)
    answer_preview: Optional[str] = Field(None, description="AI回答预览（前500字）")


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

    # 2. 预加载 Reranker 模型
    logger.info("预加载 Reranker 模型...")
    get_reranker()  # 触发模型加载
    logger.info("Reranker 模型预加载完成")

    # 3. 预热向量库和 BM25
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

    # 启动 L1 本地缓冲后台 flush
    try:
        from app.memory.fallback_buffer import start_background_flush
        start_background_flush()
    except Exception as e:
        logger.warning(f"启动 L1 缓冲 flush 失败：{e}")

    yield

    # 关闭时清理
    try:
        from app.memory.fallback_buffer import stop_background_flush
        stop_background_flush()
    except Exception as e:
        logger.warning(f"停止 L1 缓冲 flush 失败：{e}")
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
    """首页"""
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


@app.post("/api/chat/stream")
async def stream(request: ChatRequest, http_request: Request):
    """流式聊天接口（SSE）

    编排逻辑已迁移至 app.graph.streaming.StreamingOrchestrator，
    本端点仅负责参数提取和 SSE 响应包装。
    """
    request_id = getattr(http_request.state, "request_id", str(uuid.uuid4()))
    request_start_time = getattr(http_request.state, "request_start_time", time.time())
    thread_id = request.thread_id or f"thread_{request.user_id or 'default'}"

    logger.info(
        f"收到流式聊天请求：request_id={request_id}, thread_id={thread_id}, user_id={request.user_id}"
    )

    orchestrator = StreamingOrchestrator(
        question=request.question,
        user_id=request.user_id,
        thread_id=thread_id,
        image_base64=request.image_base64,
        request_id=request_id,
        request_start_time=request_start_time,
        snapshot_lock=_get_snapshot_lock(thread_id),
    )

    return StreamingResponse(
        orchestrator.run(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Request-ID": request_id,
        },
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


@app.post("/api/admin/reload-config")
async def reload_config(request: Request):
    """热更新配置（需管理员认证）

    重新读取 .env 文件，更新运行时配置，无需重启服务。
    可热更新的配置包括：缓存 TTL、速率限制、模型参数、特性开关等。
    """
    if not _verify_admin_key(request):
        return JSONResponse(status_code=403, content={"detail": "无权访问，请提供有效的 X-Admin-API-Key"})
    from app.core.config import reload_config as do_reload
    result = do_reload()
    if result.get("error"):
        return JSONResponse(status_code=500, content={"detail": result["error"]})
    return {
        "reloaded": result["reloaded"],
        "changed_fields": result["changed"],
        "message": f"配置已重新加载，{len(result['changed'])} 个字段发生变化" if result["changed"] else "配置已重新加载，无字段变化",
    }


@app.post("/api/feedback")
async def submit_feedback(request: Request):
    """用户反馈接口

    接收用户对 AI 回答的反馈（👍/👎），将负面反馈写入 bad_cases 存储。

    反馈原因：
        - answer_inaccurate: 答案不准确
        - not_answering: 没回答我的问题
        - missing_info: 缺少关键信息
        - unsafe_content: 内容不安全
        - other: 其他
    """
    try:
        body = await request.json()
        feedback = FeedbackRequest(**body)
    except Exception:
        return JSONResponse(status_code=400, content={"detail": "请求格式错误"})

    try:
        from app.memory import get_long_term_memory
        memory = get_long_term_memory()
        memory.append_bad_case(
            case_type="user_negative_feedback",
            original_query="",  # 前端未传原始问题，可通过 thread_id 查询
            rewritten_query="",
            final_question="",
            answer_preview=feedback.answer_preview or "",
            user_id=feedback.user_id,
            thread_id=feedback.thread_id or "",
            metadata={
                "reason": feedback.reason,
                "note": feedback.note or "",
            },
        )
        logger.info(
            f"用户反馈已记录：user_id={feedback.user_id}, "
            f"thread_id={feedback.thread_id}, reason={feedback.reason}"
        )
    except Exception as e:
        logger.warning(f"用户反馈记录失败：{e}")
        # 不影响用户体验，静默处理

    return {"status": "ok", "message": "反馈已收到"}


# 启动入口
if __name__ == '__main__':
    import os
    import uvicorn

    # Ollama 优化：缩减上下文窗口加速本地模型推理（默认4096太慢）
    os.environ.setdefault("OLLAMA_NUM_CTX", "1024")
    uvicorn.run("app.api.routes:app", host="0.0.0.0", port=8000, reload=True)
