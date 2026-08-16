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
import json
import shutil
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
from app.rag.loader import LOADERS

logger = get_logger(__name__)


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
    request_id: str = Field("", description="关联的请求ID（从 SSE 事件获取）")
    rating: str = Field("up", description="反馈类型：up(👍)/down(👎)")
    reason: str = Field("", description="差评原因：answer_inaccurate/not_answering/missing_info/unsafe_content/other")
    note: Optional[str] = Field(None, description="补充说明", max_length=500)
    answer_preview: Optional[str] = Field(None, description="AI回答预览（前500字）")
    question: Optional[str] = Field(None, description="用户原始问题（用于差评关联）")
    user_id: Optional[str] = Field("default", description="用户ID")
    thread_id: Optional[str] = Field(None, description="会话线程ID")


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

    # v9.16: Reranker 预热推理（消除首次请求 ~4s 冷启动延迟）
    logger.info("预热 Reranker 推理...")
    from app.rag.reranker import warmup_reranker
    warmup_reranker()
    logger.info("Reranker 预热完成")

    # 3. 预热向量库和 BM25
    logger.info("预热混合检索器...")
    get_hybrid_retriever(k=3, alpha=0.5, use_reranker=True, rerank_top_k=8)
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


@app.get("/admin")
async def admin_page():
    """知识库管理页面"""
    return FileResponse(str(STATIC_DIR / "admin.html"))


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


def _resolve_thread_id(thread_id: Optional[str], user_id: Optional[str]) -> str:
    """解析会话线程 ID

    P2-7：匿名用户（thread_id 与 user_id 均为空）不再共用 thread_default，
    否则跨用户的医疗对话在 checkpointer 中互相加载。匿名请求各生成独立会话。
    """
    if thread_id:
        return thread_id
    if user_id:
        return f"thread_{user_id}"
    return f"thread_anon_{uuid.uuid4().hex}"


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
                "thread_id": _resolve_thread_id(request.thread_id, request.user_id),
                "user_id": request.user_id,
                "store": store,
            }
        }

        input_state = {
            "question": request.question,
            "user_id": request.user_id,
            # 重置残留中间状态，防止同一 thread 复用上一轮 final_answer
            "final_answer": None,
            "retrieved_docs": None,
            "all_retrieved_docs": None,
            "rewritten_query": None,
            "final_question": None,
            "symptoms": None,
            "question_type": None,
            "retrieval_attempts": 0,
            "retrieval_confidence": None,
            "refusal_type": None,
            "sub_questions": None,
            "hyde_answer": None,
            "error": None,
            # P2-4：重置输出字段，防止 checkpointer 恢复上一轮 warnings/sources 无限累积
            "warnings": None,
            "sources": None,
        }

        result = await graph.ainvoke(input_state, config)

        # H11 修复：ENABLE_SAFETY_CHECK=False 时图内无 safety_check 边，
        # 非流式路径需与流式路径（streaming.py _run_safety_review）一致做后置安全审查，
        # 否则同步接口返回的答案跳过全部规则引擎/用药核查/LLM 审查
        cfg = get_config()
        if not getattr(cfg, "ENABLE_SAFETY_CHECK", True) and result.get("final_answer"):
            try:
                from app.graph.nodes import safety_check_node
                review_state = {
                    "final_answer": result.get("final_answer"),
                    "clinical_checkpoint": result.get("clinical_checkpoint"),
                    "user_profile": result.get("user_profile"),
                    "question": result.get("question") or request.question,
                    "symptoms": result.get("symptoms"),
                }
                review_result = safety_check_node(review_state)
                if review_result.get("final_answer"):
                    result["final_answer"] = review_result["final_answer"]
                for w in review_result.get("warnings", []):
                    if w not in (result.get("warnings") or []):
                        result.setdefault("warnings", []).append(w)
            except Exception as e:
                logger.warning(f"非流式后置安全审查失败（保留原始答案）：{e}")

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
    thread_id = _resolve_thread_id(request.thread_id, request.user_id)

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


def _sanitize_kb_filename(raw_name: Optional[str]) -> Optional[str]:
    """净化知识库上传/删除的文件名，防止路径穿越到 docs_dir 之外

    拒绝：路径分隔符（/ \\）、盘符（C:）、点目录（. ..）、空名。
    返回净化后的纯文件名，非法输入返回 None。
    """
    if not raw_name:
        return None
    if ("/" in raw_name or "\\" in raw_name
            or raw_name in (".", "..")
            or (len(raw_name) > 1 and raw_name[1] == ":")):
        return None
    name = Path(raw_name).name
    if not name or name in (".", ".."):
        return None
    return name


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

    接收用户对 AI 回答的反馈（👍/👎）：
    - 👍：记录满意度数据
    - 👎：记录差评 + 自动创建 Bad Case + 可转化为黄金测试集

    请求体：
        request_id: 关联的请求ID（从 SSE done 事件获取）
        rating: "up" (👍) 或 "down" (👎)
        reason: 差评原因（answer_inaccurate/not_answering/missing_info/unsafe_content/other）
        note: 用户补充说明
        answer_preview: AI 回答预览
        question: 原始问题
        user_id: 用户ID
        thread_id: 会话线程ID

    反馈闭环流程：
        用户 👎 → record_feedback() → 自动 append_bad_case()
        → 人工审核 → 补填 ground_truth → 加入黄金测试集
        → 每次系统迭代后重跑评估 → 验证修复
    """
    try:
        body = await request.json()
        feedback = FeedbackRequest(**body)
    except Exception:
        return JSONResponse(status_code=400, content={"detail": "请求格式错误"})

    # 校验 rating
    if feedback.rating not in ("up", "down"):
        return JSONResponse(status_code=400, content={"detail": "rating 必须为 up 或 down"})

    try:
        from app.core.metrics import get_metrics_collector
        collector = get_metrics_collector()
        feedback_id = collector.record_feedback(
            request_id=feedback.request_id,
            rating=feedback.rating,
            reason=feedback.reason,
            note=feedback.note or "",
            answer_preview=feedback.answer_preview or "",
            question=feedback.question or "",
            user_id=feedback.user_id or "default",
            thread_id=feedback.thread_id or "",
        )
        logger.info(
            f"用户反馈已记录：feedback_id={feedback_id}, rating={feedback.rating}, "
            f"request_id={feedback.request_id}, reason={feedback.reason}"
        )
    except Exception as e:
        logger.warning(f"用户反馈记录失败：{e}")

    return {"status": "ok", "feedback_id": feedback_id if 'feedback_id' in dir() else "", "message": "反馈已收到"}


@app.get("/api/metrics/nodes")
async def node_metrics_stats(request: Request, hours: int = 24):
    """查询节点耗时统计（P50/P95/P99）"""
    if not _verify_admin_key(request):
        return JSONResponse(status_code=403, content={"detail": "无权访问"})
    from app.core.metrics import get_metrics_collector
    collector = get_metrics_collector()
    return {"hours": hours, "stats": collector.get_node_stats(hours)}


@app.get("/api/metrics/requests")
async def request_metrics_stats(request: Request, hours: int = 24):
    """查询请求级耗时统计"""
    if not _verify_admin_key(request):
        return JSONResponse(status_code=403, content={"detail": "无权访问"})
    from app.core.metrics import get_metrics_collector
    collector = get_metrics_collector()
    return {"hours": hours, "stats": collector.get_request_stats(hours)}


@app.get("/api/metrics/tokens")
async def token_usage_stats(request: Request, hours: int = 24):
    """查询 Token 用量统计（按模型/节点/每日趋势 + 成本估算）"""
    if not _verify_admin_key(request):
        return JSONResponse(status_code=403, content={"detail": "无权访问"})
    from app.core.metrics import get_metrics_collector
    collector = get_metrics_collector()
    return collector.get_token_stats(hours)


@app.get("/api/metrics/feedback")
async def feedback_stats(request: Request, hours: int = 24):
    """查询反馈统计（满意度率/差评原因分布/每日趋势）"""
    if not _verify_admin_key(request):
        return JSONResponse(status_code=403, content={"detail": "无权访问"})
    from app.core.metrics import get_metrics_collector
    collector = get_metrics_collector()
    return collector.get_feedback_stats(hours)


@app.get("/api/metrics/feedback/candidates")
async def feedback_golden_candidates(request: Request, limit: int = 50):
    """获取差评中适合转化为黄金测试集的候选"""
    if not _verify_admin_key(request):
        return JSONResponse(status_code=403, content={"detail": "无权访问"})
    from app.core.metrics import get_metrics_collector
    collector = get_metrics_collector()
    return {"candidates": collector.get_feedback_candidates_for_golden_set(limit)}


@app.get("/api/admin/refusal/stats")
async def get_refusal_stats(request: Request, days: int = 7):
    """拒答日志统计"""
    if not _verify_admin_key(request):
        return JSONResponse(status_code=403, content={"detail": "无权访问"})
    from app.core.metrics import get_metrics_collector
    collector = get_metrics_collector()
    stats = collector.get_refusal_stats(days=days)
    return JSONResponse(content=stats)


@app.get("/api/admin/refusal/export")
async def export_refusal_logs(request: Request, days: int = 7):
    """导出拒答日志明细"""
    if not _verify_admin_key(request):
        return JSONResponse(status_code=403, content={"detail": "无权访问"})
    from app.core.metrics import get_metrics_collector
    collector = get_metrics_collector()
    records = collector.export_refusal_logs(days=days)
    return JSONResponse(content={"total": len(records), "records": records})


# ===== 知识库管理 API =====

# 知识库更新写锁（防止并发更新导致数据不一致）
_kb_update_lock = asyncio.Lock()
# 知识库更新状态
_kb_update_status = {
    "updating": False,
    "progress": "",
    "started_at": None,
    "error": None,
}


@app.get("/api/admin/kb/status")
async def kb_status(request: Request):
    """知识库状态查询（需管理员认证）

    返回：文档数量、向量库信息、更新状态等
    """
    if not _verify_admin_key(request):
        return JSONResponse(status_code=403, content={"detail": "无权访问"})

    config = get_config()
    docs_dir = config.DOCS_DIR

    # 扫描文档目录
    supported_ext = set(LOADERS.keys())
    files = []
    if docs_dir.exists():
        for f in docs_dir.iterdir():
            if f.is_file() and f.suffix.lower() in supported_ext:
                files.append({
                    "name": f.name,
                    "size": f.stat().st_size,
                    "type": f.suffix.lstrip("."),
                    "modified": f.stat().st_mtime,
                })

    # 向量库信息
    try:
        from app.rag.vector_store import get_vector_store, get_kb_version
        vs = get_vector_store()
        kb_version = get_kb_version()
        # ChromaDB 文档数
        doc_count = 0
        if hasattr(vs, '_collection') and vs._collection:
            doc_count = vs._collection.count()
    except Exception:
        kb_version = "unknown"
        doc_count = 0

    # Embedding 模型信息 + 一致性校验
    from app.rag.kb_updater import get_embedding_model_info, run_reconciliation, CHUNK_STRATEGY_VERSION
    embedding_info = get_embedding_model_info()
    reconciliation = run_reconciliation(docs_dir, vs) if doc_count > 0 else {}

    return {
        "docs_dir": str(docs_dir),
        "files": files,
        "file_count": len(files),
        "kb_version": kb_version,
        "vector_count": doc_count,
        "update_status": _kb_update_status,
        "embedding_model": embedding_info,
        "chunk_strategy": CHUNK_STRATEGY_VERSION,
        "reconciliation": reconciliation,
    }


@app.post("/api/admin/kb/upload")
async def kb_upload(request: Request):
    """上传文档到知识库（需管理员认证）

    支持：.txt .pdf .docx .md .xlsx .xls .csv .png .jpg .jpeg
    流程：保存文件到 docs/medical/ → 加载 → 切分 → 增量写入向量库
    """
    if not _verify_admin_key(request):
        return JSONResponse(status_code=403, content={"detail": "无权访问"})

    if _kb_update_status["updating"]:
        return JSONResponse(status_code=409, content={"detail": "知识库正在更新中，请稍后重试"})

    # 使用 multipart/form-data 上传
    from fastapi import UploadFile, File as FastAPIFile
    form = await request.form()

    uploaded_files = []
    errors = []

    config = get_config()
    docs_dir = config.DOCS_DIR
    docs_dir.mkdir(parents=True, exist_ok=True)

    for field_name, file_item in form.items():
        if not hasattr(file_item, 'filename') or not file_item.filename:
            continue

        raw_filename = file_item.filename
        filename = _sanitize_kb_filename(raw_filename)
        if not filename:
            errors.append({"file": raw_filename, "error": "非法文件名（不允许路径分隔符/目录穿越）"})
            continue
        suffix = Path(filename).suffix.lower()

        # 检查支持的格式
        try:
            from app.rag.loader import LOADERS
            supported = set(LOADERS.keys())
        except Exception:
            supported = {".txt", ".pdf", ".docx", ".md", ".xlsx", ".xls", ".csv"}

        if suffix not in supported:
            errors.append({"file": filename, "error": f"不支持的文件类型：{suffix}"})
            continue

        # 保存文件
        target_path = docs_dir / filename
        try:
            content = await file_item.read()
            with open(target_path, 'wb') as f:
                f.write(content)
            uploaded_files.append({
                "name": filename,
                "size": len(content),
                "type": suffix.lstrip("."),
            })
        except Exception as e:
            errors.append({"file": filename, "error": str(e)})

    if not uploaded_files:
        return JSONResponse(status_code=400, content={"detail": "无有效文件上传", "errors": errors})

    # 增量入库（加写锁）
    async with _kb_update_lock:
        _kb_update_status["updating"] = True
        _kb_update_status["progress"] = "增量入库中"
        _kb_update_status["started_at"] = time.time()
        _kb_update_status["error"] = None

        try:
            # 加载上传的文件
            from app.rag.loader import load_single_file, split_documents, add_metadata
            from app.rag.vector_store import add_documents_to_store, get_vector_store
            from app.rag.parent_child_store import get_parent_child_manager
            from app.rag.kb_updater import (
                enrich_chunk_metadata, filter_unchanged_chunks,
                get_existing_content_hashes, get_document_version,
                soft_delete_document, log_kb_audit,
                activate_document_version,
            )

            total_chunks = 0
            skipped_chunks = 0
            for file_info in uploaded_files:
                file_path = docs_dir / file_info["name"]
                start_time = time.time()
                try:
                    # 加载 + 添加元数据（含多源自动提取）
                    docs = load_single_file(file_path)
                    meta_report = add_metadata(docs, file_path)

                    # 将元数据提取结果附加到上传响应
                    if meta_report:
                        file_info["meta_report"] = {
                            "overall_confidence": meta_report.get("overall_confidence", "unknown"),
                            "needs_manual_review": meta_report.get("needs_manual_review", False),
                            "extracted": {
                                k: v for k, v in meta_report.items()
                                if not k.endswith("_confidence")
                                and not k.endswith("_source")
                                and k not in ("pending_review", "needs_manual_review",
                                              "overall_confidence", "metadata_source")
                                and v and v != "unknown"
                            },
                            "pending_review": [
                                {"field": p["field"], "value": p["value"],
                                 "confidence": p["confidence"], "reason": p["reason"]}
                                for p in meta_report.get("pending_review", [])
                            ],
                        }

                    # 切分
                    chunks = split_documents(docs)

                    # 版本管理：已存在则版本号+1（不先删旧版本！双缓冲）
                    vs = get_vector_store()
                    old_version = get_document_version(vs, file_info["name"])
                    version_id = old_version + 1 if old_version > 0 else 1

                    # 增强元数据：status=pending（不可检索！）
                    chunks = enrich_chunk_metadata(chunks, file_info["name"], version_id=version_id, status="pending")

                    # 增量去重：content_hash 未变化的 chunk 跳过 Embedding
                    existing_hashes = get_existing_content_hashes(vs)
                    changed_chunks, unchanged_chunks = filter_unchanged_chunks(chunks, existing_hashes)

                    if not changed_chunks:
                        file_info["chunks"] = 0
                        file_info["skipped"] = "all_unchanged"
                        skipped_chunks += len(unchanged_chunks)
                        continue

                    total_chunks += len(changed_chunks)

                    # 构建父子索引（增量：只更新本次变更文档的 parent，不动其它文档，
                    # 避免 build_index 清空全库 parent store 导致父还原能力退化）
                    parent_manager = get_parent_child_manager()
                    child_chunks = parent_manager.update_index(changed_chunks, child_chunk_size=150)

                    # 写入向量库（status=pending，检索层看不到！）
                    add_documents_to_store(child_chunks)

                    # 双缓冲：校验 → 激活新版本(status=active) → 仅废弃新版本中已删除的旧块
                    # keep_hashes 传新版本全部 chunk（changed + unchanged）的 hash，
                    # 未变块在旧版本中保持 active，避免"只改几行 → 整篇 deprecated → 内容丢失"
                    new_hashes = {c.metadata.get("content_hash", "") for c in chunks}
                    new_hashes.discard("")
                    activated = activate_document_version(
                        vs, file_info["name"], version_id, keep_hashes=new_hashes
                    )
                    if activated == 0:
                        # 校验失败：新 chunk 仍在 pending 状态，不影响旧版本检索
                        file_info["chunks"] = len(changed_chunks)
                        file_info["status"] = "pending_verification_failed"
                        logger.warning(f"文档入库但激活失败：{file_info['name']} v{version_id}，chunk 仍在 pending 状态")
                    else:
                        file_info["chunks"] = len(changed_chunks)
                        file_info["status"] = "active"

                    if unchanged_chunks:
                        file_info["skipped_unchanged"] = len(unchanged_chunks)

                    # 审计日志
                    elapsed_ms = int((time.time() - start_time) * 1000)
                    log_kb_audit(
                        doc_id=file_info["name"],
                        change_type="modify" if old_version > 0 else "add",
                        chunk_count=len(changed_chunks),
                        version_id=version_id,
                        result="success",
                        elapsed_ms=elapsed_ms,
                    )
                except Exception as e:
                    errors.append({"file": file_info["name"], "error": str(e)})
                    logger.error(f"文件入库失败：{file_info['name']} - {e}")
                    log_kb_audit(
                        doc_id=file_info["name"],
                        change_type="add",
                        result="failed",
                        error_message=str(e),
                    )

            # 清除缓存（知识库变更后旧缓存无效）
            try:
                cache = get_cache()
                cache.clear()
                semantic_cache = get_semantic_cache()
                semantic_cache.clear()
            except Exception:
                pass

            # BM25 索引失效：新文档入库后必须重建，否则 BM25 稀疏检索仍用旧文档集
            try:
                config = get_config()
                bm25_cache = config.BM25_CACHE_PATH
                if bm25_cache.exists():
                    bm25_cache.unlink()
                from app.rag.hybrid_retriever import reset_hybrid_retriever
                reset_hybrid_retriever()
            except Exception as e:
                logger.warning(f"上传后重置 BM25 失败：{e}")

            _kb_update_status["progress"] = "完成"
        except Exception as e:
            _kb_update_status["error"] = str(e)
            logger.error(f"知识库增量入库失败：{e}")
        finally:
            _kb_update_status["updating"] = False
            _kb_update_status["started_at"] = None

    return {
        "uploaded": uploaded_files,
        "total_chunks": total_chunks,
        "errors": errors,
    }


@app.delete("/api/admin/kb/documents/{filename}")
async def kb_delete_document(filename: str, request: Request):
    """从知识库删除指定文档（需管理员认证）

    流程：
        1. 从 ChromaDB 删除该文档的所有 chunks（按 source 元数据过滤）
        2. 从 ParentChildStore 删除对应 parent 文档
        3. 从磁盘删除文件
        4. 清除缓存
    """
    if not _verify_admin_key(request):
        return JSONResponse(status_code=403, content={"detail": "无权访问"})

    if _kb_update_status["updating"]:
        return JSONResponse(status_code=409, content={"detail": "知识库正在更新中，请稍后重试"})

    filename = _sanitize_kb_filename(filename)
    if not filename:
        return JSONResponse(status_code=400, content={"detail": "非法文件名（不允许路径分隔符/目录穿越）"})

    async with _kb_update_lock:
        config = get_config()
        docs_dir = config.DOCS_DIR
        file_path = docs_dir / filename

        if not file_path.exists():
            return JSONResponse(status_code=404, content={"detail": f"文件不存在：{filename}"})

        deleted_chunks = 0
        try:
            # 1. 软删除（is_deleted=True，非物理删除）
            from app.rag.vector_store import get_vector_store
            from app.rag.kb_updater import soft_delete_document, log_kb_audit

            vs = get_vector_store()
            deleted_chunks = soft_delete_document(vs, filename)

            # 2. 从磁盘删除文件
            file_path.unlink()

            # 3. 清除缓存
            try:
                cache = get_cache()
                cache.clear()
                semantic_cache = get_semantic_cache()
                semantic_cache.clear()
            except Exception:
                pass

            # 4. BM25 缓存失效
            bm25_cache = config.BM25_CACHE_PATH
            if bm25_cache.exists():
                bm25_cache.unlink()

            # 5. 审计日志
            log_kb_audit(
                doc_id=filename,
                change_type="delete",
                chunk_count=deleted_chunks,
                result="success",
            )

            logger.info(f"知识库文档已软删除：{filename}（{deleted_chunks} 个 chunks）")
        except Exception as e:
            logger.error(f"知识库文档删除失败：{filename} - {e}")
            log_kb_audit(doc_id=filename, change_type="delete", result="failed", error_message=str(e))
            return JSONResponse(status_code=500, content={"detail": str(e)})

    return {
        "deleted": True,
        "filename": filename,
        "deleted_chunks": deleted_chunks,
        "soft_delete": True,
    }


@app.post("/api/admin/kb/restore/{filename}")
async def kb_restore_document(filename: str, request: Request):
    """恢复软删除的文档（误删恢复）"""
    if not _verify_admin_key(request):
        return JSONResponse(status_code=403, content={"detail": "无权访问"})

    async with _kb_update_lock:
        from app.rag.vector_store import get_vector_store
        from app.rag.kb_updater import restore_deleted_document, log_kb_audit

        vs = get_vector_store()
        count = restore_deleted_document(vs, filename)

        if count == 0:
            return JSONResponse(status_code=404, content={"detail": f"未找到可恢复的文档：{filename}"})

        log_kb_audit(doc_id=filename, change_type="restore", chunk_count=count, result="success")

    return {"restored": True, "filename": filename, "restored_chunks": count}


@app.get("/api/admin/kb/audit-log")
async def kb_audit_log(limit: int = 50, request: Request = None):
    """查询知识库更新审计日志"""
    if not _verify_admin_key(request):
        return JSONResponse(status_code=403, content={"detail": "无权访问"})

    from app.rag.kb_updater import get_kb_audit_log
    return {"logs": get_kb_audit_log(limit)}


@app.get("/api/admin/kb/reconcile")
async def kb_reconcile(request: Request):
    """一致性校验：对比磁盘/向量库/Embedding/切分策略，检测差异"""
    if not _verify_admin_key(request):
        return JSONResponse(status_code=403, content={"detail": "无权访问"})

    from app.rag.kb_updater import run_reconciliation
    from app.rag.vector_store import get_vector_store

    config = get_config()
    vs = get_vector_store()
    report = run_reconciliation(config.DOCS_DIR, vs)
    return report


@app.get("/api/admin/kb/stale-detect")
async def kb_stale_detect(hours: int = 24, request: Request = None):
    """变更检测：查找源系统已更新但索引未同步的陈旧文档"""
    if not _verify_admin_key(request):
        return JSONResponse(status_code=403, content={"detail": "无权访问"})

    from app.rag.kb_updater import detect_stale_documents
    from app.rag.vector_store import get_vector_store

    config = get_config()
    vs = get_vector_store()
    stale = detect_stale_documents(config.DOCS_DIR, vs, max_staleness_hours=hours)
    return {"stale_documents": stale, "threshold_hours": hours}


@app.post("/api/admin/kb/rebuild")
async def kb_rebuild(request: Request):
    """重建知识库（需管理员认证）— 双集合零停机方案

    流程（影子集合 + 别名指针，全程服务不中断）：
        1. 加写锁
        2. 创建影子集合（带时间戳的独立目录，对线上完全不可见）
        3. 加载全部文档 → 切分 → 构建父子索引 → 写入影子集合
        4. 校验影子集合（chunk数、抽样召回率、模型一致性）
           ├─ 失败 → 丢弃影子集合 + 告警 → 服务不中断
           └─ 通过 → 继续
        5. 激活影子集合中的 pending chunk → active
        6. 原子切换：别名指针指向影子集合
        7. 重建 BM25 索引 + 清除缓存
        8. 释放写锁
        9. 5 分钟后延迟清理旧集合

    并发安全：
        - 写锁：同一时间只有一个重建任务
        - 检索请求始终走 active 集合，不受影子集合构建影响
        - 切换是原子的：用户要么看到完整旧版，要么看到完整新版
    """
    if not _verify_admin_key(request):
        return JSONResponse(status_code=403, content={"detail": "无权访问"})

    if _kb_update_status["updating"]:
        return JSONResponse(status_code=409, content={"detail": "知识库正在更新中，请稍后重试"})

    async with _kb_update_lock:
        _kb_update_status["updating"] = True
        _kb_update_status["progress"] = "重建中：创建影子集合"
        _kb_update_status["started_at"] = time.time()
        _kb_update_status["error"] = None

        shadow_name = None
        shadow_persist_dir = None

        try:
            from app.rag.vector_store import get_dual_collection_manager
            from app.rag.parent_child_store import get_parent_child_manager
            from app.rag.loader import load_medical_documents, split_documents
            from app.rag.kb_updater import enrich_chunk_metadata, log_kb_audit

            dcm = get_dual_collection_manager()

            # 1. 创建影子集合
            shadow_name, shadow_persist_dir = dcm.create_shadow_collection()
            _kb_update_status["progress"] = "重建中：加载文档"

            # 2. 加载文档
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(None, load_medical_documents)
            _kb_update_status["progress"] = f"重建中：切分文档（{len(docs)} 个）"

            # 3. 切分
            parent_docs = await loop.run_in_executor(None, split_documents, docs)
            _kb_update_status["progress"] = f"重建中：构建索引（{len(parent_docs)} 个 chunks）"

            # 4. 构建父子索引（用全新 ParentChildManager，
            #    避免 build_index 重置 store 期间污染线上检索的父索引）
            from app.rag.parent_child_store import ParentChildManager
            parent_manager = ParentChildManager()
            child_chunks = await loop.run_in_executor(
                None, lambda: parent_manager.build_index(parent_docs, child_chunk_size=150)
            )
            _kb_update_status["progress"] = f"重建中：写入影子集合（{len(child_chunks)} 个 child chunks）"

            # 5. 增强元数据（按真实 source 分组，保证 doc_id/chunk_id 反映文档来源，
            #    不能一次性传 source="rebuild"，否则所有 chunk 的 doc_id 被覆盖为 "rebuild"）
            from collections import defaultdict
            by_source = defaultdict(list)
            for c in child_chunks:
                by_source[c.metadata.get("source", "unknown")].append(c)
            child_chunks = []
            for src, chunks in by_source.items():
                child_chunks.extend(
                    enrich_chunk_metadata(chunks, src, version_id=1, status="active")
                )

            # 6. 写入影子集合（对线上完全不可见！）
            def _progress_cb(done, total):
                _kb_update_status["progress"] = f"重建中：写入影子集合（{done}/{total}）"

            shadow_vs = await loop.run_in_executor(
                None, lambda: dcm.build_shadow_collection(
                    child_chunks, shadow_name, shadow_persist_dir, _progress_cb
                )
            )

            _kb_update_status["progress"] = "重建中：校验影子集合"

            # 7. 校验影子集合
            validation = await loop.run_in_executor(
                None, lambda: dcm.validate_shadow_collection(shadow_vs, len(child_chunks))
            )
            if not validation["valid"]:
                # 校验失败：丢弃影子集合，服务不中断
                error_msg = f"影子集合校验失败：{validation['errors']}"
                logger.error(error_msg)
                # 清理影子集合
                try:
                    shutil.rmtree(shadow_persist_dir)
                except Exception:
                    pass
                _kb_update_status["error"] = error_msg
                _kb_update_status["progress"] = f"校验失败（已回滚）"
                return JSONResponse(status_code=500, content={
                    "detail": error_msg,
                    "validation": validation,
                })

            if validation.get("warnings"):
                logger.warning(f"影子集合校验警告：{validation['warnings']}")

            # 8. 激活 pending chunk（如有）
            await loop.run_in_executor(None, dcm.activate_pending_chunks, shadow_vs)

            _kb_update_status["progress"] = "重建中：原子切换集合"

            # 9. 原子切换：别名指针指向影子集合
            dcm.switch_active_collection(shadow_name, str(shadow_persist_dir))

            # 10. 原子换入新父索引（替换全局单例，线上检索即刻用新 parent store）并持久化
            import app.rag.parent_child_store as pcs
            pcs._parent_child_manager = parent_manager
            await loop.run_in_executor(None, parent_manager.save_to_disk)

            _kb_update_status["progress"] = "重建中：重建 BM25 索引"

            # 11. 重建 BM25 索引（基于新集合数据）
            # 删除旧 BM25 缓存，HybridRetriever 下次检索时自动重建
            config = get_config()
            bm25_cache = config.BM25_CACHE_PATH
            if bm25_cache.exists():
                bm25_cache.unlink()
            # 重置 HybridRetriever 实例，触发 BM25 重建
            try:
                from app.rag.hybrid_retriever import reset_hybrid_retriever
                reset_hybrid_retriever()
            except Exception:
                pass

            _kb_update_status["progress"] = "重建中：清除缓存"

            # 12. 清除所有缓存
            try:
                cache = get_cache()
                cache.clear()
                semantic_cache = get_semantic_cache()
                semantic_cache.clear()
            except Exception:
                pass

            # 13. 审计日志
            try:
                log_kb_audit(
                    doc_id="full_rebuild",
                    change_type="rebuild",
                    chunk_count=len(child_chunks),
                    result="success",
                    elapsed_ms=int((time.time() - _kb_update_status["started_at"]) * 1000),
                )
            except Exception:
                pass

            # 14. 延迟清理旧集合（5 分钟后）
            dcm.schedule_cleanup_old_collection(delay_seconds=300)

            _kb_update_status["progress"] = "完成"

            elapsed = time.time() - _kb_update_status["started_at"]
            logger.info(
                f"知识库重建完成（双集合）：{shadow_name}，"
                f"{len(docs)} 个文档 → {len(parent_docs)} 个 chunks → {len(child_chunks)} 个 child chunks，"
                f"耗时 {elapsed:.1f}s"
            )

            return {
                "rebuilt": True,
                "collection_name": shadow_name,
                "document_count": len(docs),
                "chunk_count": len(parent_docs),
                "child_chunk_count": len(child_chunks),
                "elapsed_seconds": round(elapsed, 1),
                "validation": validation,
            }

        except Exception as e:
            _kb_update_status["error"] = str(e)
            _kb_update_status["progress"] = f"失败：{str(e)}"
            logger.error(f"知识库重建失败：{e}")
            # 清理影子集合（如果已创建）
            if shadow_persist_dir and Path(shadow_persist_dir).exists():
                try:
                    shutil.rmtree(shadow_persist_dir)
                    logger.info(f"已清理失败的影子集合：{shadow_persist_dir}")
                except Exception:
                    pass
            return JSONResponse(status_code=500, content={"detail": str(e)})
        finally:
            _kb_update_status["updating"] = False
            _kb_update_status["started_at"] = None


@app.post("/api/admin/kb/rollback")
async def kb_rollback(request: Request):
    """紧急回滚到上一个活跃集合（需管理员认证）

    场景：新集合上线后发现数据问题，需要秒级回滚到旧版本。
    """
    if not _verify_admin_key(request):
        return JSONResponse(status_code=403, content={"detail": "无权访问"})

    try:
        from app.rag.vector_store import get_dual_collection_manager
        dcm = get_dual_collection_manager()
        success = dcm.rollback_to_previous()

        if success:
            # 清除缓存
            try:
                cache = get_cache()
                cache.clear()
                semantic_cache = get_semantic_cache()
                semantic_cache.clear()
            except Exception:
                pass

            # 重置 BM25
            config = get_config()
            bm25_cache = config.BM25_CACHE_PATH
            if bm25_cache.exists():
                bm25_cache.unlink()
            try:
                from app.rag.hybrid_retriever import reset_hybrid_retriever
                reset_hybrid_retriever()
            except Exception:
                pass

            return {"rolled_back": True, "active_collection": dcm.get_active_collection_name()}
        else:
            return JSONResponse(status_code=500, content={"detail": "回滚失败：无可用旧集合"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


@app.get("/api/admin/kb/collection-info")
async def kb_collection_info(request: Request):
    """查询当前集合信息（活跃集合 + 上一集合 + 切换历史）"""
    if not _verify_admin_key(request):
        return JSONResponse(status_code=403, content={"detail": "无权访问"})

    try:
        from app.rag.vector_store import get_dual_collection_manager
        dcm = get_dual_collection_manager()
        config = dcm.get_active_config()

        # 补充向量数信息
        try:
            from app.rag.vector_store import get_vector_store, get_kb_version
            vs = get_vector_store()
            if hasattr(vs, '_collection') and vs._collection:
                config["active_vector_count"] = vs._collection.count()
            config["kb_version"] = get_kb_version()
        except Exception:
            pass

        return config
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


# 启动入口
if __name__ == '__main__':
    import os
    import uvicorn

    # Ollama 优化：缩减上下文窗口加速本地模型推理（默认4096太慢）
    os.environ.setdefault("OLLAMA_NUM_CTX", "1024")
    uvicorn.run("app.api.routes:app", host="0.0.0.0", port=8000, reload=True)
