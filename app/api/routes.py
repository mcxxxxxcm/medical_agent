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
from typing import Optional, List
import json
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, StreamingResponse

from app.core.app_logging import get_logger
from app.graph.graph import get_graph
from app.graph.nodes import (
    stream_answer_generation,
    stream_direct_answer,
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
from pathlib import Path
from fastapi.staticfiles import StaticFiles
import jieba

logger = get_logger(__name__)


# Pydantic模型
class ChatRequest(BaseModel):
    """聊天请求模型"""
    question: str = Field(..., description="用户问题", min_length=1, max_length=1000)
    user_id: Optional[str] = Field(None, description="用户ID")
    thread_id: Optional[str] = Field(None, description="会话线程ID")


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

    yield

    # 关闭时清理
    logger.info(f"应用关闭中")
    await close_checkpointer()
    logger.info(f"资源清理完成")


# FastAPI应用
app = FastAPI(
    title="Medical Assistant API",
    description="医疗助手智能问答系统 API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    """全局异常处理"""
    logger.error(f"未处理的异常：{exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"服务器内部错误：{str(exc)}"}
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
        current_state = {
            "question": request.question,
            "user_id": request.user_id,
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

            profile_state = profile_extraction_node(current_state)
            if profile_state:
                current_state.update(profile_state)

            route_command = router_node(current_state)
            next_node = getattr(route_command, "goto", "direct_answer")
            logger.info(
                f"流式请求路由完成：request_id={request_id}, thread_id={thread_id}, next_node={next_node}"
            )

            if next_node == "direct_answer":
                async for token in stream_direct_answer(current_state):
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
                elif current_state.get("retrieved_docs"):
                    logger.info(f"开始流式生成 RAG 答案：request_id={request_id}, thread_id={thread_id}")
                    async for token in stream_answer_generation(current_state):
                        yield await emit_data(token)
                else:
                    logger.info(f"无检索文档，降级为流式直接回答：request_id={request_id}, thread_id={thread_id}")
                    async for token in stream_direct_answer(current_state):
                        yield await emit_data(token)

            total_elapsed_ms = (time.time() - request_start_time) * 1000
            logger.info(
                f"流式请求完成：request_id={request_id}, thread_id={thread_id}, total_elapsed_ms={total_elapsed_ms:.2f}"
            )
            yield "data: [DONE]\n\n"

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


@app.post("/api/cache/clear")
async def clear_cache():
    """清空缓存"""
    from app.cache.redis_cache import get_cache
    cache = get_cache()
    count = cache.clear()
    return {"cleared": count, "message": f"已清空 {count} 条缓存"}


@app.delete("/api/cache/{query}")
async def delete_cache(query: str):
    """删除指定查询的缓存"""
    from app.cache.redis_cache import get_cache
    cache = get_cache()
    success = cache.delete(query)
    return {"deleted": success}


# 启动入口
if __name__ == '__main__':
    import uvicorn

    uvicorn.run("app.api.routes:app", host="0.0.0.0", port=8000, reload=True)
