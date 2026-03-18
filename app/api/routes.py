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

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

from app.core.app_logging import get_logger
from app.graph.graph import get_graph
from app.graph.nodes import stream_answer_generation
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
    warnings: Optional[str] = Field(default_factory=list, description="警告信息")


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str
    database: str
    vector_store: str


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
    return HTTPException(
        status_code=500,
        detail=f"服务器内部错误：{str(exc)}"
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
async def stream(request: ChatRequest):
    """流式聊天接口
    功能描述：
        接受用户问题，以SSE流式返回回答

    Args：
        request：聊天请求

    Returns：
        StreamResponse：SSE流式响应
    """
    logger.info(f"收到流式聊天请求：user_id={request.user_id}")

    async def event_generator():
        try:
            graph = await get_graph()

            config = {
                "configurable": {
                    "thread_id": request.thread_id or f"thread_{request.user_id or 'default'}",
                }
            }

            input_state = {
                "question": request.question,
                "user_id": request.user_id,
            }

            # 储存中间状态
            state_snapshot = {}
            answer_generated = False

            async for event in graph.astream_events(
                    input_state,
                    config,
                    version="v2"
            ):
                kind = event["event"]
                name = event.get("name", "")

                # 捕获检索完成后的状态
                if kind == "on_chain_end" and name == "knowledge_retrieval":
                    output = event.get("data", {}).get("output", {})
                    if output:
                        state_snapshot.update(output)

                # 捕获direct_answer完成后的状态
                if kind == "on_chain_end" and name == "direct_answer":
                    output = event.get("data", {}).get("output", {})
                    answer = output.get("final_answer", "")
                    if answer:
                        state_snapshot.update(output)
                        answer_generated = True

                # 捕获用户档案
                if kind == "on_chain_end" and name == "memory_load":
                    output = event.get("data", {}).get("output", {})
                    if output:
                        state_snapshot.update(output)  # ✅ 使用 update

                # 流程结束
                if kind == "on_chain_end" and name == "LangGraph":
                    import json

                    full_state = {**input_state, **state_snapshot}

                    # 判断路径类型
                    if answer_generated and state_snapshot.get("final_answer"):
                        # direct_answer 路径：已经有答案，直接返回
                        yield f"data: {json.dumps(state_snapshot['final_answer'], ensure_ascii=False)}\n\n"

                    elif state_snapshot.get("retrieved_docs"):
                        # RAG 路径：有检索结果，流式生成答案
                        logger.info("开始流式生成答案...")
                        async for token in stream_answer_generation(full_state):
                            yield f"data: {json.dumps(token, ensure_ascii=False)}\n\n"

                    else:
                        # 兜底：直接回答路径
                        from app.graph.nodes import stream_direct_answer
                        async for token in stream_direct_answer(full_state):
                            yield f"data: {json.dumps(token, ensure_ascii=False)}\n\n"

                    logger.info("流程结束，发送 DONE")
                    yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"流式处理失败：{e}")
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
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

    # 检查数据库连接
    try:
        memory = get_long_term_memory()
        if memory.store is None:
            database_status = "unavailable"
    except Exception as e:
        database_status = f"error: {str(e)[:50]}"

    # 检查向量库
    try:
        from app.rag.vector_store import get_vector_store
        vs = get_vector_store()
        if vs is None:
            vector_store_status = "unavailable"
    except Exception as e:
        vector_store_status = f"error: {str(e)[:50]}"

    return HealthResponse(
        status="healthy",
        database=database_status,
        vector_store=vector_store_status,
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
