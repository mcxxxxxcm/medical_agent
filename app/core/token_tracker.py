"""LLM Token 用量自动采集包装器

在 LangChain ChatOpenAI 调用后，自动从 response_metadata 中提取 token 用量
并写入 SQLite metrics。

用法：
    from app.core.token_tracker import track_tokens

    # 自动采集 invoke 调用的 token 用量
    result = llm.invoke(messages)
    track_tokens("答案生成", result, request_id="req_123", thread_id="thread_1")

    # 自动采集 stream 调用的 token 用量（需要在流结束后调用）
    # stream 模式下 token 信息在最后一个 chunk 中
"""
from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage

from app.core.app_logging import get_logger
from app.core.metrics import get_metrics_collector

logger = get_logger(__name__)


def track_tokens(
    node_name: str,
    response: BaseMessage,
    request_id: str = "",
    thread_id: str = "",
    model_override: str = "",
):
    """从 LLM 响应中提取 token 用量并写入 metrics

    Args:
        node_name: 调用节点名（如"答案生成"/"查询重写"）
        response: LLM 响应消息（AIMessage）
        request_id: 请求 ID
        thread_id: 会话线程 ID
        model_override: 手动指定模型名（覆盖自动检测）
    """
    try:
        # 提取 token 用量
        # LangChain ChatOpenAI 返回格式：
        #   response.response_metadata = {"token_usage": {"prompt_tokens": 800, "completion_tokens": 300, "total_tokens": 1100}, "model_name": "glm-4-flash"}
        #   或 OpenAI 兼容格式：
        #   response.response_metadata = {"usage": {"prompt_tokens": 800, "completion_tokens": 300}}
        metadata = getattr(response, "response_metadata", {}) or {}

        # 兼容多种格式
        token_usage = metadata.get("token_usage") or metadata.get("usage") or {}
        prompt_tokens = token_usage.get("prompt_tokens", 0) or 0
        completion_tokens = token_usage.get("completion_tokens", 0) or 0

        # 模型名
        model = model_override or metadata.get("model_name", "") or token_usage.get("model", "")

        if prompt_tokens == 0 and completion_tokens == 0:
            # 流式模式或某些模型不返回 token 信息，跳过
            return

        # 写入 metrics
        collector = get_metrics_collector()
        collector.record_token_usage(
            request_id=request_id,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            node_name=node_name,
            thread_id=thread_id,
        )

        logger.debug(
            f"Token 用量已记录：node={node_name}, model={model}, "
            f"prompt={prompt_tokens}, completion={completion_tokens}, "
            f"request_id={request_id}"
        )
    except Exception as e:
        # 静默失败，不影响主流程
        logger.debug(f"Token 采集失败（不影响回答）：{e}")


def track_stream_tokens(
    node_name: str,
    chunks: list,
    request_id: str = "",
    thread_id: str = "",
    model_override: str = "",
):
    """从流式响应的最后一个 chunk 中提取 token 用量

    Args:
        node_name: 调用节点名
        chunks: 流式响应的所有 chunk 列表
        request_id: 请求 ID
        thread_id: 会话线程 ID
        model_override: 手动指定模型名
    """
    if not chunks:
        return

    # 最后一个 chunk 可能包含 token 信息
    last_chunk = chunks[-1]
    track_tokens(node_name, last_chunk, request_id, thread_id, model_override)
