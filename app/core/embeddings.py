# app/core/embeddings.py
from functools import lru_cache

from app.core.config import get_config
from langchain_openai import OpenAIEmbeddings


@lru_cache(maxsize=2)
def get_embeddings() -> OpenAIEmbeddings:
    """获取 Embedding 实例（带缓存）

    Returns:
        OpenAIEmbeddings: Embedding 实例
    """
    config = get_config()

    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.MODEL_URL,
        api_key=config.MODEL_API_KEY,
        # 限制超时和重试，避免API网络抖动时阻塞20+秒
        request_timeout=10,     # 单次请求超时10秒
        max_retries=1,          # 最多重试1次（共2次尝试）
    )
