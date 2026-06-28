# app/core/embeddings.py
import httpx
from functools import lru_cache

from app.core.config import get_config
from app.core.app_logging import get_logger
from langchain_openai import OpenAIEmbeddings

logger = get_logger(__name__)


@lru_cache(maxsize=2)
def get_embeddings() -> OpenAIEmbeddings:
    """获取 Embedding 实例（带缓存）

    Returns:
        OpenAIEmbeddings: Embedding 实例

    超时说明：
        request_timeout 在旧版 openai 库中映射为 httpx.Client(timeout=...)，
        但新版 openai>=1.0 中该参数已被弃用，实际超时可能回退到默认 600s。
        因此显式传入 httpx.Timeout 强制生效。
    """
    config = get_config()

    # 显式构造 httpx.Timeout，确保 connect/read/write/pool 各阶段都有限制
    # 避免 openai 库版本差异导致 request_timeout 不生效（实测 v1.x 默认 600s）
    timeout = httpx.Timeout(
        connect=5.0,      # TCP 连接超时 5s
        read=10.0,        # 等待响应超时 10s
        write=10.0,       # 发送请求超时 10s
        pool=5.0,         # 连接池等待超时 5s
    )

    try:
        return OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            base_url=config.MODEL_URL,
            api_key=config.MODEL_API_KEY,
            timeout=timeout,
            max_retries=1,          # 最多重试1次（共2次尝试）
        )
    except TypeError:
        # 旧版 langchain-openai 不支持 timeout=httpx.Timeout，回退到 request_timeout
        logger.warning("当前 langchain-openai 版本不支持 httpx.Timeout，回退到 request_timeout")
        return OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            base_url=config.MODEL_URL,
            api_key=config.MODEL_API_KEY,
            request_timeout=15,     # 单次请求超时15秒
            max_retries=1,
        )
