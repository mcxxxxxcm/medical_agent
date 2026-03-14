# app/core/embeddings.py
from app.core.config import get_config
from langchain_openai import OpenAIEmbeddings


def get_embeddings() -> OpenAIEmbeddings:
    """获取 Embedding 实例

    Returns:
        OpenAIEmbeddings: Embedding 实例
    """
    config = get_config()

    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.MODEL_URL,
        api_key=config.MODEL_API_KEY,
    )