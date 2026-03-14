# app/core/llm.py
from app.core.config import get_config
from langchain_openai import ChatOpenAI


def get_llm(model_name: str = None, model_url: str = None, streaming: bool = False) -> ChatOpenAI:
    """获取 LLM 实例

    Args:
        model_name: 模型名称（可选，默认使用配置）
        model_url: 模型 URL（可选，默认使用配置）
        streaming: 是否启用流式输出

    Returns:
        ChatOpenAI: LLM 实例
    """
    config = get_config()

    return ChatOpenAI(
        model=model_name or config.MODEL_NAME,
        base_url=model_url or config.MODEL_URL,
        api_key=config.MODEL_API_KEY,
        temperature=config.MODEL_TEMPERATURE,
        streaming=streaming,
    )
