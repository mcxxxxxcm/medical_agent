# app/core/llm.py
from functools import lru_cache

from app.core.config import get_config
from langchain_openai import ChatOpenAI


@lru_cache(maxsize=8)
def get_llm(model_name: str = None, model_url: str = None, streaming: bool = False) -> ChatOpenAI:
    """获取 LLM 实例（带缓存）

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


@lru_cache(maxsize=2)
def get_rewrite_llm() -> ChatOpenAI:
    """获取查询重写专用的轻量 LLM 实例

    查询重写不需要复杂推理，使用轻量模型即可：
        - 响应更快（降低首token延迟）
        - Token消耗更少
        - 可通过 REWRITE_MODEL_NAME 环境变量配置

    Returns:
        ChatOpenAI: 轻量 LLM 实例
    """
    config = get_config()
    rewrite_model = getattr(config, 'REWRITE_MODEL_NAME', None) or config.MODEL_NAME

    return ChatOpenAI(
        model=rewrite_model,
        base_url=config.MODEL_URL,
        api_key=config.MODEL_API_KEY,
        temperature=0.0,  # 查询重写不需要创造性，温度设为0
    )
