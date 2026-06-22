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
        request_timeout=30,     # 单次请求超时30秒
        max_retries=1,          # 最多重试1次
    )


@lru_cache(maxsize=2)
def get_local_llm() -> ChatOpenAI:
    """获取本地模型实例（Ollama）

    用于中间节点（查询重写/档案提取/快照更新），降低延迟：
        - 本地推理，无网络延迟
        - qwen2.5:1.5b 适配 4GB VRAM，纯GPU推理约 50+ tokens/s
        - 需先安装 Ollama 并拉取模型：ollama pull qwen2.5:1.5b

    当 LOCAL_MODEL_ENABLED=False 时，降级使用 API 模型

    Returns:
        ChatOpenAI: 本地 LLM 实例
    """
    config = get_config()

    if not config.LOCAL_MODEL_ENABLED:
        # 未启用本地模型，降级使用 API 轻量模型
        return get_rewrite_llm()

    return ChatOpenAI(
        model=config.LOCAL_MODEL_NAME,
        base_url=config.LOCAL_MODEL_URL,
        api_key=config.LOCAL_MODEL_API_KEY,
        temperature=0.0,           # 结构化提取/改写不需要创造性
        request_timeout=30,        # 本地模型一般很快，30秒足够
        max_retries=1,
        max_tokens=200,            # 查询重写/HyDE只需短输出，限制生成长度加速推理
    )


@lru_cache(maxsize=2)
def get_local_llm_json() -> ChatOpenAI:
    """获取本地模型实例（Ollama）— JSON Mode

    与 get_local_llm() 的区别：
        开启 response_format={"type": "json_object"}，在采样层面提高 JSON 字符权重，
        大幅减少 3B 模型的格式错误（单引号、多余逗号、括号不匹配等）。

    注意：
        - JSON Mode 只保证输出是合法 JSON，不保证符合特定 schema
        - 仍需后处理容错（_coerce_list_fields / json_repair）
        - prompt 中必须包含 "JSON" 字样，否则 Ollama 可能忽略 response_format

    适用节点：症状解析、快照更新、档案提取（需要结构化输出的节点）
    不适用节点：查询重写、HyDE（纯文本输出，不需要 JSON）
    """
    config = get_config()

    if not config.LOCAL_MODEL_ENABLED:
        return get_rewrite_llm()

    return ChatOpenAI(
        model=config.LOCAL_MODEL_NAME,
        base_url=config.LOCAL_MODEL_URL,
        api_key=config.LOCAL_MODEL_API_KEY,
        temperature=0.0,
        request_timeout=30,
        max_retries=1,
        max_tokens=150,  # 结构化输出只需短JSON，限制生成长度加速推理
        model_kwargs={
            "response_format": {"type": "json_object"},
        },
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
        request_timeout=15,     # 轻量模型超时15秒
        max_retries=1,
    )


@lru_cache(maxsize=2)
def get_symptom_llm() -> ChatOpenAI:
    """获取症状解析专用的快速 LLM 实例

    症状解析只需提取结构化字段，不需要复杂推理：
        - glm-4-flash：智谱最快模型，首token ~1-2秒
        - 对比 glm-4.5-air：首token ~5-7秒

    Returns:
        ChatOpenAI: 快速 LLM 实例
    """
    config = get_config()

    return ChatOpenAI(
        model=config.SYMPTOM_MODEL_NAME,
        base_url=config.MODEL_URL,
        api_key=config.MODEL_API_KEY,
        temperature=0.0,  # 结构化提取不需要创造性
        request_timeout=15,
        max_retries=1,
    )


@lru_cache(maxsize=2)
def get_vision_llm(streaming: bool = False) -> ChatOpenAI:
    """获取多模态视觉 LLM 实例（图片问诊）

    用于处理用户上传的图片（报告、皮肤、药盒等）：
        - glm-4v-plus：智谱多模态模型，支持图片+文字输入
        - 首token ~4-9秒（取决于图片复杂度）

    Returns:
        ChatOpenAI: 多模态 LLM 实例
    """
    config = get_config()

    return ChatOpenAI(
        model=config.VISION_MODEL_NAME,
        base_url=config.MODEL_URL,
        api_key=config.MODEL_API_KEY,
        temperature=config.MODEL_TEMPERATURE,
        streaming=streaming,
        request_timeout=60,     # 视觉模型需要更长超时
        max_retries=1,
    )
