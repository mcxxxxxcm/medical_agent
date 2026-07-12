"""结构化输出策略模块（v9.2）

三层降级策略：
    Layer 1: Tool Calling（最可靠）
        - LLM 原生支持 function calling
        - model → bind_tools(schema) → tool_choice="required"
        - 返回 tool_calls[0].args → Pydantic 校验
        - 支持：glm-4-flash、glm-4-plus、glm-4、qwen2.5 (Ollama)

    Layer 2: JSON Mode（中等可靠）
        - response_format={"type": "json_object"}
        - LLM 保证输出合法 JSON，但不保证 schema
        - extract_json_block + Pydantic 校验

    Layer 3: 纯文本 + 本地解析（兜底）
        - 完全无格式约束的 LLM 输出
        - extract_json_block + json_repair + Pydantic 校验
        - _coerce_list_fields 容错

策略选择逻辑：
    - ChatOpenAI (智谱 API) → 优先 Tool Calling
    - Ollama (qwen2.5) → 优先 Tool Calling (v0.3+ 支持)
    - 不支持时 → JSON Mode → 纯文本
    - 可通过 STRUCTURED_OUTPUT_STRATEGY 强制指定

用法：
    from app.graph.nodes.structured_output import invoke_structured

    result: RouterOutput = invoke_structured(
        llm, messages, RouterOutput,
    )
"""

import json
from typing import Any, Dict, List, Optional, Type

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel

from app.core.app_logging import get_logger
from app.graph.nodes.helpers import extract_json_block

logger = get_logger(__name__)

# 强制策略（环境变量覆盖）
_FORCE_STRATEGY = None  # "tool_calling" / "json_mode" / "text_only"


def _pydantic_to_openai_tool(schema: Type[BaseModel]) -> Dict:
    """将 Pydantic 模型转为 OpenAI tool 定义

    利用 LangChain 的 convert_to_openai_tool，
    将 Pydantic model 的 JSON Schema 包装为 function calling 格式。
    """
    return convert_to_openai_tool(schema)


def _try_tool_calling(
    llm,
    prompt,
    schema: Type[BaseModel],
) -> Optional[BaseModel]:
    """Layer 1: Tool Calling 策略

    流程：
        1. 将 Pydantic schema 转为 OpenAI tool 定义
        2. llm.bind_tools([tool], tool_choice=tool_name)
        3. LLM 返回 AIMessage 包含 tool_calls
        4. 从 tool_calls[0]["args"] 提取参数
        5. Pydantic model_validate 校验

    优势：
        - LLM 采样层约束输出格式，不需要 prompt 中声明 JSON 格式
        - tool_choice="required" 保证 LLM 一定调用工具
        - 返回的 args 已是结构化 dict，无需文本解析

    失败场景：
        - 模型不支持 tool calling（旧版 Ollama）
        - tool_choice 参数不被 API 接受
        - LLM 返回了 tool_calls 但 args 格式异常
    """
    try:
        tool_def = _pydantic_to_openai_tool(schema)
        tool_name = schema.__name__

        # 绑定工具，强制调用
        llm_with_tools = llm.bind_tools(
            [tool_def],
            tool_choice={"type": "function", "function": {"name": tool_name}},
        )

        response = llm_with_tools.invoke(prompt)

        # 提取 tool_calls
        if not hasattr(response, "tool_calls") or not response.tool_calls:
            logger.debug(f"Tool Calling: LLM 未返回 tool_calls，降级")
            return None

        tool_call = response.tool_calls[0]

        # 从 args 中构建 Pydantic 对象
        args = tool_call.get("args", {})
        if not args and isinstance(tool_call.get("function", {}).get("arguments"), str):
            # 兼容旧格式
            args = json.loads(tool_call["function"]["arguments"])

        result = schema.model_validate(args)
        logger.debug(f"Tool Calling 成功：{tool_name} → {result}")
        return result

    except NotImplementedError:
        logger.debug(f"Tool Calling: 模型不支持 bind_tools，降级")
        return None
    except TypeError as e:
        # bind_tools 参数不被接受（如 Ollama 旧版不支持 tool_choice）
        if "tool_choice" in str(e) or "bind_tools" in str(e):
            logger.debug(f"Tool Calling: bind_tools 参数错误，降级：{e}")
            return None
        raise
    except Exception as e:
        logger.debug(f"Tool Calling 失败，降级：{e}")
        return None


def _try_json_mode(
    llm,
    prompt,
    schema: Type[BaseModel],
) -> Optional[BaseModel]:
    """Layer 2: JSON Mode 策略

    前提：llm 已通过 model_kwargs 设置了 response_format={"type": "json_object"}
    如果 llm 没有 JSON Mode，则降级到纯文本解析。
    """
    try:
        # 检查 llm 是否配置了 JSON Mode
        model_kwargs = getattr(llm, "model_kwargs", {}) or {}
        if not model_kwargs.get("response_format", {}).get("type") == "json_object":
            # 没有 JSON Mode，直接跳到 Layer 3
            return None

        response = llm.invoke(prompt)
        raw_text = response.content if hasattr(response, "content") else str(response)

        parsed = extract_json_block(raw_text)
        if parsed is not None:
            result = schema.model_validate(parsed)
            logger.debug(f"JSON Mode 成功：{schema.__name__}")
            return result

        return None

    except Exception as e:
        logger.debug(f"JSON Mode 失败，降级：{e}")
        return None


def _try_text_parse(
    llm,
    prompt,
    schema: Type[BaseModel],
) -> Optional[BaseModel]:
    """Layer 3: 纯文本 + 本地解析（兜底）

    无格式约束，完全依赖后处理：
        extract_json_block → json.loads → json_repair → ast.literal_eval
        → Pydantic model_validate（含 field_validator 容错）
    """
    try:
        response = llm.invoke(prompt)
        raw_text = response.content if hasattr(response, "content") else str(response)

        parsed = extract_json_block(raw_text)
        if parsed is not None:
            result = schema.model_validate(parsed)
            logger.debug(f"纯文本解析成功：{schema.__name__}")
            return result

        return None

    except Exception as e:
        logger.debug(f"纯文本解析失败：{e}")
        return None


def invoke_structured(
    llm,
    prompt,
    schema: Type[BaseModel],
    *,
    max_attempts: int = 1,
    force_strategy: Optional[str] = None,
) -> BaseModel:
    """统一结构化输出入口（三层降级 + 重试）

    策略优先级：
        1. Tool Calling（bind_tools + tool_choice）
        2. JSON Mode（response_format=json_object）
        3. 纯文本 + 本地解析（extract_json_block + json_repair）

    Args:
        llm: LangChain ChatOpenAI 实例
        prompt: str 或 List[BaseMessage]
        schema: Pydantic 模型类
        max_attempts: 每层最大重试次数
        force_strategy: 强制使用指定策略（"tool_calling"/"json_mode"/"text_only"）

    Returns:
        Pydantic 模型实例

    Raises:
        ValueError: 所有策略均失败
    """
    strategy = force_strategy or _FORCE_STRATEGY

    # 按优先级尝试各策略
    strategies = []
    if strategy == "tool_calling":
        strategies = [_try_tool_calling]
    elif strategy == "json_mode":
        strategies = [_try_json_mode, _try_text_parse]
    elif strategy == "text_only":
        strategies = [_try_text_parse]
    else:
        # 默认：三层降级
        strategies = [_try_tool_calling, _try_json_mode, _try_text_parse]

    last_error = None
    for attempt in range(max_attempts):
        for strategy_fn in strategies:
            try:
                result = strategy_fn(llm, prompt, schema)
                if result is not None:
                    strategy_name = strategy_fn.__name__.replace("_try_", "")
                    logger.info(
                        f"结构化输出成功：{schema.__name__} via {strategy_name}"
                        f"（第{attempt+1}次尝试）"
                    )
                    return result
            except Exception as e:
                last_error = e
                logger.debug(f"{strategy_fn.__name__} 异常：{e}")
                continue

        if attempt < max_attempts - 1:
            logger.warning(f"{schema.__name__} 第{attempt+1}次全部策略失败，重试")

    raise ValueError(
        f"{schema.__name__} 结构化输出失败（{max_attempts}次尝试 × {len(strategies)}种策略），"
        f"最后错误：{last_error}"
    )
