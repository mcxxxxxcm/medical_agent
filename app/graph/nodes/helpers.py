"""通用工具函数和常量

包含：
    - 药物关键词常量
    - 计时装饰器（同步/异步）
    - JSON 提取与修复工具
    - LLM 结构化输出回退工具
"""
import ast
import json
import re
import time
from functools import wraps
from typing import Any, Dict, Optional

from pydantic import BaseModel

from app.core.app_logging import get_logger

logger = get_logger(__name__)

# 公共关键词集合：药物名（用于路由/症状解析/查询重写/HyDE 多处判断）
_DRUG_KEYWORDS = [
    "布洛芬", "对乙酰氨基酚", "阿莫西林", "头孢", "阿司匹林", "奥司他韦",
    "连花清瘟", "板蓝根", "感冒灵", "止咳糖浆", "蒙脱石散", "藿香正气",
    "氯雷他定", "西替利嗪", "扑尔敏", "地塞米松", "红霉素", "甲硝唑",
    "奥美拉唑", "雷尼替丁", "硝苯地平", "氨氯地平", "二甲双胍",
    "阿卡波糖", "格列美脲", "胰岛素", "阿托伐他汀", "辛伐他汀",
    "氯吡格雷", "华法林", "肝素", "青霉素", "左氧氟沙星",
    "莫西沙星", "利巴韦林", "更昔洛韦", "阿昔洛韦", "伐昔洛韦",
    "双氯芬酸", "塞来昔布", "美洛昔康", "曲马多", "可待因",
    "地氯雷他定", "氮卓斯汀", "糠酸莫米松", "丙酸氟替卡松",
]
_DRUG_INTENT_KEYWORDS = ["吃几颗", "吃几粒", "怎么吃", "怎么服用", "用量", "用法",
                          "剂量", "一天几次", "一次几颗", "一次几粒", "能吃吗",
                          "能吃不能吃", "可以吃吗", "能和", "能一起"]

def timing_decorator(node_name: str):
    """节点耗时记录装饰器"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.info(f"[{node_name}] 开始执行")

            try:
                result = func(*args, **kwargs)
                elapsed_time = (time.time() - start_time) * 1000
                logger.info(f"[{node_name}] 执行完成，耗时：{elapsed_time:.2f}ms")
                return result
            except Exception as e:
                elapsed_time = (time.time() - start_time) * 1000
                logger.error(f"[{node_name}] 执行失败，耗时：{elapsed_time:.2f}ms，错误：{str(e)}")
                raise

        return wrapper

    return decorator

def async_timing_decorator(node_name: str):
    """异步节点耗时记录装饰器"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.info(f"[{node_name}] 开始执行")

            try:
                result = await func(*args, **kwargs)
                elapsed_time = (time.time() - start_time) * 1000
                logger.info(f"[{node_name}] 执行完成，耗时：{elapsed_time:.2f}ms")
                return result
            except Exception as e:
                elapsed_time = (time.time() - start_time) * 1000
                logger.error(f"[{node_name}] 执行失败，耗时：{elapsed_time:.2f}ms，错误：{str(e)}")
                raise
        return wrapper
    return decorator

def extract_json_block(raw_text: str) -> Optional[Dict[str, Any]]:
    """从模型文本输出中尽量提取 JSON 对象（鲁棒解析）

    解析策略（按优先级）：
        1. 直接 json.loads（最快路径）
        2. 提取 Markdown 代码块中的 JSON
        3. 正则提取最大的 {...} 块
        4. json_repair 修复（处理单引号、多余逗号、缺少引号等）
        5. ast.literal_eval（处理 Python 字典格式）
    """
    if not raw_text:
        return None

    text = raw_text.strip()

    # 收集候选 JSON 字符串（按优先级排序）
    candidates = []

    # 优先级1：Markdown 代码块中的 JSON
    fenced_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    if fenced_match:
        candidates.append(fenced_match.group(1).strip())

    # 优先级2：最大的 {...} 块
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        candidates.append(brace_match.group(0).strip())

    # 优先级3：原始文本
    candidates.append(text)

    for candidate in candidates:
        # 第一层：直接 json.loads
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                _coerce_list_fields(data)
                return data
        except Exception:
            pass

        # 第二层：json_repair 修复（单引号、多余逗号、缺少引号等 3B 模型常见问题）
        try:
            from json_repair import repair_json
            repaired = repair_json(candidate)
            data = json.loads(repaired)
            if isinstance(data, dict):
                _coerce_list_fields(data)
                logger.debug(f"json_repair 修复成功")
                return data
        except ImportError:
            pass  # json_repair 未安装，跳过
        except Exception:
            pass

        # 第三层：ast.literal_eval（处理 Python 字典格式，如 {'key': 'value'}）
        try:
            data = ast.literal_eval(candidate)
            if isinstance(data, dict):
                # ast 解析的结果可能包含非 JSON 类型，转为 JSON 安全格式
                data = json.loads(json.dumps(data, default=str))
                _coerce_list_fields(data)
                logger.debug(f"ast.literal_eval 解析成功")
                return data
        except Exception:
            pass

    return None


def _coerce_list_fields(data: Dict[str, Any]) -> None:
    """将期望为列表但实际为字符串的字段自动转为列表，并展平嵌套字典元素

    3B 模型常见问题：
        "symptoms": "膝盖摔伤"  →  "symptoms": ["膝盖摔伤"]
        "body_parts": "膝盖"    →  "body_parts": ["膝盖"]
        "red_flags": [{"symptom": "头痛加重"}]  →  "red_flags": ["头痛加重"]
    """
    list_field_names = {"symptoms", "body_parts", "medications", "allergies",
                        "red_flags", "confirmed_facts", "ruled_out"}
    for field in list_field_names:
        if field in data and isinstance(data[field], str):
            # 用中文逗号、顿号、空格分割
            items = re.split(r"[，,、；;\s]+", data[field])
            data[field] = [item.strip() for item in items if item.strip()]
        elif field in data and data[field] is None:
            data[field] = []
        elif field in data and isinstance(data[field], list):
            # 展平嵌套字典元素：[{"symptom": "头痛加重"}] → ["头痛加重"]
            flattened = []
            for item in data[field]:
                if isinstance(item, dict):
                    # 取第一个非空值作为字符串
                    value = next((v for v in item.values() if v is not None), str(item))
                    flattened.append(str(value))
                elif isinstance(item, str):
                    flattened.append(item)
                elif item is not None:
                    flattened.append(str(item))
            data[field] = flattened


def invoke_structured_with_fallback(llm, prompt: str, schema: type[BaseModel]) -> BaseModel:
    """优先结构化输出，失败时回退到纯文本+JSON提取，避免重复多次调用模型"""
    try:
        structured_llm = llm.with_structured_output(schema)
        return structured_llm.invoke(prompt)
    except Exception as structured_error:
        logger.warning(f"{schema.__name__} 结构化解析失败，退回单次文本解析：{structured_error}")
        raw_response = llm.invoke(prompt)
        raw_text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
        parsed = extract_json_block(raw_text)
        if parsed is not None:
            return schema.model_validate(parsed)
        raise ValueError(f"{schema.__name__} 回退解析失败，模型原始输出：{raw_text}")


def invoke_json_once_with_fallback(
        llm,
        prompt: str,
        schema: type[BaseModel],
        fallback_parser=None,
) -> BaseModel:
    """单次模型调用后在本地完成 JSON / 文本回退解析。"""
    raw_response = llm.invoke(prompt)
    raw_text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)

    parsed = extract_json_block(raw_text)
    if parsed is not None:
        try:
            return schema.model_validate(parsed)
        except Exception as schema_error:
            logger.warning(f"{schema.__name__} JSON 校验失败，尝试本地规则解析：{schema_error}")

    if fallback_parser is not None:
        payload = fallback_parser(raw_text)
        return schema.model_validate(payload)

    raise ValueError(f"{schema.__name__} 单次调用本地解析失败，模型原始输出：{raw_text}")

