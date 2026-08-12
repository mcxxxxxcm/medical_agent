"""LangGraph节点模块
功能描述：
    定义langchain工作流中的各个节点，每个节点负责特定的处理逻辑
    节点是工作流的基本执行单元，接收状态并返回更新后的状态

设计理念：
    1、单一职责：每个节点只负责一个特定的功能
    2、状态驱动：节点通过状态进行通信，不直接依赖其他节点
    3、可测试性：节点是纯函数，易于单元测试
    4、返回更新：节点返回更新字典，不直接修改state
    5、Command路由：路由节点使用Command.goto控制流程

包引用：
    app.graph.state.MedicalAssistantState：自定义状态模块
    app.core.llm：获取LLM实例
    AgentError、LLMError：自定义异常类
"""
import json
import jieba
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.types import Command

from app.core.app_logging import get_logger
from app.core.config import get_config
from app.core.llm import (
    get_llm,
    get_local_llm,
    get_local_llm_json,
    get_rewrite_llm,
    get_symptom_llm,
    get_vision_llm,
)
from app.graph.state import MedicalAssistantState
from app.memory import get_long_term_memory
from app.rag.hybrid_retriever import get_hybrid_retriever, get_cached_hybrid_retriever
from app.skills import run_rule_based_review, get_block_template

logger = get_logger(__name__)
config = get_config()

# 从子模块导入公共工具和模型（相对导入避免循环引用）
from .helpers import (
    _DRUG_KEYWORDS,
    _DRUG_INTENT_KEYWORDS,
    async_timing_decorator,
    extract_json_block,
    invoke_json_once_with_fallback,
    invoke_structured_with_fallback,
    timing_decorator,
    _coerce_list_fields,
)
from .models import (
    ClinicalCheckpointOutput,
    GradeDocuments,
    ProfileExtractionOutput,
    QueryRewriteOutput,
    RouterOutput,
    SafetyCheckOutput,
    SymptomAnalysisOutput,
    VisionAnalysisOutput,
)


def detect_rule_based_route(question: str) -> Optional[str]:
    """优先使用规则快速判断路由（优先级：symptom > knowledge > general）

    设计理念：
        1. 先检查症状关键词（最高优先级），避免 "你好我是王艺涵发烧了" 被误判为 general
        2. 再检查知识关键词
        3. 最后检查 general（精确匹配，避免误判）
        4. 无法匹配时返回 None，交给 LLM 判断

    优化：
        使用 AC 自动机替代线性扫描，O(m) 一次扫描匹配所有关键词
    """
    text = (question or "").strip().lower()
    if not text:
        return "general"

    from app.core.keyword_matcher import get_route_symptom_matcher, get_route_knowledge_matcher

    # 1. 先检查症状关键词（最高优先级）—— AC 自动机 O(m)
    symptom_matcher = get_route_symptom_matcher()
    if symptom_matcher.contains_any(text, use_boundary=False):
        return "symptom"

    # 2. 再检查知识关键词 —— AC 自动机 O(m)
    knowledge_matcher = get_route_knowledge_matcher()
    if knowledge_matcher.contains_any(text, use_boundary=False):
        return "knowledge"

    # 3. 最后检查 general（精确匹配，避免误判）
    # 纯问候：整句话就是问候（长度很短 或 以问候开头且无实质内容）
    general_greetings = ["你好", "您好", "hello", "hi", "hey", "谢谢", "thanks", "再见", "拜拜",
                         "早上好", "晚上好"]
    general_questions = ["你是谁", "你能做什么", "你叫什么"]

    text_no_punct = re.sub(r"[，。！？,.!?\s]", "", text)
    is_pure_greeting = (
        text_no_punct in general_greetings
        or (len(text_no_punct) <= 6 and any(text_no_punct.startswith(g) for g in general_greetings))
    )
    is_general_question = any(q in text for q in general_questions)

    if is_pure_greeting or is_general_question:
        return "general"

    # 4. 无法匹配时返回 None，交给 LLM 判断
    return None


def _extract_symptoms_by_rules(question: str) -> Optional[Dict[str, Any]]:
    """基于规则的症状提取（快速路径，避免LLM调用）

    当用户描述中包含明确的症状关键词时，直接提取结构化信息。
    返回 None 表示规则无法处理，需要交给 LLM。

    优化：使用 AC 自动机替代线性遍历，O(m) 一次扫描匹配所有症状关键词
    """
    text = (question or "").strip()
    if not text:
        return None

    # 使用 AC 自动机提取症状（集中定义于 keyword_matcher.py）
    from app.core.keyword_matcher import get_symptom_matcher
    symptom_matcher = get_symptom_matcher()
    found_symptoms = symptom_matcher.get_matched_keywords(text, use_boundary=False)

    # 通用疼痛模式兜底：匹配"X疼/X痛"（如"手腕疼"、"膝盖痛"）
    # 仅在特定关键词未命中时补充，避免与已有映射重复
    if not found_symptoms:
        pain_matches = re.findall(r"([\u4e00-\u9fff]{1,4})(疼|痛)", text)
        for part_name, _ in pain_matches:
            symptom_name = f"{part_name}疼痛"
            if symptom_name not in found_symptoms:
                found_symptoms.append(symptom_name)

    if not found_symptoms:
        return None

    # 去重
    found_symptoms = list(dict.fromkeys(found_symptoms))

    # 提取严重程度（优先取最严重的）
    severity_map = {
        "很严重": "严重", "非常严重": "严重", "剧痛": "严重", "剧烈": "严重",
        "严重": "严重",
        "有点": "轻微", "轻微": "轻微", "稍微": "轻微", "一点": "轻微",
    }
    severity = None
    for keyword, level in severity_map.items():
        if keyword in text:
            severity = level
            # 不 break：继续检查是否匹配更严重的级别

    # 提取身体部位
    body_part_map = {
        "头": "头部", "脑袋": "头部",
        "嗓子": "咽喉", "喉咙": "咽喉", "咽": "咽喉",
        "胸": "胸部",
        "肚子": "腹部", "胃": "胃部", "腹": "腹部",
        "背": "背部", "后背": "背部",
        "腰": "腰部",
        "腿": "腿部", "大腿": "大腿", "小腿": "小腿",
        "脚": "足部", "脚踝": "脚踝", "脚跟": "脚跟", "脚底": "足底",
        "手": "手部", "手腕": "手腕", "手臂": "手臂", "手肘": "手肘",
        "膝": "膝盖", "膝盖": "膝盖",
        "肩": "肩膀", "肩膀": "肩膀",
        "脖": "颈部", "脖子": "颈部", "颈": "颈部",
        "眼": "眼部", "眼睛": "眼部",
        "耳": "耳部", "耳朵": "耳部",
        "鼻": "鼻部",
        "皮肤": "皮肤",
        "牙": "牙齿", "牙齿": "牙齿",
    }
    body_parts = []
    for keyword, part in body_part_map.items():
        if keyword in text:
            body_parts.append(part)
    body_parts = list(dict.fromkeys(body_parts))

    # ===== 时间锚定：相对时间 → 绝对时间戳 =====
    # 核心原则：绝不让 LLM 做时间运算，代码层完成所有时间转换和计算
    # 三层策略：L1 dateparser 规则解析 → L2 中文数字正则兜底 → L3 默认当前时刻
    system_now = datetime.now()
    onset_iso = None       # ISO 格式绝对时间（如 "2026-06-22T00:23:00"）
    onset_ts = None        # Unix 时间戳（用于精确计算差值）
    time_precision = None  # 时间精度：exact / approximate / vague
    duration = None        # 人类可读的持续时间（如 "3天"）

    # L1: dateparser 解析（支持 200+ 语言的相对时间，1-5ms）
    try:
        import dateparser
        parsed_time = dateparser.parse(
            text,
            settings={
                'RELATIVE_BASE': system_now,       # 以系统时间为基准
                'PREFER_DATES_FROM': 'past',        # 症状默认指向过去
                'RETURN_AS_TIMEZONE_AWARE': False,
                'TIMEZONE': 'Asia/Shanghai',
                'PARSERS': ['relative-time', 'absolute-time', 'custom-formats'],
            }
        )
        if parsed_time:
            onset_iso = parsed_time.strftime("%Y-%m-%dT%H:%M:%S")
            onset_ts = int(parsed_time.timestamp())
            delta = system_now - parsed_time
            days = delta.days
            hours = delta.seconds // 3600
            if days > 0:
                duration = f"{days}天"
            elif hours > 0:
                duration = f"{hours}小时"
            else:
                duration = "今天"
            time_precision = "exact"
    except ImportError:
        pass
    except Exception:
        pass

    # L2: 中文数字正则兜底（dateparser 对中文数字+单位支持有限）
    if not onset_ts:
        duration_patterns = [
            r"持续\s*([一二三四五六七八九十\d]+)\s*(天|周|个月|年)",
            r"([一二三四五六七八九十\d]+)\s*(天|周|个月|年)\s*[了以]",
            r"有\s*([一二三四五六七八九十\d]+)\s*(天|周|个月|年)",
        ]
        cn_num_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                      "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
        unit_days_map = {"天": 1, "周": 7, "个月": 30, "年": 365}

        for pattern in duration_patterns:
            m = re.search(pattern, text)
            if m:
                num_raw, unit = m.group(1), m.group(2)
                num = cn_num_map.get(num_raw)
                if num is not None:
                    days_per_unit = unit_days_map.get(unit, 1)
                    total_days = num * days_per_unit
                    onset_dt = system_now - timedelta(days=total_days)
                    onset_iso = onset_dt.strftime("%Y-%m-%dT%H:%M:%S")
                    onset_ts = int(onset_dt.timestamp())
                    duration = f"{num}{unit}"
                    time_precision = "exact"
                break

        # "几天了"等模糊表达
        if not onset_ts:
            vague_match = re.search(r"(几)\s*(天|周|个月)\s*[了以]", text)
            if vague_match:
                duration = f"?{vague_match.group(2)}"
                time_precision = "vague"

    # L3: 未提及任何时间 → 默认当前时刻（"我现在头痛"）
    if not onset_ts and not duration:
        onset_iso = system_now.strftime("%Y-%m-%dT%H:%M:%S")
        onset_ts = int(system_now.timestamp())
        duration = "今天"
        time_precision = "default"  # 系统推断，非用户明确表述

    # 构建结果
    result = {
        "symptoms": found_symptoms,
        "severity": severity,
        "body_parts": body_parts if body_parts else None,
        "duration": duration,
        "onset_date": onset_iso,         # 绝对时间（ISO格式）
        "onset_ts": onset_ts,            # Unix 时间戳（精确计算用）
        "time_precision": time_precision, # 时间精度标记
        "additional_info": None,
    }

    return result


def _calculate_duration_from_checkpoint(
    question: str, current_symptoms: List[str], checkpoint: Dict[str, Any]
) -> Optional[str]:
    """从临床快照中计算症状持续时间（代码层计算，绝不让LLM算）

    核心原则：
        - 使用 Unix 时间戳做精确差值计算
        - 计算结果作为事实注入 Prompt，LLM 只负责读取

    场景：用户首轮说"我现在头痛"→ onset_ts 记录到快照，
         后续追问"头痛几天了"→ 代码层计算 current_ts - onset_ts

    Args:
        question: 当前用户问题
        current_symptoms: 当前提取到的症状列表
        checkpoint: 临床快照（含 symptom_onset_dates）

    Returns:
        持续时间字符串（如"3天2小时"），或 None
    """
    # 只在用户追问持续时间时触发（"几天了"/"多久了"/"多长时间"）
    duration_question_patterns = [r"几天", r"多久", r"多长", r"多长时间", r"多长时间了"]
    if not any(re.search(p, question) for p in duration_question_patterns):
        return None

    # 从快照中取出症状首发日期映射
    onset_dates = checkpoint.get("symptom_onset_dates", {})
    if not onset_dates:
        return None

    system_now = datetime.now()
    system_now_ts = int(system_now.timestamp())

    # 匹配当前症状与记录的 onset_ts
    for symptom in current_symptoms:
        for recorded_symptom, onset_info in onset_dates.items():
            if symptom in recorded_symptom or recorded_symptom in symptom:
                # 优先用 Unix 时间戳精确计算
                onset_ts = onset_info.get("ts") if isinstance(onset_info, dict) else None
                if onset_ts and isinstance(onset_ts, (int, float)):
                    delta_seconds = system_now_ts - int(onset_ts)
                    if delta_seconds >= 0:
                        days = delta_seconds // 86400
                        hours = (delta_seconds % 86400) // 3600
                        if days > 0:
                            return f"{days}天{hours}小时" if hours > 0 else f"{days}天"
                        elif hours > 0:
                            return f"{hours}小时"
                        else:
                            minutes = delta_seconds // 60
                            return f"{minutes}分钟" if minutes > 0 else "刚刚"

                # 兜底：用 ISO 字符串解析
                onset_iso = onset_info.get("iso") if isinstance(onset_info, dict) else onset_info
                if onset_iso and isinstance(onset_iso, str):
                    try:
                        onset_dt = datetime.strptime(onset_iso[:19], "%Y-%m-%dT%H:%M:%S")
                        delta = system_now - onset_dt
                        days = delta.days
                        hours = delta.seconds // 3600
                        if days > 0:
                            return f"{days}天{hours}小时" if hours > 0 else f"{days}天"
                        elif hours > 0:
                            return f"{hours}小时"
                    except (ValueError, TypeError):
                        continue

    return None


def parse_router_output(raw_text: str) -> Optional[str]:
    """兼容 JSON 和纯文本标签的路由结果解析"""
    text = (raw_text or "").strip()
    if not text:
        return None

    parsed = extract_json_block(text)
    if parsed and isinstance(parsed.get("question_type"), str):
        return normalize_router_label(parsed.get("question_type"))

    plain_text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()
    plain_text = plain_text.strip(" \n\t\"'：:")
    return normalize_router_label(plain_text)


def parse_symptom_text(raw_text: str) -> Dict[str, Any]:
    """兼容非 JSON 的症状提取结果"""
    text = (raw_text or "").strip()
    if not text:
        return {}

    def extract_list(field_names: List[str]) -> Optional[List[str]]:
        for field_name in field_names:
            match = re.search(rf"{field_name}\s*[:：]\s*(.+)", text, re.IGNORECASE)
            if match:
                value = match.group(1).splitlines()[0].strip()
                items = [item.strip(" ，,、；;。") for item in re.split(r"[，,、；;\s]+", value) if item.strip(" ，,、；;。")]
                return items or None
        return None

    def extract_value(field_names: List[str]) -> Optional[str]:
        for field_name in field_names:
            match = re.search(rf"{field_name}\s*[:：]\s*(.+)", text, re.IGNORECASE)
            if match:
                return match.group(1).splitlines()[0].strip(" 。")
        return None

    severity = extract_value(["severity", "严重程度"]) or next(
        (level for level in ["轻微", "中等", "严重"] if level in text),
        None,
    )

    return {
        "symptoms": extract_list(["symptoms", "症状"]),
        "severity": severity,
        "body_parts": extract_list(["body_parts", "身体部位", "部位"]),
        "duration": extract_value(["duration", "持续时间"]),
        "additional_info": extract_value(["additional_info", "附加信息", "补充信息"]),
    }


def normalize_query_text(raw_text: str, original_question: str) -> str:
    """规范化查询重写结果"""
    text = (raw_text or "").strip()
    if not text:
        return original_question

    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = text.replace("```", "").strip()

    parsed = extract_json_block(text)
    if parsed and isinstance(parsed.get("rewritten_query"), str):
        text = parsed["rewritten_query"].strip()

    text = re.sub(r"^(rewritten_query|query|重写后查询|重写查询)\s*[:：]\s*", "", text, flags=re.IGNORECASE)
    text = text.strip(" \n\t\"'。")
    return text or original_question


def is_same_query(left: Optional[str], right: Optional[str]) -> bool:
    """判断两个查询是否等价，避免无意义重试"""
    normalize = lambda value: re.sub(r"\s+", "", (value or "")).strip("。？！?!，,；;")
    return normalize(left) == normalize(right)


def build_no_results_answer(question: str) -> str:
    """构建完全无匹配时的拒答回复（v9.15）"""
    return (
        f"抱歉，当前知识库未收录与\u201c{question}\u201d直接相关的可靠资料。\n"
        "建议您：\n"
        "1. 补充更具体的信息（如症状持续时间、伴随表现、既往病史）；\n"
        "2. 提供更准确的疾病名称或药品名称；\n"
        "3. 如症状明显加重或伴有高热/剧烈疼痛/呼吸困难，请及时就医。"
    )


def build_incomplete_answer(partial_info: str, missing_items: list) -> str:
    """构建信息不完整时的部分回答 + 缺失声明"""
    missing_text = "\u3001".join(missing_items)
    return (
        f"{partial_info}\n\n"
        f"\u26a0\ufe0f 关于{missing_text}\uff0c当前资料未提及\uff0c请务必咨询医生确认\u3002\n"
        "\n\u26a0\ufe0f 以上建议仅供参考，如有疑问请及时就医"
    )


def build_conflict_answer(question: str, conflicts: list) -> str:
    """构建多源冲突回答：展示分歧 + 建议核实"""
    conflict_lines = []
    for c in conflicts:
        conflict_lines.append(f"- {c.get('source', '\u672a\u77e5\u6765\u6e90')}\uff1a{c.get('content', '')}")
    conflict_text = "\n".join(conflict_lines)
    return (
        f"\u5173\u4e8e\u201c{question}\u201d\uff0c\u4e0d\u540c\u53c2\u8003\u6765\u6e90\u5b58\u5728\u5dee\u5f02\uff1a\n{conflict_text}\n\n"
        "\u5efa\u8bae\u4ee5\u60a8\u5c31\u8bca\u533b\u9662\u7684\u76f8\u5173\u79d1\u5ba4\u610f\u89c1\u4e3a\u51c6\u3002\u5982\u6709\u7591\u95ee\u8bf7\u54a8\u8be2\u4e13\u4e1a\u533b\u751f\u3002\n"
        "\n\u26a0\ufe0f \u4ee5\u4e0a\u5efa\u8bae\u4ec5\u4f9b\u53c2\u8003\uff0c\u5982\u6709\u7591\u95ee\u8bf7\u53ca\u65f6\u5c31\u533b"
    )


def build_out_of_scope_answer(question: str) -> str:
    """构建超出能力边界的拒答回复"""
    return (
        f"\u62b1\u6b49\uff0c\u201c{question}\u201d\u6d89\u53ca\u8bca\u65ad\u6216\u5904\u65b9\u5efa\u8bae\uff0c\u8d85\u51fa\u6211\u7684\u80fd\u529b\u8303\u56f4\u3002"
        "\u6211\u65e0\u6cd5\u63d0\u4f9b\u8bca\u65ad\u610f\u89c1\u6216\u5904\u65b9\u5efa\u8bae\uff0c\u60a8\u7684\u75c7\u72b6\u53ef\u80fd\u4e0e\u591a\u79cd\u60c5\u51b5\u76f8\u5173\uff0c"
        "\u5efa\u8bae\u5c3d\u5feb\u524d\u5f80\u76f8\u5173\u79d1\u5ba4\u9762\u8bca\uff0c\u7531\u4e13\u4e1a\u533b\u751f\u8fdb\u884c\u8bc4\u4f30\u3002"
    )


@timing_decorator("路由")
def router_node(state: MedicalAssistantState) -> Command:
    """路由节点

    功能描述：
        根据用户问题判断问题类型，使用Command.goto决定后续执行路径
        三层路由策略：规则优先 → 上下文感知 → 主模型分类

    Args：
        state：当前状态，包含question字段

    Returns：
        Command：包含路由目标的Command对象

    路由策略：
        1. 规则优先（0ms）：关键词匹配，处理确定性高的场景
        2. 上下文感知（0ms）：对话历史中已有症状/知识类对话时，
           当前轮大概率也是同类问题（如"布洛芬没用"接在"头痛"之后）
        3. 主模型分类（~1-2s）：用 glm-4-flash 做意图分类，
           准确率远高于 3B 本地模型，路由是管道最关键决策点，
           用主模型多花 1s 但走错路径浪费 10s+

    路由类型：
        symptom：症状咨询，需要症状解析
        knowledge：知识查询，直接检索
        general：一般问题，直接回答
    """
    logger.info("路由节点开始执行")

    question = state.get("question", "")
    image_base64 = state.get("image_base64")

    # 优先检测图片：有图片走vision分支
    if image_base64:
        logger.info("检测到图片输入，路由到vision分支")
        return Command(goto="vision_analysis")

    # ===== 第一层：规则优先（0ms） =====
    rule_based_route = detect_rule_based_route(question)
    route_layer = "rule"  # 记录命中的路由层
    if rule_based_route:
        logger.info(f"规则路由命中：question_type={rule_based_route}")
        question_type = rule_based_route
    else:
        # ===== 第二层：上下文感知（0ms） =====
        context_route = _detect_route_from_context(state)
        if context_route:
            logger.info(f"上下文感知路由命中：question_type={context_route}")
            question_type = context_route
            route_layer = "context"
        else:
            # ===== 第三层：主模型分类（~1-2s） =====
            question_type = _llm_route(question)
            route_layer = "llm"

    question_type = normalize_router_label(question_type)
    logger.info(f"问题类型：{question_type}（路由层：{route_layer}）")

    # 记录路由结果到 state（供后续节点和 metrics 使用）
    _record_route_metrics(state, question, question_type, route_layer)

    # 常规类型general就直接使用llm进行回复，不用调用RAG
    # v9.15: 通过 update 将 question_type 传递给后续节点
    if question_type == "symptom":
        return Command(goto="symptom_analysis", update={"question_type": question_type})
    elif question_type == "knowledge":
        return Command(goto="query_rewrite", update={"question_type": question_type})
    else:
        return Command(goto="direct_answer", update={"question_type": question_type})


def _record_route_metrics(state: dict, question: str, question_type: str, route_layer: str):
    """记录路由分层指标到 MetricsCollector

    指标用途：
        - 分层命中率：rule/context/llm 各处理了多少比例
        - 路由分布：symptom/knowledge/general 占比
        - 关联 feedback：👎 时可反查路由层和类型

    不影响主流程，异常时静默降级。
    """
    try:
        from app.core.metrics import get_metrics_collector
        collector = get_metrics_collector()
        request_id = state.get("request_id", "")
        if not request_id:
            import uuid
            request_id = str(uuid.uuid4())[:8]
        thread_id = ""
        messages = state.get("messages", [])
        if messages:
            from langchain_core.messages import BaseMessage
            # 从 messages 中提取 thread_id 不可用，使用 user_id 替代
        user_id = state.get("user_id", "")

        collector.record_node(
            request_id=request_id,
            node_name="router",
            duration_ms=0,  # 路由耗时已由 timing_decorator 记录，这里只记录元数据
            route_type=question_type,
            thread_id=thread_id,
            metadata={
                "route_layer": route_layer,
                "question_preview": question[:100],
                "user_id": user_id,
            },
        )
    except Exception as e:
        logger.debug(f"路由指标记录失败（不影响主流程）：{e}")


def _detect_route_from_context(state: MedicalAssistantState) -> Optional[str]:
    """上下文感知路由：根据对话历史推断当前问题类型

    核心思路：
        如果对话历史中已有症状/知识类对话，当前轮大概率也是同类。
        例如：用户先问"头痛怎么办"，再问"布洛芬没用"，
        第二句话单独看没有症状关键词，但上下文明确是症状咨询。

    注意：不再使用硬编码规则检测当前问题类型，
    路由判断交给规则路由（detect_rule_based_route）和 LLM 路由。
    """
    messages = state.get("messages", [])
    if not messages:
        return None

    question = state.get("question", "").strip()

    # 只看最近 4 条消息（2 轮对话）
    recent = messages[-4:] if len(messages) > 4 else messages

    has_symptom_context = False
    has_knowledge_context = False

    for msg in recent:
        content = ""
        if isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
        elif isinstance(msg, AIMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            # AI 消息中包含用药建议/症状分析 → 说明之前是症状咨询
            symptom_indicators = ["用药", "服用", "剂量", "副作用", "建议您", "药物", "缓解"]
            if any(kw in content for kw in symptom_indicators):
                has_symptom_context = True
            knowledge_indicators = ["是指", "是一种", "特征是", "主要包括"]
            if any(kw in content for kw in knowledge_indicators):
                has_knowledge_context = True
            continue  # AI 消息只做辅助判断

        # 用户消息：检查是否包含症状/知识关键词
        if content:
            route = detect_rule_based_route(content)
            if route == "symptom":
                has_symptom_context = True
            elif route == "knowledge":
                has_knowledge_context = True

    # 当前问题是否像追问（短句、含代词/指代词）
    follow_up_indicators = [
        "没用", "无效", "不管用", "不好使", "还有呢", "然后呢", "还有吗",
        "换什么", "还能", "可以吗", "呢", "那", "还有", "除了",
    ]
    is_follow_up = len(question) <= 15 or any(kw in question for kw in follow_up_indicators)

    if is_follow_up:
        if has_symptom_context:
            return "symptom"
        if has_knowledge_context:
            return "knowledge"

    # 非追问但有明确的上下文倾向
    if has_symptom_context and not has_knowledge_context:
        return "symptom"
    if has_knowledge_context and not has_symptom_context:
        return "knowledge"

    return None


def _llm_route(question: str) -> str:
    """本地模型路由：使用 qwen2.5:3b 做意图分类

    v9.2: 使用 invoke_structured 三层降级策略（Tool Calling → JSON Mode → 纯文本）
    + RouterOutput Pydantic 校验。
    """
    try:
        from app.graph.nodes.prompts import ROUTER_PROMPT
        from app.graph.nodes.models import RouterOutput
        from app.graph.nodes.structured_output import invoke_structured

        llm = get_local_llm_json()
        messages = ROUTER_PROMPT.format_messages(question=question)

        result: RouterOutput = invoke_structured(
            llm, messages, RouterOutput, max_attempts=2,
        )
        route = result.question_type
        logger.info(f"本地模型路由结果（Pydantic校验通过）：{route}")
        return route

    except Exception as e:
        logger.warning(f"本地模型路由失败，兜底 general：{e}")
        return "general"

@timing_decorator("症状解析")
def symptom_analysis_node(state: MedicalAssistantState) -> Dict[str, Any]:
    """症状解析节点（v8.4 重构：规则优先 + 时间锚定独立 + LLM 兜底移除）

    设计决策（v8.4）：
        移除 Layer 3 LLM 兜底。原因：
        1. 症状解析和查询改写做的是同一件事（理解查询医学语义），两次 LLM 调用是重复劳动
        2. LLM 兜底 2.8s 但结果常校验失败（如 severity="轻至中度"），白跑
        3. symptoms=None 时下游完全能处理：追问引导为空，RAG prompt 不追加，LLM 照常生成
        4. 时间锚定已独立于症状关键词，规则未命中时 onset_ts 仍能计算

    策略（按优先级）：
        1. 规则提取命中 → 直接返回（<5ms）
        2. 追问短路：有对话历史 → 跳过，症状从快照继承
        3. 首轮规则未命中 → 时间锚定仍执行 + symptoms=None → 降级为原始查询检索
    """
    logger.info("症状解析节点开始执行")

    question = state.get("question", "")
    messages = state.get("messages", [])
    _symptom_words = ["疼", "痛", "发烧", "咳嗽", "恶心", "呕吐", "腹泻", "头晕",
                      "胸闷", "过敏", "出血", "流血", "血", "骨折", "肿", "痒", "麻", "炎"]

    # 用药咨询快速跳过
    has_drug = any(kw in question for kw in _DRUG_KEYWORDS)
    if has_drug and not any(kw in question for kw in _symptom_words):
        logger.info(f"用药咨询问题，跳过症状解析：{question}")
        return {"symptoms": {"symptoms": [], "severity": None, "body_parts": [], "duration": None, "onset_date": None, "additional_info": None}}

    # 1. 规则提取
    rule_result = _extract_symptoms_by_rules(question)
    if rule_result:
        logger.info(f"规则提取症状成功，跳过LLM：{rule_result}")
        checkpoint = state.get("clinical_checkpoint")
        if checkpoint and not rule_result.get("duration"):
            calculated_duration = _calculate_duration_from_checkpoint(
                question, rule_result.get("symptoms", []), checkpoint
            )
            if calculated_duration:
                rule_result["duration"] = calculated_duration
                logger.info(f"从快照计算持续时间：{calculated_duration}")
        return {"symptoms": rule_result}

    # 2. 追问短路：有历史 → 症状由下方快照继承
    if messages:
        logger.info(f"追问规则未命中，短路跳过LLM（{len(messages)}条历史）：{question[:30]}")
        return _symptoms_with_checkpoint_fallback(
            {"symptoms": [], "severity": None, "body_parts": [],
             "duration": None, "onset_date": None, "additional_info": None},
            state,
        )

    # 3. 首轮规则未命中 → 不调 LLM，降级为原始查询检索
    # 时间锚定仍然执行（独立于症状关键词），确保 clinical_checkpoint 有 onset_ts
    logger.info(f"规则未命中，降级为原始查询检索（不调LLM）：{question[:50]}")
    time_grounding = _extract_time_grounding(question)
    return {"symptoms": {
        "symptoms": [], "severity": None, "body_parts": [],
        "duration": time_grounding.get("duration"),
        "onset_date": time_grounding.get("onset_iso"),
        "onset_ts": time_grounding.get("onset_ts"),
        "time_precision": time_grounding.get("time_precision"),
        "additional_info": None,
    }}


def _extract_time_grounding(question: str) -> Dict[str, Any]:
    """独立的时间锚定函数，不依赖症状关键词

    核心原则：绝不让 LLM 做时间运算，代码层完成所有时间转换和计算
    三层策略：L1 dateparser → L2 中文数字正则 → L3 默认当前时刻

    v8.4：从 _extract_symptoms_by_rules 中提取，独立于症状解析。
    即使症状规则未命中，时间锚定仍能执行，确保 clinical_checkpoint 有 onset_ts。
    """
    text = (question or "").strip()
    system_now = datetime.now()
    onset_iso = None
    onset_ts = None
    time_precision = None
    duration = None

    # L1: dateparser 解析
    try:
        import dateparser
        parsed_time = dateparser.parse(
            text,
            settings={
                'RELATIVE_BASE': system_now,
                'PREFER_DATES_FROM': 'past',
                'RETURN_AS_TIMEZONE_AWARE': False,
                'TIMEZONE': 'Asia/Shanghai',
                'PARSERS': ['relative-time', 'absolute-time', 'custom-formats'],
            }
        )
        if parsed_time:
            onset_iso = parsed_time.strftime("%Y-%m-%dT%H:%M:%S")
            onset_ts = int(parsed_time.timestamp())
            delta = system_now - parsed_time
            days = delta.days
            hours = delta.seconds // 3600
            if days > 0:
                duration = f"{days}天"
            elif hours > 0:
                duration = f"{hours}小时"
            else:
                duration = "今天"
            time_precision = "exact"
    except ImportError:
        pass
    except Exception:
        pass

    # L2: 中文数字正则兜底
    if not onset_ts:
        duration_patterns = [
            r"持续\s*([一二三四五六七八九十\d]+)\s*(天|周|个月|年)",
            r"([一二三四五六七八九十\d]+)\s*(天|周|个月|年)\s*[了以]",
            r"有\s*([一二三四五六七八九十\d]+)\s*(天|周|个月|年)",
        ]
        cn_num_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                      "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
        unit_days_map = {"天": 1, "周": 7, "个月": 30, "年": 365}

        for pattern in duration_patterns:
            m = re.search(pattern, text)
            if m:
                num_raw, unit = m.group(1), m.group(2)
                num = cn_num_map.get(num_raw)
                if num is not None:
                    days_per_unit = unit_days_map.get(unit, 1)
                    total_days = num * days_per_unit
                    onset_dt = system_now - timedelta(days=total_days)
                    onset_iso = onset_dt.strftime("%Y-%m-%dT%H:%M:%S")
                    onset_ts = int(onset_dt.timestamp())
                    duration = f"{num}{unit}"
                    time_precision = "exact"
                break

        if not onset_ts:
            vague_match = re.search(r"(几)\s*(天|周|个月)\s*[了以]", text)
            if vague_match:
                duration = f"?{vague_match.group(2)}"
                time_precision = "vague"

    # L3: 未提及任何时间 → 默认当前时刻
    if not onset_ts and not duration:
        onset_iso = system_now.strftime("%Y-%m-%dT%H:%M:%S")
        onset_ts = int(system_now.timestamp())
        duration = "今天"
        time_precision = "default"

    return {
        "onset_iso": onset_iso,
        "onset_ts": onset_ts,
        "duration": duration,
        "time_precision": time_precision,
    }


def _symptoms_with_checkpoint_fallback(symptoms_payload: dict, state: MedicalAssistantState) -> dict:
    """空症状时从临床快照补全（追问场景）"""
    if symptoms_payload.get("symptoms"):
        return {"symptoms": symptoms_payload}

    checkpoint = state.get("clinical_checkpoint")
    if not checkpoint:
        return {"symptoms": symptoms_payload}

    cp_symptoms = checkpoint.get("symptoms") or []
    if not cp_symptoms:
        return {"symptoms": symptoms_payload}

    symptoms_payload["symptoms"] = list(cp_symptoms)
    symptoms_payload["body_parts"] = checkpoint.get("body_parts") or []
    symptoms_payload["severity"] = checkpoint.get("severity")
    cp_onset = checkpoint.get("symptom_onset_dates") or {}
    if cp_onset and not symptoms_payload.get("duration"):
        first_onset = next(iter(cp_onset.values()), {}) if isinstance(cp_onset, dict) else None
        if isinstance(first_onset, dict) and first_onset.get("iso"):
            symptoms_payload["onset_date"] = first_onset.get("iso")
            symptoms_payload["onset_ts"] = first_onset.get("ts")
    logger.info(f"追问症状继承：从临床快照补充 {cp_symptoms}")
    return {"symptoms": symptoms_payload}


@timing_decorator("问题拆解")
def question_decompose_node(state: MedicalAssistantState) -> Dict[str, Any]:
    """长问题拆解节点：检测复合问题并拆解为独立子问题

    触发条件（不用 LLM 的前置检测）：
    1. 问题长度 > 20 字符（短问题不可能复合）
    2. 问题中包含分隔标记（？？、？和、？另外、？还有、同时、另外、而且、以及）

    LLM 拆解：调用 QUESTION_DECOMPOSE_PROMPT + invoke_structured
    结果存入 state["sub_questions"]，供 knowledge_retrieval_node 并行检索
    """
    question = state.get("question", "")
    sub_questions_existing = state.get("sub_questions")

    # 已有子问题列表（自纠正重试），直接返回
    if sub_questions_existing and len(sub_questions_existing) > 1:
        logger.info(f"问题拆解：已有 {len(sub_questions_existing)} 个子问题，跳过")
        return {}

    # 前置检测：复合标记检测（移除长度阈值，"发烧？便秘？"仅 8 字符但也应拆解）
    _COMPOUND_CONNECTORS = [
        "另外", "还有", "同时", "而且", "以及", "并且", "还有个", "再问",
    ]
    question_mark_count = question.count("？") + question.count("?")
    has_connector = any(c in question for c in _COMPOUND_CONNECTORS)

    # 条件1：至少 2 个问号 或 1 个问号 + 连接词
    has_punctuation_or_connector = question_mark_count >= 2 or (question_mark_count >= 1 and has_connector)

    # 条件2（无标点兜底）：症状实体检测——用 AC 自动机识别多个不同症状
    # 解决"发烧怎么处理便秘怎么处理"这类无标点复合问题
    has_multi_symptoms = False
    matched_symptoms = []
    if not has_punctuation_or_connector:
        try:
            from app.core.keyword_matcher import get_symptom_matcher
            symptom_matcher = get_symptom_matcher()
            matched_symptoms = symptom_matcher.get_matched_keywords(question, use_boundary=False)
            # 去重后 ≥2 个不同症状 → 判定复合问题
            if len(matched_symptoms) >= 2:
                has_multi_symptoms = True
        except Exception as e:
            logger.debug(f"症状实体检测失败：{e}")

    if not has_punctuation_or_connector and not has_multi_symptoms:
        logger.info(f"问题拆解：未检测到复合标记（{question_mark_count} 个问号，无连接词，症状数={len(matched_symptoms)}），不拆解")
        return {"sub_questions": [question]}

    if has_multi_symptoms and not has_punctuation_or_connector:
        logger.info(f"问题拆解：检测到 {len(matched_symptoms)} 个症状实体（{matched_symptoms}），无标点/连接词，判定复合问题")
    else:
        logger.info(f"问题拆解：检测到复合标记（{question_mark_count} 个问号，连接词={has_connector}），调用 LLM 拆解")

    # LLM 拆解（带超时保护）
    try:
        from app.graph.nodes.prompts import QUESTION_DECOMPOSE_PROMPT
        from app.graph.nodes.models import QuestionDecomposeOutput
        from app.graph.nodes.structured_output import invoke_structured

        llm = get_local_llm_json()
        messages = QUESTION_DECOMPOSE_PROMPT.format_messages(question=question)

        # v9.6.1: max_attempts=1 + force_strategy="text_parse"
        # Ollama 本地模型不支持 Tool Calling / JSON Mode，直接用 Layer 3 避免无效调用
        # 超时保护：2s 内必须返回，否则降级规则拆解
        decompose_deadline = time.time() + 2.0  # 2 秒超时

        result: QuestionDecomposeOutput = invoke_structured(
            llm, messages, QuestionDecomposeOutput, max_attempts=1, force_strategy="text_only",
        )

        if time.time() > decompose_deadline:
            raise TimeoutError("问题拆解超时（>2s），降级规则拆解")

        if result.need_decompose and len(result.sub_questions) >= 2:
            sub_questions = result.sub_questions[:4]  # 最多 4 个
            logger.info(f"问题拆解完成：'{question[:30]}...' → {len(sub_questions)} 个子问题")
            for i, sq in enumerate(sub_questions):
                logger.info(f"  子问题{i + 1}：{sq}")
            return {"sub_questions": sub_questions}
        else:
            logger.info(f"问题拆解：LLM 判定为单一问题，不拆解")
            return {"sub_questions": [question]}

    except Exception as e:
        logger.warning(f"问题拆解 LLM 调用失败，使用规则拆解：{e}")
        # 降级：按问号切分
        sub_questions = _rule_based_decompose(question)
        if len(sub_questions) >= 2:
            logger.info(f"规则拆解：'{question[:30]}...' → {len(sub_questions)} 个子问题")
        return {"sub_questions": sub_questions}


def _rule_based_decompose(question: str) -> List[str]:
    """规则拆解降级：优先按问号切分，问号不足时按症状边界切分（不用 LLM）"""
    import re
    # 按？或? 切分
    parts = re.split(r'[？?]', question)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return [question]
    # 有问号且切出多段
    if len(parts) >= 2:
        return [p + "？" for p in parts]

    # 无问号或仅1段 → 尝试按症状实体边界切分
    # 例："发烧怎么处理便秘怎么处理" → ["发烧怎么处理", "便秘怎么处理"]
    try:
        from app.core.keyword_matcher import get_symptom_matcher
        symptom_matcher = get_symptom_matcher()
        matches = symptom_matcher.findall(question, use_boundary=False)
        if len(matches) >= 2:
            # 按症状在文本中的位置切分
            # 每个症状实体的 start 位置作为新子问题的起始
            sub_questions = []
            for i, m in enumerate(matches):
                start = m["start"]
                end = matches[i + 1]["start"] if i + 1 < len(matches) else len(question)
                fragment = question[start:end].strip()
                if fragment:
                    sub_questions.append(fragment)
            if len(sub_questions) >= 2:
                logger.info(f"按症状边界切分：{sub_questions}")
                return sub_questions
    except Exception as e:
        logger.debug(f"症状边界切分失败：{e}")

    return [question]


@timing_decorator("知识检索")
def knowledge_retrieval_node(state: MedicalAssistantState) -> Dict[str, Any]:
    """知识检索节点
    
    功能描述：
        从向量库中检索与用户问题相关的医疗文档
        知识RAG流程的核心节点，负责文档检索
    
    Args：
        state：当前状态，包含question字段
    
    Returns：
        Dict[str, Any]：需要更新的状态字段
    
    设计理念：
        1、向量检索：使用向量相似度来检索相关文档
        2、来源追踪：记录文档来源
        3、数量控制：限制返回文档数量
        4、性能优化：记录检索耗时
    
    检索参数：
        k：返回的文档数量，默认为3
        search_type：检索类型，默认为similarity
    """
    logger.info("知识检索节点开始执行")

    # v9.0 查询预处理：纠错 + 同义词标准化 + 停用词清理
    # 提高 BM25 命中率和 Embedding 质量
    def _preprocess_query(q: str) -> str:
        if not q:
            return q
        # 同义词标准化（常见口语→书面语）
        _SYNONYMS = {
            "咋办": "怎么办", "怎么治": "怎么办", "如何治疗": "怎么办",
            "头疼": "头痛", "拉肚子": "腹泻", "拉稀": "腹泻",
            "糖料病": "糖尿病", "糖料": "糖尿病",
            "血压高": "高血压", "血糖高": "高血糖",
            "哮喘病": "哮喘", "气管炎": "支气管炎",
            "退烧": "退热", "发烧": "发热",
            "止疼": "止痛", "镇痛": "止痛",
            "咳嗽有痰": "咳嗽咳痰", "干咳": "咳嗽无痰",
            "感冒灵颗粒": "感冒灵", "芬必得": "布洛芬缓释胶囊",
            "美林": "布洛芬混悬液", "泰诺": "对乙酰氨基酚",
            "扑热息痛": "对乙酰氨基酚", "芬必得": "布洛芬",
            # v9.3: 口语症状词 → 书面语（修复"流鼻血"召回错误）
            "流鼻血": "鼻出血", "鼻子出血": "鼻出血", "鼻流血": "鼻出血",
            "拉血": "便血", "大便出血": "便血",
            "吐血": "咯血", "咳血": "咯血",
            "尿血": "血尿",
            # v9.16: 口语化同义词扩充（+35条，覆盖7大类）
            # 消化系统
            "肚子疼": "腹痛", "胃疼": "胃痛", "反酸": "胃酸反流",
            "烧心": "胃灼热", "吐了": "呕吐", "恶心想吐": "恶心呕吐",
            "拉不出": "便秘", "胀气": "腹胀",
            # 呼吸系统
            "流鼻涕": "鼻分泌物增多", "嗓子疼": "咽痛", "喉咙疼": "咽痛",
            "喘不上气": "呼吸困难", "气短": "呼吸困难",
            # 神经系统
            "偏头疼": "偏头痛", "脑袋嗡嗡响": "耳鸣", "睡不着": "失眠",
            "老犯困": "嗜睡",
            # 妇科
            "不来月经": "闭经",
            # 皮肤
            "起红点": "皮疹", "起疹子": "皮疹", "痒": "瘙痒", "脱皮": "脱屑",
            # 全身
            "起不来": "乏力", "没劲": "乏力", "没力气": "乏力",
            "发低烧": "低热", "发高烧": "高热", "打冷战": "畏寒",
            "出冷汗": "冷汗", "抽筋": "肌肉痉挛",
            # 用药相关
            "退烧药": "解热镇痛药", "消炎药": "抗感染药",
            "感冒药": "感冒用药", "止痛药": "镇痛药", "拉肚子药": "止泻药",
        }
        for colloquial, standard in _SYNONYMS.items():
            q = q.replace(colloquial, standard)
        # 去除语气词前缀
        q = re.sub(r'^(我想问一下|请问|问一下|请教一下|咨询一下|麻烦问一下)\s*', '', q)
        # 去除语气词后缀
        q = re.sub(r'\s*(啊|呀|呢|吧|嘛|哦|嗯|哈)+[？?！!。]*$', '', q)
        return q.strip()

    # 获取原始查询和重写查询
    original_query = state.get("question", "")
    rewritten_query = state.get("rewritten_query", "")
    hyde_answer = state.get("hyde_answer")
    retrieval_attempts = int(state.get("retrieval_attempts") or 0) + 1

    # 检索时优先使用重写查询（提高检索质量）
    search_query = rewritten_query or original_query

    # v9.0 查询预处理：对原始查询做同义词/纠错/语气词清理
    # 重写查询已经是高质量的自包含查询，不需要再预处理
    # v9.4 修复：query_rewrite_node 跳过重写时设 rewritten_query=question（非空），
    #   导致 `not rewritten_query` 永远为 False，预处理从未执行
    #   改为：只有真正被 LLM 重写过的查询才跳过预处理
    _was_actually_rewritten = rewritten_query and rewritten_query != original_query
    if not _was_actually_rewritten:
        preprocessed = _preprocess_query(original_query)
        if preprocessed != original_query:
            logger.info(f"查询预处理：'{original_query}' → '{preprocessed}'")
            search_query = preprocessed

    try:
        # v9.16: 根据 question_type 动态选择 K 值
        # symptom 类型需要更多候选（多跳推理），knowledge 类型5个足够
        question_type = state.get("question_type", "general")
        if question_type == "symptom":
            k = config.RETRIEVAL_K_SYMPTOM
        elif question_type == "knowledge":
            k = config.RETRIEVAL_K_KNOWLEDGE
        else:
            k = config.RETRIEVAL_K_DEFAULT
        retriever = get_cached_hybrid_retriever(k=k, alpha=0.5, use_reranker=True, rerank_top_k=10)

        # v9.6: 多子问题并行检索
        sub_questions = state.get("sub_questions") or [search_query]
        if len(sub_questions) > 1:
            logger.info(f"多子问题并行检索：{len(sub_questions)} 个子问题")
            start_time = time.time()

            # v9.6.2: 真正并行检索（ThreadPoolExecutor），而非串行 for 循环
            # 每个子问题的 retriever.invoke 包含 Embedding + Dense + BM25 + Reranker，
            # 串行执行 2 个子问题约 2×2s=4s，并行约 2s
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def _retrieve_sub_question(idx: int, sub_q: str) -> tuple:
                """单个子问题检索（线程安全）"""
                preprocessed_sub_q = _preprocess_query(sub_q)
                if preprocessed_sub_q != sub_q:
                    logger.info(f"  子问题{idx + 1} 预处理：'{sub_q}' → '{preprocessed_sub_q}'")
                    sub_q = preprocessed_sub_q
                sub_docs = retriever.invoke(
                    sub_q,
                    original_query=sub_q,  # 用子问题自身作为缓存 key，避免误命中其他子问题
                )
                for doc in sub_docs:
                    doc.metadata["sub_question"] = sub_q
                    doc.metadata["sub_question_idx"] = idx + 1
                return idx, sub_q, sub_docs

            all_sub_docs = []
            with ThreadPoolExecutor(max_workers=min(len(sub_questions), 4)) as pool:
                futures = {pool.submit(_retrieve_sub_question, i, sq): i for i, sq in enumerate(sub_questions)}
                for future in as_completed(futures):
                    try:
                        idx, sub_q, sub_docs = future.result()
                        all_sub_docs.extend(sub_docs)
                        logger.info(f"  子问题{idx + 1} '{sub_q[:30]}' → {len(sub_docs)} 篇文档")
                    except Exception as e:
                        idx = futures[future]
                        logger.warning(f"  子问题{idx + 1} 检索失败：{e}")

            # 去重：同 source + 同 page_content 的文档只保留一个
            seen = set()
            unique_docs = []
            for doc in all_sub_docs:
                doc_key = (doc.metadata.get("source", ""), doc.page_content[:100])
                if doc_key not in seen:
                    seen.add(doc_key)
                    unique_docs.append(doc)

            docs = unique_docs
            retrieval_time = (time.time() - start_time) * 1000
            logger.info(f"多子问题检索完成：{len(all_sub_docs)} 篇 → 去重后 {len(docs)} 篇，耗时：{retrieval_time:.2f}ms")
        else:
            # 单一问题检索（原有逻辑）
            start_time = time.time()
            docs = retriever.invoke(
                search_query,
                original_query=original_query,
                hyde_answer=hyde_answer,
            )
            retrieval_time = (time.time() - start_time) * 1000
            logger.info(f"检索到{len(docs)}个相关文档，耗时：{retrieval_time:.2f}ms，第 {retrieval_attempts} 次检索")

        sources = []
        for doc in docs:
            sources.append({
                "source": doc.metadata.get("source", "未知"),
                "file_path": doc.metadata.get("file_path", "未知"),
                "content": doc.page_content[:100]
            })

        return {
            "retrieved_docs": docs,
            "all_retrieved_docs": docs,
            "sources": sources,
            "retrieval_attempts": retrieval_attempts,
        }

    except Exception as e:
        logger.error(f"知识检索失败：{str(e)}")
        return {
            "retrieved_docs": [],
            "sources": [],
            "retrieval_attempts": retrieval_attempts,
            "error": f"知识检索失败：{str(e)}"
        }


# ===== v9.15: 三层拒答 — 检索层辅助函数 =====

def _timeliness_modifier(doc_metadata: dict) -> float:
    """基于文档元数据的时效衰减因子（0~1.0）"""
    # 过期文档
    expire_date = doc_metadata.get("doc_expire_date")
    if expire_date and expire_date != "unknown":
        try:
            if datetime.fromisoformat(str(expire_date)) < datetime.now():
                return 0.0
        except (ValueError, TypeError):
            pass
    # 基于生效日期衰减
    effective_date = doc_metadata.get("doc_effective_date")
    if effective_date and effective_date != "unknown":
        try:
            eff = datetime.fromisoformat(str(effective_date))
            age_days = (datetime.now() - eff).days
            if age_days > 1095:  return 0.6
            elif age_days > 730: return 0.75
            elif age_days > 365: return 0.85
            else:                return 1.0
        except (ValueError, TypeError):
            pass
    # 回退：文件修改时间
    file_modified = doc_metadata.get("file_modified") or doc_metadata.get("last_modified")
    if file_modified:
        try:
            mod_ts = float(file_modified)
            age_days = (time.time() - mod_ts) / 86400
            if age_days > 1095:  return 0.6
            elif age_days > 730: return 0.75
            elif age_days > 365: return 0.85
            else:                return 1.0
        except (ValueError, TypeError):
            pass
    return 0.8  # 无时间元数据，保守降权


AUTHORITY_MODIFIER = {
    "national_guideline": 1.0,
    "society_consensus": 0.95,
    "textbook": 0.90,
    "institutional_protocol": 0.85,
    "expert_opinion": 0.75,
    "training_material": 0.65,
    "faq": 0.55,
    "unknown": 0.60,
}


def _authority_modifier(doc_metadata: dict) -> float:
    level = doc_metadata.get("doc_authority_level", "unknown") or "unknown"
    return AUTHORITY_MODIFIER.get(level, 0.60)


def _compute_retrieval_confidence(
    max_rerank_score: float,
    score_gap: float,
    relevant_count: int,
    query_overlap_ratio: float,
    timeliness_mod: float,
    authority_mod: float,
) -> float:
    """四维加权 × 时效衰减 × 权威权重"""
    # 维度1: Reranker最高分 (0.3)
    if max_rerank_score >= 0.10:  s1 = 0.30
    elif max_rerank_score >= 0.03: s1 = 0.20
    elif max_rerank_score >= 0.01: s1 = 0.10
    else:                           s1 = 0.00
    # 维度2: 得分差距 (0.2)
    if score_gap >= 0.05:  s2 = 0.20
    elif score_gap >= 0.02: s2 = 0.10
    else:                    s2 = 0.00
    # 维度3: 相关文档数 (0.3)
    if relevant_count >= 3:  s3 = 0.30
    elif relevant_count >= 2: s3 = 0.20
    elif relevant_count >= 1: s3 = 0.10
    else:                      s3 = 0.00
    # 维度4: 关键词重叠 (0.2)
    if query_overlap_ratio >= 0.8:  s4 = 0.20
    elif query_overlap_ratio >= 0.5: s4 = 0.10
    else:                             s4 = 0.00

    base = s1 + s2 + s3 + s4
    return round(base * timeliness_mod * authority_mod, 4)


def _dedup_by_version(docs: list) -> list:
    """同源多版本文档去重：保留最新版，旧版标记 _superseded"""
    by_source = defaultdict(list)
    for doc in docs:
        source = doc.metadata.get("source", "") or doc.metadata.get("file_path", "unknown")
        # 提取基础名称（去掉版本号）
        base = source.rsplit("_v", 1)[0] if "_v" in source else source
        by_source[base].append(doc)

    result = []
    for base, versions in by_source.items():
        if len(versions) > 1:
            # 按effective_date或version降序排序
            def _sort_key(d):
                v = d.metadata.get("doc_version", "0")
                ed = d.metadata.get("doc_effective_date", "")
                return (ed, v)
            versions.sort(key=_sort_key, reverse=True)
            # 保留最新版
            result.append(versions[0])
            for old in versions[1:]:
                old.metadata["_superseded"] = True
                logger.info(f"旧版文档已标记废弃：{old.metadata.get('source', '')} (version={old.metadata.get('doc_version', '')})")
        else:
            result.append(versions[0])
    return result


@timing_decorator("文档评分")
def grade_documents_node(state: MedicalAssistantState) -> Command:
    """文档相关性评分节点（轻量启发式版）

    功能描述：
        基于检索结果和轻量启发式过滤文档
        避免逐文档调用 LLM 导致首 token 延迟过高
    """
    logger.info("文档评分节点开始执行")

    question = state.get("question", "")
    retrieved_docs = state.get("retrieved_docs", [])
    retrieval_attempts = int(state.get("retrieval_attempts") or 0)
    rewritten_query = state.get("rewritten_query") or question
    question_type = state.get("question_type", "general")

    # 振荡检测：比较重试前后的 Reranker 分数，无显著提升则跳过重试
    OSCILLATION_SCORE_DELTA = 0.05  # Reranker top-1 分数提升低于此值视为无改善
    OSCILLATION_DOC_DELTA = 1       # 相关文档数增加低于此值视为无改善
    _prev_max_score = float(state.get("_prev_max_score") or 0)
    _prev_relevant_count = int(state.get("_prev_relevant_count") or 0)

    if not retrieved_docs:
        # 记录当前分数为空值，后续重试时比较
        if retrieval_attempts >= 2 or is_same_query(rewritten_query, question):
            logger.warning("无检索文档且已达到重试上限/查询未变化，直接生成兜底答案")
            return Command(
                goto="answer_generation",
                update={
                    "retrieved_docs": [],
                    "final_answer": build_no_results_answer(question),
                    "warnings": ["知识库暂无直接命中结果，以下回答为保守兜底建议"],
                },
            )

        logger.warning("无检索文档，路由到查询重写")
        return Command(
            goto="query_rewrite",
            update={"_prev_max_score": 0, "_prev_relevant_count": 0},
        )

    # ===== v9.15: 三层拒答 — 检索层 =====
    # Step 1: 时效过滤（过期文档直接剔除）
    timely_docs = []
    for doc in retrieved_docs:
        tm = _timeliness_modifier(doc.metadata)
        doc.metadata["_timeliness_modifier"] = tm
        doc.metadata["_authority_modifier"] = _authority_modifier(doc.metadata)
        if tm > 0.0:
            timely_docs.append(doc)
        else:
            logger.info(f"过期文档已剔除：{doc.metadata.get('source', '')} (expire_date={doc.metadata.get('doc_expire_date', '')})")

    if len(timely_docs) < len(retrieved_docs):
        logger.warning(f"时效过滤：{len(retrieved_docs)} → {len(timely_docs)}（剔除 {len(retrieved_docs) - len(timely_docs)} 篇过期文档）")
        retrieved_docs = timely_docs

    # Step 2: 版本去重（同源多版本保留最新）
    retrieved_docs = _dedup_by_version(retrieved_docs)

    # Reranker 低分检测：最高分极低说明知识库无相关文档
    max_rerank_score = max(
        (doc.metadata.get("rerank_score", 0) for doc in retrieved_docs),
        default=0,
    )
    # v9.0 知识库无覆盖早退：Reranker 最高分 < 0.01 说明完全不在知识库范围
    # v9.15: symptom/knowledge 类型不再走 direct_answer，改为拒答
    RERANK_NO_COVERAGE_THRESHOLD = 0.01
    if max_rerank_score < RERANK_NO_COVERAGE_THRESHOLD and max_rerank_score > 0:
        logger.warning(
            f"Reranker 最高分仅 {max_rerank_score:.4f}（< {RERANK_NO_COVERAGE_THRESHOLD}），"
            f"知识库完全无覆盖"
        )
        if question_type in ("symptom", "knowledge"):
            # v9.15: 医疗问题 + 知识库无覆盖 = 拒答，不让LLM自由编造
            from app.core.metrics import get_metrics_collector
            get_metrics_collector().record_refusal(
                refusal_type="no_match",
                question=question,
                question_type=question_type,
                reason=f"Reranker最高分仅{max_rerank_score:.4f}，知识库无覆盖",
                confidence=0.0,
                max_rerank_score=max_rerank_score,
                retrieved_doc_count=len(retrieved_docs),
                relevant_doc_count=0,
                thread_id=state.get("thread_id", ""),
                request_id=state.get("request_id", ""),
            )
            return Command(
                goto="answer_generation",
                update={
                    "retrieved_docs": [],
                    "final_answer": build_no_results_answer(question),
                    "warnings": ["知识库暂无相关文档，无法提供可靠回答"],
                },
            )
        else:
            # general类型 → 直接回答（问候等）
            return Command(
                goto="direct_answer",
                update={
                    "retrieved_docs": [],
                    "warnings": ["知识库无覆盖范围，以下回答基于通用知识"],
                },
            )
    RERANK_IRRELEVANT_THRESHOLD = 0.02
    if max_rerank_score < RERANK_IRRELEVANT_THRESHOLD and max_rerank_score > 0:
        logger.warning(
            f"Reranker 最高分仅 {max_rerank_score:.4f}（阈值 {RERANK_IRRELEVANT_THRESHOLD}），"
            f"知识库无相关文档"
        )
        if question_type in ("symptom", "knowledge"):
            # v9.15: 医疗问题 + 知识库无覆盖 = 拒答，不让LLM自由编造
            from app.core.metrics import get_metrics_collector
            get_metrics_collector().record_refusal(
                refusal_type="no_match",
                question=question,
                question_type=question_type,
                reason=f"Reranker最高分仅{max_rerank_score:.4f}，知识库无覆盖",
                confidence=0.0,
                max_rerank_score=max_rerank_score,
                retrieved_doc_count=len(retrieved_docs),
                relevant_doc_count=0,
                thread_id=state.get("thread_id", ""),
                request_id=state.get("request_id", ""),
            )
            return Command(
                goto="answer_generation",
                update={
                    "retrieved_docs": [],
                    "final_answer": build_no_results_answer(question),
                    "warnings": ["知识库暂无相关文档，无法提供可靠回答"],
                },
            )
        else:
            # general类型 → 直接回答（问候等）
            return Command(
                goto="direct_answer",
                update={
                    "retrieved_docs": [],
                    "warnings": ["知识库无覆盖范围，以下回答基于通用知识"],
                },
            )

    relevant_docs = filter_relevant_docs(question, retrieved_docs)
    relevant_count = len(relevant_docs)
    logger.info(f"文档启发式过滤结果：{len(retrieved_docs)} -> {relevant_count} 相关")

    if not relevant_docs:
        # 振荡检测：有重试历史但无改善时不再重试
        if _prev_max_score > 0:
            score_delta = max_rerank_score - _prev_max_score
            if score_delta < OSCILLATION_SCORE_DELTA:
                logger.warning(
                    f"自纠正无改善（score_delta={score_delta:.4f} < {OSCILLATION_SCORE_DELTA}），"
                    f"跳过重试，直接生成兜底答案"
                )
                return Command(
                    goto="answer_generation",
                    update={
                        "retrieved_docs": [],
                        "final_answer": build_no_results_answer(question),
                        "warnings": ["知识库暂无直接命中结果，以下回答为保守兜底建议"],
                    },
                )

        if retrieval_attempts >= 2 or is_same_query(rewritten_query, question):
            logger.warning("文档过滤后为空且不再重试，直接生成兜底答案")
            return Command(
                goto="answer_generation",
                update={
                    "retrieved_docs": [],
                    "final_answer": build_no_results_answer(question),
                    "warnings": ["知识库暂无直接命中结果，以下回答为保守兜底建议"],
                },
            )

        logger.info("所有文档被启发式过滤，路由到查询重写")
        return Command(
            goto="query_rewrite",
            update={
                "_prev_max_score": max_rerank_score,
                "_prev_relevant_count": 0,
            },
        )

    # 振荡检测：有重试历史时检查改善幅度
    if _prev_max_score > 0 and retrieval_attempts > 0:
        score_delta = max_rerank_score - _prev_max_score
        doc_delta = relevant_count - _prev_relevant_count
        if score_delta < OSCILLATION_SCORE_DELTA and doc_delta < OSCILLATION_DOC_DELTA:
            logger.warning(
                f"自纠正改善不足（score_delta={score_delta:.4f}, doc_delta={doc_delta}），"
                f"不再重试，使用当前结果"
            )

    # ===== v9.15: 计算检索置信度 =====
    if relevant_docs:
        top1_score = relevant_docs[0].metadata.get("rerank_score", 0)
        top2_score = relevant_docs[1].metadata.get("rerank_score", 0) if len(relevant_docs) > 1 else 0
        score_gap = top1_score - top2_score

        # 计算关键词重叠率
        query_terms = set(jieba.cut(question))
        overlap_count = 0
        for doc in relevant_docs[:3]:
            for term in query_terms:
                if term in doc.page_content:
                    overlap_count += 1
                    break
        overlap_ratio = overlap_count / max(len(relevant_docs[:3]), 1)

        # 取相关文档中最高时效/权威modifier
        best_timeliness = max((d.metadata.get("_timeliness_modifier", 0.8) for d in relevant_docs), default=0.8)
        best_authority = max((d.metadata.get("_authority_modifier", 0.6) for d in relevant_docs), default=0.6)

        confidence = _compute_retrieval_confidence(
            max_rerank_score=top1_score,
            score_gap=score_gap,
            relevant_count=len(relevant_docs),
            query_overlap_ratio=overlap_ratio,
            timeliness_mod=best_timeliness,
            authority_mod=best_authority,
        )

        logger.info(f"检索置信度：{confidence:.4f} (rerank={top1_score:.4f}, gap={score_gap:.4f}, docs={len(relevant_docs)}, overlap={overlap_ratio:.2f}, timeliness={best_timeliness:.2f}, authority={best_authority:.2f})")

        # 低置信度记录拒答日志
        if confidence < 0.4 and question_type in ("symptom", "knowledge"):
            from app.core.metrics import get_metrics_collector
            get_metrics_collector().record_refusal(
                refusal_type="low_confidence",
                question=question,
                question_type=question_type,
                reason=f"检索置信度{confidence:.4f}<0.4，相关信息不足",
                confidence=confidence,
                max_rerank_score=top1_score,
                retrieved_doc_count=len(retrieved_docs),
                relevant_doc_count=len(relevant_docs),
                thread_id=state.get("thread_id", ""),
                request_id=state.get("request_id", ""),
            )
    else:
        confidence = 0.0

    return Command(
        goto="answer_generation",
        update={
            "retrieved_docs": relevant_docs,
            "retrieval_confidence": confidence,
        }
    )


def get_user_context_prompt(user_profile: Optional[Dict[str, Any]]) -> str:
    """构建用户上下文提示（冻结层，永不压缩）

    用户档案作为冻结层独立存储，不会在对话摘要中丢失。
    此函数仅生成档案文本内容，由调用方决定注入位置和标记。
    """
    if not user_profile:
        return ""

    context_parts = []
    if user_profile.get("name"):
        context_parts.append(f"姓名：{user_profile['name']}")
    if user_profile.get("age"):
        context_parts.append(f"年龄：{user_profile['age']}岁")
    if user_profile.get("gender"):
        context_parts.append(f"性别：{user_profile['gender']}")
    if user_profile.get("allergies"):
        context_parts.append(f"过敏史：{', '.join(user_profile['allergies'])}")

    return "\n".join(context_parts) if context_parts else ""


def normalize_router_label(raw_label: str) -> str:
    """规范化路由分类标签

    兜底策略：无法识别时走 general（直接回答），避免模糊输入触发不必要的 RAG 检索
    """
    label = (raw_label or "").strip().lower()
    if label in {"symptom", "knowledge", "general"}:
        return label

    if "symptom" in label or "症状" in label:
        return "symptom"
    if "knowledge" in label or "知识" in label:
        return "knowledge"
    if "general" in label or "一般" in label or "问候" in label:
        return "general"

    return "general"  # 兜底：无法识别时走直接回答，避免不必要的 RAG 开销


def strip_rag_documents_from_history(history_text: str) -> str:
    """剥离历史对话中的RAG文档内容，替换为 doc_id 引用标记（MicroCompact策略）

    借鉴 Claude Code 的 MicroCompact：旧工具结果转存磁盘，上下文只留引用。
    医疗场景中，RAG文档原文在历史对话中占大量token，但LLM只需知道"曾经参考过什么"。

    处理规则：
    - 匹配 [文档N 来源：xxx]\n... 格式的文档块
    - 将文档原文存入 Redis（doc_id 为键，TTL=2h）
    - 替换为 [参考文档 doc_xxxx: 来源标题] 引用标记
    - 需要恢复原文时，通过 doc_id 从 Redis 读取
    """
    if not history_text:
        return history_text

    # 匹配 [文档N 来源：xxx doc_id:doc_xxxx]\n... 格式（含 doc_id）
    # 也兼容旧格式 [文档N 来源：xxx]\n...
    pattern = r'\[文档(\d+)\s+来源[：:](.*?)(?:\s+doc_id:(doc_\w+))?\]\n(.*?)(?=\[文档\d|\Z)'

    from app.cache.redis_cache import get_cache
    cache = get_cache()

    def replace_doc(match):
        doc_num = match.group(1)
        source = match.group(2).strip()
        existing_doc_id = match.group(3)  # 可能为 None
        content = match.group(4).strip()

        if existing_doc_id:
            # 已有 doc_id（当前轮文档已存入 Redis），直接引用
            doc_id = existing_doc_id
            # 确认 Redis 中存在，不存在则重新存储
            if not cache.get_doc(doc_id.replace("doc_", "")):
                cache.store_doc(doc_id.replace("doc_", ""), content, source)
        else:
            # 旧格式无 doc_id，生成新的并存入 Redis
            doc_id = f"doc_{uuid.uuid4().hex[:8]}"
            cache.store_doc(doc_id.replace("doc_", ""), content, source)

        # 取内容前20字作为摘要
        summary = content[:20] + "..." if len(content) > 20 else content
        return f"[参考文档 {doc_id}: {source} - {summary}]"

    result = re.sub(pattern, replace_doc, history_text, flags=re.DOTALL)
    return result


def get_conversation_history_text(state: MedicalAssistantState, max_rounds: int = 0,
                                   compress_ai_answers: bool = False,
                                   enable_context_compression: bool = True) -> str:
    """构建对话历史文本（含四层上下文压缩），不含临床快照

    临床快照由 build_rag_prompt 单独注入，避免重复
    max_rounds: 最多注入最近N轮对话（1轮=1条Human+1条AI），0=动态计算（v9.17）
    compress_ai_answers: RAG场景下压缩AI回答为首句摘要，防止复制旧答案
    enable_context_compression: v9.5 新增——启用四层上下文压缩（L1→L3→L2→L4）
    """
    messages = state.get("messages", [])

    if not messages:
        return ""

    # v9.17: 动态max_rounds（基于token占用率，而非固定轮数）
    if max_rounds <= 0:
        from app.core.config import get_config
        _config = get_config()
        from app.graph.nodes.context_manager import get_adaptive_max_rounds
        max_rounds = get_adaptive_max_rounds(
            messages,
            context_window=getattr(_config, 'LLM_CONTEXT_WINDOW_TOKENS', 8192),
            compression_ratio=getattr(_config, 'CONTEXT_COMPRESSION_RATIO', 0.7),
        )

    # 只取最近 max_rounds 轮（2*max_rounds 条消息）
    max_msgs = max_rounds * 2
    if len(messages) > max_msgs:
        messages = messages[-max_msgs:]

    # v9.5: 四层上下文压缩（L1→L3→L2→L4）
    if enable_context_compression:
        from app.graph.nodes.context_manager import compress_context
        messages = compress_context(messages)

    # MicroCompact：对旧AI消息中的RAG文档内容进行压缩
    last_ai_index = -1
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            last_ai_index = i
            break

    history_parts = []
    for i, msg in enumerate(messages):
        if isinstance(msg, SystemMessage):
            # 跳过临床快照的 SystemMessage，由 build_rag_prompt 单独注入
            # 但保留 context_summary 类型的 SystemMessage
            if getattr(msg, 'name', None) == 'context_summary':
                history_parts.append(f"【摘要】：{msg.content}")
            continue
        elif isinstance(msg, HumanMessage):
            history_parts.append(f"用户：{msg.content}")
        elif isinstance(msg, AIMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)

            if compress_ai_answers:
                # v9.4: RAG场景下压缩AI回答，仅保留首句摘要
                # 防止LLM从历史对话中复制旧答案作为"事实"来源
                # 关键事实由临床快照和检索文档提供，不需要从历史AI回答中获取
                first_sentence = content.split('。')[0].split('\n')[0]
                if len(first_sentence) > 80:
                    first_sentence = first_sentence[:80] + "..."
                content = f"{first_sentence}...[已回复]"
            else:
                # 原始逻辑：保留完整AI回答（用于非RAG场景）
                _medical_keywords = ["mg", "ml", "剂量", "用药", "服用", "过敏", "禁忌", "副作用",
                                     "布洛芬", "对乙酰氨基酚", "阿莫西林", "头孢", "阿司匹林",
                                     "不要", "避免", "注意", "警告", "严重"]
                has_medical_info = any(kw in content for kw in _medical_keywords)
                truncate_len = 800 if has_medical_info else 400
                if len(content) > truncate_len:
                    content = content[:truncate_len] + "..."
                if i != last_ai_index:
                    content = strip_rag_documents_from_history(content)

            history_parts.append(f"助手：{content}")

    return "\n".join(history_parts)


def format_retrieved_sources(retrieved_docs: Optional[List[Any]], content_limit: int = 200) -> List[Dict[str, str]]:
    """格式化检索来源信息"""
    if not retrieved_docs:
        return []

    return [
        {
            "source": doc.metadata.get("source", "未知来源"),
            "file_path": doc.metadata.get("file_path", ""),
            "content": doc.page_content[:content_limit],
        }
        for doc in retrieved_docs
    ]


def has_query_overlap(question: str, doc_content: str) -> bool:
    """基于关键词重叠的轻量相关性判断

    改进：
    1. 要求至少2个关键词命中才算相关，避免单字/泛化词误匹配
    2. 过滤口语虚词（怎么办/怎么样/好不好等），避免因这些词不命中文档
       而把真正相关的文档（如含"头痛"的文档）误判为不相关
    """
    # 口语虚词过滤：这些词在正式医学文档中几乎不会出现，
    # 但用户查询中很常见（如"头痛怎么办"中的"怎么办"）
    ORAL_FILLERS = {"怎么办", "怎么样", "好不好", "怎么治", "怎么处理",
                    "怎么缓解", "什么药", "吃什么", "如何", "该如何"}

    tokens = [token for token in re.findall(r"[一-鿿A-Za-z0-9]+", question.lower())
              if len(token) >= 2 and token not in ORAL_FILLERS]
    if not tokens:
        # 查询只剩口语虚词时，降级为单字匹配
        raw_tokens = [t for t in re.findall(r"[一-鿿A-Za-z0-9]+", question.lower()) if len(t) >= 2]
        if not raw_tokens:
            return False
        content = (doc_content or "").lower()
        return any(t in content for t in raw_tokens)

    content = (doc_content or "").lower()
    match_count = sum(1 for token in tokens if token in content)
    # 至少匹配1个实质关键词（过滤掉虚词后，1个关键词命中即算相关）
    return match_count >= 1


def filter_relevant_docs(question: str, retrieved_docs: List[Any]) -> List[Any]:
    """使用 rerank 分数和关键词重叠做轻量过滤，替代逐文档 LLM 评分

    策略：
    - Reranker 已执行：信任排序结果，保留 Top 文档（Reranker分数范围不固定，不能用绝对阈值）
    - Reranker 未执行：用关键词重叠做启发式过滤
    - 全部过滤掉时返回空列表，由上层走 direct_answer（不喂不相关文档给 LLM）

    设计决策（v8.2）：
    - 去除兜底逻辑 `filtered_docs or retrieved_docs[:1]`
    - 旧逻辑：全过滤掉 → 硬塞第1篇（可能是最不相关的）→ LLM 产生幻觉
    - 新逻辑：全过滤掉 → 返回空列表 → 走 direct_answer + 警告"知识库无相关文档"
    - 理由：喂一篇不相关的文档比不喂文档更糟，不相关文档会误导 LLM 编造答案
    """
    if not retrieved_docs:
        return []

    # 检查是否有 rerank 分数（Reranker 真正执行过）
    has_rerank_scores = any(
        doc.metadata.get("rerank_score") is not None for doc in retrieved_docs
    )

    # v9.6.2: 多子问题检索的文档不能用统一的 Reranker 排序做位置判断
    # 因为不同子问题的文档分别 Rerank，分数不可交叉比较
    # 且每个子问题的 top-k 文档已经过各自 Reranker 精排，本身就是高质量的
    # 直接信任子问题的 Reranker 排序，不做启发式过滤
    has_sub_questions = any(
        doc.metadata.get("sub_question") is not None for doc in retrieved_docs
    )

    if has_sub_questions:
        # 多子问题检索：各子问题已分别 Reranker 精排，跳过启发式过滤
        # 原因：
        # 1. 每个子问题的 top-k 文档已过各自 Reranker 精排，质量有保证
        # 2. 邻域扩展的兄弟章节可能内容与 sub_question 不同（如发热查询扩展出便秘章节）
        #    但它们是原始检索结果的结构邻居，有价值
        # 3. 用 has_query_overlap 做交叉过滤会导致：
        #    兄弟章节的 sub_question 不匹配内容 → 误杀 → 丢失关键文档
        # 4. 去重已在上层完成，不会重复
        logger.info(f"多子问题检索：跳过启发式过滤，保留全部 {len(retrieved_docs)} 篇文档")
        return retrieved_docs

    if has_rerank_scores:
        # 单一问题 Reranker 排序：信任排序结果，保留 Top 文档
        filtered_docs = []
        for index, doc in enumerate(retrieved_docs):
            overlap = has_query_overlap(question, doc.page_content)
            # 前2名无条件保留（Reranker排序可信），其余需关键词重叠
            if index < 2 or overlap:
                filtered_docs.append(doc)
            else:
                logger.info(f"文档启发式过滤（Reranker排序靠后+无重叠）：{doc.page_content[:50]}...")

        # Reranker 已执行时，至少保留前2名（Reranker排序可信）
        return filtered_docs

    # 单一问题 + Reranker 未执行：用关键词重叠做启发式过滤
    filtered_docs = []
    for index, doc in enumerate(retrieved_docs):
        overlap = has_query_overlap(question, doc.page_content)
        threshold_fallback = bool(doc.metadata.get("rerank_threshold_fallback"))
        keep_doc = overlap or (index == 0 and threshold_fallback)
        if keep_doc:
            filtered_docs.append(doc)
        else:
            logger.info(f"文档启发式过滤：{doc.page_content[:50]}...")

    # 不兜底！全部过滤掉 → 返回空列表 → 上层走 direct_answer
    if not filtered_docs:
        logger.warning(
            f"启发式过滤后无相关文档（{len(retrieved_docs)} 篇全被过滤），"
            f"将走直接回答而非 RAG 生成"
        )
    return filtered_docs


def _build_followup_hints(symptoms: Optional[Dict[str, Any]]) -> str:
    """根据症状提取结果，生成追问引导（参考蚂蚁阿福的主动追问机制）

    当用户描述模糊时，在回答末尾追加追问，引导用户补充关键信息，
    模拟真人医生的问诊逻辑。
    """
    if not symptoms:
        return ""

    missing_fields = []
    symptom_list = symptoms.get("symptoms", [])
    severity = symptoms.get("severity")
    body_parts = symptoms.get("body_parts")
    duration = symptoms.get("duration")

    if not symptom_list:
        return ""

    # 检查缺失的关键信息
    if not body_parts:
        missing_fields.append("具体部位（如：头部、腹部、四肢等）")
    if not duration:
        missing_fields.append("持续时间（如：3天、1周等）")
    if not severity:
        missing_fields.append("严重程度（如：轻微、中等、剧烈）")

    if not missing_fields:
        return ""

    hints = "、".join(missing_fields)
    return f"\n\n💡 为了更准确地帮助您，您可以补充以下信息：{hints}。这些信息有助于我给出更有针对性的建议。"


def build_rag_prompt(question: str, retrieved_docs: Optional[List[Any]], user_profile: Optional[Dict[str, Any]], state: MedicalAssistantState, symptoms: Optional[Dict[str, Any]] = None) -> str:
    """构建 RAG 问答提示词（三层上下文架构）

    v9.1: 使用 ChatPromptTemplate 构建，通过 format_messages 得到消息列表，
    再传给 llm.invoke(messages)，而非手动拼接 f-string。
    """
    from app.graph.nodes.prompts import RAG_ANSWER_PROMPT, RAG_ANSWER_NO_CONTEXT_PROMPT

    # L2 会话层：临床状态快照
    checkpoint = state.get("clinical_checkpoint")
    checkpoint_text = format_clinical_checkpoint(checkpoint) if checkpoint else ""

    # L1 永久层：用户档案独立注入
    frozen_profile_section = ""
    profile_text = get_user_context_prompt(user_profile)
    if profile_text:
        frozen_profile_section = f"【L1 用户档案（永不压缩）】\n{profile_text}\n"

    # L3 短期窗口：注入近期对话历史
    # v9.4: RAG场景启用 compress_ai_answers，防止LLM从历史中复制旧答案作为事实
    history_text = get_conversation_history_text(state, compress_ai_answers=True)
    history_section = f"【L3 对话历史（仅供理解上下文，不是事实来源）】\n{history_text}\n" if history_text else ""

    if not retrieved_docs:
        # 无检索文档时的降级 Prompt
        messages = RAG_ANSWER_NO_CONTEXT_PROMPT.format_messages(
            frozen_profile_section=frozen_profile_section,
            history_section=history_section,
            question=question,
        )
        return messages

    formatted_docs = []
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get("source", "未知来源")
        content = doc.page_content
        if len(content) > 2000:
            content = content[:2000] + "..."
        doc_id = f"{uuid.uuid4().hex[:8]}"
        try:
            from app.cache.redis_cache import get_cache
            get_cache().store_doc(doc_id, doc.page_content, source)
        except Exception:
            pass
        formatted_docs.append(f"[{source}]\n{content}")
    context = "\n\n".join(formatted_docs)

    # 生成追问引导
    followup = _build_followup_hints(symptoms)

    # 时间差事实注入（代码层计算）
    time_facts_section = ""
    if checkpoint and checkpoint.get("symptom_onset_dates"):
        system_now = datetime.now()
        system_now_ts = int(system_now.timestamp())
        time_facts = []
        onset_dates = checkpoint["symptom_onset_dates"]
        for symptom_name, onset_info in onset_dates.items():
            if not isinstance(onset_info, dict):
                continue
            onset_ts = onset_info.get("ts")
            onset_iso = onset_info.get("iso", "")
            precision = onset_info.get("precision", "default")
            if onset_ts and isinstance(onset_ts, (int, float)):
                delta_seconds = system_now_ts - int(onset_ts)
                if delta_seconds >= 0:
                    days = delta_seconds // 86400
                    hours = (delta_seconds % 86400) // 3600
                    if days > 0:
                        duration_str = f"{days}天{hours}小时" if hours > 0 else f"{days}天"
                    elif hours > 0:
                        duration_str = f"{hours}小时"
                    else:
                        minutes = delta_seconds // 60
                        duration_str = f"{minutes}分钟"
                    precision_note = "（约数）" if precision == "approximate" else ""
                    time_facts.append(f"- {symptom_name}：首发于 {onset_iso}，距今 {duration_str}{precision_note}")
        if time_facts:
            time_facts_section = f"【时间事实（系统计算，无需推算）】\n" + "\n".join(time_facts) + "\n"

    # L2 快照独立注入
    checkpoint_section = f"【L2 临床快照】\n{checkpoint_text}\n" if checkpoint_text else ""
    followup_section = f"；追问：{followup}" if followup else ""

    # 使用 ChatPromptTemplate 构建
    messages = RAG_ANSWER_PROMPT.format_messages(
        frozen_profile_section=frozen_profile_section,
        context=context,
        time_facts_section=time_facts_section,
        checkpoint_section=checkpoint_section,
        history_section=history_section,
        question=question,
        followup_section=followup_section,
    )
    return messages


def build_direct_answer_prompt(question: str, user_profile: Optional[Dict[str, Any]], state: MedicalAssistantState):
    """构建直接回答提示词（三层上下文架构）

    v9.1: 使用 ChatPromptTemplate 构建。
    """
    from app.graph.nodes.prompts import DIRECT_ANSWER_PROMPT

    checkpoint = state.get("clinical_checkpoint")
    checkpoint_text = format_clinical_checkpoint(checkpoint) if checkpoint else ""

    # L1 永久层：用户档案
    frozen_profile_section = ""
    profile_text = get_user_context_prompt(user_profile)
    if profile_text:
        frozen_profile_section = f"【L1 用户档案（永不压缩）】\n{profile_text}\n"

    # L2 会话层：临床快照
    checkpoint_section = f"【L2 临床快照】\n{checkpoint_text if checkpoint_text else '无'}\n"

    # L3 短期窗口：注入近期对话历史
    history_text = get_conversation_history_text(state, max_rounds=2)
    history_section = f"【L3 对话历史】\n{history_text}\n" if history_text else ""

    # 使用 ChatPromptTemplate 构建
    messages = DIRECT_ANSWER_PROMPT.format_messages(
        frozen_profile_section=frozen_profile_section,
        checkpoint_section=checkpoint_section,
        history_section=history_section,
        question=question,
    )
    return messages


@async_timing_decorator("答案生成")
async def answer_generation_node(state: MedicalAssistantState, config: RunnableConfig) -> Dict[str, Any]:
    """答案生成节点（v9.18 改为异步流式）

    功能描述：
        基于检索到的文档生成用户友好的医疗答案
        这是RAG流程的生成节点，负责答案生成

    v9.18 (P0-1)：改为 async + get_llm(streaming=True).astream()，
    使 token 能通过 graph.astream(stream_mode="messages") 原生流出，
    同时兼容非流式 ainvoke（内部累加后一次性返回 final_answer）。
    """
    logger.info("答案生成节点开始执行")

    original_question = state.get("question", "")
    # 优先使用重写后的完整问题（含上下文），无重写时用原问题
    prompt_question = state.get("final_question") or state.get("rewritten_query") or original_question
    retrieved_docs = state.get("retrieved_docs")
    existing_answer = state.get("final_answer")
    context_prompt = get_user_context_prompt(state.get("user_profile"))
    if context_prompt:
        logger.info("已注入用户上下文")

    try:
        if existing_answer:
            logger.info("检测到预生成兜底答案，跳过 LLM 再生成")
            result = {
                "final_answer": existing_answer,
                "messages": [
                    HumanMessage(content=original_question),
                    AIMessage(content=existing_answer)
                ]
            }
            if retrieved_docs:
                result["sources"] = format_retrieved_sources(retrieved_docs)
            return result

        prompt = build_rag_prompt(
            question=prompt_question,
            retrieved_docs=retrieved_docs,
            user_profile=state.get("user_profile"),
            state=state,
            symptoms=state.get("symptoms"),
        )

        llm = get_llm(streaming=True)
        start_time = time.time()
        full_answer = ""
        chunks = []

        # 显式传入 config，确保 Python < 3.11 下也能被 messages 模式捕获 token
        async for chunk in llm.astream(prompt, config=config):
            chunks.append(chunk)
            token = chunk.content if isinstance(chunk.content, str) else ""
            if token:
                full_answer += token

        answer = full_answer.strip()
        generation_time = (time.time() - start_time) * 1000

        # v9.0: Token 用量自动采集（流式最后一个 chunk 含 usage）
        try:
            from app.core.token_tracker import track_stream_tokens
            track_stream_tokens(
                node_name="答案生成",
                chunks=chunks,
                request_id=state.get("request_id", ""),
                thread_id=state.get("thread_id", ""),
            )
        except Exception:
            pass  # 静默，不影响主流程

        logger.info(f"生成答案完成，长度：{len(answer)}字符，耗时：{generation_time:.2f}ms")

        result = {
            "final_answer": answer,
            "messages": [
                HumanMessage(content=original_question),
                AIMessage(content=answer)
            ]
        }

        if retrieved_docs:
            result["sources"] = format_retrieved_sources(retrieved_docs)

        return result

    except Exception as e:
        logger.error(f"答案生成失败：{str(e)}")
        return {
            "final_answer": "抱歉，生成答案时出现错误。",
            "error": str(e),
            "messages": [
                HumanMessage(content=original_question),
                AIMessage(content="抱歉，生成答案时出现错误。")
            ]
        }

@timing_decorator("安全检查")
def safety_check_node(state: MedicalAssistantState) -> Dict[str, Any]:
    """医疗合规与安全审查节点（Skill 增强）

    基于「医疗回答合规与安全审查 Skill」的结构化 SOP 执行：
        1. 规则引擎审查（0ms）：诊断性断言检测 + 紧急风险拦截 + 免责声明注入
        2. LLM 深度审查（仅规则引擎标记高风险时触发）：用药安全核查
        3. 审查决策：pass（透传）/ revise（替换）/ block（拒答）

    Args：
        state：当前状态，包含 final_answer 和 clinical_checkpoint

    Returns：
        Dict[str, Any]：更新 final_answer（如有修订）+ warnings
    """
    logger.info("医疗合规安全审查节点开始执行")
    start_time = time.time()

    content = state.get("final_answer", "")
    if not content:
        return {"warnings": []}

    clinical_checkpoint = state.get("clinical_checkpoint")

    # ===== 第一步：规则引擎审查（0ms） =====
    rule_result = run_rule_based_review(content, clinical_checkpoint)
    risk_tags = rule_result.get("risk_tags", [])
    status = rule_result["status"]
    revised_answer = rule_result["revised_answer"]

    logger.info(f"规则引擎审查完成：status={status}, risk_tags={risk_tags}, "
                f"耗时={(time.time() - start_time) * 1000:.1f}ms")

    # ===== 第二步：LLM 深度审查（仅高风险时触发） =====
    warnings = []

    if risk_tags or status == "revise":
        try:
            from app.graph.nodes.prompts import SAFETY_CHECK_PROMPT
            from app.graph.nodes.structured_output import invoke_structured
            llm = get_local_llm_json()

            # v9.2: ChatPromptTemplate + invoke_structured 三层降级
            messages = SAFETY_CHECK_PROMPT.format_messages(
                answer=revised_answer,
                clinical_snapshot=clinical_checkpoint or "无",
            )

            result: SafetyCheckOutput = invoke_structured(
                llm, messages, SafetyCheckOutput,
            )

            warnings = result.detected_issues.copy()
            if result.requires_medical_attention:
                warnings.append("紧急提醒：建议立刻就医")
                # LLM 判定需要就医但回答中未包含 → 升级为 block
                if result.risk_level == "high" and "emergency_risk_missed" not in risk_tags:
                    status = "block"

            logger.info(f"LLM 深度审查完成：risk_level={result.risk_level}, "
                        f"总耗时={time.time() - start_time:.2f}s")

        except Exception as e:
            logger.warning(f"LLM 深度审查失败（规则引擎结果仍生效）：{e}")

    # ===== 第三步：执行审查决策 =====
    if status == "block":
        revised_answer = get_block_template()
        warnings.append("回答因安全风险已被拦截，返回安全引导模板")
        logger.warning("安全审查决策：BLOCK — 回答被拦截")
    elif status == "revise":
        logger.info("安全审查决策：REVISE — 回答已修订")

    # 免责声明
    if "本回答仅供参考" not in " ".join(warnings):
        warnings.append("本回答仅供参考，不能替代专业医生的诊断和治疗建议")

    result_update = {"warnings": warnings}
    if status in ("revise", "block"):
        result_update["final_answer"] = revised_answer

    return result_update

@timing_decorator("长期记忆加载")
def memory_load_node(state: MedicalAssistantState) -> Dict[str, Any]:
    """长期记忆加载节点
    功能描述：
        从长期记忆中读取用户档案和健康历史
        作为工作流的第一个节点，为后续节点提供用户上下文
        新增：从 L1 加载症状首发时间，填充 L2 symptom_onset_dates

    Args：
        state：当前状态，包含user_id

    Returns：
        Dict[str, Any]：需要更新的状态字段
    """
    logger.info("长期记忆加载节点开始执行")
    user_id = state.get("user_id")

    if not user_id:
        logger.info("未提供user_id，跳过长期记忆加载")
        return {"user_profile": None}

    try:
        memory = get_long_term_memory()

        # 读取用户档案
        user_profile = memory.get_user_profile(user_id)

        if user_profile:
            logger.info(f"已加载用户档案：user_id={user_id}")
            # 记录档案关键信息
            profile_info = []
            if user_profile.get("age"):
                profile_info.append(f"年龄:{user_profile['age']}")
            if user_profile.get("gender"):
                profile_info.append(f"性别:{user_profile['gender']}")
            if user_profile.get("allergies"):
                profile_info.append(f"过敏:{len(user_profile['allergies'])}项")
            logger.info(f"用户档案信息：{', '.join(profile_info)}")
        else:
            logger.info(f"用户档案不存在：user_id={user_id}")

        # 从 L1 加载症状首发时间，填充 L2 symptom_onset_dates
        # 场景：新会话时 L2 为空，但 L1 有历史症状记录
        result = {"user_profile": user_profile}
        try:
            l1_onset_dates = memory.get_all_symptom_onsets(user_id)
            if l1_onset_dates:
                logger.info(f"从L1加载症状首发时间：{list(l1_onset_dates.keys())}")
                # 仅在 L2 快照为空或没有 symptom_onset_dates 时填充
                existing_checkpoint = state.get("clinical_checkpoint")
                existing_onset = (existing_checkpoint or {}).get("symptom_onset_dates") or {}
                # 合并：同一症状取最早的首发时间（ts 最小值）
                # L1 是跨会话事实真相源，L2 是当前会话工作缓存
                merged_onset = _merge_onset_dates(l1_onset_dates, existing_onset)
                if merged_onset != existing_onset:
                    # 需要更新 clinical_checkpoint
                    checkpoint = existing_checkpoint or {}
                    checkpoint["symptom_onset_dates"] = merged_onset
                    result["clinical_checkpoint"] = checkpoint
                    logger.info(f"L1→L2 症状首发时间已合并：{list(merged_onset.keys())}")
        except Exception as e:
            logger.warning(f"从L1加载症状首发时间失败（不影响主流程）：{e}")

        return result

    except Exception as e:
        logger.error(f"加载长期记忆失败：{str(e)}")
        return {"user_profile": None}

@timing_decorator("用户档案提取")
def profile_extraction_node(state: MedicalAssistantState) -> Dict[str, Any]:
    """用户档案提取节点
    功能描述：从用户问题中提取关键信息并更新用户档案
    """
    logger.info("用户档案提取节点开始执行")

    question = state.get("question", "")
    user_id = state.get("user_id")
    user_profile = state.get("user_profile")  # 已有档案

    if not user_id:
        logger.info("没有提供user_id，跳过档案提取")
        return {}

    # 检查问题是否包含个人信息
    info_keywords = ["我叫", "我的名字", "我今年", "我岁", "我对", "过敏", "我有", "病史"]

    has_personal_info = any(keyword in question for keyword in info_keywords)

    if not has_personal_info:
        logger.info(f"问题不包含个人信息，跳过档案提取")
        return {}

    try:
        from app.graph.nodes.prompts import PROFILE_EXTRACTION_PROMPT
        from app.graph.nodes.structured_output import invoke_structured
        llm = get_local_llm_json()

        # v9.2: ChatPromptTemplate + invoke_structured 三层降级
        messages = PROFILE_EXTRACTION_PROMPT.format_messages(question=question)

        result: ProfileExtractionOutput = invoke_structured(
            llm, messages, ProfileExtractionOutput,
        )

        # 过滤掉null值
        updates = {k: v for k, v in result.model_dump().items() if v is not None}

        if updates:
            memory = get_long_term_memory()
            memory.update_user_profile(user_id, updates)
            logger.info(f"已更新用户档案：{updates}")

            # 返回更新后的完整档案
            updated_profile = memory.get_user_profile(user_id)
            return {"user_profile": updated_profile}

        return {}

    except Exception as e:
        logger.warning(f"用户档案提取失败：{str(e)}")
        return {}

def _is_simple_greeting(question: str) -> Optional[str]:
    """判断是否为简单问候/寒暄，如果是则返回预设回复，否则返回 None"""
    text = (question or "").strip()
    text_lower = text.lower()
    text_no_punct = re.sub(r"[，。！？,.!?\s]", "", text_lower)

    # 精确匹配问候映射表
    greeting_map = {
        "你好": "你好！我是医疗助手，有什么健康问题可以帮你解答吗？",
        "您好": "您好！我是医疗助手，有什么健康问题可以帮你解答吗？",
        "hello": "Hello! 我是医疗助手，有什么健康问题可以帮你解答吗？",
        "hi": "Hi! 我是医疗助手，有什么健康问题可以帮你解答吗？",
        "hey": "Hey! 我是医疗助手，有什么健康问题可以帮你解答吗？",
        "谢谢": "不客气！如果还有其他健康问题，随时可以问我。",
        "thanks": "不客气！如果还有其他健康问题，随时可以问我。",
        "再见": "再见！祝您身体健康，有问题随时来找我。",
        "拜拜": "再见！祝您身体健康，有问题随时来找我。",
        "早上好": "早上好！我是医疗助手，有什么健康问题可以帮你解答吗？",
        "晚上好": "晚上好！我是医疗助手，有什么健康问题可以帮你解答吗？",
    }
    if text_no_punct in greeting_map:
        return greeting_map[text_no_punct]

    # 短文本且以问候开头（如"你好！"、"您好~"）
    if len(text_no_punct) <= 6:
        for greet in greeting_map:
            if text_no_punct.startswith(greet) and len(text_no_punct) == len(greet):
                return greeting_map[greet]

    # 通用问题
    general_questions = {
        "你是谁": "我是医疗助手，可以为你提供健康咨询、症状分析和医疗知识问答。请告诉我你的问题！",
        "你能做什么": "我可以为你提供：1. 症状分析与就医建议 2. 医疗健康知识问答 3. 用药指导参考。请告诉我你需要什么帮助！",
        "你叫什么": "我是医疗助手，专注于健康咨询和医疗知识问答。有什么可以帮你的吗？",
    }
    for q, answer in general_questions.items():
        if q in text:
            return answer

    return None


@timing_decorator("图片问诊")
def vision_analysis_node(state: MedicalAssistantState) -> Dict[str, Any]:
    """图片问诊节点（方案C：VLM结构化提取 → OCR校准 → 不确定性追问 → RAG生成）

    流程：
        1. VLM结构化提取：强制输出标准化JSON（image_type/description/data/directions/confidence）
        2. OCR校准（仅数据类图片）：报告/处方/药盒 → PaddleOCR精确提取数值 → 覆盖VLM猜测
        3. 不确定性处理：confidence=low 或 needs_followup=True → 直接追问用户
        4. 构造RAG查询：将VLM+OCR提取结果转为检索查询 → 路由到RAG管线

    VLM只负责"提取图片信息"，不负责"生成医学回答"。
    医学回答由RAG管线基于知识库文档生成。
    """
    logger.info("图片问诊节点开始执行（方案C：VLM提取+RAG生成）")

    question = state.get("question", "")
    image_base64 = state.get("image_base64", "")

    if not image_base64:
        logger.warning("图片问诊节点未收到图片")
        return {"final_answer": "抱歉，未能接收到您的图片，请重新上传。"}

    try:
        # ===== Step 1: VLM 结构化提取 =====
        vision_result = _vlm_structured_extract(image_base64, question)
        logger.info(
            f"VLM结构化提取完成：type={vision_result.image_type}, "
            f"confidence={vision_result.confidence}, "
            f"directions={vision_result.possible_directions}"
        )

        # ===== Step 2: OCR 校准（仅数据类图片） =====
        data_image_types = {"lab_report", "prescription", "medication_label"}
        if vision_result.image_type in data_image_types and vision_result.extracted_data:
            ocr_data = _ocr_extract_from_base64(image_base64)
            if ocr_data:
                vision_result = _merge_ocr_into_vision(vision_result, ocr_data)
                logger.info(f"OCR校准完成，覆盖 {len(ocr_data)} 个数值")

        # ===== Step 3: 不确定性处理 =====
        if vision_result.needs_followup and vision_result.followup_question:
            logger.info(f"图片分析不确定，追问用户：{vision_result.followup_question}")
            return Command(
                update={
                    "final_answer": vision_result.followup_question,
                    "warnings": ["⚠️ 图片信息不足，请补充后再试"],
                },
                goto="safety_check",
            )

        if vision_result.confidence == "low":
            return Command(
                update={
                    "final_answer": "图片较为模糊，难以准确识别。建议您重新拍摄一张更清晰的照片，"
                                    "或者用文字描述您的症状和问题。",
                    "warnings": ["⚠️ 图片识别置信度低，结果不可靠"],
                },
                goto="safety_check",
            )

        # ===== Step 4: 构造 RAG 查询，路由到 RAG 管线 =====
        rag_query = _build_rag_query_from_vision(vision_result, question)
        logger.info(f"图片问诊构造RAG查询：{rag_query}")

        # v9.18: 通过 get_stream_writer 流式输出图片分析摘要（原生流式）
        # 非流式上下文（ainvoke）下 get_stream_writer 为 no-op，静默降级
        _emit_vision_summary(vision_result)

        # 将结构化提取结果和RAG查询写入 state，交给后续 RAG 管线
        return Command(
            update={
                "question": rag_query,
                "image_base64": None,  # 清除图片，避免RAG管线再次触发vision分支
                "final_question": rag_query,
                "warnings": ["⚠️ AI图片解读仅供参考，不能替代专业医生的诊断"],
            },
            goto="knowledge_retrieval",
        )

    except Exception as e:
        logger.error(f"图片问诊失败：{str(e)}")
        return Command(
            update={"final_answer": "抱歉，图片解读遇到了问题，请稍后重试或尝试文字描述您的症状。"},
            goto="safety_check",
        )


def _vlm_structured_extract(image_base64: str, question: str) -> VisionAnalysisOutput:
    """使用 VLM 进行结构化图片分析

    调用 glm-4v-plus，强制输出 VisionAnalysisOutput 结构化 JSON。
    """
    from langchain_core.messages import HumanMessage

    llm = get_vision_llm()
    image_url = f"data:image/jpeg;base64,{image_base64}"

    # 使用 Prompt + structured output
    from app.graph.nodes.prompts import VISION_STRUCTURED_EXTRACT_PROMPT
    prompt = VISION_STRUCTURED_EXTRACT_PROMPT
    messages = prompt.format_messages(question=question or "请帮我解读这张图片", image_url=image_url)

    # 构建多模态消息（替换 prompt 格式化的纯文本为多模态格式）
    text_part = messages[-1].content if isinstance(messages[-1].content, str) else str(messages[-1].content)
    multimodal_message = HumanMessage(content=[
        {"type": "text", "text": text_part},
        {"type": "image_url", "image_url": {"url": image_url}},
    ])

    result: VisionAnalysisOutput = invoke_structured_with_fallback(
        llm, [multimodal_message], VisionAnalysisOutput, max_attempts=2,
    )
    return result


def _ocr_extract_from_base64(image_base64: str) -> List[Dict[str, str]]:
    """对 base64 图片执行 OCR，提取结构化数值数据

    使用 PaddleOCR PP-Structure 进行版面分析+表格识别。
    返回格式：[{"name": "指标名", "value": "数值", "unit": "单位", "reference": "参考范围"}]
    """
    import base64
    import tempfile

    try:
        from app.rag.loader import _ocr_image_to_markdown
    except ImportError:
        logger.warning("PaddleOCR 未安装，跳过OCR校准")
        return []

    try:
        # base64 → 临时文件 → OCR
        image_bytes = base64.b64decode(image_base64)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        try:
            md_content = _ocr_image_to_markdown(tmp_path)
        finally:
            os.unlink(tmp_path)

        if not md_content.strip():
            return []

        # 从 Markdown 表格中解析结构化数据
        return _parse_ocr_markdown_table(md_content)

    except Exception as e:
        logger.warning(f"OCR提取失败（不影响主流程）：{e}")
        return []


def _parse_ocr_markdown_table(md_content: str) -> List[Dict[str, str]]:
    """从 OCR 输出的 Markdown 表格中解析结构化数据

    尝试识别 | 指标名 | 结果 | 单位 | 参考范围 | 格式的表格。
    如果不是表格格式，返回空列表（让VLM数据保留）。
    """
    lines = md_content.strip().split("\n")
    table_lines = [l for l in lines if l.strip().startswith("|")]

    if len(table_lines) < 3:
        return []  # 不足3行不是有效表格

    # 解析表头
    headers = [h.strip() for h in table_lines[0].split("|") if h.strip()]
    if len(headers) < 2:
        return []

    # 映射列名到字段
    name_idx = _find_column_index(headers, ["指标", "项目", "名称", "药物", "药品", "检查"])
    value_idx = _find_column_index(headers, ["结果", "数值", "测定值", "含量", "剂量"])
    unit_idx = _find_column_index(headers, ["单位"])
    ref_idx = _find_column_index(headers, ["参考", "范围", "正常值", "参考区间"])

    if name_idx is None:
        return []  # 无法识别指标名列

    # 解析数据行（跳过分隔行）
    data = []
    for line in table_lines[2:]:
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if len(cells) <= name_idx:
            continue

        entry = {"name": cells[name_idx]}
        if value_idx is not None and value_idx < len(cells):
            entry["value"] = cells[value_idx]
        if unit_idx is not None and unit_idx < len(cells):
            entry["unit"] = cells[unit_idx]
        if ref_idx is not None and ref_idx < len(cells):
            entry["reference"] = cells[ref_idx]

        data.append(entry)

    return data


def _find_column_index(headers: List[str], keywords: List[str]) -> Optional[int]:
    """在表头中查找包含关键词的列索引"""
    for i, h in enumerate(headers):
        for kw in keywords:
            if kw in h:
                return i
    return None


def _merge_ocr_into_vision(vision_result: VisionAnalysisOutput,
                            ocr_data: List[Dict[str, str]]) -> VisionAnalysisOutput:
    """用 OCR 数据校准 VLM 的 extracted_data

    策略：OCR 数值覆盖 VLM 猜测（OCR 对数字更可靠）。
    VLM 中有但 OCR 没有的数据保留（VLM 可能识别了 OCR 遗漏的指标）。
    """
    if not vision_result.extracted_data or not ocr_data:
        return vision_result

    # 建立 OCR name → entry 映射
    ocr_map = {}
    for entry in ocr_data:
        name = entry.get("name", "").strip()
        if name:
            ocr_map[name] = entry

    # 覆盖 VLM 数据中的数值
    merged_data = []
    for v_entry in vision_result.extracted_data:
        v_name = v_entry.get("name", "").strip()
        ocr_entry = ocr_map.get(v_name)

        if ocr_entry:
            # OCR 有该指标 → 覆盖数值
            merged = {**v_entry}  # 保留VLM的所有字段
            if "value" in ocr_entry:
                merged["value"] = ocr_entry["value"]
            if "unit" in ocr_entry:
                merged["unit"] = ocr_entry["unit"]
            if "reference" in ocr_entry:
                merged["reference"] = ocr_entry["reference"]
            merged["source"] = "ocr_verified"  # 标记已校准
            merged_data.append(merged)
        else:
            # VLM 有但 OCR 没有 → 保留，但标记为未校准
            merged = {**v_entry}
            merged["source"] = "vlm_only"
            merged_data.append(merged)

    # OCR 中有但 VLM 没有的指标 → 追加
    vlm_names = {e.get("name", "").strip() for e in vision_result.extracted_data}
    for ocr_entry in ocr_data:
        ocr_name = ocr_entry.get("name", "").strip()
        if ocr_name and ocr_name not in vlm_names:
            ocr_entry_copy = {**ocr_entry, "source": "ocr_only"}
            merged_data.append(ocr_entry_copy)

    # 更新 vision_result
    vision_result.extracted_data = merged_data
    return vision_result


def _build_rag_query_from_vision(vision_result: VisionAnalysisOutput,
                                  user_question: str) -> str:
    """将 VLM+OCR 提取结果构造为 RAG 检索查询

    策略：
        - 数据类（报告/处方/药盒）：用 extracted_data 中的异常项构造查询
        - 外观类（皮肤/伤口）：用 objective_description + possible_directions 构造查询
        - 用户原始问题始终保留
    """
    parts = []

    # 用户原始问题
    if user_question and user_question.strip():
        parts.append(user_question.strip())

    # VLM 客观描述
    desc = vision_result.objective_description.strip()
    if desc:
        parts.append(f"（图片内容：{desc}）")

    # 数据类：提取异常指标
    data_types = {"lab_report", "prescription", "medication_label"}
    if vision_result.image_type in data_types and vision_result.extracted_data:
        abnormal_items = []
        for entry in vision_result.extracted_data:
            name = entry.get("name", "")
            value = entry.get("value", "")
            reference = entry.get("reference", "")
            if name and value:
                item_str = f"{name}{value}"
                if reference:
                    item_str += f"（参考{reference}）"
                abnormal_items.append(item_str)
        if abnormal_items:
            parts.append("异常指标：" + "、".join(abnormal_items[:5]))

    # 可能方向
    if vision_result.possible_directions:
        parts.append("可能方向：" + " ".join(vision_result.possible_directions[:3]))

    return " ".join(parts)


def _build_vision_summary(vision_result: VisionAnalysisOutput) -> str:
    """构建图片分析摘要文本（图片识别 + 提取数据 + 分析方向）

    v9.18 (P0-1)：原 stream_vision_answer 中的摘要逻辑提取为纯函数，
    由 vision_analysis_node 通过 get_stream_writer 流式输出。
    """
    summary_parts = []
    if vision_result.objective_description:
        summary_parts.append(f"📷 图片识别：{vision_result.objective_description}")
    if vision_result.extracted_data:
        data_lines = []
        for entry in vision_result.extracted_data[:8]:
            name = entry.get("name", "")
            value = entry.get("value", "")
            unit = entry.get("unit", "")
            reference = entry.get("reference", "")
            src = entry.get("source", "")
            tag = "✓" if src == "ocr_verified" else ""
            line = f"  • {name}: {value}{unit}"
            if reference:
                line += f"（参考{reference}）"
            line += tag
            data_lines.append(line)
        summary_parts.append("📋 提取数据：\n" + "\n".join(data_lines))
    if vision_result.possible_directions:
        summary_parts.append("🔍 分析方向：" + "、".join(vision_result.possible_directions[:3]))

    if not summary_parts:
        return ""
    return "\n\n".join(summary_parts) + "\n\n---\n\n"


def _emit_vision_summary(vision_result: VisionAnalysisOutput) -> None:
    """通过 get_stream_writer 流式输出图片分析摘要（custom 事件）

    非流式上下文（如 ainvoke）下 get_stream_writer 为 no-op，静默降级。
    """
    try:
        writer = get_stream_writer()
        summary = _build_vision_summary(vision_result)
        if summary:
            writer(summary)
    except Exception:
        pass


@async_timing_decorator("直接回答")
async def direct_answer_node(state: MedicalAssistantState, config: RunnableConfig) -> Dict[str, Any]:
    """直接回答节点（v9.18 改为异步流式）

    功能描述：
        对于一般性问题，不需要检索知识库，直接调用llm回复
        优化：简单问候/寒暄直接返回预设回复，不调用LLM

    v9.18 (P0-1)：改为 async + get_llm(streaming=True).astream()，
    使 token 能通过 graph.astream(stream_mode="messages") 原生流出。
    """
    logger.info("直接回答节点开始执行")

    question = state.get("question", "")
    user_profile = state.get("user_profile")

    # 优化：简单问候直接返回，不调用LLM
    quick_answer = _is_simple_greeting(question)
    if quick_answer:
        logger.info(f"简单问候直接返回，跳过LLM调用：{question}")
        return {
            "final_answer": quick_answer,
            "messages": [
                HumanMessage(content=question),
                AIMessage(content=quick_answer)
            ]
        }

    try:
        prompt = build_direct_answer_prompt(
            question=question,
            user_profile=user_profile,
            state=state,
        )
        llm = get_llm(streaming=True)
        full_answer = ""
        chunks = []

        # 显式传入 config，确保 Python < 3.11 下也能被 messages 模式捕获 token
        async for chunk in llm.astream(prompt, config=config):
            chunks.append(chunk)
            token = chunk.content if isinstance(chunk.content, str) else ""
            if token:
                full_answer += token

        answer = full_answer.strip()

        # v9.0: Token 用量自动采集（流式最后一个 chunk 含 usage）
        try:
            from app.core.token_tracker import track_stream_tokens
            track_stream_tokens(
                node_name="直接回答",
                chunks=chunks,
                request_id=state.get("request_id", ""),
                thread_id=state.get("thread_id", ""),
            )
        except Exception:
            pass

        logger.info(f"直接回答完成：{answer}")

        return {
            "final_answer": answer,
            "messages": [
                HumanMessage(content=question),
                AIMessage(content=answer)
            ]
        }
    except Exception as e:
        logger.error(f"直接回答失败{str(e)}")
        return {
            "final_answer": "您好，有什么可以帮助你的吗？",
            "messages": [
                HumanMessage(content=question),
                AIMessage(content="您好，有什么可以帮助你的吗？")
            ]
        }

@timing_decorator("查询重写")
def query_rewrite_node(state: MedicalAssistantState) -> Dict[str, Any]:
    """查询重写 + 问题拆解 + HyDE 假想答案生成

    功能描述：
        1. 查询重写：将追问补全为自包含的完整问题（融入对话历史中的症状/药物/诊断）
        2. 问题拆解：从完整问题中提取检索用关键词（用于 BM25 稀疏检索）
        3. HyDE 假想答案：生成假想医学回答（用于 Dense 向量检索）

    核心原则：
        - 有对话历史 → 强制重写 + 拆解，不再问"是否需要"（追问必然依赖上下文）
        - 无对话历史 → 跳过重写（首轮问题一定自包含）
        - FINAL（完整问句）→ 用于答案生成和 HyDE
        - SEARCH（关键词）→ 用于 BM25 检索

    检索策略：
        - Dense 检索：用 hyde_answer 做 embedding（语义空间更接近文档）
        - Sparse 检索（BM25）：用 rewritten_query（关键词，假想答案会引入噪声）
    """
    logger.info("查询重写节点开始执行")

    question = state.get("question", "")
    messages = state.get("messages", [])
    user_id = state.get("user_id") or "anonymous"
    thread_id = state.get("thread_id", "")

    rewritten_query = question
    final_question = question
    hyde_answer = None

    # ===== 自包含性前置检测（方案A P0）：指代词/省略结构 → 强制重写 =====
    _anaphora_detected = False
    if messages and _has_anaphora_pattern(question):
        _anaphora_detected = True
        logger.info(f"检测到指代词/省略结构，强制进入重写：{question}")

    # ===== 步骤0：无历史 或 查询自包含 → 跳过重写 =====
    # 只有追问/指代词（如"还有什么药？"）才需要从历史补全上下文
    if not messages:
        logger.info(f"首轮对话，查询自包含，跳过重写：{question}")
    elif not _anaphora_detected:
        logger.info(f"查询自包含（无指代词/省略结构），跳过重写：{question}")
    else:
        # ===== 步骤1：追问 → 强制重写 + 问题拆解 =====
        try:
            from app.graph.nodes.prompts import QUERY_REWRITE_PROMPT
            from app.graph.nodes.models import QueryRewriteOutput
            from app.graph.nodes.structured_output import invoke_structured

            llm = get_local_llm_json()
            history_summary = _build_rewrite_context(messages)

            # v9.2: ChatPromptTemplate + invoke_structured 三层降级
            messages_prompt = QUERY_REWRITE_PROMPT.format_messages(
                history_summary=history_summary,
                question=question,
            )

            result: QueryRewriteOutput = invoke_structured(
                llm, messages_prompt, QueryRewriteOutput, max_attempts=2,
            )

            # Pydantic 校验通过，直接使用结构化字段
            parsed_final = result.final_question
            parsed_search = result.search_keywords

            # v9.0: Token 用量自动采集
            try:
                from app.core.token_tracker import track_tokens
                track_tokens(
                    node_name="查询重写",
                    response=result,  # Pydantic 对象无 response_metadata，跳过
                    request_id=state.get("request_id", ""),
                    thread_id=state.get("thread_id", ""),
                )
            except Exception:
                pass

            if parsed_final and not is_same_query(parsed_final, question):
                final_question = parsed_final
                logger.info(f"重写完成（Pydantic校验通过）：{question[:30]} -> {final_question[:50]}")
            else:
                logger.info(f"重写结果与原问题一致，保留原问题：{question[:30]}")
                final_question = question

            if parsed_search:
                rewritten_query = parsed_search
                logger.info(f"检索子查询：{rewritten_query}")
            else:
                # 兜底：用 FINAL 或原问题做检索
                rewritten_query = final_question
                logger.info(f"SEARCH 为空，用 FINAL/原问题做检索")

            # 重写守卫：丢失核心信息时回退
            if final_question != question:
                final_question = _rewrite_guard_check(question, final_question)
            if rewritten_query != question:
                rewritten_query = _rewrite_guard_check(question, rewritten_query)

            # ===== Bad Case 自动采集 =====
            _record_bad_case_if_needed(
                original=question,
                final_question=final_question,
                rewritten_query=rewritten_query,
                anaphora_detected=_anaphora_detected,
                history_summary=history_summary,
                user_id=user_id,
                thread_id=thread_id,
            )

        except Exception as e:
            logger.error(f"查询重写失败：{str(e)}")
            final_question = question
            rewritten_query = question

    # ===== 步骤2：HyDE 假想答案生成 =====
    # v8.5 决策：基于 A/B 测试数据，HyDE 在当前架构下为负收益组件，默认关闭
    # 测试结果：10 条查询，Recall -13.3%，耗时 +1574ms，4 条负向仅 2 条正向
    # 原因：规则引擎+症状解析已做查询标准化，Embedding 召回率已足够高，
    #       HyDE 生成的假想答案反而把 Dense 检索引向错误方向
    # 保留 ENABLE_HYDE 开关，供未来长尾模糊查询按需启用
    enable_hyde = getattr(config, "ENABLE_HYDE", False)
    if enable_hyde and not _is_self_contained:
        try:
            llm = get_local_llm()
            hyde_prompt = f"""请针对以下医学问题，写一段简短的假想性医学回答（2-3句话）。
不需要完全正确，但要使用医学专业术语，使其在语义上接近医学文献。

问题：{final_question}

假想回答："""

            hyde_response = llm.invoke(hyde_prompt)
            hyde_answer = hyde_response.content if hasattr(hyde_response, "content") else str(hyde_response)
            hyde_answer = hyde_answer.strip()

            if len(hyde_answer) < 20:
                logger.warning(f"HyDE 假想答案过短（{len(hyde_answer)}字），丢弃：{hyde_answer}")
                hyde_answer = None
            else:
                logger.info(f"HyDE 假想答案生成完成（{len(hyde_answer)}字）：{hyde_answer[:50]}...")

        except Exception as e:
            logger.warning(f"HyDE 假想答案生成失败，使用查询做 dense 检索：{e}")
            hyde_answer = None
    else:
        if enable_hyde:
            logger.info(f"查询自包含，跳过 HyDE：{final_question[:50]}")
        # else: HyDE 默认关闭，不生成假想答案

    return {
        "rewritten_query": rewritten_query,
        "final_question": final_question,
        "hyde_answer": hyde_answer,
    }


def _rewrite_guard_check(original: str, rewritten: str) -> str:
    """重写守卫：如果重写结果丢失了原问题的核心信息，回退到原问题

    这不是硬编码规则，而是安全兜底：防止 LLM 重写时丢失用户明确提到的关键信息。
    """
    # 检查原问题中的药物名是否在重写结果中丢失
    original_drugs = [kw for kw in _DRUG_KEYWORDS if kw in original]
    lost_drugs = [d for d in original_drugs if d not in rewritten]
    if lost_drugs:
        logger.warning(f"重写丢失药物名 {lost_drugs}，回退到原问题")
        return original

    # 检查原问题中的核心症状是否丢失
    _core_symptoms = ["头痛", "头疼", "肚子疼", "腹痛", "发烧", "咳嗽", "恶心", "呕吐",
                      "腹泻", "拉肚子", "胸闷", "胸痛", "过敏", "头晕", "失眠"]
    original_symptoms = [s for s in _core_symptoms if s in original]
    lost_symptoms = [s for s in original_symptoms if s not in rewritten]
    if lost_symptoms:
        logger.warning(f"重写丢失核心症状 {lost_symptoms}，回退到原问题")
        return original

    return rewritten


# ===== 自包含性前置检测（方案A P0） =====

# 指代词/省略结构黑名单：包含这些词的追问在无上下文时语义残缺
_ANAPHORA_PATTERNS = [
    "其他", "还有", "别的", "另外",
    "这个", "那个", "它", "其",
    "上述", "前面说的", "刚才", "之前说的",
    "继续",
    # 注意："呢"、"再"、"也" 是多义字，单独出现不等于指代
    # "流鼻血了怎么办呢？"中的"呢"是语气词，不是指代
    # "再吃一粒布洛芬"中的"再"是频率副词，不是省略
    # 这些由 Layer 2（短查询+无实体）和 Layer 3（疑问词开头）兜底
]

# 以这些疑问词开头但缺少名词实体的查询，大概率依赖上下文
_QUESTION_STARTS = ["怎么", "如何", "什么", "哪些", "还有"]

# 领域核心实体关键词（症状/药物/疾病名）— 查询包含这些则视为自包含
_DOMAIN_ENTITY_KEYWORDS = [
    # 症状
    "头痛", "头疼", "发烧", "发热", "咳嗽", "流鼻涕", "鼻塞",
    "流鼻血", "鼻出血",  # v8.4 新增
    "腹痛", "肚子疼", "胃疼", "胃痛", "恶心", "呕吐", "腹泻",
    "胸闷", "胸痛", "呼吸困难", "过敏", "头晕", "失眠", "乏力",
    "瘙痒", "肿胀", "麻木", "出血", "流血", "便秘", "皮疹",
    "便血", "咯血", "尿血", "血尿",  # v8.4 新增
    # 常见药物
    "布洛芬", "对乙酰氨基酚", "阿莫西林", "头孢", "阿司匹林",
    "奥司他韦", "连花清瘟", "感冒灵", "板蓝根", "蒙脱石散",
    "氯雷他定", "西替利嗪", "红霉素", "甲硝唑",
    # 常见疾病
    "感冒", "肺炎", "胃炎", "高血压", "糖尿病", "痛风", "哮喘",
    "甲亢", "甲减", "贫血", "冠心病", "肝炎", "肾炎",
]


def _merge_onset_dates(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """深度合并症状首发时间，同一症状取 ts 最小值（最早记录）

    Args:
        base: 基础字典（通常是旧快照/L1 记录）
        override: 覆盖字典（通常是新快照/当前轮解析）

    Returns:
        合并后的字典，同一症状取 ts 更小（更早）的记录
    """
    merged = dict(base)
    for name, info in override.items():
        if name not in merged:
            merged[name] = info
            continue
        # 两条记录都有 → 取 ts 更小的（更早的首发时间）
        existing_info = merged[name]
        existing_ts = (existing_info or {}).get("ts") if isinstance(existing_info, dict) else None
        new_ts = (info or {}).get("ts") if isinstance(info, dict) else None
        # 新记录没有 ts → 保留旧的
        if new_ts is None:
            continue
        # 旧记录没有 ts → 用新的
        if existing_ts is None:
            merged[name] = info
            continue
        # 都有 ts → 取更小的（更早）
        if new_ts < existing_ts:
            merged[name] = info
            logger.debug(f"首发时间更新：{name} ts {existing_ts} → {new_ts}（取更早记录）")
    return merged


# ===== v9.2 漏洞2修复：临床快照字段级合并策略 =====

# 快照字段上限（防止 LLM 编造过多条目）
_CHECKPOINT_FIELD_LIMITS = {
    "medication_history": 15,    # 最多 15 条用药记录
    "red_flags": 10,             # 最多 10 个高危症状
    "symptom_timeline": 20,      # 最多 20 条时间线
    "confirmed_facts": 20,       # 最多 20 条已确认事实
    "ruled_out": 15,             # 最多 15 条排除项
}


def _merge_medication_history(
    existing: Optional[List[Dict]], new: Optional[List[Dict]]
) -> List[Dict]:
    """合并用药记录：追加去重，不丢失旧记录

    策略：
        - 旧记录中已有的药物（按 drug 名称匹配），保留旧的，仅补充新的 dosage/effect
        - 新记录中出现的旧药物的新信息（如更换剂量），更新旧记录的对应字段
        - 全新药物追加到末尾
    """
    if not existing:
        return (new or [])[:_CHECKPOINT_FIELD_LIMITS["medication_history"]]
    if not new:
        return existing[:_CHECKPOINT_FIELD_LIMITS["medication_history"]]

    # 以 drug 名称为 key 的索引
    existing_map = {}
    for med in existing:
        drug_name = (med or {}).get("drug", "")
        if drug_name:
            existing_map[drug_name.lower()] = med

    merged = list(existing)  # 以旧记录为基础
    for med in (new or []):
        if not isinstance(med, dict) or not med.get("drug"):
            continue
        drug_name = med["drug"].lower()
        if drug_name in existing_map:
            # 已有药物：仅补充空字段（不覆盖非空值）
            old_med = existing_map[drug_name]
            for key in ("dosage", "effect", "frequency"):
                if not old_med.get(key) and med.get(key):
                    old_med[key] = med[key]
        else:
            # 新药物：追加
            merged.append(med)

    # 上限裁剪
    return merged[:_CHECKPOINT_FIELD_LIMITS["medication_history"]]


def _merge_list_field(
    existing: Optional[List], new: Optional[List], field_name: str = ""
) -> List:
    """通用列表合并：追加去重（基于字符串规范化比较）

    策略：旧列表为基础，新列表中不在旧列表的条目追加
    适用于：red_flags, ruled_out, confirmed_facts
    """
    limit = _CHECKPOINT_FIELD_LIMITS.get(field_name, 20)
    if not existing:
        return (new or [])[:limit]
    if not new:
        return existing[:limit]

    # 规范化去重（忽略首尾空格和标点差异）
    existing_set = set()
    for item in existing:
        normalized = str(item).strip().rstrip("。，,.")
        existing_set.add(normalized)

    merged = list(existing)
    for item in new:
        normalized = str(item).strip().rstrip("。，,.")
        if normalized and normalized not in existing_set:
            merged.append(item)
            existing_set.add(normalized)

    return merged[:limit]


def _merge_chief_complaint(existing: Optional[str], new: Optional[str]) -> Optional[str]:
    """合并主诉：优先保留旧主诉（不变性），仅当旧为空时用新值

    设计理由：主诉是会话的锚点，一旦确定不应被后续 LLM 输出覆盖。
    LLM 可能在后续轮次中"遗忘"或"篡改"主诉。
    """
    if existing and existing.strip():
        return existing
    return new


def _apply_checkpoint_merge(
    existing_checkpoint: Optional[Dict], new_checkpoint: Dict
) -> Dict:
    """应用字段级合并策略，防止 LLM 全量覆盖导致数据丢失

    核心原则：
        - confirmed_facts：只增不减（过敏史/既往史不可删除）
        - medication_history：追加去重，补充空字段但不覆盖
        - red_flags：追加去重
        - chief_complaint：不变性（旧值优先）
        - ruled_out：追加去重
        - symptom_timeline：由 LLM 更新（允许修正）
        - symptom_onset_dates：由 _merge_onset_dates 单独处理
    """
    if not existing_checkpoint:
        # 首次快照，直接应用上限裁剪
        for field, limit in _CHECKPOINT_FIELD_LIMITS.items():
            val = new_checkpoint.get(field)
            if isinstance(val, list) and len(val) > limit:
                new_checkpoint[field] = val[:limit]
                logger.info(f"快照字段 {field} 超过上限 {limit}，已裁剪")
        return new_checkpoint

    # 字段级合并
    # 1. chief_complaint：不变性
    new_checkpoint["chief_complaint"] = _merge_chief_complaint(
        existing_checkpoint.get("chief_complaint"),
        new_checkpoint.get("chief_complaint"),
    )

    # 2. medication_history：追加去重
    new_checkpoint["medication_history"] = _merge_medication_history(
        existing_checkpoint.get("medication_history"),
        new_checkpoint.get("medication_history"),
    )

    # 3. red_flags：追加去重
    new_checkpoint["red_flags"] = _merge_list_field(
        existing_checkpoint.get("red_flags"),
        new_checkpoint.get("red_flags"),
        field_name="red_flags",
    )

    # 4. confirmed_facts：只增不减（关键安全性约束）
    new_checkpoint["confirmed_facts"] = _merge_list_field(
        existing_checkpoint.get("confirmed_facts"),
        new_checkpoint.get("confirmed_facts"),
        field_name="confirmed_facts",
    )

    # 5. ruled_out：追加去重
    new_checkpoint["ruled_out"] = _merge_list_field(
        existing_checkpoint.get("ruled_out"),
        new_checkpoint.get("ruled_out"),
        field_name="ruled_out",
    )

    # 6. symptom_timeline：LLM 可更新（允许修正），但需裁剪
    timeline = new_checkpoint.get("symptom_timeline") or []
    if len(timeline) > _CHECKPOINT_FIELD_LIMITS["symptom_timeline"]:
        new_checkpoint["symptom_timeline"] = timeline[:_CHECKPOINT_FIELD_LIMITS["symptom_timeline"]]

    return new_checkpoint


def _has_anaphora_pattern(query: str) -> bool:
    """检测查询是否包含指代词/省略结构（不自包含）

    四层检测（从快到慢）：
        1. 包含指代词/省略词 → 不自包含
        2. 极短查询（<15字）且缺少领域实体 → 不自包含
           （含实体的短查询如"头痛怎么缓解？"是自包含的，不能误杀）
        3. 以疑问词开头但缺少领域实体 → 不自包含
        4. 查询纯疑问词/纯语气词 → 不自包含

    设计原则：含领域实体的短查询是自包含的，不应被误判。
    """
    text = (query or "").strip()
    if not text:
        return False

    # 1. 包含指代词/省略结构
    if any(p in text for p in _ANAPHORA_PATTERNS):
        return True

    # 2. 极短查询（<15字）+ 缺少领域实体 → 不自包含
    #    含实体的短查询（如"布洛芬的副作用是什么？"12字）是自包含的
    if len(text) < 15:
        if not any(kw in text for kw in _DOMAIN_ENTITY_KEYWORDS):
            return True

    # 3. 以疑问词开头但缺少领域核心实体
    if any(text.startswith(q) for q in _QUESTION_STARTS):
        if not any(kw in text for kw in _DOMAIN_ENTITY_KEYWORDS):
            return True

    return False


def _record_bad_case_if_needed(
        original: str,
        final_question: str,
        rewritten_query: str,
        anaphora_detected: bool,
        history_summary: str,
        user_id: str,
        thread_id: str,
):
    """在重写后自动检测并记录 bad case

    触发条件：
        1. 检测到指代词但重写结果与原问题一致 → LLM 未理解上下文依赖
        2. 重写守卫回退 → 丢失核心实体
    """
    try:
        from app.memory import get_long_term_memory
        memory = get_long_term_memory()
    except Exception:
        return  # 长期记忆不可用，静默跳过

    # Case 1: 指代词检测命中但重写结果与原问题一致 → LLM 未补全上下文
    if anaphora_detected and is_same_query(final_question, original):
        try:
            memory.append_bad_case(
                case_type="rewrite_same_as_original",
                original_query=original,
                rewritten_query=rewritten_query,
                final_question=final_question,
                history_summary=history_summary,
                user_id=user_id,
                thread_id=thread_id,
                metadata={"anaphora_detected": True},
            )
        except Exception as e:
            logger.warning(f"Bad case 记录失败：{e}")

    # Case 2: 指代词检测命中但重写后仍缺少领域实体 → 重写不充分
    if anaphora_detected and final_question != original:
        has_entity = any(kw in final_question for kw in _DOMAIN_ENTITY_KEYWORDS)
        if not has_entity:
            try:
                memory.append_bad_case(
                    case_type="rewrite_missed_anaphora",
                    original_query=original,
                    rewritten_query=rewritten_query,
                    final_question=final_question,
                    history_summary=history_summary,
                    user_id=user_id,
                    thread_id=thread_id,
                    metadata={"anaphora_detected": True, "has_entity_after_rewrite": False},
                )
            except Exception as e:
                logger.warning(f"Bad case 记录失败：{e}")


def _build_rewrite_context(messages: list, max_rounds: int = 2) -> str:
    """从对话历史中构建查询重写所需的上下文

    取最近 max_rounds 轮，AI 回复在截断前先扫描全文提取医疗实体（药物/症状），
    确保药物名称不管出现在回复的哪个位置都不会因截断丢失。

    优化：使用 AC 自动机替代线性遍历，O(m) 一次扫描提取所有实体
    """
    if not messages:
        return ""

    from app.core.keyword_matcher import get_drug_matcher, get_symptom_matcher
    drug_matcher = get_drug_matcher()
    symptom_matcher = get_symptom_matcher()

    # 只取最近 max_rounds 轮（2*max_rounds 条消息）
    max_msgs = max_rounds * 2
    recent = messages[-max_msgs:] if len(messages) > max_msgs else messages

    parts = []
    for msg in recent:
        if isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            parts.append(f"用户：{content}")
        elif isinstance(msg, AIMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)

            # 截断前用 AC 自动机扫描全文提取医疗实体
            found_entities = set()
            found_entities.update(drug_matcher.get_matched_originals(content, use_boundary=False))
            found_entities.update(symptom_matcher.get_matched_originals(content, use_boundary=False))

            has_medical = bool(found_entities)
            truncate_len = 500 if has_medical else 250
            if len(content) > truncate_len:
                head_len = truncate_len * 2 // 3
                tail_len = truncate_len - head_len
                content = content[:head_len] + "..." + content[-tail_len:]
                # 将全文扫描提取的实体前置，弥补截断丢失的信息
                if found_entities:
                    entity_hint = "、".join(sorted(found_entities, key=lambda x: len(x))[:12])
                    content = f"[提及：{entity_hint}] {content}"
            parts.append(f"助手：{content}")
        # 跳过 SystemMessage

    return "\n".join(parts)


def should_update_snapshot(state: MedicalAssistantState) -> str:
    """判断是否需要更新临床状态快照（L2会话层）

    三层上下文管理架构：
    - L1 永久层：Profile（跨会话持久化，永不压缩）
    - L2 会话层：Clinical Snapshot（单会话，JSON结构化快照）
    - L3 短期窗口：Messages（滑动窗口，保留最近3轮）

    触发条件：messages > SNAPSHOT_TRIGGER 时，将早期消息提取为快照后删除
    """
    snapshot_trigger = config.SNAPSHOT_TRIGGER
    messages = state.get("messages", [])

    if len(messages) > snapshot_trigger:
        logger.info(f"消息数量{len(messages)}超过阈值{snapshot_trigger}，触发快照更新")
        return "update_snapshot"

    from langgraph.graph import END
    return END


@timing_decorator("临床状态快照")
def update_clinical_snapshot_node(state: MedicalAssistantState) -> Dict[str, Any]:
    """L2会话层：临床状态快照节点

    三层上下文管理策略：
    - 当 messages > SNAPSHOT_TRIGGER 时触发
    - 从早期消息中提取/更新结构化临床快照（JSON）
    - 删除已提取的早期消息，保留最近 KEEP_RECENT_MESSAGES 条（3轮）
    - 快照增量更新，非全量重建

    与 L1 Profile 的分工：
    - Profile：跨会话持久信息（姓名、年龄、性别、过敏史）
    - Snapshot：当前会话的临床状态（主诉、症状时间线、用药记录、诊断）
    """
    keep_recent_messages = config.KEEP_RECENT_MESSAGES

    logger.info(f"临床状态快照节点开始执行")

    messages = state.get("messages", [])
    existing_checkpoint = state.get("clinical_checkpoint")

    # v9.17: 动态触发判断（基于token占用率，而非固定消息数量）
    from app.graph.nodes.context_manager import should_compress_by_token_ratio
    from app.core.config import get_config as _get_config
    _config = _get_config()

    if not should_compress_by_token_ratio(
        messages,
        context_window=getattr(_config, 'LLM_CONTEXT_WINDOW_TOKENS', 8192),
        compression_ratio=getattr(_config, 'CONTEXT_COMPRESSION_RATIO', 0.7),
    ):
        logger.info(f"上下文占用率未达压缩阈值，跳过快照")
        return {}

    # 获取需要提取的消息（早于keep_recent_messages的消息）
    messages_to_extract = messages[:-keep_recent_messages]

    if not messages_to_extract:
        return {}

    # v9.17: 原文存档（压缩前保存，不丢弃）
    from app.graph.nodes.context_manager import archive_messages_to_disk
    archive_id = archive_messages_to_disk(
        messages_to_extract,
        user_id=state.get("user_id", ""),
        thread_id=state.get("thread_id", ""),
    )

    logger.info(f"从{len(messages_to_extract)}条早期消息中提取临床状态快照")

    try:
        llm = get_local_llm_json()  # JSON Mode：Ollama response_format=json_object

        # 格式化消息为文本
        formatted_messages = []
        for msg in messages_to_extract:
            role = "用户" if isinstance(msg, HumanMessage) else "助手"
            if isinstance(msg, SystemMessage):
                role = "系统"
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            formatted_messages.append(f"[{role}]: {content}")

        messages_text = "\n".join(formatted_messages)

        # 冻结层提示：Profile信息已单独存储，快照中不重复
        frozen_layer_note = "注意：用户档案信息（过敏史、既往病史等）已单独存储在冻结层（L1），不需要在快照中重复记录，但快照中应引用这些信息（如'结合用户青霉素过敏史'）。"

        if existing_checkpoint:
            from app.graph.nodes.prompts import CHECKPOINT_UPDATE_PROMPT
            existing_json = json.dumps(existing_checkpoint, ensure_ascii=False, indent=2)
            messages = CHECKPOINT_UPDATE_PROMPT.format_messages(
                existing_snapshot=existing_json,
                new_messages=messages_text,
            )
        else:
            from app.graph.nodes.prompts import CHECKPOINT_NEW_PROMPT
            messages = CHECKPOINT_NEW_PROMPT.format_messages(
                new_messages=messages_text,
            )

        # v9.2: invoke_structured 三层降级
        from app.graph.nodes.structured_output import invoke_structured
        result: ClinicalCheckpointOutput = invoke_structured(
            llm, messages, ClinicalCheckpointOutput,
        )
        new_checkpoint = result.model_dump()

        # v9.2 漏洞2修复：字段级合并策略，防止 LLM 全量覆盖导致数据丢失
        # 核心安全约束：confirmed_facts 只增不减，chief_complaint 不变性，
        # medication_history 追加去重，red_flags 追加去重
        new_checkpoint = _apply_checkpoint_merge(existing_checkpoint, new_checkpoint)

        # v9.17: 记录原文存档ID，便于回溯
        if archive_id:
            new_checkpoint["archive_ids"] = (existing_checkpoint or {}).get("archive_ids", []) + [archive_id]

        merge_log_parts = []
        if existing_checkpoint:
            for field in ("chief_complaint", "medication_history", "red_flags", "confirmed_facts", "ruled_out"):
                old_len = len(existing_checkpoint.get(field) or []) if isinstance(existing_checkpoint.get(field), list) else (1 if existing_checkpoint.get(field) else 0)
                new_len = len(new_checkpoint.get(field) or []) if isinstance(new_checkpoint.get(field), list) else (1 if new_checkpoint.get(field) else 0)
                if old_len != new_len:
                    merge_log_parts.append(f"{field}:{old_len}→{new_len}")
            if merge_log_parts:
                logger.info(f"快照字段合并变更：{', '.join(merge_log_parts)}")

        # 合并症状首发日期（代码层维护，LLM 不参与计算）
        # 关键：同一症状取最早的首发时间（ts 最小值），而非简单覆盖
        # 存储结构：{"头痛": {"iso": "2026-06-22T10:00:00", "ts": 1784567890, "precision": "exact"}}
        existing_onset_dates = (existing_checkpoint or {}).get("symptom_onset_dates", {}) or {}
        # LLM 可能输出空的 symptom_onset_dates，用旧快照的值兜底
        onset_dates = _merge_onset_dates(existing_onset_dates, new_checkpoint.get("symptom_onset_dates") or {})

        # 合并当前轮症状解析的时间记录（覆盖旧值，以最新为准）
        symptoms_data = state.get("symptoms")
        if symptoms_data and symptoms_data.get("symptoms") and (symptoms_data.get("onset_ts") or symptoms_data.get("onset_date")):
            for symptom_name in symptoms_data["symptoms"]:
                onset_dates[symptom_name] = {
                    "iso": symptoms_data.get("onset_date", ""),
                    "ts": symptoms_data.get("onset_ts"),
                    "precision": symptoms_data.get("time_precision", "default"),
                }
        if onset_dates:
            new_checkpoint["symptom_onset_dates"] = onset_dates
            logger.info(f"症状首发日期已记录：{onset_dates}")

            # ===== 异步同步到 L1 长期记忆（不阻塞响应） =====
            # 核心原则：L2 是工作缓存，L1 是事实真相源
            # 每次快照更新时，将新的症状首发时间写入 L1（Append-Only）
            try:
                user_id = state.get("user_id")
                if user_id:
                    memory = get_long_term_memory()
                    question = state.get("question", "")
                    for symptom_name, onset_info in onset_dates.items():
                        # 只同步当前轮新增/更新的（有 ts 的）
                        if isinstance(onset_info, dict) and onset_info.get("ts"):
                            # 检查 L1 中是否已有相同症状的更早记录
                            existing_l1 = memory.get_latest_symptom_onset(user_id, symptom_name)
                            if existing_l1 and existing_l1.get("ts") and existing_l1["ts"] <= onset_info["ts"]:
                                # L1 已有更早的记录，不同步（保留最早首发时间）
                                continue
                            memory.append_symptom_event(
                                user_id=user_id,
                                symptom_name=symptom_name,
                                onset_iso=onset_info.get("iso", ""),
                                onset_ts=onset_info["ts"],
                                precision=onset_info.get("precision", "default"),
                                source_query=question[:100],
                            )
            except Exception as e:
                logger.warning(f"症状事件同步到L1失败，写入本地缓冲：{e}")
                try:
                    from app.memory.fallback_buffer import enqueue_symptom_event
                    buf_user_id = state.get("user_id")
                    buf_question = state.get("question", "")
                    if buf_user_id:
                        for symptom_name, onset_info in onset_dates.items():
                            if isinstance(onset_info, dict) and onset_info.get("ts"):
                                enqueue_symptom_event(
                                    user_id=buf_user_id,
                                    symptom_name=symptom_name,
                                    onset_iso=onset_info.get("iso", ""),
                                    onset_ts=onset_info["ts"],
                                    precision=onset_info.get("precision", "default"),
                                    source_query=buf_question[:100],
                                )
                except Exception as buf_err:
                    logger.error(f"本地缓冲写入也失败（事件将丢失）：{buf_err}")

        # ===== 用药记录同步到 L1 =====
        try:
            user_id = state.get("user_id")
            if user_id and new_checkpoint.get("medication_history"):
                memory = get_long_term_memory()
                question = state.get("question", "")
                for med in new_checkpoint["medication_history"]:
                    if isinstance(med, dict) and med.get("drug"):
                        memory.append_medication_event(
                            user_id=user_id,
                            drug=med["drug"],
                            dosage=med.get("dosage"),
                            effect=med.get("effect"),
                            source_query=question[:100],
                        )
        except Exception as e:
            logger.warning(f"用药记录同步到L1失败，写入本地缓冲：{e}")
            try:
                from app.memory.fallback_buffer import enqueue_medication_event
                user_id = state.get("user_id")
                question = state.get("question", "")
                if user_id and new_checkpoint.get("medication_history"):
                    for med in new_checkpoint["medication_history"]:
                        if isinstance(med, dict) and med.get("drug"):
                            enqueue_medication_event(
                                user_id=user_id,
                                drug=med["drug"],
                                dosage=med.get("dosage"),
                                effect=med.get("effect"),
                                source_query=question[:100],
                            )
            except Exception as buf_err:
                logger.error(f"本地缓冲写入也失败（事件将丢失）：{buf_err}")

        # 删除已提取的消息（滑动窗口：保留最近3轮，删除其余）
        delete_messages = [RemoveMessage(id=m.id) for m in messages_to_extract if hasattr(m, 'id') and m.id]

        logger.info(f"临床状态快照更新完成，删除 {len(delete_messages)} 条早期消息，保留最近 {keep_recent_messages} 条")
        return {
            "clinical_checkpoint": new_checkpoint,
            "messages": delete_messages,
        }

    except Exception as e:
        logger.error(f"临床状态快照更新失败：{str(e)}")
        return {}

def format_clinical_checkpoint(checkpoint: Dict[str, Any]) -> str:
    """将临床状态快照格式化为结构化文本

    Args：
        checkpoint：结构化临床状态快照字典

    Returns：
        格式化后的中文文本
    """
    if not checkpoint:
        return ""

    lines = []

    # 主诉
    chief_complaint = checkpoint.get("chief_complaint")
    lines.append(f"主诉：{chief_complaint or '无'}")

    # 症状时间线
    symptom_timeline = checkpoint.get("symptom_timeline")
    if symptom_timeline:
        lines.append("症状时间线：")
        for item in symptom_timeline:
            symptom = item.get("symptom", "未知")
            onset = item.get("onset", "未知")
            severity = item.get("severity", "未知")
            evolution = item.get("evolution", "未知")
            lines.append(f"  - {symptom}：{onset}开始，{severity}，{evolution}")
    else:
        lines.append("症状时间线：无")

    # 用药记录
    medication_history = checkpoint.get("medication_history")
    if medication_history:
        lines.append("用药记录：")
        for item in medication_history:
            drug = item.get("drug", "未知")
            dosage = item.get("dosage", "未知")
            effect = item.get("effect", "未知")
            lines.append(f"  - {drug}（{dosage}）：{effect}")
    else:
        lines.append("用药记录：无")

    # 高危症状
    red_flags = checkpoint.get("red_flags")
    lines.append(f"高危症状：{', '.join(red_flags) if red_flags else '无'}")

    # 已确认信息
    confirmed_facts = checkpoint.get("confirmed_facts")
    lines.append(f"已确认信息：{', '.join(confirmed_facts) if confirmed_facts else '无'}")

    # 已排除
    ruled_out = checkpoint.get("ruled_out")
    lines.append(f"已排除：{', '.join(ruled_out) if ruled_out else '无'}")

    return "\n".join(lines)


@timing_decorator("获取上下文（临床快照）")
def get_context_with_summary(state: MedicalAssistantState) -> List:
    """获取带临床状态快照的上下文消息
    功能描述：
        用于在调用llm前构建完整的上下文
        将临床状态快照格式化为结构化文本，作为SystemMessage添加到消息列表开头
        对旧AI消息中的RAG文档内容进行MicroCompact压缩
    """
    messages = state.get("messages", [])
    checkpoint = state.get("clinical_checkpoint")

    # MicroCompact：对旧AI消息中的RAG文档内容进行压缩
    # 找到最后一条AI消息的索引，保留其原文不变
    last_ai_index = -1
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            last_ai_index = i
            break

    processed_messages = []
    for i, msg in enumerate(messages):
        if isinstance(msg, AIMessage) and i != last_ai_index:
            # 旧AI消息：剥离RAG文档内容
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            stripped_content = strip_rag_documents_from_history(content)
            if stripped_content != content:
                # 创建新的AIMessage，保留原id
                new_msg = AIMessage(content=stripped_content)
                if hasattr(msg, 'id') and msg.id:
                    new_msg.id = msg.id
                processed_messages.append(new_msg)
            else:
                processed_messages.append(msg)
        else:
            processed_messages.append(msg)

    if checkpoint:
        checkpoint_text = format_clinical_checkpoint(checkpoint)
        checkpoint_message = SystemMessage(content=f"【临床状态快照】\n{checkpoint_text}\n\n以下是最近的对话：")
        return [checkpoint_message] + processed_messages

    return processed_messages


async def stream_answer_generation(state: MedicalAssistantState):
    """流式答案生成，用于SSE流式输出
    功能描述：
        异步生成答案，每个token通过yield返回
        用户可以边生成便看到答案，无需等待全部完成

    Yields：
        str：每个token片段
    """
    logger.info(f"流式答案生成节点开始执行")
    start_time = time.time()

    # 优先使用重写后的完整问题（含上下文），无重写时用原问题
    question = state.get("final_question") or state.get("rewritten_query") or state.get("question", "")
    retrieved_docs = state.get("retrieved_docs")
    context_prompt = get_user_context_prompt(state.get("user_profile"))
    if context_prompt:
        logger.info("已注入用户上下文")

    prompt = build_rag_prompt(
        question=question,
        retrieved_docs=retrieved_docs,
        user_profile=state.get("user_profile"),
        state=state,
        symptoms=state.get("symptoms"),
    )

    # 日志记录 prompt 大小，用于 TTFT 分析
    prompt_chars = sum(len(msg.content) for msg in prompt if hasattr(msg, 'content'))
    prompt_est_tokens = prompt_chars // 2  # 中文约2字/token的粗略估算
    logger.info(f"RAG Prompt 大小：{prompt_chars}字符，估算约{prompt_est_tokens} tokens")

    llm = get_llm(streaming=True)
    full_answer = ""

    try:
        async for chunk in llm.astream(prompt):
            token = chunk.content
            if token:
                full_answer += token
                yield token

        elapsed_time = (time.time() - start_time) * 1000
        logger.info(f"流式生成完成，总长度：{len(full_answer)}字符，耗时：{elapsed_time:.2f}ms")

    except Exception as e:
        logger.error(f"流式生成失败：{str(e)}")
        yield "抱歉，生成答案时出现错误"
