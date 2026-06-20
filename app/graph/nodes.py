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
import re
import uuid
import time
from typing import Dict, Any, List, Optional, Literal
from functools import wraps

from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage, AIMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.core.config import get_config
from app.graph.state import MedicalAssistantState
from app.core.llm import get_llm, get_rewrite_llm, get_symptom_llm, get_vision_llm
from app.memory import get_long_term_memory
from app.core.app_logging import get_logger
from app.rag.hybrid_retriever import get_hybrid_retriever

logger = get_logger(__name__)
config = get_config()


# 计时装饰器
def timing_decorator(node_name: str):
    """节点耗时记录装饰器"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.info(f"⏱️ [{node_name}] 开始执行")

            try:
                result = func(*args, **kwargs)
                elapsed_time = (time.time() - start_time) * 1000
                logger.info(f"⏱️ [{node_name}] 执行完成，耗时：{elapsed_time:.2f}ms")
                return result
            except Exception as e:
                elapsed_time = (time.time() - start_time) * 1000
                logger.error(f"⏱️ [{node_name}] 执行失败，耗时：{elapsed_time:.2f}ms，错误：{str(e)}")
                raise

        return wrapper

    return decorator

def async_timing_decorator(node_name: str):
    """异步节点耗时记录装饰器"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.info(f"计时器：{node_name}开始执行")

            try:
                result = await func(*args, **kwargs)
                elapsed_time = (time.time() - start_time) * 1000
                logger.info(f"计时器：{node_name}执行完成，耗时：{elapsed_time:.2f}ms")
                return result
            except Exception as e:
                elapsed_time = (time.time() - start_time) * 1000
                logger.error(f"计时器：{node_name}执行失败，耗时：{elapsed_time:.2f}ms，错误：{str(e)}")
                raise
        return wrapper
    return decorator

# 结构化输出模型定义（改进原始方案，使用langchain的with_structured_output替代手动JSON解析）
class RouterOutput(BaseModel):
    """路由分类输出（router_node）"""
    question_type: Literal["symptom", "knowledge", "general"] = Field(
        description="问题类型：symptom=症状查询，knowledge=知识查询，general=一般问题"
    )


class SymptomAnalysisOutput(BaseModel):
    """症状分析输出"""
    symptoms: Optional[List[str]] = Field(default=None, description="症状列表")
    severity: Optional[Literal["轻微", "中等", "严重"]] = Field(default=None, description="严重程度")
    body_parts: Optional[List[str]] = Field(default=None, description="身体部位")
    duration: Optional[str] = Field(default=None, description="持续时间")
    additional_info: Optional[str] = Field(default=None, description="附加信息")


class SafetyCheckOutput(BaseModel):
    """安全检查输出"""
    is_safe: bool = Field(description="是否安全")
    risk_level: Literal["low", "medium", "high"] = Field(description="风险等级")
    detected_issues: List[str] = Field(default_factory=list, description="检测到的问题")
    requires_medical_attention: bool = Field(description="是否需要就医")


class GradeDocuments(BaseModel):
    """文档相关性评分（Agentic RAG 模式）"""
    binary_score: str = Field(
        description="文档相关性评分：'yes' 表示相关，'no' 表示不相关"
    )


class QueryRewriteOutput(BaseModel):
    """查询重写输出"""
    rewritten_query: str = Field(description="重写后的查询")


class ProfileExtractionOutput(BaseModel):
    """用户档案提取输出"""
    name: Optional[str] = Field(default=None, description="姓名")
    age: Optional[int] = Field(default=None, description="年龄")
    gender: Optional[str] = Field(default=None, description="性别")
    allergies: Optional[List[str]] = Field(default=None, description="过敏史")


class ClinicalCheckpointOutput(BaseModel):
    """结构化临床状态快照"""
    chief_complaint: Optional[str] = Field(default=None, description="核心主诉（如：持续性头痛3天）")
    symptom_timeline: Optional[List[Dict[str, str]]] = Field(default=None, description="症状时间线，每项含symptom/onset/severity/evolution")
    medication_history: Optional[List[Dict[str, str]]] = Field(default=None, description="用药记录，每项含drug/dosage/effect")
    red_flags: Optional[List[str]] = Field(default=None, description="高危症状列表")
    confirmed_facts: Optional[List[str]] = Field(default=None, description="已确认的既往史/过敏史")
    ruled_out: Optional[List[str]] = Field(default=None, description="已排除的疾病或原因")


def extract_json_block(raw_text: str) -> Optional[Dict[str, Any]]:
    """从模型文本输出中尽量提取 JSON 对象"""
    if not raw_text:
        return None

    text = raw_text.strip()
    candidates = [text]

    fenced_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    if fenced_match:
        candidates.insert(0, fenced_match.group(1).strip())

    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        candidates.append(brace_match.group(0).strip())

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except Exception:
            continue

    return None


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


def detect_rule_based_route(question: str) -> Optional[str]:
    """优先使用规则快速判断路由（优先级：symptom > knowledge > general）

    设计理念：
        1. 先检查症状关键词（最高优先级），避免 "你好我是王艺涵发烧了" 被误判为 general
        2. 再检查知识关键词
        3. 最后检查 general（精确匹配，避免误判）
        4. 无法匹配时返回 None，交给 LLM 判断
    """
    text = (question or "").strip().lower()
    if not text:
        return "general"

    # 1. 先检查症状关键词（最高优先级）
    symptom_keywords = [
        "怎么办", "咋办", "吃什么药", "挂什么科", "严不严重", "缓解", "疼", "痛", "痒", "肿",
        "发烧", "咳嗽", "流鼻涕", "头痛", "头疼", "嗓子疼", "喉咙痛", "腹痛", "肚子疼",
        "恶心", "呕吐", "腹泻", "拉肚子", "胸闷", "胸痛", "呼吸困难", "发炎", "不舒服",
        "过敏", "便秘", "失眠", "头晕", "乏力", "麻木", "出血", "溃疡", "骨折",
        "扭伤", "抽筋", "水肿", "贫血", "结石", "鼻炎", "咽炎", "皮炎", "湿疹",
        "哮喘", "痛风", "甲亢", "甲减", "颈椎", "腰椎", "关节炎",
    ]
    if any(keyword in text for keyword in symptom_keywords):
        return "symptom"

    # 2. 再检查知识关键词
    knowledge_keywords = [
        "是什么", "什么是", "原因", "症状", "治疗", "预防", "护理", "诊断", "检查",
        "高血压", "糖尿病", "感冒", "肺炎", "胃炎", "肝炎", "肾炎", "支气管炎",
        "冠心病", "脑梗", "脂肪肝", "胃溃疡", "甲状腺", "贫血", "痛风", "哮喘",
        "怎么吃", "注意什么", "禁忌", "副作用", "用量", "用法",
    ]
    if any(keyword in text for keyword in knowledge_keywords):
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
    """
    text = (question or "").strip()
    if not text:
        return None

    # 症状关键词映射（关键词 -> 标准症状名）
    symptom_map = {
        "发烧": "发烧", "发热": "发烧",
        "咳嗽": "咳嗽", "咳": "咳嗽",
        "感冒": "感冒", "伤风": "感冒",
        "流鼻涕": "流鼻涕", "鼻塞": "鼻塞",
        "头痛": "头痛", "头疼": "头痛",
        "嗓子疼": "嗓子疼", "喉咙痛": "嗓子疼", "咽痛": "嗓子疼",
        "腹痛": "腹痛", "肚子疼": "腹痛", "胃疼": "腹痛", "胃痛": "腹痛",
        "恶心": "恶心",
        "呕吐": "呕吐", "吐": "呕吐",
        "腹泻": "腹泻", "拉肚子": "腹泻",
        "胸闷": "胸闷",
        "胸痛": "胸痛",
        "呼吸困难": "呼吸困难",
        "过敏": "过敏",
        "痒": "瘙痒", "瘙痒": "瘙痒",
        "肿": "肿胀", "肿胀": "肿胀",
        "发炎": "发炎", "炎症": "发炎",
        "头晕": "头晕", "眩晕": "头晕", "晕": "头晕",
        "乏力": "乏力", "没力气": "乏力", "疲劳": "乏力",
        "失眠": "失眠", "睡不着": "失眠",
        "出血": "出血", "流血": "出血",
        "麻木": "麻木", "发麻": "麻木",
        "抽筋": "抽筋", "痉挛": "抽筋",
        "皮疹": "皮疹", "起疹子": "皮疹",
        "水肿": "水肿", "浮肿": "水肿",
        "便秘": "便秘",
        # 常见疾病名称（用户常以疾病名提问，如"高血压怎么办"）
        "高血压": "高血压",
        "糖尿病": "糖尿病",
        "肺炎": "肺炎",
        "胃炎": "胃炎",
        "肝炎": "肝炎",
        "肾炎": "肾炎",
        "支气管炎": "支气管炎",
        "鼻炎": "鼻炎",
        "肠炎": "肠炎",
        "关节炎": "关节炎",
        "冠心病": "冠心病",
        "心脏病": "心脏病",
        "哮喘": "哮喘",
        "痛风": "痛风",
        "贫血": "贫血",
        "甲亢": "甲亢", "甲状腺": "甲亢",
        "胃溃疡": "胃溃疡",
        "颈椎病": "颈椎病",
        "腰椎": "腰椎病",
    }

    # 严重程度关键词
    severity_map = {
        "很严重": "严重", "非常严重": "严重", "剧痛": "严重", "剧烈": "严重",
        "有点": "轻微", "轻微": "轻微", "稍微": "轻微", "一点": "轻微",
        "严重": "严重",
    }

    # 身体部位关键词
    body_part_map = {
        "头": "头部", "脑袋": "头部",
        "嗓子": "咽喉", "喉咙": "咽喉", "咽": "咽喉",
        "胸": "胸部",
        "肚子": "腹部", "胃": "胃部", "腹": "腹部",
        "背": "背部", "腰": "腰部",
        "腿": "腿部", "脚": "足部", "手": "手部", "臂": "手臂",
        "眼": "眼部", "眼睛": "眼部",
        "耳": "耳部", "耳朵": "耳部",
        "鼻": "鼻部",
        "皮肤": "皮肤",
    }

    # 提取症状
    found_symptoms = []
    for keyword, symptom_name in symptom_map.items():
        if keyword in text:
            found_symptoms.append(symptom_name)

    if not found_symptoms:
        return None

    # 去重
    found_symptoms = list(dict.fromkeys(found_symptoms))

    # 提取严重程度
    severity = None
    for keyword, level in severity_map.items():
        if keyword in text:
            severity = level
            break

    # 提取身体部位
    body_parts = []
    for keyword, part in body_part_map.items():
        if keyword in text:
            body_parts.append(part)
    body_parts = list(dict.fromkeys(body_parts))

    # 构建结果
    result = {
        "symptoms": found_symptoms,
        "severity": severity,
        "body_parts": body_parts if body_parts else None,
        "duration": None,
        "additional_info": None,
    }

    return result


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
    """构建无检索结果时的兜底回复"""
    return (
        f"暂时没有在知识库中检索到与“{question}”直接相关的可靠资料。"
        "建议补充更具体的信息，例如症状持续时间、伴随表现、既往病史或想了解的疾病名称。"
        "如症状明显加重、持续不缓解或伴有高热/剧烈疼痛/呼吸困难，请及时就医。"
    )


@timing_decorator("路由")
def router_node(state: MedicalAssistantState) -> Command:
    """路由节点

    功能描述：
        根据用户问题判断问题类型，使用Command.goto决定后续执行路径
        这是工作流的第一个节点，负责问题分类和路由决策

    Args：
        state：当前状态，包含question字段

    Returns：
        Command：包含路由目标的Command对象

    设计理念：
        1、智能分类：使用LLM进行问题类型分类
        2、Command路由：使用Command.goto控制流程，不存储route_decision
        3、容错处理：分类失败时使用默认路由
        4、日志记录：记录分类结果和路由决策

    路由类型：
        symptom：症状咨询，需要症状解析
        knowledge：知识查询，直接检索
        general：一般问题，直接检索
    """
    logger.info("路由节点开始执行")

    question = state.get("question", "")
    image_base64 = state.get("image_base64")

    # 优先检测图片：有图片走vision分支（参考蚂蚁阿福的图片问诊）
    if image_base64:
        logger.info("检测到图片输入，路由到vision分支")
        return Command(goto="vision_analysis")

    rule_based_route = detect_rule_based_route(question)
    if rule_based_route:
        logger.info(f"规则路由命中：question_type={rule_based_route}")
        question_type = rule_based_route
    else:
        try:
            llm = get_llm()
            prompt = f"""请判断以下问题的类型，只返回类型名称：
问题：{question}

类型选项：
- symptom：症状查询（如"我头痛怎么办"、"感冒了吃什么药"）
- knowledge：知识查询（如"什么是高血压"、"糖尿病的症状"）
- general：一般问题（如"你好"、"谢谢"、"你是谁"）
- refuse：非医疗相关问题（如"帮我写代码"、"今天天气"、"炒股建议"），这类问题应礼貌拒绝

请只返回类型名称（symptom/knowledge/general/refuse）："""

            raw_response = llm.invoke(prompt)
            raw_text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
            parsed_route = parse_router_output(raw_text)
            if parsed_route:
                question_type = parsed_route
            else:
                logger.warning(f"路由解析失败，使用 knowledge 兜底，模型原始输出：{raw_text}")
                question_type = "knowledge"

        except Exception as fallback_error:
            logger.warning(f"路由结构化输出失败，使用 knowledge 兜底：{fallback_error}")
            question_type = "knowledge"

    question_type = normalize_router_label(question_type)
    logger.info(f"问题类型：{question_type}")

    # 拒答：非医疗相关问题（参考蚂蚁阿福的安全边界）
    if question_type == "refuse":
        return Command(
            goto="direct_answer",
            update={"final_answer": "抱歉，我是医疗健康助手，只能回答与健康相关的问题。如果您有健康方面的疑问，欢迎随时向我咨询！"}
        )

    # 常规类型general就直接使用llm进行回复，不用调用RAG
    if question_type == "symptom":
        return Command(goto="symptom_analysis")
    elif question_type == "knowledge":
        return Command(goto="query_rewrite")
    else:
        return Command(goto="direct_answer")

@timing_decorator("症状解析")
def symptom_analysis_node(state: MedicalAssistantState) -> Dict[str, Any]:
    """症状解析节点

    功能描述：
        从用户问题中提取结构化的症状信息
        优化：规则优先提取，规则未命中时调用快速模型（glm-4-flash）

    Args：
        state：当前状态，包含question字段

    Returns：
        Dict[str, Any]：需要更新的状态字段
    """
    logger.info("症状解析节点开始执行")

    question = state.get("question", "")

    # 规则优先提取症状
    rule_result = _extract_symptoms_by_rules(question)
    if rule_result:
        logger.info(f"规则提取症状成功，跳过LLM：{rule_result}")
        return {"symptoms": rule_result}

    # 规则未命中：调用快速模型提取（glm-4-flash，首token ~1-2秒）
    try:
        llm = get_symptom_llm()

        prompt = f"""你是一位专业的症状分析助手，负责从用户描述中提取结构化的症状信息。

用户描述：
{question}

请提取症状名称、严重程度、身体部位、持续时间等信息。
请仅返回 JSON 对象，不要输出解释，不要输出 Markdown 代码块以外的说明。
字段包含：symptoms、severity、body_parts、duration、additional_info。
如果无法提取某项信息，该字段设为 null。"""
        result: SymptomAnalysisOutput = invoke_json_once_with_fallback(
            llm,
            prompt,
            SymptomAnalysisOutput,
            fallback_parser=parse_symptom_text,
        )
        payload = result.model_dump()

        logger.info(f"快速模型提取症状成功：{payload.get('symptoms')}")

        return {"symptoms": payload}

    except Exception as e:
        logger.error(f"症状解析失败：{str(e)}")
        return {"symptoms": {"symptoms": [], "severity": None, "body_parts": [], "duration": None, "additional_info": None}}

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

    # 获取原始查询和重写查询
    original_query = state.get("question", "")
    rewritten_query = state.get("rewritten_query", "")
    retrieval_attempts = int(state.get("retrieval_attempts") or 0) + 1

    # 检索时优先使用重写查询（提高检索质量）
    search_query = rewritten_query or original_query

    try:
        retriever = get_hybrid_retriever(k=5, alpha=0.5, use_reranker=True, rerank_top_k=10)

        start_time = time.time()
        docs = retriever.invoke(
            search_query,
            original_query=original_query,  # 传递原始查询用于redis缓存
        )
        retrieval_time = (time.time() - start_time) * 1000

        sources = []
        for doc in docs:
            sources.append({
                "source": doc.metadata.get("source", "未知"),
                "file_path": doc.metadata.get("file_path", "未知"),
                "content": doc.page_content[:100]
            })

        logger.info(f"检索到{len(docs)}个相关文档，耗时：{retrieval_time:.2f}ms，第 {retrieval_attempts} 次检索")

        return {
            "retrieved_docs": docs,
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

    if not retrieved_docs:
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
        return Command(goto="query_rewrite")

    relevant_docs = filter_relevant_docs(question, retrieved_docs)
    logger.info(f"文档启发式过滤结果：{len(retrieved_docs)} -> {len(relevant_docs)} 相关")

    if not relevant_docs:
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
        return Command(goto="query_rewrite")

    return Command(
        goto="answer_generation",
        update={"retrieved_docs": relevant_docs}
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
    """规范化路由分类标签"""
    label = (raw_label or "").strip().lower()
    if label in {"symptom", "knowledge", "general"}:
        return label

    if "symptom" in label or "症状" in label:
        return "symptom"
    if "knowledge" in label or "知识" in label:
        return "knowledge"
    if "general" in label or "一般" in label or "问候" in label:
        return "general"

    return "knowledge"


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


def get_conversation_history_text(state: MedicalAssistantState) -> str:
    """构建对话历史文本（含RAG文档MicroCompact压缩），不含临床快照

    临床快照由 build_rag_prompt 单独注入，避免重复
    """
    messages = state.get("messages", [])

    if not messages:
        return ""

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
            continue
        elif isinstance(msg, HumanMessage):
            history_parts.append(f"用户：{msg.content}")
        elif isinstance(msg, AIMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
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

    改进：要求至少2个关键词命中才算相关，避免单字/泛化词误匹配
    """
    tokens = [token for token in re.findall(r"[一-鿿A-Za-z0-9]+", question.lower()) if len(token) >= 2]
    if not tokens:
        return True

    content = (doc_content or "").lower()
    match_count = sum(1 for token in tokens if token in content)
    # 至少匹配2个关键词，或匹配超过一半的关键词
    return match_count >= min(2, len(tokens) / 2 + 0.5)


def filter_relevant_docs(question: str, retrieved_docs: List[Any]) -> List[Any]:
    """使用 rerank 分数和关键词重叠做轻量过滤，替代逐文档 LLM 评分

    策略：
    - Reranker 已执行：信任排序结果，保留 Top 文档（Reranker分数范围不固定，不能用绝对阈值）
    - Reranker 未执行：用关键词重叠做启发式过滤
    """
    if not retrieved_docs:
        return []

    # 检查是否有 rerank 分数（Reranker 真正执行过）
    has_rerank_scores = any(
        doc.metadata.get("rerank_score") is not None for doc in retrieved_docs
    )

    if has_rerank_scores:
        # Reranker 已执行：信任排序结果，保留 Top 文档
        # Reranker 的分数范围取决于模型（bge-reranker 可以为负值），不能用绝对阈值
        # 只过滤掉明显不相关的尾部文档（关键词完全不重叠且非前2名）
        filtered_docs = []
        for index, doc in enumerate(retrieved_docs):
            overlap = has_query_overlap(question, doc.page_content)
            # 前2名无条件保留（Reranker排序可信），其余需关键词重叠
            if index < 2 or overlap:
                filtered_docs.append(doc)
            else:
                logger.info(f"文档启发式过滤（Reranker排序靠后+无重叠）：{doc.page_content[:50]}...")

        return filtered_docs or retrieved_docs[:2]

    # Reranker 未执行：用关键词重叠做启发式过滤
    filtered_docs = []
    for index, doc in enumerate(retrieved_docs):
        overlap = has_query_overlap(question, doc.page_content)
        threshold_fallback = bool(doc.metadata.get("rerank_threshold_fallback"))
        keep_doc = overlap or (index == 0 and threshold_fallback)
        if keep_doc:
            filtered_docs.append(doc)
        else:
            logger.info(f"文档启发式过滤：{doc.page_content[:50]}...")

    return filtered_docs or retrieved_docs[:1]


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
    """构建 RAG 问答提示词（含对话历史 + 循证标注 + 主动追问 + 冻结层用户档案）"""
    checkpoint = state.get("clinical_checkpoint")
    checkpoint_text = format_clinical_checkpoint(checkpoint) if checkpoint else ""

    # 冻结层：用户档案独立注入，不嵌入 enhanced_question，确保摘要时不会丢失
    frozen_profile_section = ""
    profile_text = get_user_context_prompt(user_profile)
    if profile_text:
        frozen_profile_section = f"【用户档案（冻结层，永不压缩）】\n{profile_text}\n"

    # 对话历史：注入近期对话，让 LLM 理解上下文（如用户之前提到的用药情况）
    history_text = get_conversation_history_text(state)
    history_section = f"【对话历史】\n{history_text}\n" if history_text else ""

    if not retrieved_docs:
        return f"{frozen_profile_section}{history_section}请回答以下问题：\n{question}"

    formatted_docs = []
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get("source", "未知来源")
        content = doc.page_content
        # 截断文档内容，控制 prompt token 数（约300字≈150 token）
        if len(content) > 300:
            content = content[:300] + "..."
        # 当前轮文档存入 Redis，供后续轮次 MicroCompact 引用
        doc_id = f"{uuid.uuid4().hex[:8]}"
        try:
            from app.cache.redis_cache import get_cache
            get_cache().store_doc(doc_id, doc.page_content, source)  # 存完整原文
        except Exception:
            pass  # 存储失败不影响当前轮回答
        formatted_docs.append(f"[{source}]\n{content}")
    context = "\n\n".join(formatted_docs)

    # 生成追问引导
    followup = _build_followup_hints(symptoms)

    return f"""你是医疗助手，基于文档和对话历史回答问题。{frozen_profile_section}
【文档】
{context}

{f"【临床快照】{checkpoint_text}" if checkpoint_text else ""}
{history_section}【问题】{question}

要求：基于文档和对话历史回答，结合用户之前提到的信息（如用药、症状等）；无相关信息则说明；结尾加"⚠️ 以上建议仅供参考，如有疑问请及时就医"{f"；追问：{followup}" if followup else ""}。"""

    # 注意：history_section 已包含临床快照（通过 get_context_with_summary），
    # 但此处 checkpoint_text 单独注入确保快照始终可见，即使对话历史为空


def build_direct_answer_prompt(question: str, user_profile: Optional[Dict[str, Any]], state: MedicalAssistantState) -> str:
    """构建直接回答提示词（含冻结层用户档案）"""
    checkpoint = state.get("clinical_checkpoint")
    checkpoint_text = format_clinical_checkpoint(checkpoint) if checkpoint else ""

    # 冻结层：用户档案独立注入，不嵌入 enhanced_question，确保摘要时不会丢失
    frozen_profile_section = ""
    profile_text = get_user_context_prompt(user_profile)
    if profile_text:
        frozen_profile_section = f"【用户档案（冻结层，永不压缩）】\n{profile_text}\n"

    return f"""你是一个友好的医疗助手。

{frozen_profile_section}【临床状态快照】
{checkpoint_text if checkpoint_text else ""}

【用户问题】
{question}

请简洁友好地回复用户。如果是问候语，请热情回复。如果是感谢，请礼貌回应。
回复要简短，不要超过50个字。
"""


@timing_decorator("答案生成")
def answer_generation_node(state: MedicalAssistantState) -> Dict[str, Any]:
    """答案生成节点

    功能描述：
        基于检索到的文档生成用户友好的医疗答案
        这是RAG流程的生成节点，负责答案生成

    Args：
        state：当前状态，包含question和retrieved_docs字段

    Returns：
        Dict[str, Any]：需要更新的状态字段

    设计理念：
        1、RAG生成：结合检索到的文档生成答案
        2、用户友好：生成易于理解的答案
        3、安全提醒：添加必要的医疗安全提醒
        4、结构化输出：生成清晰的答案结构
    """
    logger.info("答案生成节点开始执行")

    question = state.get("question", "")
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
                    HumanMessage(content=question),
                    AIMessage(content=existing_answer)
                ]
            }
            if retrieved_docs:
                result["sources"] = format_retrieved_sources(retrieved_docs)
            return result

        prompt = build_rag_prompt(
            question=question,
            retrieved_docs=retrieved_docs,
            user_profile=state.get("user_profile"),
            state=state,
            symptoms=state.get("symptoms"),
        )

        llm = get_llm()
        start_time = time.time()
        response = llm.invoke(prompt)
        answer = response.content.strip()
        generation_time = (time.time() - start_time) * 1000

        logger.info(f"生成答案完成，长度：{len(answer)}字符，耗时：{generation_time:.2f}ms")

        result = {
            "final_answer": answer,
            "messages": [
                HumanMessage(content=question),
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
                HumanMessage(content=question),
                AIMessage(content="抱歉，生成答案时出现错误。")
            ]
        }

@timing_decorator("安全检查")
def safety_check_node(state: MedicalAssistantState) -> Dict[str, Any]:
    """安全检查节点
    
    功能描述：
        审核生成的内容是否安全合规
        识别风险并添加必要的警告信息
    
    Args：
        state：当前状态，包含final_answer字段
    
    Returns：
        Dict[str, Any]：需要更新的状态字段
    
    设计理念：
        1、内容审核
        2、风险评估
        3、紧急提醒
        4、警告累计（使用add reducer）
    
    风险等级：
        high：涉及紧急医疗情况
        medium：涉及处方药、需要专业诊断
        low：一般健康建议
    """
    logger.info("安全检查节点开始执行")

    content = state.get("final_answer", "")

    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(SafetyCheckOutput)

        prompt = f"""你是一位医疗安全审核专家，负责审核医疗建议的安全性。
【待审核内容】
{content}

请判断：
1. 内容是否安全
2. 风险等级
3. 检测到的问题
4. 是否需要就医

风险判断标准：
- high：涉及紧急医疗情况、严重症状、可能延误治疗
- medium：涉及处方药、需要专业诊断、症状不明确
- low：一般健康建议、家庭护理、预防措施"""

        result = structured_llm.invoke(prompt)

        warnings = result.detected_issues.copy()
        if result.requires_medical_attention:
            warnings.append("紧急提醒：建议立刻就医")
        warnings.append("本回答仅供参考，不能替代专业医生的诊断和治疗建议")

        logger.info(f"安全检查完成，风险等级：{result.risk_level}")

        return {"warnings": warnings}

    except Exception as e:
        logger.error(f"安全检查失败：{str(e)}")
        return {"warnings": ["本回答仅供参考，不能替代专业医生的诊断和治疗建议"]}

@timing_decorator("长期记忆加载")
def memory_load_node(state: MedicalAssistantState) -> Dict[str, Any]:
    """长期记忆加载节点
    功能描述：
        从长期记忆中读取用户档案和健康历史
        作为工作流的第一个节点，为后续节点提供用户上下文

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
        # # ✅ 新增：加载最近查询历史，暂不需要此功能。使用checkpointer能够找到历史
        # recent_queries = memory.get_query_history(user_id, limit=3)
        # 注入到 state 或日志
        # if recent_queries:
        #     logger.info(f"加载最近{len(recent_queries)}条查询记录")

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
        return {"user_profile": user_profile}

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
        llm = get_llm()
        structured_llm = llm.with_structured_output(ProfileExtractionOutput)

        prompt = f"""从以下用户问题中提取用户的个人信息：
用户问题：{question}

请提取姓名、年龄、性别、过敏史等信息。
如果问题中没有提到某项信息，该字段设为 null。"""

        result: ProfileExtractionOutput = structured_llm.invoke(prompt)

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

    except json.JSONDecodeError as e:
        logger.warning(f"JSON解析失败：{str(e)}")
        return {}
    except Exception as e:
        logger.warning(f"用户档案提取失败{str(e)}")
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
    """图片问诊节点（参考蚂蚁阿福的多模态问诊）

    处理用户上传的图片（体检报告、皮肤、药盒、处方等），
    使用多模态VLM直接理解图片内容并生成回答。

    技术路线：
        - 皮肤/外观类：纯VLM直接识别
        - 报告/处方类：VLM理解 + 结构化提取
        - 药盒类：VLM识别 + 知识库校验
    """
    logger.info("图片问诊节点开始执行")

    question = state.get("question", "")
    image_base64 = state.get("image_base64", "")
    user_profile = state.get("user_profile")

    if not image_base64:
        logger.warning("图片问诊节点未收到图片，降级为直接回答")
        return {"final_answer": "抱歉，未能接收到您的图片，请重新上传。"}

    try:
        llm = get_vision_llm()

        # 构建用户上下文（冻结层）
        context_prompt = get_user_context_prompt(user_profile)
        user_info = f"\n【用户档案（冻结层，永不压缩）】\n{context_prompt}\n" if context_prompt else ""

        # 构建多模态消息（LangChain标准格式）
        from langchain_core.messages import HumanMessage

        text_content = f"""你是一位专业的AI医疗助手，正在为用户解读医疗相关图片。

【重要提醒】
1. 你的回答仅供参考，不能替代专业医生的诊断和治疗建议
2. 如果发现紧急或严重情况，请立即建议用户就医
3. 基于图片内容如实描述，不要编造信息

{user_info}【用户问题】
{question if question else "请帮我解读这张图片"}

【回答要求】
1. 首先描述图片内容（如：这是一份血常规报告/皮肤照片/药盒等）
2. 提取关键信息（如：异常指标及数值、皮疹特征、药品名称等）
3. 用通俗易懂的语言解释含义
4. 给出初步建议和注意事项
5. 结尾加上安全提醒："⚠️ 以上解读仅供参考，如有疑问请及时就医"
6. 如果图片不是医疗相关内容，请礼貌说明并引导用户上传正确的图片"""

        message = HumanMessage(content=[
            {"type": "text", "text": text_content},
            {"type": "image", "base64": image_base64, "mime_type": "image/jpeg"},
        ])

        response = llm.invoke([message])
        answer = response.content.strip() if hasattr(response, "content") else str(response)

        logger.info(f"图片问诊完成，回答长度：{len(answer)}字符")

        return {
            "final_answer": answer,
            "warnings": ["⚠️ AI图片解读仅供参考，不能替代专业医生的诊断"],
        }

    except Exception as e:
        logger.error(f"图片问诊失败：{str(e)}")
        return {"final_answer": "抱歉，图片解读遇到了问题，请稍后重试或尝试文字描述您的症状。"}


async def stream_vision_answer(state: MedicalAssistantState):
    """流式图片问诊（异步生成器）"""
    logger.info("流式图片问诊节点开始执行")

    question = state.get("question", "")
    image_base64 = state.get("image_base64", "")
    user_profile = state.get("user_profile")

    if not image_base64:
        yield "抱歉，未能接收到您的图片，请重新上传。"
        return

    try:
        llm = get_vision_llm(streaming=True)

        context_prompt = get_user_context_prompt(user_profile)
        user_info = f"\n【用户信息】\n{context_prompt}" if context_prompt else ""

        from langchain_core.messages import HumanMessage

        text_content = f"""你是一位专业的AI医疗助手，正在为用户解读医疗相关图片。

【重要提醒】
1. 你的回答仅供参考，不能替代专业医生的诊断和治疗建议
2. 如果发现紧急或严重情况，请立即建议用户就医
3. 基于图片内容如实描述，不要编造信息

{user_info}
【用户问题】
{question if question else "请帮我解读这张图片"}

【回答要求】
1. 首先描述图片内容（如：这是一份血常规报告/皮肤照片/药盒等）
2. 提取关键信息（如：异常指标及数值、皮疹特征、药品名称等）
3. 用通俗易懂的语言解释含义
4. 给出初步建议和注意事项
5. 结尾加上安全提醒："⚠️ 以上解读仅供参考，如有疑问请及时就医"
6. 如果图片不是医疗相关内容，请礼貌说明并引导用户上传正确的图片"""

        message = HumanMessage(content=[
            {"type": "text", "text": text_content},
            {"type": "image", "base64": image_base64, "mime_type": "image/jpeg"},
        ])

        async for chunk in llm.astream([message]):
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content

    except Exception as e:
        logger.error(f"流式图片问诊失败：{str(e)}")
        yield "抱歉，图片解读遇到了问题，请稍后重试。"


@timing_decorator("同步直接回答")
def direct_answer_node(state: MedicalAssistantState) -> Dict[str, Any]:
    """直接回答节点
    功能描述：
        对于一般性问题，不需要检索知识库，直接调用llm回复
        优化：简单问候/寒暄直接返回预设回复，不调用LLM

    Agrs：
        state：当前状态，包含question字段
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
        llm = get_llm()
        response = llm.invoke(prompt)
        answer = response.content.strip()

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
    """查询重写节点

    功能描述：
        对原始问题进行检索友好的重写
        目标是提升后续知识检索召回率
        优化：明确医疗关键词直接跳过重写，模糊问题使用轻量模型重写
        关键：重写时注入临床快照和对话上下文，将模糊指代重写为具体查询
    """
    logger.info("查询重写节点开始执行")

    question = state.get("question", "")

    # 优化：对于明确的医疗提问，直接跳过重写
    # 但模糊/短问题即使包含医疗关键词也需要上下文感知重写
    _skip_rewrite_keywords = [
        # 症状相关（具体症状词，问题中包含这些词且有足够上下文时跳过重写）
        "发烧", "咳嗽", "流鼻涕", "头痛", "头疼", "嗓子疼", "喉咙痛", "腹痛", "肚子疼",
        "恶心", "呕吐", "腹泻", "拉肚子", "胸闷", "胸痛", "呼吸困难", "不舒服",
        "过敏", "便秘", "失眠", "头晕", "乏力", "麻木", "出血", "溃疡", "骨折",
        "扭伤", "抽筋", "水肿", "贫血", "结石", "鼻炎", "咽炎", "皮炎", "湿疹",
        "哮喘", "痛风", "甲亢", "甲减", "颈椎", "腰椎", "关节炎",
        # 疾病相关（具体疾病名）
        "高血压", "糖尿病", "感冒", "肺炎", "胃炎", "肝炎", "肾炎", "支气管",
        "冠心病", "脑梗", "脂肪肝", "甲状腺",
    ]

    # 检测是否为模糊/依赖上下文的问题
    # 这些模式即使包含医疗关键词也需要重写，因为它们缺少具体症状/疾病信息
    _context_dependent_patterns = [
        "换什么药", "吃什么药", "能吃什么", "该吃什么", "可以吃什么",
        "怎么办", "咋办", "怎么处理", "怎么治疗",
        "还有什么", "有没有其他", "能不能", "要不要",
        "呢", "吗", "还有", "继续", "然后",
    ]

    # 如果问题匹配了"依赖上下文"模式，走上下文感知重写
    is_context_dependent = any(p in question for p in _context_dependent_patterns)
    # 短问题（<8字）大概率依赖上下文
    is_short_question = len(question.strip()) < 8

    if not is_context_dependent and not is_short_question:
        if any(keyword in question for keyword in _skip_rewrite_keywords):
            logger.info(f"问题包含明确的医疗关键词且自包含，跳过重写：{question}")
            return {"rewritten_query": question}

    try:
        # 构建上下文感知的重写提示
        # 注入临床快照和最近对话，让模糊问题能被重写为具体查询
        context_parts = []

        # 临床快照
        checkpoint = state.get("clinical_checkpoint")
        if checkpoint:
            checkpoint_text = format_clinical_checkpoint(checkpoint)
            context_parts.append(f"【临床状态快照】\n{checkpoint_text}")

        # 最近2轮对话（避免过长）
        messages = state.get("messages", [])
        recent_msgs = messages[-4:] if len(messages) > 4 else messages  # 最近2轮=4条消息
        if recent_msgs:
            history_lines = []
            for msg in recent_msgs:
                if isinstance(msg, HumanMessage):
                    history_lines.append(f"用户：{msg.content}")
                elif isinstance(msg, AIMessage):
                    # AI消息截断，避免过长
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    history_lines.append(f"助手：{content[:100]}...")
            if history_lines:
                context_parts.append(f"【最近对话】\n" + "\n".join(history_lines))

        context_section = "\n\n".join(context_parts) if context_parts else "无上下文信息"

        # 优化：使用轻量模型进行查询重写，降低延迟
        llm = get_rewrite_llm()
        prompt = f"""你是一位医疗检索查询优化助手，请将用户问题重写为更适合知识库检索的短查询。

{context_section}

用户问题：
{question}

要求：
1. 结合上下文将模糊指代重写为具体查询（如"换什么药"→"对乙酰氨基酚无效后替代止痛药选择"）
2. 只输出重写后的查询
3. 保留核心医学实体和症状词
4. 不要解释，不要列出步骤，不要输出 JSON
5. 如果无需重写，直接返回原问题"""

        raw_response = llm.invoke(prompt)
        raw_text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
        rewritten_query = normalize_query_text(raw_text, question)

        if is_same_query(rewritten_query, question):
            logger.info("查询重写结果与原问题一致，直接复用原问题")
        else:
            logger.info(f"查询重写完成：{question[:30]} -> {rewritten_query[:30]}")

        return {"rewritten_query": rewritten_query}

    except Exception as e:
        logger.error(f"查询重写失败：{str(e)}")
        return {"rewritten_query": question}


def should_summarize(state: MedicalAssistantState) -> str:
    """判断是否需要总结
    Returns：
        "summarize"：需要
        "END"：不需要
    """
    max_messages = config.MAX_MESSAGES
    keep_recent_messages = config.KEEP_RECENT_MESSAGES
    summarize_trigger = config.SUMMARY_TRIGGER

    messages = state.get("messages", [])

    if len(messages) > summarize_trigger:
        logger.info(f"消息数量{len(messages)}超过阈值{summarize_trigger}，触发总结")
        return "summarize"

    from langgraph.graph import END
    return END

@timing_decorator("临床状态快照")
def summarize_conversation_node(state: MedicalAssistantState) -> Dict[str, Any]:
    """临床状态快照节点
    功能描述：
        将早期对话历史总结为结构化临床状态快照（JSON Checkpoint）
        保存最近的消息，总结并删除早期消息

    设计理念：
        1、当messages数量超过阈值时触发
        2、保留最近N条消息
        3、将早期消息总结为结构化临床快照存储在clinical_checkpoint
        4、快照会累计更新，而非每次重新生成
        5、使用ClinicalCheckpointOutput结构化输出，确保字段一致性
    """
    max_messages = config.MAX_MESSAGES
    keep_recent_messages = config.KEEP_RECENT_MESSAGES
    summarize_trigger = config.SUMMARY_TRIGGER

    logger.info(f"临床状态快照节点开始执行")

    messages = state.get("messages", [])
    existing_checkpoint = state.get("clinical_checkpoint")

    if len(messages) <= keep_recent_messages:
        logger.info(f"消息数量{len(messages)}未超过保留数量，跳过快照")
        return {}

    # 获取需要总结的消息（早于keep_recent_messages数量的消息都将总结）
    messages_to_summarize = messages[:-keep_recent_messages]

    if not messages_to_summarize:
        return {}

    logger.info(f"开始总结{len(messages_to_summarize)}条早期消息为临床状态快照")

    try:
        llm = get_llm()

        # 格式化消息为文本
        formatted_massages = []
        for msg in messages_to_summarize:
            role = "用户" if isinstance(msg, HumanMessage) else "助手"
            if isinstance(msg, SystemMessage):
                role = "系统"
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            formatted_massages.append(f"[{role}]: {content}")

        messages_text = "\n".join(formatted_massages)

        # 构建快照提示
        # 冻结层提示：用户档案已单独存储，快照中不需要重复记录，但应引用
        frozen_layer_note = "注意：用户档案信息（过敏史、既往病史等）已单独存储在冻结层，不需要在快照中重复记录，但快照中应引用这些信息（如'结合用户青霉素过敏史'）。"

        if existing_checkpoint:
            existing_json = json.dumps(existing_checkpoint, ensure_ascii=False, indent=2)
            checkpoint_prompt = f"""你是一位专业的医疗信息提取助手。以下是当前的临床状态快照（JSON格式）：

{existing_json}

请结合以下新的对话内容，更新并扩展临床状态快照。

【重要规则】
1. 必须保留所有已有的临床细节（持续时间、药物效果等），不得遗漏
2. 新信息与已有信息冲突时，以最新信息为准
3. 不得编造任何未在对话中提及的信息
4. 所有字段必须使用中文填写
5. 如果某项信息在对话中未提及，对应字段设为null
6. {frozen_layer_note}

对话内容：
{messages_text}

请输出更新后的完整临床状态快照JSON："""
        else:
            checkpoint_prompt = f"""你是一位专业的医疗信息提取助手。请从以下医疗助手对话中提取结构化临床状态快照。

【提取规则】
1. 仔细提取所有临床相关信息，包括症状、用药等
2. 不得编造任何未在对话中提及的信息
3. 所有字段必须使用中文填写
4. 如果某项信息在对话中未提及，对应字段设为null
5. 症状时间线中每项必须包含：symptom（症状）、onset（发作时间）、severity（严重程度）、evolution（演变）
6. 用药记录中每项必须包含：drug（药物）、dosage（剂量）、effect（效果）
7. {frozen_layer_note}

对话内容：
{messages_text}

请输出临床状态快照JSON："""

        result: ClinicalCheckpointOutput = invoke_json_once_with_fallback(
            llm,
            checkpoint_prompt,
            ClinicalCheckpointOutput,
        )
        new_checkpoint = result.model_dump()

        # 删除已总结的消息
        delete_messages = [RemoveMessage(id=m.id) for m in messages_to_summarize if hasattr(m, 'id') and m.id]

        logger.info(f"临床状态快照生成完成，删除 {len(delete_messages)} 条消息")
        return {
            "clinical_checkpoint": new_checkpoint,
            "messages": delete_messages,
        }

    except Exception as e:
        logger.error(f"临床状态快照生成失败：{str(e)}")
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

    question = state.get("question", "")
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
    prompt_chars = len(prompt)
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


async def stream_direct_answer(state: MedicalAssistantState):
    """流式直接回答节点
    优化：简单问候/寒暄直接返回预设回复，不调用LLM
    """
    logger.info("流式直接回答节点开始执行")
    start_time=time.time()

    question = state.get("question", "")
    user_profile = state.get("user_profile")

    # 优化：简单问候直接返回，不调用LLM
    quick_answer = _is_simple_greeting(question)
    if quick_answer:
        logger.info(f"简单问候直接返回，跳过LLM调用：{question}")
        yield quick_answer
        return

    prompt = build_direct_answer_prompt(
        question=question,
        user_profile=user_profile,
        state=state,
    )

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
        logger.error(f"流式直接回答失败：{str(e)}")
        yield "您好，有什么可以帮助你的吗？"
