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
from app.core.llm import get_llm
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

    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(RouterOutput)
        prompt = f"""请判断以下问题的类型，只返回类型名称：
问题：{question}

类型选项：
- symptom：症状查询（如"我头痛怎么办"、"感冒了吃什么药"）
- knowledge：知识查询（如"什么是高血压"、"糖尿病的症状"、"英语成绩查询"）
- general：一般问题（如"你好"、"谢谢"）

请只返回类型名称（symptom/knowledge/general）："""

        result: RouterOutput = structured_llm.invoke(prompt)
        question_type = result.question_type

        logger.info(f"问题类型：{question_type}")

        # 常规类型general就直接使用llm进行回复，不用调用RAG
        if question_type == "symptom":
            return Command(goto="symptom_analysis")
        elif question_type == "knowledge":
            return Command(goto="query_rewrite")
        else:
            return Command(goto="direct_answer")

    except Exception as e:
        logger.error(f"路由节点执行失败：{str(e)}")
        return Command(goto="knowledge_retrieval")

@timing_decorator("症状解析")
def symptom_analysis_node(state: MedicalAssistantState) -> Dict[str, Any]:
    """症状解析节点
    
    功能描述：
        从用户问题中提取结构化的症状信息
        包括症状名称、严重程度、持续时间等
    
    Args：
        state：当前状态，包含question字段
    
    Returns：
        Dict[str, Any]：需要更新的状态字段
    
    设计理念：
        1、结构化输出
        2、置信度评估
        3、容错处理
        4、JSON格式
    
    症状信息结构：
        {
            "symptoms": ["症状1", "症状2"],
            "severity": "轻微/中等/严重",
            "body_parts": ["身体部位1", "身体部位2"],
            "duration": "持续时间描述",
            "additional_info": "其他相关信息"
        }
    """
    logger.info("症状解析节点开始执行")

    question = state.get("question", "")

    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(SymptomAnalysisOutput)

        prompt = f"""你是一位专业的症状分析助手，负责从用户描述中提取结构化的症状信息。

用户描述：
{question}

请提取症状名称、严重程度、身体部位、持续时间等信息。
如果无法提取某项信息，该字段设为 null。"""
        result: SymptomAnalysisOutput = structured_llm.invoke(prompt)

        logger.info(f"成功提取症状：{result.symptoms}")

        return {"symptoms": result.model_dump()}

    except Exception as e:
        logger.error(f"症状解析失败：{str(e)}")
        return {"symptoms": None}

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

    # 检索时优先使用重写查询（提高检索质量）
    search_query = rewritten_query or original_query

    try:
        retriever = get_hybrid_retriever(k=5, alpha=0.5, use_reranker=True)

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

        logger.info(f"检索到{len(docs)}个相关文档，耗时：{retrieval_time:.2f}ms")

        return {
            "retrieved_docs": docs,
            "sources": sources
        }

    except Exception as e:
        logger.error(f"知识检索失败：{str(e)}")
        return {
            "retrieved_docs": [],
            "sources": [],
            "error": f"知识检索失败：{str(e)}"
        }


@timing_decorator("文档评分")
def grade_documents_node(state: MedicalAssistantState) -> Command:
    """文档相关性评分节点（Agentic RAG 模式）

    功能描述：
        评估检索到的文档是否与用户问题相关
        如果不相关，路由到 query_rewrite 重新检索
        如果相关，路由到 answer_generation 生成答案

    参考 LangChain Agentic RAG 教程中的 grade_documents 模式
    """
    logger.info("文档评分节点开始执行")

    question = state.get("question", "")
    retrieved_docs = state.get("retrieved_docs", [])

    if not retrieved_docs:
        logger.warning("无检索文档，路由到查询重写")
        return Command(goto="query_rewrite")

    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(GradeDocuments)

        # 逐个文档评分，过滤不相关的
        relevant_docs = []
        for doc in retrieved_docs:
            prompt = f"""你是一个文档相关性评估专家。请判断以下检索文档是否与用户问题相关。

用户问题：{question}

检索文档内容：
{doc.page_content}

如果文档包含与用户问题相关的关键词或语义，评为相关。只回答 'yes' 或 'no'。"""

            result = structured_llm.invoke(prompt)
            if result.binary_score.lower() == "yes":
                relevant_docs.append(doc)
            else:
                logger.info(f"文档不相关，已过滤：{doc.page_content[:50]}...")

        logger.info(f"文档评分结果：{len(retrieved_docs)} -> {len(relevant_docs)} 相关")

        if not relevant_docs:
            logger.info("所有文档不相关，路由到查询重写")
            return Command(goto="query_rewrite")

        # 更新状态并路由到答案生成
        return Command(
            goto="answer_generation",
            update={"retrieved_docs": relevant_docs}
        )

    except Exception as e:
        logger.error(f"文档评分失败：{str(e)}，直接生成答案")
        return Command(goto="answer_generation")

def get_user_context_prompt(user_profile: Optional[Dict[str, Any]]) -> str:
    """构建用户上下文提示"""
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

    return f"【用户信息】\n{chr(10).join(context_parts)}\n" if context_parts else ""


def get_conversation_history_text(state: MedicalAssistantState) -> str:
    """构建带摘要的对话历史文本"""
    context_messages = get_context_with_summary(state)
    if not context_messages:
        return ""

    history_parts = []
    for msg in context_messages:
        if isinstance(msg, SystemMessage):
            history_parts.append(f"{msg.content}")
        elif isinstance(msg, HumanMessage):
            history_parts.append(f"用户：{msg.content}")
        else:
            history_parts.append(f"助手：{msg.content}")
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


def build_rag_prompt(question: str, retrieved_docs: Optional[List[Any]], user_profile: Optional[Dict[str, Any]], state: MedicalAssistantState) -> str:
    """构建 RAG 问答提示词"""
    context_prompt = get_user_context_prompt(user_profile)
    conversation_history = get_conversation_history_text(state)
    enhanced_question = f"{context_prompt}【用户问题】\n{question}" if context_prompt else question

    if not retrieved_docs:
        return f"请回答以下问题：\n{enhanced_question}"

    formatted_docs = []
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get("source", "未知来源")
        content = doc.page_content
        formatted_docs.append(f"[文档{i} 来源：{source}]\n{content}")
    context = "\n\n".join(formatted_docs)

    return f"""你是一位专业的医疗健康助手，基于提供的医疗文档内容回答用户问题。

【重要提醒】
1. 你的回答仅供参考，不能替代专业医生的诊断和治疗建议
2. 如果问题涉及紧急医疗情况，请立即建议用户就医
3. 回答时要基于提供的文档内容，不要编造信息

【参考文档】
{context}

【对话历史总结】
{conversation_history if conversation_history else ""}

{enhanced_question}

【回答要求】
1. 基于参考文档内容，准确回答用户问题
2. 如果文档中没有相关信息，请明确说明
3. 回答要清晰易懂，避免过于专业的术语
4. 必要时提供实用的家庭护理建议
5. 结尾加上安全提醒："如有疑问，请及时就医"

请用中文回答："""


def build_direct_answer_prompt(question: str, user_profile: Optional[Dict[str, Any]], state: MedicalAssistantState) -> str:
    """构建直接回答提示词"""
    context_prompt = get_user_context_prompt(user_profile)
    conversation_history = get_conversation_history_text(state)
    enhanced_question = f"{context_prompt}【用户问题】\n{question}" if context_prompt else question

    return f"""你是一个友好的医疗助手。

【对话历史总结】
{conversation_history if conversation_history else ""}

{enhanced_question}

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
    context_prompt = get_user_context_prompt(state.get("user_profile"))
    if context_prompt:
        logger.info("已注入用户上下文")

    try:
        prompt = build_rag_prompt(
            question=question,
            retrieved_docs=retrieved_docs,
            user_profile=state.get("user_profile"),
            state=state,
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

@timing_decorator("同步直接回答")
def direct_answer_node(state: MedicalAssistantState) -> Dict[str, Any]:
    """直接回答节点
    功能描述：
        对于一般性问题，不需要检索知识库，直接调用llm回复

    Agrs：
        state：当前状态，包含question字段
    """
    logger.info("直接回答节点开始执行")

    question = state.get("question", "")
    user_profile = state.get("user_profile")

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
            "messages": [  # ✅ 添加消息
                HumanMessage(content=question),
                AIMessage(content=answer)
            ]
        }
    except Exception as e:
        logger.error(f"直接回答失败{str(e)}")
        return {
            "final_answer": "您好，有什么可以帮助你的吗？",
            "messages": [  # ✅ 添加消息
                HumanMessage(content=question),
                AIMessage(content="您好，有什么可以帮助你的吗？")
            ]
        }

@timing_decorator("查询重写")
def query_rewrite_node(state: MedicalAssistantState) -> Dict[str, Any]:
    """查询重写节点
    功能描述：
        将用户模糊问题重写为更精确的检索查询
        提高检索准确率
    示例：
        "感冒了怎么办" -> "普通感冒的家庭护理建议 感冒症状处理"
        "头疼吃什么药" -> "头痛止痛药物 头痛用药指导"
    """
    logger.info("查询重写节点开始执行")

    question = state.get("question", "")

    # 优化：对于明确的提问，直接跳过重写
    medical_keywords=[
        "炎", "病", "症", "治疗", "怎么办", "吃什么药",
        "症状", "原因", "预防", "护理", "诊断", "检查",
        "高血压", "糖尿病", "感冒", "发烧", "咳嗽", "头痛",
        "支气管", "肺炎", "胃炎", "肝炎", "肾炎"
    ]

    if any(keyword in question for keyword in medical_keywords):
        logger.info(f"问题包含明确的医疗关键词，跳过重写：{question}")
        return {"rewritten_query": question}

    # 优化：对于模糊的问题，调用llm进行重写，后续优化方向：本地部署小模型进行查询重写

    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(QueryRewriteOutput)

        prompt = f"""你一个查询优化专家，负责将用户问题重写为更合适检索的查询。
原始问题：{question}

请将问题重写为更精确的检索查询，要求：
1、保留原始问题的核心意图
2、添加相关的医术术语或同义词
3、拓展查询范围以提高召回率
4、保持简洁，不超过50个字"""

        result = structured_llm.invoke(prompt)
        rewritten_query = result.rewritten_query

        logger.info(f"查询重写：{question}--->{rewritten_query}")

        return {"rewritten_query": rewritten_query}

    except Exception as e:
        logger.warning(f"查询重写失败：{str(e)}，使用原始查询")
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

@timing_decorator("对话总结")
def summarize_conversation_node(state: MedicalAssistantState) -> Dict[str, Any]:
    """对话总结节点
    功能描述：
        将早期对话历史总结为摘要
        保存最近的消息，总结并删除早期消息

    设计理念：
        1、当messages数量超过阈值时触发
        2、保留最近N条消息
        3、将早期消息总结为摘要存储在conversation_summary
        4、摘要会累计更新，而非每次重新生成
    """
    max_messages = config.MAX_MESSAGES
    keep_recent_messages = config.KEEP_RECENT_MESSAGES
    summarize_trigger = config.SUMMARY_TRIGGER

    logger.info(f"对话总结节点开始执行")

    messages = state.get("messages", [])
    existing_summary = state.get("conversation_summary", "")

    if len(messages) <= keep_recent_messages:
        logger.info(f"消息数量{len(messages)}未超过保留数量，跳过总结")
        return {}

    # 获取需要总结的消息（早于keep_recent_messages数量的消息都将总结）
    messages_to_summarize = messages[:-keep_recent_messages]

    if not messages_to_summarize:
        return {}

    logger.info(f"开始总结{len(messages_to_summarize)}条早期消息")

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

        # 构建总结提示
        if existing_summary:
            summary_prompt = f"""这是一个医疗助手对话的现有摘要：
{existing_summary}

请结合以下新的对话内容，更新并扩展摘要。保留重要的医疗信息、症状描述、用药建议等关键内容：

对话内容：
{messages_text}

请生成更新后的完整摘要（不超过300字）："""
        else:
            summary_prompt = f"""请总结以下医疗助手对话的关键信息。
重点关注：
1. 用户描述的症状和健康问题
2. 给出的医疗建议和用药指导
3. 用户的个人信息（年龄、过敏史等）
4. 重要的上下文信息

对话内容：
{messages_text}

请生成简洁的摘要（不超过300字）："""

        response = llm.invoke(summary_prompt)
        new_summary = response.content.strip()

        # 删除已总结的消息
        delete_messages = [RemoveMessage(id=m.id) for m in messages_to_summarize if hasattr(m, 'id') and m.id]

        logger.info(f"总结完成，删除 {len(delete_messages)} 条消息")
        return {
            "conversation_summary": new_summary,
            "messages": delete_messages,
        }

    except Exception as e:
        logger.error(f"对话总结失败：{str(e)}")
        return {}

@timing_decorator("获取上下文（对话总结）")
def get_context_with_summary(state: MedicalAssistantState) -> List:
    """获取带摘要的上下文消息
    功能描述：
        用于在调用llm前构建完整的上下文
        将摘要作为SystemMessage添加到消息列表开头
    """
    messages = state.get("messages", [])
    summary = state.get("conversation_summary")

    if summary:
        summary_message = SystemMessage(content=f"【对话历史摘要】\n{summary}\n\n以下是最近的对话：")
        return [summary_message] + list(messages)

    return list(messages)


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
        logger.error(f"流式生成失败：{str(e)}")
        yield "抱歉，生成答案时出现错误"


async def stream_direct_answer(state: MedicalAssistantState):
    """流式直接回答节点"""
    logger.info("流式直接回答节点开始执行")
    start_time=time.time()

    question = state.get("question", "")
    user_profile = state.get("user_profile")

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
