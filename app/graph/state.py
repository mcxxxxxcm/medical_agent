"""LangGraph状态定义模块
功能描述：
    定义LangGraph工作流的核心状态结构
    采用单一State + input/output schema的官方推荐模式

设计理念：
    1、单一State：所有字段定义在一个TypedDict中
    2、Schema分离：通过input_schema/output_schema控制对外接口
    3、Reducer机制：使用Annotated + reducer处理累积字段
    4、路由决策：使用Command.goto处理，不存储到State

状态分类：
    输入字段：question, user_id
    输出字段：final_answer, warnings, sources
    中间字段：retrieved_docs, symptoms（节点间传递）
    标准字段：messages（LangGraph对话历史）
"""
from operator import add
from typing import TypedDict, Optional, List, Dict, Any, Annotated

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class MedicalAssistantState(TypedDict):
    """医疗助手状态
    
    设计原则：
        1. 单一State：Graph的核心状态，包含所有节点需要的字段
        2. 字段分类清晰：输入/输出/中间数据/标准字段
        3. 累积字段使用Annotated + reducer
    
    字段说明：
        【输入字段】
        question: 用户问题，必填
        user_id: 用户标识，用于多用户隔离，可选
        
        【LangGraph标准字段】
        messages: 对话消息列表，使用add reducer自动累积
        
        【输出字段】
        final_answer: 最终答案
        warnings: 警告信息列表，使用add reducer自动累积
        sources: 答案来源信息
        
        【中间字段 - 节点间传递】
        retrieved_docs: 检索到的文档列表
        symptoms: 症状分析结果
        
        【错误处理】
        error: 错误信息
    """

    # ===== 输入字段 =====
    question: str
    user_id: Optional[str]

    # ===== LangGraph标准字段 =====
    messages: Annotated[List[BaseMessage], add_messages]

    # ===== 输出字段 =====
    final_answer: Optional[str]
    warnings: List[str]  # 覆盖警告信息
    # warnings: Annotated[List[str], add] 累加警告信息
    sources: Annotated[List[Dict[str, str]], add]

    # ===== 中间字段 =====
    retrieved_docs: Optional[List[Document]]
    symptoms: Optional[Dict[str, Any]]

    # ===== 错误处理 =====
    error: Optional[str]

    # ===== 用户上下文（长期记忆）=====
    user_profile: Optional[Dict[str, Any]]

    # ===== 查询重写 =====
    rewritten_query: Optional[str]

    # ===== 对话摘要 =====
    conversation_summary: Optional[str]


class InputSchema(TypedDict):
    """输入Schema - 用户调用时传入"""
    question: str
    user_id: Optional[str]


class OutputSchema(TypedDict):
    """输出Schema - 返回给用户"""
    final_answer: Optional[str]
    sources: Optional[List[Dict[str, str]]]
    warnings: List[str]


def create_initial_state(question: str, user_id: Optional[str] = None) -> MedicalAssistantState:
    """创建初始状态"""
    return {
        "question": question,
        "user_id": user_id,
        "messages": [],
        "final_answer": None,
        "warnings": [],
        "sources": [],
        "retrieved_docs": None,
        "symptoms": None,
        "error": None,
        "user_profile": None,
        "rewritten_query": None,
        "conversation_summary": None,
    }


def add_warning(warning: str) -> Dict[str, Any]:
    """生成警告更新（warnings 使用 add reducer，返回列表即可）"""
    return {"warnings": [warning]}


def set_error(error: str) -> Dict[str, Any]:
    """生成错误更新"""
    return {"error": error}


def extract_output(state: MedicalAssistantState) -> OutputSchema:
    """提取输出状态"""
    return {
        "final_answer": state.get("final_answer"),
        "sources": state.get("sources"),
        "warnings": state.get("warnings", []),
    }
