"""Pydantic 结构化输出模型

供各节点的结构化输出使用，支持 LangChain with_structured_output。
"""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

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
    symptom_timeline: Optional[List[Dict[str, Optional[str]]]] = Field(default=None, description="症状时间线，每项含symptom/onset/severity/evolution")
    medication_history: Optional[List[Dict[str, Optional[str]]]] = Field(default=None, description="用药记录，每项含drug/dosage/effect")
    red_flags: Optional[List[str]] = Field(default=None, description="高危症状列表")
    confirmed_facts: Optional[List[str]] = Field(default=None, description="已确认的既往史/过敏史")
    ruled_out: Optional[List[str]] = Field(default=None, description="已排除的疾病或原因")
    symptom_onset_dates: Optional[Dict[str, Dict[str, Any]]] = Field(default=None, description="症状首发日期映射，如{'头痛':{'iso':'2026-06-21T10:00:00','ts':1784567890,'precision':'exact'}}")
