"""Pydantic 结构化输出模型

供各节点的结构化输出使用，支持 LangChain with_structured_output。
每个模型严格对应 prompt 中的输出格式要求，确保 LLM 输出经过类型校验。
"""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class RouterOutput(BaseModel):
    """路由分类输出（router_node）

    对应 ROUTER_PROMPT 的输出格式要求。
    LLM 必须返回三个合法类型之一，否则校验失败兜底 general。
    """
    question_type: Literal["symptom", "knowledge", "general"] = Field(
        description="问题类型：symptom=症状查询，knowledge=知识查询，general=一般问题"
    )

    @field_validator("question_type", mode="before")
    @classmethod
    def normalize_question_type(cls, v):
        """兼容 LLM 输出的各种变体"""
        if isinstance(v, str):
            v = v.strip().lower()
            # 常见拼写错误/变体映射
            mapping = {
                "symptoms": "symptom",
                "症状": "symptom",
                "知识": "knowledge",
                "knowledge_query": "knowledge",
                "general_query": "general",
                "闲聊": "general",
                "问候": "general",
            }
            return mapping.get(v, v)
        return v


class SymptomAnalysisOutput(BaseModel):
    """症状分析输出（symptom_analysis_node）

    对应症状解析的结构化字段。
    注意：v8.4 后症状解析以规则引擎为主，此模型保留供未来 LLM 回退使用。
    """
    symptoms: Optional[List[str]] = Field(default=None, description="症状列表")
    severity: Optional[Literal["轻微", "中等", "严重"]] = Field(default=None, description="严重程度")
    body_parts: Optional[List[str]] = Field(default=None, description="身体部位")
    duration: Optional[str] = Field(default=None, description="持续时间")
    additional_info: Optional[str] = Field(default=None, description="附加信息")


class SafetyCheckOutput(BaseModel):
    """安全检查输出（safety_check_node）

    对应 SAFETY_CHECK_PROMPT 的 JSON 输出格式。
    """
    is_safe: bool = Field(description="是否安全")
    risk_level: Literal["low", "medium", "high"] = Field(description="风险等级")
    detected_issues: List[str] = Field(default_factory=list, description="检测到的问题")
    requires_medical_attention: bool = Field(description="是否需要就医")


class GradeDocuments(BaseModel):
    """文档相关性评分（Agentic RAG 模式）

    对应 grade_documents_node 的 LLM 评分输出。
    当前节点已改为启发式规则，此模型保留供未来 LLM 评分回退。
    """
    binary_score: str = Field(
        description="文档相关性评分：'yes' 表示相关，'no' 表示不相关"
    )


class QueryRewriteOutput(BaseModel):
    """查询重写输出（query_rewrite_node）

    对应 QUERY_REWRITE_PROMPT 的 FINAL + SEARCH 双行输出。
    v9.1 扩展：从单字段 rewritten_query 扩展为 final_question + search_keywords，
    匹配实际 Prompt 格式。
    """
    final_question: str = Field(
        description="补全后的自包含完整问题（FINAL 行）"
    )
    search_keywords: Optional[str] = Field(
        default=None,
        description="BM25 检索关键词，空格分隔（SEARCH 行）"
    )

    @field_validator("final_question", mode="before")
    @classmethod
    def strip_final_prefix(cls, v):
        """移除 LLM 可能残留的 FINAL: 前缀"""
        if isinstance(v, str):
            v = v.strip()
            if v.upper().startswith("FINAL:"):
                v = v[6:].strip()
            if v.upper().startswith("FINAL："):
                v = v[5:].strip()
        return v

    @field_validator("search_keywords", mode="before")
    @classmethod
    def strip_search_prefix(cls, v):
        """移除 LLM 可能残留的 SEARCH: 前缀"""
        if isinstance(v, str):
            v = v.strip()
            if v.upper().startswith("SEARCH:"):
                v = v[7:].strip()
            if v.upper().startswith("SEARCH："):
                v = v[6:].strip()
        return v


class ProfileExtractionOutput(BaseModel):
    """用户档案提取输出（profile_extraction_node）

    对应 PROFILE_EXTRACTION_PROMPT 的 JSON 输出格式。
    """
    name: Optional[str] = Field(default=None, description="姓名")
    age: Optional[int] = Field(default=None, description="年龄")
    gender: Optional[str] = Field(default=None, description="性别")
    allergies: Optional[List[str]] = Field(default=None, description="过敏史")

    @field_validator("age", mode="before")
    @classmethod
    def coerce_age(cls, v):
        """兼容 LLM 输出的字符串年龄"""
        if v is None:
            return None
        if isinstance(v, str):
            # "30岁" → 30
            digits = "".join(c for c in v if c.isdigit())
            return int(digits) if digits else None
        if isinstance(v, (int, float)):
            return int(v)
        return None

    @field_validator("allergies", mode="before")
    @classmethod
    def coerce_allergies(cls, v):
        """兼容 LLM 输出的字符串过敏史"""
        if v is None or v == "":
            return None
        if isinstance(v, str):
            return [item.strip() for item in v.replace("，", ",").split(",") if item.strip()]
        return v


class ClinicalCheckpointOutput(BaseModel):
    """结构化临床状态快照（update_snapshot_node）

    对应 CHECKPOINT_UPDATE_PROMPT / CHECKPOINT_NEW_PROMPT 的 JSON 输出格式。
    """
    chief_complaint: Optional[str] = Field(default=None, description="核心主诉（如：持续性头痛3天）")
    symptom_timeline: Optional[List[Dict[str, Optional[str]]]] = Field(default=None, description="症状时间线，每项含symptom/onset/severity/evolution")
    medication_history: Optional[List[Dict[str, Optional[str]]]] = Field(default=None, description="用药记录，每项含drug/dosage/effect")
    red_flags: Optional[List[str]] = Field(default=None, description="高危症状列表")
    confirmed_facts: Optional[List[str]] = Field(default=None, description="已确认的既往史/过敏史")
    ruled_out: Optional[List[str]] = Field(default=None, description="已排除的疾病或原因")
    symptom_onset_dates: Optional[Dict[str, Dict[str, Any]]] = Field(default=None, description="症状首发日期映射，如{'头痛':{'iso':'2026-06-21T10:00:00','ts':1784567890,'precision':'exact'}}")


class ContextSummaryOutput(BaseModel):
    """上下文摘要输出（L4 压缩层）

    对应 context_manager.py 中 _SUMMARY_PROMPT 的 JSON 输出格式。
    保留 5 类关键信息，确保 LLM 不会丢失对话中的关键上下文。
    """
    current_goal: str = Field(description="当前目标（用户在咨询什么健康问题）")
    key_findings: List[str] = Field(default_factory=list, description="关键发现和决策（已确认的症状、诊断、方案选择）")
    files_referenced: List[str] = Field(default_factory=list, description="参考过的文档来源列表")
    remaining_work: List[str] = Field(default_factory=list, description="尚未解决的问题（待确认的过敏史、未回答的追问）")
    user_constraints: List[str] = Field(default_factory=list, description="用户约束（过敏药物、年龄、孕哺状态、拒绝的治疗方案）")

    @field_validator('current_goal')
    @classmethod
    def ensure_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            return "未知"
        return v.strip()


class QuestionDecomposeOutput(BaseModel):
    """长问题拆解输出（question_decompose_node）

    对应 QUESTION_DECOMPOSE_PROMPT 的 JSON 输出格式。
    need_decompose=True 时 sub_questions 包含 2~4 个独立子问题。
    """
    need_decompose: bool = Field(
        description="是否需要拆解：True=问题包含多个独立子问题，False=单一问题不拆解"
    )
    sub_questions: List[str] = Field(
        default_factory=list,
        description="拆解后的子问题列表（need_decompose=False 时为空列表或包含原问题）"
    )

    @field_validator("sub_questions", mode="before")
    @classmethod
    def ensure_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return v
