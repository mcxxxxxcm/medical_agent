"""LangGraph 节点模块包

子模块划分：
    helpers.py           — 工具函数和常量（装饰器、JSON 解析、LLM 回退）
    models.py            — Pydantic 结构化输出模型
    nodes.py             — 主模块（所有节点函数，保持向后兼容）
    prompts.py           — ChatPromptTemplate 集中管理
    structured_output.py — 结构化输出三层降级策略（Tool Calling → JSON Mode → 纯文本）
    context_manager.py   — 四层上下文压缩策略（L1→L3→L2→L4）

所有公开接口通过本包重导出，外部代码可继续使用：
    from app.graph.nodes import router_node, ...
"""

# 从 helpers 子模块导出
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

# 从 structured_output 子模块导出
from .structured_output import invoke_structured

# 从 context_manager 子模块导出
from .context_manager import (
    compress_context,
    apply_l1_intermediate_cleanup,
    apply_l2_tool_output_trim,
    apply_l3_persist_large_outputs,
    apply_l4_llm_summary,
    load_persisted_output,
)

# 从 models 子模块导出
from .models import (
    ClinicalCheckpointOutput,
    ContextSummaryOutput,
    GradeDocuments,
    QuestionDecomposeOutput,
    ProfileExtractionOutput,
    QueryRewriteOutput,
    RouterOutput,
    SafetyCheckOutput,
    SymptomAnalysisOutput,
)

# 从主模块重新导出所有节点函数（保持向后兼容）
from .nodes import (
    answer_generation_node,
    build_direct_answer_prompt,
    build_no_results_answer,
    build_rag_prompt,
    detect_rule_based_route,
    direct_answer_node,
    filter_relevant_docs,
    format_clinical_checkpoint,
    format_retrieved_sources,
    strip_source_markers,
    get_context_with_summary,
    get_conversation_history_text,
    get_user_context_prompt,
    grade_documents_node,
    has_query_overlap,
    is_same_query,
    knowledge_retrieval_node,
    memory_load_node,
    normalize_query_text,
    normalize_router_label,
    parse_router_output,
    parse_symptom_text,
    profile_extraction_node,
    question_decompose_node,
    query_rewrite_node,
    router_node,
    safety_check_node,
    should_update_snapshot,
    stream_answer_generation,
    strip_rag_documents_from_history,
    symptom_analysis_node,
    update_clinical_snapshot_node,
    vision_analysis_node,
)
