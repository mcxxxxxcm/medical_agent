"""BadCase 三大失败大类归因定义
__version__ = 9.53

将细粒度 case_type 归因到统一的失败大类，用于管理后台展示与审核：
    1. retrieval_fail（检索失败）：正确答案在知识库中但未召回或排序靠后
    2. knowledge_gap（知识缺失）：知识库中根本没有答案所需的信息
    3. generation_fail（生成失败）：检索到正确上下文但 LLM 总结/推理出错
    4. other（其他）：成因不明或非真 badcase，需审核时人工指定

存储仍保留细粒度 case_type 不变（回归测试、标注、导出脚本均依赖），本模块
提供 case_type → 大类 的默认归类，供采集默认值和统计兜底使用。
"""
from typing import Dict, Optional

# 三大失败大类（含 other），顺序即前端下拉显示顺序
CATEGORIES = ("retrieval_fail", "knowledge_gap", "generation_fail", "other")

CATEGORY_ZH: Dict[str, str] = {
    "retrieval_fail": "检索失败",
    "knowledge_gap": "知识缺失",
    "generation_fail": "生成失败",
    "other": "其他",
}

# 细粒度 case_type → 中文子类型标签
CASE_TYPE_ZH: Dict[str, str] = {
    "rewrite_missed_anaphora": "指代未重写",
    "rewrite_lost_entity": "丢失实体",
    "rewrite_same_as_original": "未重写",
    "low_score_no_clarify": "低分未澄清",
    "hallucination_suspected": "疑似幻觉",
    "retrieval_miss": "检索失败",
    "route_misclassification": "路由误分类",
    "user_negative_feedback": "用户负反馈",
    "manual_flag": "人工标记",
    "user_misclick": "误点",
    "acceptable_answer": "答案可接受",
    "unknown": "未知",
}

# case_type → 默认大类（可自动归类的做预选，含糊的归 other 待人工指定）
_CASE_TYPE_CATEGORY_DEFAULT: Dict[str, str] = {
    "retrieval_miss": "retrieval_fail",
    "route_misclassification": "retrieval_fail",
    "rewrite_missed_anaphora": "retrieval_fail",
    "rewrite_lost_entity": "retrieval_fail",
    "rewrite_same_as_original": "retrieval_fail",
    "low_score_no_clarify": "retrieval_fail",
    "hallucination_suspected": "generation_fail",
    # user_negative_feedback / manual_flag 成因不明，默认 other，审核时人工指定
    "user_negative_feedback": "other",
    "manual_flag": "other",
    "user_misclick": "other",
    "acceptable_answer": "other",
    "unknown": "other",
}


def default_category(case_type: Optional[str]) -> str:
    """返回 case_type 对应的默认失败大类，未知类型回退 other"""
    return _CASE_TYPE_CATEGORY_DEFAULT.get(case_type or "unknown", "other")


def category_zh(category: str) -> str:
    """三大类枚举 → 中文，未知回退 other"""
    return CATEGORY_ZH.get(category or "", CATEGORY_ZH["other"])


def case_type_zh(case_type: Optional[str]) -> str:
    """细粒度 case_type → 中文子类型，未知回退原值或"未知\""""
    zh = CASE_TYPE_ZH.get(case_type or "")
    return zh if zh else (case_type if case_type else "未知")