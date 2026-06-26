"""医疗 Agent Skills 模块

当前 Skills：
    1. medical_safety_review — 医疗回答合规与安全审查
       - 规则引擎：0ms 诊断性断言检测 + 紧急风险拦截 + 免责声明注入
       - LLM 审查：规则引擎标记为高风险时，调用 LLM 做深度审查
       - 输出：{status: pass|revise|block, revised_answer, risk_tags}
    2. symptom_triage — 症状分诊与紧急度评估
       - 规则引擎：紧急度分级（🔴/🟡/🟢） + 危险症状组合检测 + 持续时间评估
       - 输出：{status: pass|revise, triage_result, advice_text, risk_tags}
    3. medication_guide — 用药指导规则引擎
       - 药物实体识别（AC 自动机）、禁忌人群交叉检查、用量安全范围检查
       - 药物相互作用初筛、规范性校验（5 字段完整性）
       - 输出：{status: pass|revise, revised_answer, risk_tags, details}
"""
from .safety_review_engine import (
    run_rule_based_review,
    get_block_template,
    detect_diagnostic_assertions,
    check_emergency_signals,
    check_disclaimer,
    DISCLAIMER,
    BLOCK_TEMPLATE,
)

from .symptom_triage_engine import (
    run_symptom_triage,
    classify_urgency,
    check_combination_risks,
    generate_triage_advice,
)

from .medication_guide_engine import (
    run_medication_guide_review,
    detect_drugs_in_answer,
    check_contraindications,
    check_drug_interactions,
    DRUG_SAFETY_RULES,
)

__all__ = [
    # safety_review_engine
    "run_rule_based_review",
    "get_block_template",
    "detect_diagnostic_assertions",
    "check_emergency_signals",
    "check_disclaimer",
    "DISCLAIMER",
    "BLOCK_TEMPLATE",
    # symptom_triage_engine
    "run_symptom_triage",
    "classify_urgency",
    "check_combination_risks",
    "generate_triage_advice",
    # medication_guide_engine
    "run_medication_guide_review",
    "detect_drugs_in_answer",
    "check_contraindications",
    "check_drug_interactions",
    "DRUG_SAFETY_RULES",
]
