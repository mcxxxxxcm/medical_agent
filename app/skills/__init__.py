"""医疗 Agent Skills 模块

当前 Skills：
    1. medical_safety_review — 医疗回答合规与安全审查
       - 规则引擎：0ms 诊断性断言检测 + 紧急风险拦截 + 免责声明注入
       - LLM 审查：规则引擎标记为高风险时，调用 LLM 做深度审查
       - 输出：{status: pass|revise|block, revised_answer, risk_tags}
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

__all__ = [
    "run_rule_based_review",
    "get_block_template",
    "detect_diagnostic_assertions",
    "check_emergency_signals",
    "check_disclaimer",
    "DISCLAIMER",
    "BLOCK_TEMPLATE",
]
