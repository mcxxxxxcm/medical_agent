"""医疗安全审查规则引擎

在 LLM 审查之前执行 0ms 的规则检测，覆盖：
1. 诊断性断言检测（正则匹配绝对化诊断表述）
2. 紧急风险信号检测（交叉检查临床快照）
3. 免责声明检测
4. 审查决策聚合
"""
import re
from typing import Dict, Any, List, Tuple, Optional

# 诊断性断言模式（正则）
DIAGNOSTIC_ASSERTION_PATTERNS = [
    r"确诊为",
    r"就是.{0,6}病",
    r"肯定是",
    r"一定是",
    r"毫无疑问是",
    r"百分之百是",
    r"绝对是",
    r"你患了",
    r"你得了",
    r"诊断你是",
]

# 诊断性断言的替换建议
DIAGNOSTIC_REPLACEMENTS = {
    "确诊为": "可能提示为",
    "就是": "可能是",
    "肯定是": "可能是",
    "一定是": "可能是",
    "毫无疑问是": "可能是",
    "百分之百是": "可能是",
    "绝对是": "可能是",
    "你患了": "您可能患有",
    "你得了": "您可能出现了",
    "诊断你是": "初步判断您可能是",
}

# 紧急症状关键词
EMERGENCY_SYMPTOMS = [
    "胸痛", "呼吸困难", "剧烈头痛", "意识不清", "晕厥",
    "大出血", "吐血", "便血", "持续高烧不退",
    "抽搐", "癫痫", "严重过敏", "过敏性休克",
    "急性腹痛", "药物中毒",
]

# 标准免责声明
DISCLAIMER = "⚠️ 以上建议仅供参考，如有疑问请及时就医"

# 预设安全拒答模板
BLOCK_TEMPLATE = (
    "该问题涉及个体化诊疗，我目前无法给出确切建议。"
    "建议您携带以下信息线下就诊：\n"
    "1. 当前症状及持续时间\n"
    "2. 既往病史和用药史\n"
    "3. 近期检查报告（如有）\n\n"
    f"{DISCLAIMER}"
)


def detect_diagnostic_assertions(answer: str) -> List[Dict[str, Any]]:
    """检测诊断性断言
    
    Returns:
        匹配结果列表，每项包含:
        - pattern: 匹配的模式
        - match: 匹配的文本
        - position: 位置
        - replacement: 建议替换
    """
    results = []
    for pattern in DIAGNOSTIC_ASSERTION_PATTERNS:
        for match in re.finditer(pattern, answer):
            results.append({
                "pattern": pattern,
                "match": match.group(),
                "position": match.start(),
                "replacement": DIAGNOSTIC_REPLACEMENTS.get(match.group(), "可能"),
            })
    return results


def check_emergency_signals(answer: str, clinical_checkpoint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """紧急风险二次拦截
    
    检查用户症状快照中是否有危急重症信号，
    以及回答中是否给出了就医指引。
    
    Returns:
        {
            "has_emergency_symptom": bool,
            "emergency_symptoms": [...],
            "answer_addressed_emergency": bool,
            "needs_emergency_alert": bool,  # True 表示需要追加紧急提示
        }
    """
    # 从临床快照中提取症状（使用 AC 自动机精确匹配）
    emergency_in_snapshot = []
    if clinical_checkpoint:
        try:
            from app.core.keyword_matcher import get_emergency_matcher
            emergency_matcher = get_emergency_matcher()

            # 检查 symptoms 字段
            symptoms = clinical_checkpoint.get("symptoms", [])
            if isinstance(symptoms, list):
                for sym in symptoms:
                    if isinstance(sym, str) and emergency_matcher.contains_any(sym, use_boundary=False):
                        emergency_in_snapshot.append(sym)

            # 检查 symptom_onset_dates 字段
            onset_dates = clinical_checkpoint.get("symptom_onset_dates", {})
            if isinstance(onset_dates, dict):
                for sym_name in onset_dates:
                    if emergency_matcher.contains_any(sym_name, use_boundary=False):
                        if sym_name not in emergency_in_snapshot:
                            emergency_in_snapshot.append(sym_name)
        except Exception:
            # 降级为原有逻辑
            for sym in (clinical_checkpoint.get("symptoms", []) if isinstance(clinical_checkpoint.get("symptoms"), list) else []):
                if isinstance(sym, str):
                    for emerg in EMERGENCY_SYMPTOMS:
                        if emerg in sym or sym in emerg:
                            emergency_in_snapshot.append(sym)

    # 也检查当前问题中的紧急症状（使用 AC 自动机）
    answer_has_emergency = False
    try:
        from app.core.keyword_matcher import get_emergency_matcher
        answer_has_emergency = get_emergency_matcher().contains_any(answer, use_boundary=False)
    except Exception:
        for emerg in EMERGENCY_SYMPTOMS:
            if emerg in answer:
                answer_has_emergency = True
            if emerg not in emergency_in_snapshot:
                emergency_in_snapshot.append(emerg)
    
    # 检查回答中是否有就医指引
    medical_care_indicators = [
        "就医", "就诊", "急诊", "医院", "120", "看医生",
        "及时就医", "立即就医", "尽快就医", "咨询医生",
    ]
    answer_addressed = any(ind in answer for ind in medical_care_indicators)
    
    needs_alert = bool(emergency_in_snapshot) and not answer_addressed
    
    return {
        "has_emergency_symptom": bool(emergency_in_snapshot),
        "emergency_symptoms": emergency_in_snapshot,
        "answer_addressed_emergency": answer_addressed,
        "needs_emergency_alert": needs_alert,
    }


def check_disclaimer(answer: str) -> Dict[str, Any]:
    """检查免责声明
    
    Returns:
        {
            "has_disclaimer": bool,
            "needs_injection": bool,
        }
    """
    # 检查是否包含免责声明的关键部分
    disclaimer_keywords = ["仅供参考", "及时就医", "不能替代", "专业医生"]
    has_disclaimer = any(kw in answer for kw in disclaimer_keywords)
    
    return {
        "has_disclaimer": has_disclaimer,
        "needs_injection": not has_disclaimer,
    }


def revise_diagnostic_assertions(answer: str, assertions: List[Dict[str, Any]]) -> str:
    """替换诊断性断言
    
    将绝对化诊断表述替换为风险提示句式。
    """
    revised = answer
    # 从后往前替换，避免位置偏移
    for assertion in sorted(assertions, key=lambda x: x["position"], reverse=True):
        match_text = assertion["match"]
        replacement = assertion["replacement"]
        revised = revised[:assertion["position"]] + replacement + revised[assertion["position"] + len(match_text):]
    return revised


def inject_disclaimer(answer: str) -> str:
    """注入免责声明"""
    return answer.rstrip() + "\n\n" + DISCLAIMER


def inject_emergency_alert(answer: str, emergency_symptoms: List[str]) -> str:
    """注入紧急就医提示"""
    alert = f"\n\n⚠️ 紧急提醒：检测到您提到了{'、'.join(emergency_symptoms)}等症状，建议立即就医或拨打120！"
    return answer.rstrip() + alert


def run_rule_based_review(
    answer: str,
    clinical_checkpoint: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """执行规则引擎审查（0ms，不调用 LLM）
    
    Returns:
        {
            "status": "pass" | "revise" | "block",
            "revised_answer": str,
            "risk_tags": List[str],
            "details": {...},
        }
    """
    risk_tags = []
    revisions_needed = []
    
    # 1. 诊断性断言检测
    assertions = detect_diagnostic_assertions(answer)
    if assertions:
        risk_tags.append("diagnostic_assertion")
        revisions_needed.append("diagnostic_assertion")
    
    # 2. 紧急风险检测
    emergency_check = check_emergency_signals(answer, clinical_checkpoint)
    if emergency_check["needs_emergency_alert"]:
        risk_tags.append("emergency_risk_missed")
        revisions_needed.append("emergency_alert")
    
    # 3. 免责声明检测
    disclaimer_check = check_disclaimer(answer)
    if disclaimer_check["needs_injection"]:
        revisions_needed.append("disclaimer")
    
    # 执行修订
    revised_answer = answer
    
    if "diagnostic_assertion" in revisions_needed:
        revised_answer = revise_diagnostic_assertions(revised_answer, assertions)
    
    if "emergency_alert" in revisions_needed:
        revised_answer = inject_emergency_alert(revised_answer, emergency_check["emergency_symptoms"])
    
    if "disclaimer" in revisions_needed:
        revised_answer = inject_disclaimer(revised_answer)
    
    # 决策
    if not revisions_needed:
        status = "pass"
    else:
        status = "revise"
    
    return {
        "status": status,
        "revised_answer": revised_answer if status == "revise" else answer,
        "risk_tags": risk_tags,
        "details": {
            "diagnostic_assertions": assertions,
            "emergency_check": emergency_check,
            "disclaimer_check": disclaimer_check,
        },
    }


def get_block_template() -> str:
    """获取预设安全拒答模板"""
    return BLOCK_TEMPLATE
