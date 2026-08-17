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

# 含通配符的模式 → 仅替换其字面前缀（保留疾病名）
# 例如 "就是.{0,6}病" 匹配 "就是感冒病"，只把 "就是" 替换为 "可能是"，保留 "感冒病"
DIAGNOSTIC_PATTERN_KEYWORDS = {
    r"就是.{0,6}病": "就是",
}

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
        - keyword: 需替换的关键字前缀（含通配符模式仅替换前缀，保留疾病名）
        - match: 匹配的文本
        - position: 位置
        - replacement: 建议替换
    """
    results = []
    for pattern in DIAGNOSTIC_ASSERTION_PATTERNS:
        for match in re.finditer(pattern, answer):
            keyword = DIAGNOSTIC_PATTERN_KEYWORDS.get(pattern, match.group())
            results.append({
                "pattern": pattern,
                "keyword": keyword,
                "match": match.group(),
                "position": match.start(),
                "replacement": DIAGNOSTIC_REPLACEMENTS.get(keyword, "可能"),
            })
    return results


def _checkpoint_symptom_names(clinical_checkpoint: Optional[Dict[str, Any]]) -> List[str]:
    """从临床快照提取症状名（H2：无顶层 symptoms，实际在 symptom_timeline 中）"""
    if not clinical_checkpoint:
        return []
    symptoms = []
    timeline = clinical_checkpoint.get("symptom_timeline")
    if isinstance(timeline, list):
        for item in timeline:
            if isinstance(item, dict):
                sym = item.get("symptom")
                if isinstance(sym, str) and sym.strip():
                    symptoms.append(sym.strip())
    legacy = clinical_checkpoint.get("symptoms")
    if isinstance(legacy, list):
        for s in legacy:
            if isinstance(s, str) and s.strip() and s.strip() not in symptoms:
                symptoms.append(s.strip())
    return symptoms


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

            # H2 修复：ClinicalCheckpointOutput 无顶层 symptoms，症状在 symptom_timeline 中
            symptoms = _checkpoint_symptom_names(clinical_checkpoint)
            for sym in symptoms:
                if emergency_matcher.contains_any(sym, use_boundary=False):
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
            for sym in _checkpoint_symptom_names(clinical_checkpoint):
                for emerg in EMERGENCY_SYMPTOMS:
                    if emerg in sym or sym in emerg:
                        emergency_in_snapshot.append(sym)

    # 也检查回答本身是否含紧急症状（使用 AC 自动机）
    # M12 修复：此前 answer_has_emergency 计算后未参与决策，
    # 快照无紧急症状、仅回答含"胸痛"且未给就医指引时拦截落空。
    answer_emergency_symptoms = []
    try:
        from app.core.keyword_matcher import get_emergency_matcher
        answer_emergency_symptoms = get_emergency_matcher().get_matched_keywords(answer, use_boundary=False)
    except Exception:
        # P2-12：append 必须在命中条件内，否则异常时把全部紧急症状塞入快照误报紧急
        for emerg in EMERGENCY_SYMPTOMS:
            if emerg in answer and emerg not in answer_emergency_symptoms:
                answer_emergency_symptoms.append(emerg)

    # 合并快照 + 回答中的紧急症状（用于注入提示时完整列出）
    all_emergency_symptoms = list(emergency_in_snapshot)
    for sym in answer_emergency_symptoms:
        if sym not in all_emergency_symptoms:
            all_emergency_symptoms.append(sym)

    # 检查回答中是否有就医指引
    medical_care_indicators = [
        "就医", "就诊", "急诊", "医院", "120", "看医生",
        "及时就医", "立即就医", "尽快就医", "咨询医生",
    ]
    answer_addressed = any(ind in answer for ind in medical_care_indicators)

    has_emergency = bool(emergency_in_snapshot) or bool(answer_emergency_symptoms)
    needs_alert = has_emergency and not answer_addressed

    return {
        "has_emergency_symptom": has_emergency,
        "emergency_symptoms": all_emergency_symptoms,
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

    H13 修复：只替换断言关键字前缀，保留疾病名。
    此前整段替换（如"就是感冒病"→"可能"）会把疾病名删掉。
    """
    revised = answer
    # 从后往前替换，避免位置偏移
    for assertion in sorted(assertions, key=lambda x: x["position"], reverse=True):
        keyword = assertion.get("keyword", assertion["match"])
        replacement = assertion["replacement"]
        start = assertion["position"]
        end = start + len(keyword)
        revised = revised[:start] + replacement + revised[end:]
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
