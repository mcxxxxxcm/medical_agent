"""症状分诊规则引擎

在症状分析节点后执行规则化分诊判断：
1. 症状紧急度分级（🔴/🟡/🟢）
2. 危险症状组合检测
3. 持续时间维度评估
4. 分诊建议生成
"""
import time
from typing import Dict, Any, List, Optional


# 紧急症状（🔴 立即就医/120）
EMERGENCY_SYMPTOMS = [
    "胸痛", "呼吸困难", "意识不清", "晕厥", "大出血",
    "吐血", "便血", "抽搐", "癫痫", "严重过敏",
    "过敏性休克", "急性腹痛",
]

# 紧急症状组合（单个不紧急，组合后紧急）
EMERGENCY_COMBINATIONS = [
    {"symptoms": {"头痛", "发热", "颈僵"}, "risk": "脑膜炎高风险", "level": "🔴"},
    {"symptoms": {"胸痛", "呼吸困难", "冷汗"}, "risk": "心梗高风险", "level": "🔴"},
    {"symptoms": {"头痛", "呕吐", "视力模糊"}, "risk": "颅内压增高可能", "level": "🔴"},
    {"symptoms": {"发热", "皮疹", "呼吸困难"}, "risk": "严重过敏反应", "level": "🔴"},
    {"symptoms": {"腹痛", "发热", "呕吐"}, "risk": "急腹症可能", "level": "🔴"},
]

# 建议就诊症状（🟡 48小时内）
URGENT_SYMPTOMS = [
    "持续高烧", "反复发作", "加重趋势", "用药后无改善",
]

# 持续时间阈值
DURATION_URGENT_THRESHOLD_HOURS = 72  # 超过72小时建议就诊

# 等级映射
LEVEL_MAP = {
    "🔴": "紧急",
    "🟡": "建议就诊",
    "🟢": "居家观察",
}

# 科室推荐映射
DEPARTMENT_SUGGESTIONS = {
    "头痛": "神经内科",
    "发热": "内科",
    "胸痛": "心内科或急诊",
    "腹痛": "消化内科或急诊",
    "呼吸困难": "呼吸内科或急诊",
    "皮疹": "皮肤科",
    "呕吐": "消化内科",
    "视力模糊": "眼科或神经内科",
}


def classify_urgency(
    symptoms: List[str],
    severity: Optional[str] = None,
    duration_hours: Optional[float] = None,
    clinical_checkpoint: Optional[Dict] = None,
) -> Dict:
    """症状紧急度分级

    Returns:
        {
            "level": "🔴" | "🟡" | "🟢",
            "level_name": "紧急" | "建议就诊" | "居家观察",
            "reason": str,
            "matched_emergency": [...],
            "matched_combinations": [...],
            "duration_assessment": str,
        }
    """
    matched_emergency = []
    matched_combinations = []

    # 1. 检查单个紧急症状
    symptom_set = set(symptoms)
    for emerg in EMERGENCY_SYMPTOMS:
        for sym in symptoms:
            if emerg in sym or sym in emerg:
                matched_emergency.append(emerg)
                break

    # 2. 检查危险症状组合
    combo_results = check_combination_risks(symptoms)
    for combo in combo_results:
        matched_combinations.append(combo)

    # 3. 持续时间评估
    duration_assessment = assess_duration(symptoms, clinical_checkpoint)
    if duration_hours is None and duration_assessment.get("duration_hours") is not None:
        duration_hours = duration_assessment["duration_hours"]

    # 4. 分级决策（优先级：紧急症状 > 危险组合 > 持续时间 > 严重程度 > 默认）
    level = "🟢"
    reason = "轻微症状，无危险信号，建议居家观察"

    if matched_emergency:
        level = "🔴"
        reason = f"检测到紧急症状：{'、'.join(matched_emergency)}，建议立即就医"
    elif matched_combinations:
        level = "🔴"
        risks = [c["risk"] for c in matched_combinations]
        reason = f"检测到危险症状组合：{'、'.join(risks)}，建议立即就医"
    elif severity in ("severe", "严重", "剧烈"):
        level = "🟡"
        reason = "症状严重程度较高，建议48小时内就诊"
    elif duration_hours is not None and duration_hours > DURATION_URGENT_THRESHOLD_HOURS:
        level = "🟡"
        reason = f"症状已持续{duration_hours:.0f}小时（超过{DURATION_URGENT_THRESHOLD_HOURS}小时），建议就诊"
    elif any(urg in " ".join(symptoms) for urg in URGENT_SYMPTOMS):
        level = "🟡"
        matched_urgent = [u for u in URGENT_SYMPTOMS if u in " ".join(symptoms)]
        reason = f"检测到需关注症状：{'、'.join(matched_urgent)}，建议48小时内就诊"

    return {
        "level": level,
        "level_name": LEVEL_MAP[level],
        "reason": reason,
        "matched_emergency": matched_emergency,
        "matched_combinations": matched_combinations,
        "duration_assessment": duration_assessment.get("summary", ""),
    }


def check_combination_risks(symptoms: List[str]) -> List[Dict]:
    """检查危险症状组合"""
    results = []
    symptom_set = set(symptoms)

    for combo in EMERGENCY_COMBINATIONS:
        required = combo["symptoms"]
        # 模糊匹配：症状字符串中包含关键词即算命中
        matched = set()
        for req in required:
            for sym in symptoms:
                if req in sym or sym in req:
                    matched.add(req)
                    break

        if matched >= required:  # 全部命中
            results.append({
                "matched_symptoms": list(required),
                "risk": combo["risk"],
                "level": combo["level"],
            })

    return results


def assess_duration(
    symptoms: List[str],
    clinical_checkpoint: Optional[Dict] = None,
) -> Dict:
    """基于持续时间评估风险

    从 clinical_checkpoint 中提取 onset_ts 计算持续时间
    """
    if not clinical_checkpoint:
        return {"duration_hours": None, "summary": "无持续时间数据"}

    onset_dates = clinical_checkpoint.get("symptom_onset_dates", {})
    if not isinstance(onset_dates, dict) or not onset_dates:
        return {"duration_hours": None, "summary": "无持续时间数据"}

    now = time.time()
    max_duration_hours = 0.0
    duration_details = []

    for sym_name, onset_ts in onset_dates.items():
        if isinstance(onset_ts, (int, float)):
            hours = (now - onset_ts) / 3600
            if hours > max_duration_hours:
                max_duration_hours = hours
            duration_details.append({"symptom": sym_name, "duration_hours": round(hours, 1)})

    if not duration_details:
        return {"duration_hours": None, "summary": "无持续时间数据"}

    if max_duration_hours > DURATION_URGENT_THRESHOLD_HOURS:
        summary = f"最长症状已持续{max_duration_hours:.0f}小时，超过{DURATION_URGENT_THRESHOLD_HOURS}小时阈值"
    else:
        summary = f"症状持续{max_duration_hours:.0f}小时，未超过就诊阈值"

    return {
        "duration_hours": round(max_duration_hours, 1),
        "summary": summary,
        "details": duration_details,
    }


def _suggest_departments(symptoms: List[str]) -> str:
    """根据症状推荐就诊科室"""
    departments = set()
    for sym in symptoms:
        for key, dept in DEPARTMENT_SUGGESTIONS.items():
            if key in sym or sym in key:
                departments.add(dept)
                break
    if not departments:
        return "内科"
    return "或".join(sorted(departments))


def generate_triage_advice(classification: Dict) -> str:
    """生成分诊建议文本

    格式：
    【分诊结果】🔴/🟡/🟢 等级
    【主要症状】...
    【持续时间】...
    【建议行动】...
    【观察信号】...
    """
    level = classification["level"]
    level_name = classification["level_name"]
    symptoms = classification.get("symptoms", [])
    duration_assessment = classification.get("duration_assessment", "未提供")
    reason = classification.get("reason", "")

    # 建议行动
    if level == "🔴":
        action = f"建议立即就医或拨打120，挂号急诊科。原因：{reason}"
    elif level == "🟡":
        dept = _suggest_departments(symptoms)
        action = f"建议48小时内就诊，挂号{dept}。原因：{reason}"
    else:
        action = "可居家观察，注意休息和饮水。如症状变化请重新评估"

    # 观察信号
    if level == "🔴":
        watch_signals = "如出现意识改变、症状急剧加重、用药无法缓解，请立即拨打120"
    elif level == "🟡":
        emergency_signals = []
        for combo in classification.get("matched_combinations", []):
            emergency_signals.append(combo["risk"])
        if not emergency_signals:
            emergency_signals = ["症状明显加重", "出现新发严重症状"]
        watch_signals = f"如出现{'、'.join(emergency_signals)}，请立即就医"
    else:
        watch_signals = "如出现发热加重、症状持续超过3天无缓解、或出现新的严重症状，建议就诊"

    symptoms_text = "、".join(symptoms) if symptoms else "未明确"

    lines = [
        f"【分诊结果】{level} {level_name}",
        f"【主要症状】{symptoms_text}",
        f"【持续时间】{duration_assessment}",
        f"【建议行动】{action}",
        f"【观察信号】{watch_signals}",
    ]
    return "\n".join(lines)


def run_symptom_triage(
    symptoms: List[str],
    severity: Optional[str] = None,
    duration_hours: Optional[float] = None,
    clinical_checkpoint: Optional[Dict] = None,
) -> Dict:
    """执行症状分诊

    Returns:
        {
            "status": "pass" | "revise",
            "triage_result": Dict,  # 分诊详细结果
            "advice_text": str,     # 格式化的分诊建议
            "risk_tags": List[str],
        }
    """
    risk_tags = []

    # 分级
    classification = classify_urgency(
        symptoms=symptoms,
        severity=severity,
        duration_hours=duration_hours,
        clinical_checkpoint=clinical_checkpoint,
    )
    classification["symptoms"] = symptoms

    # 风险标签
    if classification["matched_emergency"]:
        risk_tags.append("emergency_symptom")
    if classification["matched_combinations"]:
        risk_tags.append("dangerous_combination")
    if classification["duration_assessment"] and "超过" in classification["duration_assessment"]:
        risk_tags.append("duration_exceeded")

    # 生成建议文本
    advice_text = generate_triage_advice(classification)

    # 状态：🟢 为 pass，其余为 revise（需要人工关注或额外处理）
    status = "pass" if classification["level"] == "🟢" else "revise"

    return {
        "status": status,
        "triage_result": classification,
        "advice_text": advice_text,
        "risk_tags": risk_tags,
    }
