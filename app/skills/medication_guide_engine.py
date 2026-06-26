"""用药指导规则引擎

在 LLM 生成用药相关回答后执行规则校验：
1. 药物实体识别（AC 自动机）
2. 禁忌人群交叉检查
3. 用量安全范围检查
4. 药物相互作用初筛
5. 规范性校验（5 字段完整性）
"""
import re
from typing import Dict, Any, List, Optional


# 药物安全知识库（内置关键规则，非完整药品数据库）
# 格式：药物名 → {禁忌人群, 每日上限, 常见相互作用}
DRUG_SAFETY_RULES = {
    "布洛芬": {
        "contraindications": ["孕妇", "消化道溃疡", "严重肝肾功能不全", "阿司匹林哮喘"],
        "max_daily_dosage": "1200mg",
        "interactions": ["阿司匹林", "华法林", "甲氨蝶呤", "地塞米松"],
        "pediatric_note": "6月以上儿童可使用，但需按体重计算剂量，请遵医嘱",
    },
    "对乙酰氨基酚": {
        "contraindications": ["严重肝功能不全"],
        "max_daily_dosage": "2000mg（成人）",
        "interactions": ["华法林", "卡马西平", "苯妥英钠"],
        "pediatric_note": "3月以上儿童可使用，需按体重计算剂量，请遵医嘱",
    },
    "阿莫西林": {
        "contraindications": ["青霉素过敏", "传染性单核细胞增多症"],
        "max_daily_dosage": "4g（成人）",
        "interactions": ["甲氨蝶呤", "华法林", "别嘌醇"],
        "pediatric_note": "儿童可用，需按体重计算剂量，请遵医嘱",
    },
    "阿司匹林": {
        "contraindications": ["儿童（Reye综合征风险）", "孕妇晚期", "消化道溃疡", "出血性疾病"],
        "max_daily_dosage": "4g（成人抗炎剂量）",
        "interactions": ["布洛芬", "华法林", "甲氨蝶呤", "地塞米松"],
        "pediatric_note": "16岁以下儿童禁用（Reye综合征风险）",
    },
}

# 禁忌人群标记（从 clinical_checkpoint / user_profile 中检测）
CONTRAINDICATION_KEYWORDS = {
    "孕妇": ["怀孕", "孕妇", "妊娠", "孕期", "孕"],
    "儿童": ["儿童", "小儿", "婴儿", "幼儿", "宝宝", "岁"],
    "老年": ["老年", "高龄", "老人"],
    "肝功能不全": ["肝功能不全", "肝病", "肝硬化", "肝炎"],
    "肾功能不全": ["肾功能不全", "肾病", "肾衰", "透析"],
    "消化道溃疡": ["胃溃疡", "十二指肠溃疡", "消化道溃疡", "胃出血"],
}

# 用药建议 5 字段
REQUIRED_FIELDS = ["适应症", "用法用量", "注意事项", "禁忌", "如症状持续请就医"]

# 免责声明
MEDICATION_DISCLAIMER = "⚠️ 以上用药建议仅供参考，不可替代专业医师或药师的指导。如有疑问，请咨询医生或药师。"


class DrugACAutomaton:
    """基于 AC 自动机的药物实体识别器

    对 DRUG_SAFETY_RULES 中的药物名构建 AC 自动机，
    实现多模式串同时匹配。
    """

    def __init__(self, keywords: List[str]):
        self.keywords = keywords
        self._goto: Dict[int, Dict[str, int]] = {0: {}}
        self._fail: Dict[int, int] = {0: 0}
        self._output: Dict[int, List[str]] = {}
        self._build()

    def _build(self):
        """构建 goto + fail + output"""
        # 构建 goto 函数
        for keyword in self.keywords:
            state = 0
            for ch in keyword:
                if ch not in self._goto[state]:
                    new_state = len(self._goto)
                    self._goto[state][ch] = new_state
                    self._goto[new_state] = {}
                    state = new_state
                else:
                    state = self._goto[state][ch]
            self._output.setdefault(state, []).append(keyword)

        # 构建 fail 函数（BFS）
        from collections import deque
        queue = deque()
        for ch, s in self._goto[0].items():
            self._fail[s] = 0
            queue.append(s)

        while queue:
            r = queue.popleft()
            for ch, s in self._goto[r].items():
                queue.append(s)
                f = self._fail[r]
                while f and ch not in self._goto.get(f, {}):
                    f = self._fail[f]
                self._fail[s] = self._goto.get(f, {}).get(ch, 0)
                if self._fail[s] == s:
                    self._fail[s] = 0
                self._output.setdefault(s, []).extend(self._output.get(self._fail[s], []))

    def search(self, text: str) -> List[str]:
        """在文本中搜索所有匹配的药物名（去重）"""
        state = 0
        found = []
        for ch in text:
            while state and ch not in self._goto.get(state, {}):
                state = self._fail[state]
            state = self._goto.get(state, {}).get(ch, 0)
            if state in self._output:
                found.extend(self._output[state])
        return list(dict.fromkeys(found))  # 去重保序


# 模块级单例
_drug_automaton: Optional[DrugACAutomaton] = None


def _get_drug_automaton() -> DrugACAutomaton:
    """获取药物 AC 自动机单例"""
    global _drug_automaton
    if _drug_automaton is None:
        _drug_automaton = DrugACAutomaton(list(DRUG_SAFETY_RULES.keys()))
    return _drug_automaton


def detect_drugs_in_answer(answer: str) -> List[str]:
    """使用 AC 自动机检测回答中的药物名

    Args:
        answer: 待检测的文本

    Returns:
        匹配到的药物名列表（去重）
    """
    automaton = _get_drug_automaton()
    matched = automaton.search(answer)

    # 正则兜底：匹配"XX片"、"XX胶囊"、"XX颗粒"等剂型后缀
    suffix_pattern = re.compile(
        r"([\u4e00-\u9fa5]{2,6})(片|胶囊|颗粒|口服液|混悬液|滴剂|糖浆|散剂|丸|膏|贴|栓|注射液|注射用)"
    )
    for m in suffix_pattern.finditer(answer):
        drug_name = m.group(1)
        if drug_name in DRUG_SAFETY_RULES and drug_name not in matched:
            matched.append(drug_name)

    return matched


def _detect_user_conditions(
    clinical_checkpoint: Optional[Dict] = None,
    user_profile: Optional[Dict] = None,
) -> List[str]:
    """从用户档案/临床快照中检测特殊人群标记

    Returns:
        匹配到的禁忌人群类别列表
    """
    detected = []
    texts_to_check = []

    if clinical_checkpoint:
        symptoms = clinical_checkpoint.get("symptoms", [])
        if isinstance(symptoms, list):
            texts_to_check.extend(str(s) for s in symptoms)
        chief = clinical_checkpoint.get("chief_complaint", "")
        if chief:
            texts_to_check.append(str(chief))

    if user_profile:
        for field in ["age_group", "special_conditions", "medical_history", "notes"]:
            val = user_profile.get(field, "")
            if val:
                if isinstance(val, list):
                    texts_to_check.extend(str(v) for v in val)
                else:
                    texts_to_check.append(str(val))

    combined_text = " ".join(texts_to_check)

    for category, keywords in CONTRAINDICATION_KEYWORDS.items():
        for kw in keywords:
            if kw in combined_text:
                if category not in detected:
                    detected.append(category)
                break

    return detected


def check_contraindications(
    drug: str,
    clinical_checkpoint: Optional[Dict] = None,
    user_profile: Optional[Dict] = None,
) -> Dict[str, Any]:
    """检查药物禁忌与用户人群的交叉

    Args:
        drug: 药物名称
        clinical_checkpoint: 临床快照
        user_profile: 用户档案

    Returns:
        {
            "drug": str,
            "has_contraindication": bool,
            "matched_contraindications": [...],
            "user_conditions": [...],
        }
    """
    drug_info = DRUG_SAFETY_RULES.get(drug)
    if not drug_info:
        return {
            "drug": drug,
            "has_contraindication": False,
            "matched_contraindications": [],
            "user_conditions": [],
        }

    user_conditions = _detect_user_conditions(clinical_checkpoint, user_profile)
    contraindications = drug_info.get("contraindications", [])

    matched = []
    for ci in contraindications:
        for uc in user_conditions:
            # 精确包含或语义匹配
            if uc in ci or ci in uc:
                matched.append(ci)
                break

    return {
        "drug": drug,
        "has_contraindication": bool(matched),
        "matched_contraindications": matched,
        "user_conditions": user_conditions,
    }


def check_drug_interactions(drugs: List[str]) -> Dict[str, Any]:
    """检查多药物相互作用

    Args:
        drugs: 药物名称列表

    Returns:
        {
            "has_interaction": bool,
            "interactions": [{"drug_a": ..., "drug_b": ..., "note": ...}, ...],
        }
    """
    interactions = []
    for i, drug_a in enumerate(drugs):
        info_a = DRUG_SAFETY_RULES.get(drug_a)
        if not info_a:
            continue
        for drug_b in drugs[i + 1:]:
            if drug_b in info_a.get("interactions", []):
                interactions.append({
                    "drug_a": drug_a,
                    "drug_b": drug_b,
                    "note": f"{drug_a} 与 {drug_b} 存在已知相互作用，建议避免联用或在医生指导下使用",
                })
            info_b = DRUG_SAFETY_RULES.get(drug_b)
            if info_b and drug_a in info_b.get("interactions", []):
                # 避免重复记录
                already = any(
                    ia["drug_a"] == drug_b and ia["drug_b"] == drug_a
                    for ia in interactions
                )
                if not already:
                    interactions.append({
                        "drug_a": drug_b,
                        "drug_b": drug_a,
                        "note": f"{drug_b} 与 {drug_a} 存在已知相互作用，建议避免联用或在医生指导下使用",
                    })

    return {
        "has_interaction": bool(interactions),
        "interactions": interactions,
    }


def check_dosage_safety(drug: str, answer: str) -> Dict[str, Any]:
    """检查回答中的剂量是否超过安全上限

    Args:
        drug: 药物名称
        answer: 回答文本

    Returns:
        {
            "drug": str,
            "max_daily_dosage": str,
            "exceeds_limit": bool,
            "extracted_dosages": [...],
        }
    """
    drug_info = DRUG_SAFETY_RULES.get(drug)
    if not drug_info:
        return {
            "drug": drug,
            "max_daily_dosage": "未知",
            "exceeds_limit": False,
            "extracted_dosages": [],
        }

    max_daily = drug_info["max_daily_dosage"]
    # 提取数字部分作为上限（mg）
    max_match = re.search(r"(\d+)", max_daily)
    if not max_match:
        return {
            "drug": drug,
            "max_daily_dosage": max_daily,
            "exceeds_limit": False,
            "extracted_dosages": [],
        }
    max_val = int(max_match.group(1))

    # 从回答中提取剂量数字（mg）
    dosage_pattern = re.compile(r"(\d+)\s*mg")
    extracted = []
    for m in dosage_pattern.finditer(answer):
        val = int(m.group(1))
        extracted.append({"value": val, "text": m.group(0)})

    # 检查是否包含"每日"相关的超量描述
    exceeds = False
    daily_pattern = re.compile(r"每日.*?(\d+)\s*mg|每天.*?(\d+)\s*mg|一天.*?(\d+)\s*mg")
    for m in daily_pattern.finditer(answer):
        for g in m.groups():
            if g and int(g) > max_val:
                exceeds = True

    # 简单判断：单个数值超过上限也视为可能超量
    for d in extracted:
        if d["value"] > max_val:
            exceeds = True

    return {
        "drug": drug,
        "max_daily_dosage": max_daily,
        "exceeds_limit": exceeds,
        "extracted_dosages": extracted,
    }


def check_output_completeness(answer: str, drug: str) -> Dict[str, Any]:
    """检查用药建议的 5 字段完整性

    Args:
        answer: 回答文本
        drug: 药物名称

    Returns:
        {
            "is_complete": bool,
            "missing_fields": [...],
            "present_fields": [...],
        }
    """
    missing = []
    present = []
    for field in REQUIRED_FIELDS:
        if field in answer:
            present.append(field)
        else:
            missing.append(field)

    return {
        "is_complete": len(missing) == 0,
        "missing_fields": missing,
        "present_fields": present,
    }


def _inject_missing_fields(answer: str, drug: str, missing: List[str]) -> str:
    """为缺失字段追加模板提示"""
    if not missing:
        return answer

    drug_info = DRUG_SAFETY_RULES.get(drug)
    supplements = []

    field_templates = {
        "适应症": f"【适应症】请查阅药品说明书或咨询药师了解{drug}的适应症。",
        "用法用量": f"【用法用量】请遵医嘱或按药品说明书服用。",
        "注意事项": f"【注意事项】服用{drug}期间请关注不良反应，如有不适应及时就医。",
        "禁忌": f"【禁忌】该药物禁忌信息未在知识库中完整覆盖，建议查阅药品说明书。",
        "如症状持续请就医": "如症状持续请就医。",
    }

    for field in missing:
        supplements.append(field_templates.get(field, f"【{field}】请查阅药品说明书。"))

    return answer.rstrip() + "\n\n" + "\n".join(supplements)


def _inject_interaction_warning(answer: str, interactions: List[Dict]) -> str:
    """追加药物相互作用警告"""
    if not interactions:
        return answer
    lines = ["\n\n⚠️ 药物相互作用提醒："]
    for ia in interactions:
        lines.append(f"  - {ia['note']}")
    return answer.rstrip() + "\n".join(lines)


def _inject_contraindication_warning(answer: str, matched: List[str], drug: str) -> str:
    """追加禁忌人群警告"""
    if not matched:
        return answer
    warning = (
        f"\n\n⚠️ 禁忌提醒：{drug} 禁用于{'、'.join(matched)}人群。"
        "如属于上述人群，请勿自行用药，务必咨询医生。"
    )
    return answer.rstrip() + warning


def _inject_pediatric_note(answer: str, drug: str) -> str:
    """追加儿童用药提示"""
    drug_info = DRUG_SAFETY_RULES.get(drug)
    if not drug_info:
        return answer
    pediatric_note = drug_info.get("pediatric_note")
    if not pediatric_note:
        return answer
    # 仅在回答中提及儿童相关内容时追加
    child_keywords = ["儿童", "小儿", "婴儿", "幼儿", "宝宝"]
    if any(kw in answer for kw in child_keywords):
        return answer.rstrip() + f"\n\n👶 儿童用药提示：{pediatric_note}"
    return answer


def _inject_unknown_drug_warning(answer: str, drug: str) -> str:
    """对未收录药物追加提示"""
    return answer.rstrip() + f"\n\n⚠️ 暂未收录该药品（{drug}）信息，请查阅药品说明书或咨询药师。"


def _inject_disclaimer(answer: str) -> str:
    """追加免责声明"""
    return answer.rstrip() + "\n\n" + MEDICATION_DISCLAIMER


def run_medication_guide_review(
    answer: str,
    clinical_checkpoint: Optional[Dict] = None,
    user_profile: Optional[Dict] = None,
) -> Dict[str, Any]:
    """执行用药指导规则引擎审查

    Args:
        answer: LLM 生成的回答
        clinical_checkpoint: 临床快照（包含症状等信息）
        user_profile: 用户档案（包含年龄、特殊状态等）

    Returns:
        {
            "status": "pass" | "revise",
            "revised_answer": str,
            "risk_tags": List[str],
            "details": {...},
        }
    """
    risk_tags = []
    revisions_needed = []

    # 1. 药物实体识别
    drugs = detect_drugs_in_answer(answer)
    if not drugs:
        # 没有检测到药物名，无需用药审查
        return {
            "status": "pass",
            "revised_answer": answer,
            "risk_tags": [],
            "details": {"drugs_detected": []},
        }

    # 记录未收录药物
    known_drugs = [d for d in drugs if d in DRUG_SAFETY_RULES]
    unknown_drugs = [d for d in drugs if d not in DRUG_SAFETY_RULES]

    # 2. 禁忌人群交叉检查
    contraindication_results = {}
    has_contraindication = False
    for drug in known_drugs:
        result = check_contraindications(drug, clinical_checkpoint, user_profile)
        contraindication_results[drug] = result
        if result["has_contraindication"]:
            has_contraindication = True
            risk_tags.append(f"contraindication:{drug}")
            revisions_needed.append(("contraindication", drug, result["matched_contraindications"]))

    # 3. 药物相互作用检测
    interaction_result = check_drug_interactions(known_drugs)
    if interaction_result["has_interaction"]:
        risk_tags.append("drug_interaction")
        revisions_needed.append(("interaction", None, interaction_result["interactions"]))

    # 4. 用量安全范围检查
    dosage_results = {}
    for drug in known_drugs:
        result = check_dosage_safety(drug, answer)
        dosage_results[drug] = result
        if result["exceeds_limit"]:
            risk_tags.append(f"dosage_exceeded:{drug}")
            revisions_needed.append(("dosage", drug, None))

    # 5. 输出规范性检查（对每种药物逐一检查）
    completeness_results = {}
    all_missing_fields = set()
    for drug in known_drugs:
        result = check_output_completeness(answer, drug)
        completeness_results[drug] = result
        all_missing_fields.update(result["missing_fields"])
    if all_missing_fields:
        risk_tags.append("incomplete_fields")
        revisions_needed.append(("completeness", None, all_missing_fields))

    # 未收录药物风险
    if unknown_drugs:
        risk_tags.append("unknown_drug")
        revisions_needed.append(("unknown_drug", None, unknown_drugs))

    # 执行修订
    revised_answer = answer

    # 追加禁忌警告
    for rev_type, drug, data in revisions_needed:
        if rev_type == "contraindication" and drug:
            revised_answer = _inject_contraindication_warning(revised_answer, data, drug)
        elif rev_type == "interaction":
            revised_answer = _inject_interaction_warning(revised_answer, data)
        elif rev_type == "dosage" and drug:
            dosage_info = DRUG_SAFETY_RULES.get(drug, {})
            max_daily = dosage_info.get("max_daily_dosage", "未知")
            revised_answer = revised_answer.rstrip() + (
                f"\n\n⚠️ 剂量安全提醒：{drug} 成人每日上限为 {max_daily}，请勿超量服用。"
            )
        elif rev_type == "completeness" and data:
            # 对每个已知药物补充缺失字段
            for d in known_drugs:
                comp = completeness_results.get(d, {})
                if comp.get("missing_fields"):
                    revised_answer = _inject_missing_fields(revised_answer, d, comp["missing_fields"])
        elif rev_type == "unknown_drug" and data:
            for ud in data:
                revised_answer = _inject_unknown_drug_warning(revised_answer, ud)

    # 儿童用药提示
    for drug in known_drugs:
        revised_answer = _inject_pediatric_note(revised_answer, drug)

    # 追加免责声明
    revised_answer = _inject_disclaimer(revised_answer)

    # 决策
    status = "revise" if revisions_needed else "pass"

    return {
        "status": status,
        "revised_answer": revised_answer if status == "revise" else answer,
        "risk_tags": risk_tags,
        "details": {
            "drugs_detected": drugs,
            "known_drugs": known_drugs,
            "unknown_drugs": unknown_drugs,
            "contraindication_results": contraindication_results,
            "interaction_result": interaction_result,
            "dosage_results": dosage_results,
            "completeness_results": completeness_results,
        },
    }
