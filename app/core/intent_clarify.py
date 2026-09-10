"""意图澄清判定与槽位抽取（阶段0：收敛澄清逻辑）

__version__ = 9.57

背景
----
此前澄清判定散在 app/graph/nodes/nodes.py 的私有函数里，跟路由混在一起，
没有独立测试面。本模块把这些判据收敛成一个独立、可单测的判定接口，
并采纳 errorLog 方案的"槽位驱动澄清"思想，新增槽位抽取与场景化追问。

行为保真准则
------------
- 判定链的顺序与闸门 __must 与 nodes._decode_answerable 完全一致（见
  should_clarify 的分支顺序），不得擅自改序，否则会破坏 v9.48 修好的死角。
- 复用的既有谓词（药物碎片 / 具体临床主语 / 指代检测）均惰性 import nodes，
  保证只此一处实现、与节点行为同步，杜绝复制词表导致的两地发散。

新增能力（errorLog 的槽位驱动）
-------------------------------
- extract_slots：从 query 抽取药物名 / 症状部位 / 用户类型 / 持续时间 / 剂型量词
- missing_slots：按检索意图查缺（仿 REQUIRED_SLOTS），产出应追问的槽位
- build_clarify_question：针对缺失槽位生成具体追问（不是"请提供更多信息"）
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# ============================================================================
# 槽位词典（新增能力，独立于既有谓词）
# ============================================================================

_DRUG_NAMES = [
    "布洛芬", "对乙酰氨基酚", "阿莫西林", "头孢", "阿司匹林", "奥司他韦",
    "连花清瘟", "感冒灵", "板蓝根", "蒙脱石散", "氯雷他定", "西替利嗪",
    "红霉素", "甲硝唑", "二甲双胍", "格列美脲", "胰岛素", "阿卡波糖",
    "左甲状腺素", "氨氯地平", "卡托普利", "缬沙坦", "阿托伐他汀",
    "瑞舒伐他汀", "氯吡格雷", "华法林", "奥美拉唑",
]

_USER_TYPES = ["孕妇", "婴儿", "幼儿", "宝宝", "小儿", "哺乳", "哺乳期",
               "老年人", "老人", "老年", "儿童"]

_SYMPTOM_PART_KEYWORDS = [
    "头", "脑", "眼", "耳", "鼻", "喉", "颈", "胸", "心", "肺", "肝", "胆",
    "脾", "肾", "胃", "腹", "腰", "背", "肩", "臂", "手", "脚", "腿", "膝",
    "踝", "骨", "牙", "舌", "唇", "口", "皮肤", "肌肉", "神经", "关节",
]

_DURATION_KEYWORDS = [
    "小时", "天", "周", "星期", "个月", "月", "年", "多久", "持续",
    "几天", "几周", "几天了",
]

_MEASURE_WORDS = ["粒", "片", "颗", "包", "丸", "袋", "毫克", "克", "毫升", "ml"]

# 检查/检验类意图关键词（badcase 全链路意图分类用）
_EXAM_KEYWORDS = [
    "化验", "检查", "报告", "指标", "血常规", "尿常规", "大便常规",
    "肝功能", "肾功能", "血脂", "血糖", "血压",
    "ct", "核磁", "彩超", "超声", "b超", "胃镜", "肠镜",
    "报告单", "数值", "偏高", "偏低", "异常",
]

# 各检索意图的必要槽位（仿 errorLog REQUIRED_SLOTS，按医疗安全调权重）
_REQUIRED_SLOTS = {
    "symptom": ["symptom", "duration"],
    "drug": ["drug_name", "user_type"],
    "exam": ["exam_name", "abnormal_item"],
    "care": ["symptom"],
}

# ============================================================================
# 判定接口
# ============================================================================

def _dep():
    """惰性引入 nodes 既有谓词，避免模块加载时拖起整个 graph 栈。"""
    from app.graph.nodes import nodes as _n
    return _n


def _has_specific_health_topic(query: str) -> bool:
    return bool(_dep()._has_specific_health_topic(query))


def _is_drug_consumption_fragment(query: str) -> bool:
    return bool(_dep()._is_drug_consumption_fragment(query))


def _has_strong_anaphora(query: str) -> bool:
    return bool(_dep()._has_strong_anaphora(query))


def _has_anaphora_pattern(query: str) -> bool:
    return bool(_dep()._has_anaphora_pattern(query))


def _has_domain_entity(query: str) -> bool:
    text = (query or "").strip()
    return any(kw in text for kw in _dep()._DOMAIN_ENTITY_KEYWORDS)


def should_clarify(
    question: str,
    *,
    history_has_entity: bool = False,
) -> Tuple[bool, str]:
    """统一可答性判定（与 _decode_answerable 判定链顺序严格一致）。

    返回 (是否应澄清, 原因)。原因为空串表示放行检索。
    """
    q = (question or "").strip()
    if not q:
        return True, "空问题"

    # 1. 问句自带领域实体（药/症状/疾病）→ 可检索
    if _has_domain_entity(q):
        return False, ""

    # 2. 审物/过量事件报告（"吃了三粒怎么办"）→ 澄清：无具体药物主语
    if _is_drug_consumption_fragment(q) and not _has_specific_health_topic(q):
        return True, "审视事件缺药物主语"

    # 3. 上下文/临床快照可补出实体 → 可检索（合法用药咨询追问）
    if history_has_entity:
        return False, ""

    # 4. 强代词悬空（"这个药/它…")且无具体临床主语 → 澄清
    if _has_strong_anaphora(q) and not _has_specific_health_topic(q):
        return True, "强代词悬空"

    # 5. 细检测不自包含（短查询/疑问词开头无实体）且 >=3 字且无具体临床主语 → 澄清
    if (
        len(q) >= 3
        and _has_anaphora_pattern(q)
        and not _has_specific_health_topic(q)
    ):
        return True, "量词碎片/短查询无实体"

    return False, ""


# ============================================================================
# 槽位抽取（新增能力）
# ============================================================================

def extract_slots(question: str) -> Dict[str, Optional[str]]:
    """从 query 抽取关键槽位，返回 dict，未命中槽为 None。"""
    q = (question or "").strip()
    slots: Dict[str, Optional[str]] = {
        "drug_name": None,
        "symptom": None,
        "user_type": None,
        "duration": None,
        "measure": None,
    }
    if not q:
        return slots

    for drug in _DRUG_NAMES:
        if drug in q:
            slots["drug_name"] = drug
            break

    for ut in _USER_TYPES:
        if ut in q:
            slots["user_type"] = ut
            break

    for part in _SYMPTOM_PART_KEYWORDS:
        if f"{part}疼" in q or f"{part}痛" in q or f"{part}不舒服" in q:
            slots["symptom"] = f"{part}不适" if part in ("腹", "胃", "胸") else f"{part}疼"
            break
        if part in q:
            slots["symptom"] = part
            break

    for d in _DURATION_KEYWORDS:
        if d in q:
            slots["duration"] = d
            break

    for m in _MEASURE_WORDS:
        if m in q:
            slots["measure"] = m
            break

    return slots


def missing_slots(question: str, intent: str) -> List[str]:
    """按检索意图返回缺失的必要槽位（errorLog REQUIRED_SLOTS 思想）。

    intent ∈ {symptom, drug, exam, care}。返回缺失槽名列表，空表示可检索。
    """
    slots = extract_slots(question)
    required = _REQUIRED_SLOTS.get(intent, [])
    missing = [s for s in required if not slots.get(s)]
    return missing


def build_clarify_question(missing_slots: List[str]) -> str:
    """针对缺失槽位生成具体、易答的追问文案（不用空泛的"请补充信息"）。"""
    if not missing_slots:
        return ""
    prompts = [
        ("drug_name", "请问您具体服用/使用的是哪一种药？比如布洛芬、头孢或其他？"),
        ("user_type", "请问是您本人服用/出现症状，还是儿童、孕妇或老人等特殊人群？"),
        ("symptom", "请问您具体是哪里不舒服？比如头疼、腹痛、咳嗽还是其他部位？"),
        ("duration", "请问这个情况持续多久了？有没有逐渐加重？"),
        ("exam_name", "请问需要解读的是哪项检查或报告？比如血常规、肝功能还是其他？"),
        ("abnormal_item", "请问报告上具体是哪一项指标异常？数值大概是多少？"),
        ("measure", "请问一次服用的是多大剂量/几粒（片）呢？"),
    ]
    slot_prompts = [txt for (k, txt) in prompts if k in missing_slots]
    if not slot_prompts:
        return ""
    joined = "\n".join(f"• {p}" for p in slot_prompts)
    return (
        "为了给您更准确的建议，麻烦补充以下几点：\n"
        f"{joined}\n\n"
        "⚠️ 若症状明显加重或伴有高热/剧烈疼痛/呼吸困难，请及时就医。"
    )


# ============================================================================
# 全链路意图分类（badcase 管理：检索→草稿→列表→审核共用）
# ============================================================================

# 各意图必要槽位中的"关键槽位"：缺失即不可可靠检索/成稿。
# 相比 _REQUIRED_SLOTS，剔除 user_type/duration/abnormal_item 等安全限定或
# 非必需槽位，避免把"布洛芬怎么吃"这类自包含用药咨询误判为需澄清。
_CRITICAL_SLOTS = {
    "symptom": ["symptom"],
    "drug": ["drug_name"],
    "exam": ["exam_name"],
    "care": ["symptom"],
}

INTENT_LABELS = {
    "drug": "用药咨询",
    "exam": "检查/检验",
    "symptom": "症状咨询",
    "care": "日常护理",
    "knowledge": "知识问答",
    "general": "一般交流",
}

_SLOT_ZH = {
    "drug_name": "药物名",
    "user_type": "适用人群",
    "symptom": "症状/部位",
    "duration": "持续时间",
    "exam_name": "检查项目",
    "abnormal_item": "异常指标",
    "measure": "服用剂量",
}


def _clarify_hint(intent: str, missing: List[str], reason: str) -> str:
    if missing:
        names = "、".join(_SLOT_ZH.get(s, s) for s in missing)
        return (
            f"此问题属「{INTENT_LABELS.get(intent, intent)}」，缺少关键信息：{names}，"
            "线上会触发澄清追问。"
        )
    if reason:
        return f"此问题缺关键信息：{reason}。"
    return ""


def classify_intent(query: str) -> Dict[str, Any]:
    """badcase 全链路意图分类：判定查询意图 + 缺失关键槽位。

    返回 {"intent", "intent_label", "missing_slots", "clarify_needed", "hint"}。
    意图判定优先 drug > exam > symptom > knowledge > general，复用既有
    keyword_matcher 的 AC 自动机 matcher 与 nodes 的药物审视谓词，保证与
    线上路由/澄清判定同源，不被坏例后台单独维护一套词表。
    """
    empty = {
        "intent": "general",
        "intent_label": INTENT_LABELS["general"],
        "missing_slots": [],
        "clarify_needed": False,
        "hint": "",
    }
    q = (query or "").strip()
    if not q:
        return empty

    from app.core.keyword_matcher import (
        get_drug_matcher,
        get_route_knowledge_matcher,
        get_route_symptom_matcher,
    )

    # 1. 药：药物名词命中 或 审视行为碎片（"吃了三粒怎么办"）
    if get_drug_matcher().contains_any(q) or _is_drug_consumption_fragment(q):
        intent = "drug"
    # 2. 检查/检验
    # 例外：_EXAM_KEYWORDS 的"血压"会命中疾病名"高血压/低血压"（如"高血压怎么预防"），
    # 这类是病症问询而非检查诉求，剔除后交由下方 symptom 判定，避免误收窄到报告库。
    elif any(kw in q.replace("高血压", "").replace("低血压", "") for kw in _EXAM_KEYWORDS):
        intent = "exam"
    # 3. 症状
    elif get_route_symptom_matcher().contains_any(q, use_boundary=True):
        intent = "symptom"
    # 4. 知识
    elif get_route_knowledge_matcher().contains_any(q, use_boundary=True):
        intent = "knowledge"
    # 5. 兜底
    else:
        intent = "general"

    missing = missing_slots(q, intent)
    should_clar, clarify_reason = should_clarify(q)
    missing_critical = [s for s in _CRITICAL_SLOTS.get(intent, []) if s in missing]
    clarify_needed = bool(should_clar) or bool(missing_critical)
    hint = _clarify_hint(intent, missing, clarify_reason) if clarify_needed else ""
    return {
        "intent": intent,
        "intent_label": INTENT_LABELS.get(intent, "general"),
        "missing_slots": missing,
        "clarify_needed": clarify_needed,
        "hint": hint,
    }