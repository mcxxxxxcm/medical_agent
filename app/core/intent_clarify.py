"""意图澄清判定与槽位抽取（阶段0：收敛澄清逻辑）

__version__ = 9.55

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

from typing import Dict, List, Optional, Tuple

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