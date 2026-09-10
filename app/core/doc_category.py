"""知识库逻辑分类（阶段一：单库内按 category 软过滤；阶段二：物理分库的预埋命名空间）

__version__ = 9.57

设计背景
--------
医疗知识库不能一股脑塞一个向量库：药品说明书、疾病诊疗、检查报告、护理指南、急救处置的
语义空间与权威源天然不同，混在一个 collection 里检索会互相污染，也无法按来源做合规隔离。
但直接物理分库会让"意图/类型判错 → 零召回"的成本从"排序问题"升级成"干脆查不到"，且需
维护 N 套独立索引。因此分两阶段：

- 阶段一（本模块落地）：单库内给文档打 `category` 标签，检索时按意图做**软过滤**，过滤后
  候选不足则**回退全库**——先试出哪条 意图→类型 映射可靠、哪些标签可靠。
- 阶段二（硬性政策，见 README/CHANGELOG）：**文档数达到 1000+ 篇 或 来源>3 类时必须物理
  分库**。本模块已按 category 预埋"未来物理库名"（disease_kb/drug_kb/...），届时直接按
  collection 拆分、检索路由层零改动。

分类策略
--------
- `classify_doc_category(filename)`：纯文件名关键词判别，确定性、零 LLM 调用（摄入时逐文件
  成本可忽略）。单一主类 + 兜底：跨域文档（如「高血压管理与用药指南」既含"用药"又含"管理"）
  命中多个关键词时取高优先级主类，跨域召回由单库兜底兜住，阶段一不追求多标签。
- `intent_to_categories(intent)`：把 `app.core.intent_clarify.classify_intent` 的输出意图映射
  到 category；`knowledge/general` 返回 None = 不过滤（全库检索）。
"""
from __future__ import annotations

from typing import Dict, List, Optional

# category 及其未来物理库名（阶段二拆 collection 时直接用 value 作为 collection 名）
DOC_CATEGORIES: Dict[str, str] = {
    "disease": "疾病诊疗",
    "drug": "药品说明",
    "report": "检查报告",
    "nurse": "护理指南",
    "emergency": "急救处置",
    "guide": "就医引导",
    "general": "通用",
}

# category → 阶段二物理 collection 名（预埋，物理分库时用）
PHYSICAL_COLLECTION_NAMES: Dict[str, str] = {
    "disease": "disease_kb",
    "drug": "drug_kb",
    "report": "report_kb",
    "nurse": "nurse_kb",
    "emergency": "emergency_kb",
    "guide": "guide_kb",
    "general": "general_kb",
}


def _contains_any(stem: str, keywords: List[str]) -> bool:
    return any(kw in stem for kw in keywords)


def classify_doc_category(filename: str) -> str:
    """按文件名给文档归类（确定性、零 LLM）。

    优先级：emergency > report > drug > nurse > disease > guide > general。
    跨域文档取高优先级主类；未命中任何关键词回退 general。
    """
    stem = (filename or "")
    # 去掉扩展名（兼容 .md/.txt/.pdf/.xlsx/.webdoc.pdf）
    lower = stem.lower()
    for tail in (".webdoc.pdf", ".pdf", ".docx", ".xlsx", ".xls", ".md", ".txt"):
        if lower.endswith(tail):
            stem = stem[: -len(tail)]
            break

    # 急救：最具体，优先
    if _contains_any(stem, [
        "急救", "急性心肌梗死", "急性脑卒中", "卒中", "窒息", "气道异物",
        "过敏性休克", "外伤", "心搏骤停", "气道",
    ]):
        return "emergency"
    # 检查报告
    if _contains_any(stem, ["报告解读", "血常规", "尿常规", "肝功能", "肾功能", "生化", "血糖血脂"]):
        return "report"
    # 药品
    if _contains_any(stem, ["用药", "药物", "药", "抗生素", "退热镇痛", "禁忌"]):
        return "drug"
    # 护理
    if _contains_any(stem, ["护理", "日常管理", "饮食", "家庭护理", "康复"]):
        return "nurse"
    # 疾病/症状/诊疗（刻意不含"指南"——几乎每份文档都叫"XX指南"，会误吞挂号/护理/急诊等引导类）
    if _contains_any(stem, ["疾病", "诊疗", "诊断", "症状", "发热", "急诊"]):
        return "disease"
    # 就医引导
    if _contains_any(stem, ["就医", "挂号", "科室", "医保", "转诊", "报销"]):
        return "guide"
    return "general"


def category_zh(category: str) -> str:
    """category → 中文标签，未知回退 general"""
    return DOC_CATEGORIES.get(category or "", DOC_CATEGORIES["general"])


def intent_to_categories(intent: str) -> Optional[List[str]]:
    """检索意图 → 应软过滤的 category 列表。

    - drug → 只查药品库；exam → 只查检查报告库；symptom → 只查疾病库
    - knowledge/general/未知 → None（不过滤，全库检索，避免知识类问题被收窄）
    """
    mapping = {
        "drug": ["drug"],
        "exam": ["report"],
        "symptom": ["disease"],
    }
    return mapping.get(intent)  # 返回 list 或 None