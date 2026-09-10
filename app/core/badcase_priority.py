"""Bad Case 医学敏感性分级（L2 自动分级）
__version__ = 9.54

给待审 bad case 判一个"人工审阅优先级"：
    - high  ：医学高风险（剂量/特殊人群/急救/用药等），强制 100% 人工终审
    - normal：相对低风险，可参与抽样复核

判据 = ① 规则词表命中（问题或原答案含高风险词）② 已标 confidence=low。
选词刻意偏向"答错可能给危险建议"的场景：剂量、儿童/孕妇等特殊人群、急救、
中毒/过敏休克、具体药物。泛泛的"怎么护理/注意什么"不升级到 high。

全部为纯字符串规则，无外部依赖；失败不影响 badcase 后台展示。
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# 医学高风险词表（命中任一即 high）。按类别分组便于追溯命中原因。
HIGH_PRIORITY_TOKENS: List[str] = [
    # 剂量与用量
    "剂量", "用量", "吃多少", "吃几", "每次", "每日", "mg", "克", "毫克", "毫升", "ml", "片", "粒",
    # 特殊人群
    "儿童", "婴儿", "幼儿", "宝宝", "小儿", "孕妇", "怀孕", "哺乳", "老年", "老人", "岁", "月龄",
    # 急救 / 危险信号
    "急救", "中毒", "过敏", "休克", "晕倒", "昏迷", "抽搐", "急诊", "120", "危及", "呼吸困难",
    "窒息", "大出血", "出血", "心跳", "主动脉", "脑梗", "中风", "胸痛", "急症",
    # 用药 / 药物
    "药", "服用", "用药", "药物", "消炎", "抗生素", "退烧", "退热", "止痛", "阿司匹林",
    "布洛芬", "对乙酰氨基酚", "激素", "胰岛素", "二甲双胍", "硝苯", "胶囊", "注射液",
]

def _normalize(text: str) -> str:
    return (text or "").replace(" ", "").replace("\n", "").lower()


def priority_reason(case: Dict) -> str:
    """返回命中高风险类别的中文说明（供前端展示），未命中返回空串。"""
    q = _normalize(case.get("original_query") or case.get("final_question") or "")
    ans = _normalize(case.get("answer_preview") or "")
    text = q + ans
    for tok in HIGH_PRIORITY_TOKENS:
        if tok.lower() in text:
            return f"命中高风险词：{tok}"
    return ""


def _confidence_low(case: Dict) -> bool:
    md = case.get("metadata") or {}
    try:
        return str(md.get("confidence") or "").lower() == "low"
    except (AttributeError, TypeError):
        return False


def classify_priority(case: Dict) -> Tuple[str, str]:
    """返回 (priority, reason)。priority ∈ {high, normal}。"""
    reason = priority_reason(case)
    if reason:
        return "high", reason
    if _confidence_low(case):
        return "high", "自动标注置信度 low，需人工重点审"
    return "normal", ""


def apply_priorities(cases: List[Dict]) -> List[Dict]:
    """批量把 priority / _priority_reason 合入 case 副本，返回新列表。"""
    out: List[Dict] = []
    for c in cases:
        copy = dict(c)
        p, reason = classify_priority(copy)
        copy["priority"] = p
        copy["_priority_reason"] = reason
        out.append(copy)
    return out


def count_pending_by_priority(cases: List[Dict]) -> Dict[str, int]:
    """统计未审核 bad case 中高优先 / 常规各有多少。入参应为『全量待审』case 列表。"""
    high = sum(1 for c in cases if classify_priority(c)[0] == "high")
    return {"high": high, "normal": len(cases) - high}