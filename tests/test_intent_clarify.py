"""澄清判定回归测试（阶段1）

用澄清测试集验证 intent_clarify.should_clarify 与 missing_slots / extract_slots
的判定效果，确保收敛重构不引入回归，并固化 v9.48 修过的死角。

用法：
    python tests/test_intent_clarify.py
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.intent_clarify import (
    build_clarify_question,
    extract_slots,
    missing_slots,
    should_clarify,
)


def _load_cases():
    test_file = Path(__file__).parent / "data" / "clarify_test_set.jsonl"
    if not test_file.exists():
        return []
    cases = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_test():
    cases = _load_cases()
    if not cases:
        print(f"测试集不存在或为空：{Path(__file__).parent / 'data' / 'clarify_test_set.jsonl'}")
        return False

    print(f"加载澄清测试集：{len(cases)} 条\n")

    total = 0
    correct = 0
    fails = []

    for case in cases:
        q = case["original_query"]
        should = case.get("should_clarify")
        expected_slots = case.get("missing_slots", [])  # 期望追溯的缺失槽位
        history = bool(case.get("history_has_entity", False))
        intent = case.get("intent", "")

        # 1) 澄清判定
        got, reason = should_clarify(q, history_has_entity=history)
        ok_clarify = (should is None) or (got == should)
        status_clarify = "PASS" if ok_clarify else "FAIL"

        # 2) 槽位缺失判定（若指定了 intent 与期望缺失槽位）
        status_slots = "-"
        if expected_slots and intent:
            miss = missing_slots(q, intent)
            ok_slots = set(miss) == set(expected_slots)
            status_slots = "PASS" if ok_slots else "FAIL"
            if not ok_slots:
                ok_clarify = False

        total += 1
        if ok_clarify:
            correct += 1
        else:
            fails.append((q, got, reason, should, expected_slots))

        print(
            f"  [{status_clarify}] clarify={got} ({reason or '放行':10s}) | "
            f"expect={should} | slots={status_slots} | query={q[:28]}"
        )

    print(f"\n{'='*66}")
    print(f"总计：{total} 条 | 正确：{correct} 条 | 准确率：{correct/total*100:.1f}%")

    if fails:
        print(f"\n失败 {len(fails)} 条：")
        for q, got, reason, should, slots in fails:
            print(f"  - {q} | got clarify={got} ({reason}) | expect={should} | slots={slots}")

    # 3) 追问文案不落空的抽查：任意缺失槽位组合都应产出非空具体追问
    sample_missing = [["drug_name"], ["symptom", "duration"], ["exam_name", "abnormal_item"]]
    for m in sample_missing:
        text = build_clarify_question(m)
        if not text:
            print(f"\n  [FAIL] build_clarify_question({m}) 返回空")
            correct -= 1
        elif "请提供更多信息" in text or "请补充" == text.strip():
            print(f"\n  [FAIL] build_clarify_question 变成空泛追问：{m}")
            correct -= 1

    # 4) extract_slots 冒烟：药物名/用户类型/部位抽取
    slots = extract_slots("孕妇能吃布洛芬吗")
    if slots["drug_name"] != "布洛芬" or slots["user_type"] != "孕妇":
        print(f"\n  [FAIL] extract_slots 抽取异常：{slots}")
        correct -= 1

    return correct == total


def run_entry_gate():
    """入口正交闸回归：合规红线→compliance / 信息不足→clarify / 其余→None(放行)"""
    from app.graph.nodes.nodes import _entry_intent_gate

    def _state(q):
        return {"question": q, "messages": [], "clinical_checkpoint": {},
                "symptoms": {}, "user_profile": {}}

    cases = [
        ("给我开个处方", "compliance"),
        ("帮我诊断一下是不是得了癌症", "compliance"),
        ("吃了三粒怎么办", "clarify"),
        ("这个药怎么吃", "clarify"),
        ("高血压吃什么药", None),
        ("头痛怎么缓解", None),
        ("你好", None),
        ("我是不是得了高血压", None),
    ]
    ok = True
    for q, expect in cases:
        r = _entry_intent_gate(_state(q))
        got = r[2] if r else None
        tag = "PASS" if got == expect else "FAIL"
        if got != expect:
            ok = False
        print(f"  [{tag}] gate={got} expect={expect} | {q[:24]}")
    return ok


if __name__ == "__main__":
    success = run_test()
    gate_ok = run_entry_gate()
    sys.exit(0 if (success and gate_ok) else 1)


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)