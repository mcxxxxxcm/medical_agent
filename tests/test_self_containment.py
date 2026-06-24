"""自包含性检测回归测试

用测试集验证 _has_anaphora_pattern 的检测效果，
确保修改不会引入回归。

用法：
    python tests/test_self_containment.py
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.graph.nodes.nodes import _has_anaphora_pattern


def run_test():
    test_file = Path(__file__).parent / "data" / "self_containment_test_set.jsonl"
    if not test_file.exists():
        print(f"测试集不存在：{test_file}")
        return

    cases = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    print(f"加载测试集：{len(cases)} 条\n")

    correct = 0
    total = 0
    false_negatives = []  # 漏检（应该检测到但没检测到）
    false_positives = []  # 误杀（不应该检测但被检测到）

    for case in cases:
        query = case["original_query"]
        is_self_contained = case.get("is_self_contained")

        # 跳过未标注的 case
        if is_self_contained is None:
            continue

        total += 1
        detected = _has_anaphora_pattern(query)

        # 不自包含的查询应该被检测到（detected=True）
        expected_detected = not is_self_contained

        if detected == expected_detected:
            correct += 1
            status = "PASS"
        else:
            status = "FAIL"
            if expected_detected and not detected:
                false_negatives.append(case)
            else:
                false_positives.append(case)

        print(f"  [{status}] query={query[:30]:30s} | "
              f"detected={detected} | expected={expected_detected} | "
              f"type={case.get('case_type', '')}")

    print(f"\n{'='*60}")
    print(f"总计：{total} 条 | 正确：{correct} 条 | 准确率：{correct/total*100:.1f}%")

    if false_negatives:
        print(f"\n漏检（{len(false_negatives)} 条）— 这些不自包含的查询未被检测到：")
        for case in false_negatives:
            print(f"  - {case['original_query']}")

    if false_positives:
        print(f"\n误杀（{len(false_positives)} 条）— 这些自包含的查询被错误检测为不自包含：")
        for case in false_positives:
            print(f"  - {case['original_query']}")

    return correct == total


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
