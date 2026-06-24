"""Bad Case 导出脚本

将 PostgresStore 中自动采集的 bad case 导出为 JSONL 测试集文件，
供人工审核和回归测试使用。

用法：
    python scripts/export_bad_cases.py [--output OUTPUT] [--case-type TYPE] [--unreviewed-only]

示例：
    # 导出所有未审核的 bad case
    python scripts/export_bad_cases.py --unreviewed-only

    # 导出指代词类 bad case
    python scripts/export_bad_cases.py --case-type rewrite_missed_anaphora

    # 指定输出路径
    python scripts/export_bad_cases.py --output tests/data/my_test_set.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def export_bad_cases(
        output: str = "tests/data/bad_cases_export.jsonl",
        case_type: str = None,
        unreviewed_only: bool = False,
        user_id: str = "system",
):
    """从 PostgresStore 导出 bad case 到 JSONL 文件"""
    from app.memory import get_long_term_memory

    memory = get_long_term_memory()
    cases = memory.get_bad_cases(
        user_id=user_id,
        case_type=case_type,
        reviewed=False if unreviewed_only else None,
        limit=500,
    )

    if not cases:
        print("没有找到符合条件的 bad case")
        return

    output_path = PROJECT_ROOT / output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for case in cases:
            # 转换为测试集格式
            test_entry = {
                "id": case.get("case_id", ""),
                "case_type": case.get("case_type", ""),
                "original_query": case.get("original_query", ""),
                "history_summary": case.get("history_summary", ""),
                "rewritten_query": case.get("rewritten_query", ""),
                "final_question": case.get("final_question", ""),
                "expected_rewrite": case.get("expected_rewrite", ""),
                "is_self_contained": case.get("is_self_contained"),
                "top_doc_score": case.get("top_doc_score", 0),
                "grade_result": case.get("grade_result", ""),
                "answer_preview": case.get("answer_preview", ""),
                "reviewed": case.get("reviewed", False),
                "created_at": case.get("created_at", ""),
                "metadata": case.get("metadata", {}),
            }
            f.write(json.dumps(test_entry, ensure_ascii=False) + "\n")

    print(f"已导出 {len(cases)} 条 bad case → {output_path}")

    # 统计
    type_counts = {}
    for case in cases:
        t = case.get("case_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    print("\n类型分布：")
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")

    reviewed_count = sum(1 for c in cases if c.get("reviewed"))
    unreviewed_count = len(cases) - reviewed_count
    print(f"\n审核状态：已审核 {reviewed_count} / 未审核 {unreviewed_count}")


def main():
    parser = argparse.ArgumentParser(description="导出 bad case 为 JSONL 测试集")
    parser.add_argument("--output", default="tests/data/bad_cases_export.jsonl",
                        help="输出文件路径（相对于项目根目录）")
    parser.add_argument("--case-type", default=None,
                        choices=["rewrite_missed_anaphora", "rewrite_same_as_original",
                                 "rewrite_lost_entity", "low_score_no_clarify",
                                 "hallucination_suspected", "retrieval_miss",
                                 "route_misclassification", "user_negative_feedback",
                                 "manual_flag"],
                        help="按类型过滤")
    parser.add_argument("--unreviewed-only", action="store_true",
                        help="仅导出未审核的 bad case")
    parser.add_argument("--user-id", default="system",
                        help="用户ID（默认 system 查全局）")

    args = parser.parse_args()
    export_bad_cases(
        output=args.output,
        case_type=args.case_type,
        unreviewed_only=args.unreviewed_only,
        user_id=args.user_id,
    )


if __name__ == "__main__":
    main()
