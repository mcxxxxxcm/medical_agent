"""Bad Case 回归测试 CLI

用法：
    python scripts/run_bad_case_regression.py
    python scripts/run_bad_case_regression.py --test-set tests/data/custom.jsonl
    python scripts/run_bad_case_regression.py --report-only data/evaluation/bad_case_report.json
"""

import argparse
import json
import sys
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_regression(test_set: str, report_path: str, verbose: bool = False):
    """执行回归测试并保存报告"""
    from app.evaluation.bad_case_runner import BadCaseRegressionRunner

    runner = BadCaseRegressionRunner(test_set_path=test_set)

    # 加载测试集
    try:
        test_cases = runner.load_test_set()
    except FileNotFoundError as e:
        print(f"错误：{e}")
        sys.exit(1)

    if not test_cases:
        print("测试集为空，无用例可执行")
        return

    print(f"已加载 {len(test_cases)} 条测试用例")

    # 执行批量回归测试
    results = runner.run_batch(test_cases)

    # 生成报告
    report = runner.generate_report(results)

    # 打印报告摘要
    print("\n" + "=" * 60)
    print("Bad Case 回归测试报告")
    print("=" * 60)
    print(f"总用例数：{report['total']}")
    print(f"通过：{report['passed']}")
    print(f"失败：{report['failed']}")
    print(f"跳过：{report['skipped']}")
    print(f"通过率：{report['pass_rate']:.1%}")
    print(f"时间：{report['timestamp']}")

    # 按 case_type 分组打印
    if report.get("by_case_type"):
        print("\n按类型统计：")
        for ct, stats in sorted(report["by_case_type"].items()):
            print(f"  {ct}: 通过 {stats['passed']}/{stats['total']}（{stats['pass_rate']:.1%}）")

    # 打印失败用例
    if verbose and report.get("failed_cases"):
        print("\n失败用例：")
        for fc in report["failed_cases"]:
            print(f"  [{fc['case_type']}] {fc['original_query'][:40]}")
            print(f"    期望：{fc['expected'][:50]}")
            print(f"    实际：{fc['actual'][:50]}")
            if fc.get("diff"):
                print(f"    差异：{fc['diff']}")
            print()

    # 保存报告
    runner.save_report(report, path=report_path)
    print(f"\n报告已保存：{report_path}")


def show_report_only(report_path: str):
    """仅展示已有报告"""
    file_path = PROJECT_ROOT / report_path if not Path(report_path).is_absolute() else Path(report_path)

    if not file_path.exists():
        print(f"报告文件不存在：{file_path}")
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    print(f"报告时间：{report.get('timestamp', '未知')}")
    print(f"总用例数：{report.get('total', 0)}")
    print(f"通过：{report.get('passed', 0)}")
    print(f"失败：{report.get('failed', 0)}")
    print(f"跳过：{report.get('skipped', 0)}")
    print(f"通过率：{report.get('pass_rate', 0):.1%}")

    if report.get("by_case_type"):
        print("\n按类型统计：")
        for ct, stats in sorted(report["by_case_type"].items()):
            print(f"  {ct}: 通过 {stats['passed']}/{stats['total']}（{stats['pass_rate']:.1%}）")

    if report.get("failed_cases"):
        print(f"\n失败用例（{len(report['failed_cases'])}条）：")
        for fc in report["failed_cases"]:
            print(f"  [{fc.get('case_type', '?')}] {fc.get('original_query', '')[:40]}")
            print(f"    期望：{fc.get('expected', '')[:50]}")
            print(f"    实际：{fc.get('actual', '')[:50]}")


def main():
    parser = argparse.ArgumentParser(description="Bad Case 回归测试 CLI")
    parser.add_argument(
        "--test-set",
        default="tests/data/bad_cases_test_set.jsonl",
        help="测试集 JSONL 文件路径（相对于项目根目录）",
    )
    parser.add_argument(
        "--report-path",
        default="data/evaluation/bad_case_report.json",
        help="报告输出路径（相对于项目根目录）",
    )
    parser.add_argument(
        "--report-only",
        default=None,
        help="仅展示已有报告（传入报告路径）",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细输出失败用例信息",
    )

    args = parser.parse_args()

    if args.report_only:
        show_report_only(args.report_only)
    else:
        run_regression(
            test_set=args.test_set,
            report_path=args.report_path,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
