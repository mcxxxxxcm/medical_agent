"""RAGAS 评估脚本

用法：
    # 使用默认测试集评估
    python scripts/evaluate_rag.py

    # 使用 bad case 导出数据评估
    python scripts/evaluate_rag.py --test-set tests/data/bad_cases_export.jsonl

    # 对比两个版本
    python scripts/evaluate_rag.py --compare data/evaluation/eval_v20250101_120000.json data/evaluation/eval_v20250102_140000.json

    # 全量重跑（不使用增量）
    python scripts/evaluate_rag.py --no-incremental
"""
import argparse
import json
import sys
from pathlib import Path

# 将项目根目录加入 sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.rag.evaluation import RAGEvaluator, _RAGAS_AVAILABLE
from app.core.app_logging import get_logger

logger = get_logger(__name__)

DEFAULT_TEST_SET = "tests/data/rag_eval_test_set.jsonl"


def main():
    parser = argparse.ArgumentParser(description="RAG 评估脚本 (v2)")

    parser.add_argument(
        "--test-set",
        type=str,
        default=DEFAULT_TEST_SET,
        help=f"测试集 JSONL 文件路径（默认: {DEFAULT_TEST_SET}）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/evaluation",
        help="评估结果输出目录（默认: data/evaluation）",
    )
    parser.add_argument(
        "--no-incremental",
        action="store_true",
        help="禁用增量评估，全量重跑",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("V1", "V2"),
        help="对比两个版本的评估结果文件路径",
    )

    args = parser.parse_args()

    evaluator = RAGEvaluator(output_dir=args.output_dir)

    # ---- 对比模式 ----
    if args.compare:
        v1_path, v2_path = args.compare
        for p in (v1_path, v2_path):
            if not Path(p).exists():
                logger.error(f"文件不存在: {p}")
                sys.exit(1)

        comparison = evaluator.compare_versions(v1_path, v2_path)

        print("\n" + "=" * 60)
        print("评估版本对比")
        print("=" * 60)
        print(f"  V1 版本: {comparison['v1_version']}")
        print(f"  V2 版本: {comparison['v2_version']}")
        print("-" * 60)

        for item in comparison["comparison"]:
            direction_icon = {
                "improved": "↑",
                "regressed": "↓",
                "unchanged": "→",
            }.get(item["direction"], "?")
            print(
                f"  {item['metric']:40s}: "
                f"{item['v1']:.4f} → {item['v2']:.4f}  "
                f"{direction_icon} {item['delta']:+.4f} ({item['direction']})"
            )

        print("=" * 60 + "\n")
        return

    # ---- 评估模式 ----
    eval_mode = "RAGAS" if _RAGAS_AVAILABLE else "规则引擎"
    logger.info(f"评估模式: {eval_mode}")

    test_data = evaluator.load_test_set(args.test_set)
    if not test_data:
        logger.error(f"测试集为空或不存在: {args.test_set}")
        sys.exit(1)

    incremental = not args.no_incremental
    results = evaluator.evaluate_batch(test_data, incremental=incremental)
    evaluator.save_results(results)


if __name__ == "__main__":
    main()
