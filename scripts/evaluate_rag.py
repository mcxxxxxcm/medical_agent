"""RAGAS 评估脚本
功能描述：
    执行 RAG 系统的完整评估流程
    支持自定义测试数据和评估指标

使用方法：
    python scripts/evaluate_rag.py --use-reranker --k 5 --metrics faithfulness,answer_relevance
"""
import argparse
import json
from pathlib import Path

from app.rag.evaluation import (
    get_evaluator,
    DEFAULT_TEST_DATA
)
from app.core.app_logging import get_logger

logger = get_logger(__name__)


def load_test_data(filepath: str) -> list:
    """从文件加载测试数据

    Args:
        filepath: JSON 文件路径

    Returns:
        测试数据列表
    """
    path = Path(filepath)
    if not path.exists():
        logger.warning(f"测试数据文件不存在：{filepath}，使用默认数据集")
        return DEFAULT_TEST_DATA

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"从文件加载测试数据：{filepath}，样本数：{len(data)}")
    return data


def main():
    parser = argparse.ArgumentParser(description="RAGAS 评估脚本")

    # 基础参数
    parser.add_argument(
        "--test-data",
        type=str,
        default="",
        help="测试数据文件路径（JSON格式），默认使用内置测试集"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/evaluation/results.json",
        help="评估结果输出文件路径"
    )

    # 检索参数
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        default=True,
        help="是否使用 Reranker（默认启用）"
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="禁用 Reranker"
    )
    parser.add_argument(
        "-k",
        type=int,
        default=5,
        help="检索文档数量（默认：5）"
    )

    # 评估指标
    parser.add_argument(
        "--metrics",
        type=str,
        default="faithfulness,answer_relevance,context_relevance",
        help="评估指标列表，逗号分隔。可选：faithfulness,answer_relevance,context_relevance,context_precision,context_recall"
    )

    args = parser.parse_args()

    # 解析指标
    metrics = [m.strip() for m in args.metrics.split(",")]

    # 加载测试数据
    test_data = load_test_data(args.test_data) if args.test_data else DEFAULT_TEST_DATA

    # 初始化评估器
    use_reranker = args.use_reranker and not args.no_reranker
    evaluator = get_evaluator(
        use_reranker=use_reranker,
        k=args.k
    )

    # 执行评估
    results = evaluator.evaluate(
        test_data=test_data,
        metrics=metrics
    )

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"评估结果已保存：{output_path}")

    # 打印摘要
    print("\n" + "=" * 60)
    print("RAGAS 评估结果")
    print("=" * 60)
    print(f"测试数据: {args.test_data or '默认测试集'}")
    print(f"样本数量: {len(test_data)}")
    print(f"使用 Reranker: {use_reranker}")
    print(f"检索数量: {args.k}")
    print(f"评估指标: {', '.join(metrics)}")
    print("-" * 60)

    scores = results.get("scores", {})
    for metric_name, score in scores.items():
        print(f"{metric_name:30s}: {score:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()