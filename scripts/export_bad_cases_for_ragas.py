"""Bad Case → RAGAS 评估联动脚本

功能：
1. 从 PostgresStore 导出已审核的 Bad Case
2. 为每条 Bad Case 补充 ground_truth（期望答案）
3. 生成 RAGAS 兼容的 JSONL 测试集
4. 可选：自动运行 RAGAS 评估

用法：
    # 仅导出（人工补充 ground_truth 后再跑评估）
    python scripts/export_bad_cases_for_ragas.py

    # 导出 + 自动运行 RAGAS 评估
    python scripts/export_bad_cases_for_ragas.py --eval

关键设计：
    Bad Case 的 expected_rewrite 是"期望的查询重写结果"（如"布洛芬儿童用量？"）
    RAGAS 的 ground_truth 是"期望的完整答案"（如"儿童布洛芬剂量为5-10mg/kg..."）
    两者含义不同，不能直接混用。

    本脚本的策略：
    - 有 ground_truth 字段的 Bad Case → 直接用于 RAGAS
    - 无 ground_truth 但有 expected_rewrite → 用 expected_rewrite 做一次检索+生成，
      将 LLM 输出作为 ground_truth 候选（需人工审核）
    - 标记 source="bad_case"，与标准测试集区分
"""
import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def export_bad_cases(output_path: str, include_unreviewed: bool = False) -> int:
    """导出 Bad Case 为 RAGAS 兼容格式

    Returns:
        导出的样本数
    """
    from app.memory.long_term_memory import get_long_term_memory
    from app.core.app_logging import get_logger
    logger = get_logger(__name__)

    memory = get_long_term_memory()

    # 获取已审核的 Bad Case
    reviewed_cases = memory.get_bad_cases(reviewed=True, limit=200)
    if include_unreviewed:
        unreviewed = memory.get_bad_cases(reviewed=False, limit=200)
        all_cases = reviewed_cases + unreviewed
    else:
        all_cases = reviewed_cases

    if not all_cases:
        print("没有找到 Bad Case 记录")
        return 0

    print(f"找到 {len(all_cases)} 条 Bad Case（已审核 {len(reviewed_cases)} 条）")

    # 转换为 RAGAS 兼容格式
    ragas_samples = []
    needs_ground_truth = 0

    for case in all_cases:
        original_query = case.get("original_query", "")
        expected_rewrite = case.get("expected_rewrite", "")
        ground_truth = case.get("ground_truth", "")  # 人工补充的期望答案
        case_type = case.get("case_type", "")
        case_id = case.get("case_id", "")
        severity = case.get("severity", "")

        if not original_query:
            continue

        # 如果已有 ground_truth，直接用
        if ground_truth:
            ragas_samples.append({
                "question": expected_rewrite or original_query,
                "ground_truth": ground_truth,
                "category": f"bad_case_{case_type}",
                "source": "bad_case",
                "case_id": case_id,
                "severity": severity,
            })
        else:
            # 无 ground_truth：用 expected_rewrite 作为问题，标记需要补充
            needs_ground_truth += 1
            ragas_samples.append({
                "question": expected_rewrite or original_query,
                "ground_truth": "",  # 待人工补充
                "category": f"bad_case_{case_type}",
                "source": "bad_case",
                "case_id": case_id,
                "severity": severity,
                "original_query": original_query,
                "expected_rewrite": expected_rewrite,
                "_needs_ground_truth": True,
            })

    # 写入 JSONL
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for sample in ragas_samples:
            # 内部标记不写入文件
            export_sample = {k: v for k, v in sample.items() if not k.startswith("_")}
            f.write(json.dumps(export_sample, ensure_ascii=False) + "\n")

    print(f"\n导出完成：{len(ragas_samples)} 条 → {output}")
    if needs_ground_truth > 0:
        print(f"⚠️  其中 {needs_ground_truth} 条缺少 ground_truth，需要人工补充期望答案")
        print(f"   打开 {output}，为每条样本的 ground_truth 字段填写期望答案")
        print(f"   然后运行：python scripts/evaluate_rag.py --test-set {output}")

    return len(ragas_samples)


def run_eval(test_set_path: str):
    """运行 RAGAS 评估"""
    from app.rag.evaluation import get_evaluator
    from app.core.app_logging import get_logger
    logger = get_logger(__name__)

    evaluator = get_evaluator()
    test_data = evaluator.load_test_set(test_set_path)

    # 过滤掉没有 ground_truth 的样本（无法评估 Faithfulness）
    evaluable = [s for s in test_data if s.get("ground_truth")]
    if not evaluable:
        print("没有可评估的样本（全部缺少 ground_truth）")
        return

    print(f"可评估样本：{len(evaluable)}/{len(test_data)}")
    result = evaluator.evaluate_batch(evaluable, incremental=False)
    print(f"\n评估完成：{result['evaluated']} 条，耗时 {result['duration_seconds']:.1f}s")
    for metric, score in result["scores"].items():
        print(f"  {metric}: {score:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Bad Case → RAGAS 评估联动")
    parser.add_argument("--eval", action="store_true", help="导出后自动运行 RAGAS 评估")
    parser.add_argument("--include-unreviewed", action="store_true", help="包含未审核的 Bad Case")
    parser.add_argument("--output", default="tests/data/bad_case_ragas_test_set.jsonl",
                        help="输出路径（默认 tests/data/bad_case_ragas_test_set.jsonl）")
    args = parser.parse_args()

    count = export_bad_cases(args.output, args.include_unreviewed)

    if count > 0 and args.eval:
        print("\n" + "=" * 60)
        print("运行 RAGAS 评估...")
        print("=" * 60)
        run_eval(args.output)


if __name__ == "__main__":
    main()
