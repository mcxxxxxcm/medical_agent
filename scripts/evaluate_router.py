"""路由分类评估脚本

对 route_node 的三层路由策略进行离线评估：
    - 整体准确率
    - 分类别 Precision / Recall / F1
    - 分层命中率（rule / context / llm）
    - 分维度统计（按 category、difficulty）
    - 边界case专项分析

用法：
    # 仅评估规则层（无需LLM，快速）
    python scripts/evaluate_router.py --layer rule

    # 全量评估（含LLM路由，需Ollama）
    python scripts/evaluate_router.py --layer all

    # 指定测试集
    python scripts/evaluate_router.py --test-set tests/data/route_test_set.jsonl

    # 生成对比报告（与上次基线对比）
    python scripts/evaluate_router.py --compare data/evaluation/router_baseline.json
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加项目根目录到 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.app_logging import get_logger

logger = get_logger(__name__)


# ===================================================================
# 测试集加载
# ===================================================================

def load_test_set(path: str) -> List[Dict]:
    """加载 JSONL 格式的路由测试集"""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"测试集文件不存在：{file_path}")

    cases = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
                cases.append(case)
            except json.JSONDecodeError as e:
                logger.warning(f"跳过无效 JSON 行 {line_no}：{e}")

    logger.info(f"已加载 {len(cases)} 条路由测试用例：{file_path}")
    return cases


# ===================================================================
# 路由执行器
# ===================================================================

def run_route_rule_only(question: str) -> Dict[str, Any]:
    """仅执行规则层路由（无需LLM，0ms延迟）"""
    from app.graph.nodes.nodes import detect_rule_based_route, normalize_router_label

    rule_result = detect_rule_based_route(question)
    if rule_result:
        return {
            "predicted_label": normalize_router_label(rule_result),
            "route_layer": "rule",
        }
    # 规则未命中，标记为 unknown（交给LLM层，但本模式不调用LLM）
    return {
        "predicted_label": "unknown",
        "route_layer": "miss",
    }


def run_route_with_context(question: str, history_summary: str = "") -> Dict[str, Any]:
    """执行规则层 + 上下文层路由（无需LLM）"""
    from app.graph.nodes.nodes import (
        detect_rule_based_route,
        _detect_route_from_context,
        normalize_router_label,
    )
    from langchain_core.messages import HumanMessage, AIMessage

    # 1. 规则层
    rule_result = detect_rule_based_route(question)
    if rule_result:
        return {
            "predicted_label": normalize_router_label(rule_result),
            "route_layer": "rule",
        }

    # 2. 上下文层（构建最小化state）
    if history_summary:
        messages = _build_messages_from_history(history_summary)
    else:
        messages = []

    state = {
        "question": question,
        "messages": messages,
    }
    context_result = _detect_route_from_context(state)
    if context_result:
        return {
            "predicted_label": normalize_router_label(context_result),
            "route_layer": "context",
        }

    return {
        "predicted_label": "unknown",
        "route_layer": "miss",
    }


def run_route_full(question: str, history_summary: str = "") -> Dict[str, Any]:
    """完整三层路由（含LLM兜底）"""
    from app.graph.nodes.nodes import (
        detect_rule_based_route,
        _detect_route_from_context,
        _llm_route,
        normalize_router_label,
    )

    # 1. 规则层
    rule_result = detect_rule_based_route(question)
    if rule_result:
        return {
            "predicted_label": normalize_router_label(rule_result),
            "route_layer": "rule",
        }

    # 2. 上下文层
    if history_summary:
        messages = _build_messages_from_history(history_summary)
    else:
        messages = []

    state = {
        "question": question,
        "messages": messages,
    }
    context_result = _detect_route_from_context(state)
    if context_result:
        return {
            "predicted_label": normalize_router_label(context_result),
            "route_layer": "context",
        }

    # 3. LLM层
    try:
        llm_result = _llm_route(question)
        return {
            "predicted_label": normalize_router_label(llm_result),
            "route_layer": "llm",
        }
    except Exception as e:
        logger.warning(f"LLM路由失败：{e}")
        return {
            "predicted_label": "general",
            "route_layer": "llm_fallback",
        }


def _build_messages_from_history(history_summary: str) -> List[Any]:
    """将 history_summary 转为 HumanMessage/AIMessage 列表"""
    import re
    from langchain_core.messages import HumanMessage, AIMessage

    if not history_summary or not history_summary.strip():
        return []

    text = history_summary.strip()
    parts = re.split(r"(?:用户[：:]\s*|助手[：:]\s*)", text)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) >= 2:
        messages = []
        for i, part in enumerate(parts):
            if i % 2 == 0:
                messages.append(HumanMessage(content=part))
            else:
                messages.append(AIMessage(content=part))
        return messages

    return [HumanMessage(content=text), AIMessage(content="好的，我了解了。")]


# ===================================================================
# 评估指标计算
# ===================================================================

def compute_metrics(y_true: List[str], y_pred: List[str], labels: List[str] = None) -> Dict:
    """计算分类评估指标

    Returns:
        {
            "accuracy": float,
            "per_label": {label: {"precision", "recall", "f1", "support", "tp", "fp", "fn"}},
            "macro_avg": {"precision", "recall", "f1"},
            "weighted_avg": {"precision", "recall", "f1"},
        }
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    # 混淆矩阵
    label_to_idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    cm = [[0] * n for _ in range(n)]

    for t, p in zip(y_true, y_pred):
        ti = label_to_idx.get(t, -1)
        pi = label_to_idx.get(p, -1)
        if ti >= 0 and pi >= 0:
            cm[ti][pi] += 1

    # 各类别指标
    per_label = {}
    for label in labels:
        i = label_to_idx[label]
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n)) - tp
        fn = sum(cm[i][c] for c in range(n)) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    # 宏平均
    macro_p = sum(v["precision"] for v in per_label.values()) / n
    macro_r = sum(v["recall"] for v in per_label.values()) / n
    macro_f1 = sum(v["f1"] for v in per_label.values()) / n

    # 加权平均
    total = sum(v["support"] for v in per_label.values())
    weighted_p = sum(v["precision"] * v["support"] for v in per_label.values()) / total if total > 0 else 0
    weighted_r = sum(v["recall"] * v["support"] for v in per_label.values()) / total if total > 0 else 0
    weighted_f1 = sum(v["f1"] * v["support"] for v in per_label.values()) / total if total > 0 else 0

    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0

    return {
        "accuracy": round(accuracy, 4),
        "per_label": per_label,
        "macro_avg": {
            "precision": round(macro_p, 4),
            "recall": round(macro_r, 4),
            "f1": round(macro_f1, 4),
        },
        "weighted_avg": {
            "precision": round(weighted_p, 4),
            "recall": round(weighted_r, 4),
            "f1": round(weighted_f1, 4),
        },
    }


# ===================================================================
# 主评估流程
# ===================================================================

def evaluate(test_set_path: str, layer: str = "rule") -> Dict:
    """执行路由评估

    Args:
        test_set_path: 测试集 JSONL 路径
        layer: 评估范围 - "rule"(仅规则层), "context"(规则+上下文), "all"(全量)

    Returns:
        评估报告字典
    """
    cases = load_test_set(test_set_path)

    # 选择路由执行器
    if layer == "rule":
        route_fn = run_route_rule_only
    elif layer == "context":
        route_fn = run_route_with_context
    else:
        route_fn = run_route_full

    # 逐条评估
    results = []
    y_true = []
    y_pred = []
    layer_distribution = Counter()
    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    difficulty_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    boundary_stats = {"correct": 0, "total": 0}
    mismatch_cases = []

    for i, case in enumerate(cases, 1):
        query = case["query"]
        golden_label = case["golden_label"]
        history_summary = case.get("history_summary", "")

        # 执行路由
        if layer == "rule":
            route_result = route_fn(query)
        else:
            route_result = route_fn(query, history_summary)

        predicted_label = route_result["predicted_label"]
        route_layer = route_result["route_layer"]
        is_correct = predicted_label == golden_label

        y_true.append(golden_label)
        y_pred.append(predicted_label)
        layer_distribution[route_layer] += 1

        # 按维度统计
        category = case.get("category", "unknown")
        category_stats[category]["total"] += 1
        if is_correct:
            category_stats[category]["correct"] += 1

        difficulty = case.get("difficulty", "unknown")
        difficulty_stats[difficulty]["total"] += 1
        if is_correct:
            difficulty_stats[difficulty]["correct"] += 1

        if case.get("is_boundary"):
            boundary_stats["total"] += 1
            if is_correct:
                boundary_stats["correct"] += 1

        result_entry = {
            "id": case.get("id", f"case_{i}"),
            "query": query,
            "golden_label": golden_label,
            "predicted_label": predicted_label,
            "route_layer": route_layer,
            "is_correct": is_correct,
            "is_boundary": case.get("is_boundary", False),
        }

        if not is_correct:
            result_entry["mismatch_reason"] = case.get("boundary_reason", "")
            mismatch_cases.append(result_entry)

        results.append(result_entry)

        status = "PASS" if is_correct else "FAIL"
        logger.info(f"[{status}] {case.get('id','?')} | golden={golden_label} pred={predicted_label} layer={route_layer} | {query[:30]}")

    # 计算指标
    valid_labels = ["symptom", "knowledge", "general"]
    # 如果有 unknown 预测，加入标签集
    if "unknown" in y_pred:
        valid_labels.append("unknown")

    metrics = compute_metrics(y_true, y_pred, labels=valid_labels)

    # 分类别准确率
    category_accuracy = {}
    for cat, stats in sorted(category_stats.items()):
        cat_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        category_accuracy[cat] = {
            "accuracy": round(cat_acc, 4),
            "correct": stats["correct"],
            "total": stats["total"],
        }

    # 分难度准确率
    difficulty_accuracy = {}
    for diff, stats in sorted(difficulty_stats.items()):
        diff_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        difficulty_accuracy[diff] = {
            "accuracy": round(diff_acc, 4),
            "correct": stats["correct"],
            "total": stats["total"],
        }

    # 边界case准确率
    boundary_accuracy = (
        boundary_stats["correct"] / boundary_stats["total"]
        if boundary_stats["total"] > 0 else 0
    )

    # 报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "test_set": str(test_set_path),
        "eval_layer": layer,
        "total_cases": len(cases),
        "metrics": metrics,
        "layer_distribution": dict(layer_distribution),
        "layer_hit_rate": {
            k: round(v / len(cases), 4) for k, v in layer_distribution.items()
        },
        "category_accuracy": category_accuracy,
        "difficulty_accuracy": difficulty_accuracy,
        "boundary_accuracy": {
            "accuracy": round(boundary_accuracy, 4),
            "correct": boundary_stats["correct"],
            "total": boundary_stats["total"],
        },
        "mismatch_cases": mismatch_cases,
    }

    return report


def print_report(report: Dict):
    """打印人类可读的评估报告"""
    print("\n" + "=" * 70)
    print("  路由分类评估报告")
    print("=" * 70)

    m = report["metrics"]
    print(f"\n📊 整体指标")
    print(f"  准确率:       {m['accuracy']:.1%}")
    print(f"  宏平均 F1:    {m['macro_avg']['f1']:.1%}")
    print(f"  加权平均 F1:  {m['weighted_avg']['f1']:.1%}")

    print(f"\n📋 分类别指标")
    print(f"  {'类别':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-'*52}")
    for label, v in m["per_label"].items():
        print(f"  {label:<12} {v['precision']:>10.1%} {v['recall']:>10.1%} {v['f1']:>10.1%} {v['support']:>10}")

    print(f"\n🔀 分层命中率")
    total = report["total_cases"]
    for layer, count in sorted(report["layer_distribution"].items()):
        hit_rate = report["layer_hit_rate"].get(layer, 0)
        print(f"  {layer:<15} {count:>4} 条 ({hit_rate:.1%})")

    print(f"\n📂 分类别准确率")
    for cat, v in report["category_accuracy"].items():
        print(f"  {cat:<12} {v['accuracy']:.1%} ({v['correct']}/{v['total']})")

    print(f"\n🎯 分难度准确率")
    for diff, v in report["difficulty_accuracy"].items():
        print(f"  {diff:<12} {v['accuracy']:.1%} ({v['correct']}/{v['total']})")

    ba = report["boundary_accuracy"]
    if ba["total"] > 0:
        print(f"\n⚠️  边界case准确率: {ba['accuracy']:.1%} ({ba['correct']}/{ba['total']})")

    mismatches = report["mismatch_cases"]
    if mismatches:
        print(f"\n❌ 错误用例 ({len(mismatches)} 条)")
        for mc in mismatches[:20]:
            reason = mc.get("mismatch_reason", "")
            reason_str = f" ({reason})" if reason else ""
            print(f"  [{mc['id']}] golden={mc['golden_label']} pred={mc['predicted_label']} "
                  f"layer={mc['route_layer']}{reason_str}")
            print(f"         query: {mc['query'][:50]}")

    print("\n" + "=" * 70)


def save_report(report: Dict, path: str):
    """保存报告到JSON文件"""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"报告已保存：{file_path}")


def compare_with_baseline(report: Dict, baseline_path: str):
    """与基线报告对比"""
    baseline_path = Path(baseline_path)
    if not baseline_path.exists():
        logger.warning(f"基线文件不存在：{baseline_path}")
        return

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    print("\n" + "=" * 70)
    print("  与基线对比")
    print("=" * 70)

    curr_acc = report["metrics"]["accuracy"]
    base_acc = baseline["metrics"]["accuracy"]
    delta = curr_acc - base_acc
    sign = "↑" if delta > 0 else "↓" if delta < 0 else "→"
    print(f"\n  准确率: {curr_acc:.1%} {sign} {abs(delta):.1%} (基线 {base_acc:.1%})")

    curr_f1 = report["metrics"]["weighted_avg"]["f1"]
    base_f1 = baseline["metrics"]["weighted_avg"]["f1"]
    delta_f1 = curr_f1 - base_f1
    sign_f1 = "↑" if delta_f1 > 0 else "↓" if delta_f1 < 0 else "→"
    print(f"  加权F1: {curr_f1:.1%} {sign_f1} {abs(delta_f1):.1%} (基线 {base_f1:.1%})")

    # 分类别对比
    print(f"\n  {'类别':<12} {'当前F1':>10} {'基线F1':>10} {'变化':>10}")
    print(f"  {'-'*42}")
    for label in ["symptom", "knowledge", "general"]:
        curr = report["metrics"]["per_label"].get(label, {})
        base = baseline["metrics"]["per_label"].get(label, {})
        curr_f1_l = curr.get("f1", 0)
        base_f1_l = base.get("f1", 0)
        d = curr_f1_l - base_f1_l
        s = "↑" if d > 0 else "↓" if d < 0 else "→"
        print(f"  {label:<12} {curr_f1_l:>10.1%} {base_f1_l:>10.1%} {s}{abs(d):>9.1%}")

    print("\n" + "=" * 70)


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="路由分类评估脚本")
    parser.add_argument(
        "--test-set",
        default="tests/data/route_test_set.jsonl",
        help="测试集路径（默认 tests/data/route_test_set.jsonl）",
    )
    parser.add_argument(
        "--layer",
        choices=["rule", "context", "all"],
        default="rule",
        help="评估范围：rule=仅规则层, context=规则+上下文, all=全量含LLM",
    )
    parser.add_argument(
        "--output",
        default="data/evaluation/router_report.json",
        help="报告输出路径",
    )
    parser.add_argument(
        "--compare",
        default=None,
        help="基线报告路径（可选，用于对比）",
    )
    parser.add_argument(
        "--save-baseline",
        default=None,
        help="保存当前报告为基线（指定路径）",
    )

    args = parser.parse_args()

    # 执行评估
    logger.info(f"开始路由评估：layer={args.layer}, test_set={args.test_set}")
    report = evaluate(args.test_set, args.layer)

    # 打印报告
    print_report(report)

    # 保存报告
    save_report(report, args.output)

    # 保存为基线
    if args.save_baseline:
        save_report(report, args.save_baseline)
        print(f"\n已保存为基线：{args.save_baseline}")

    # 与基线对比
    if args.compare:
        compare_with_baseline(report, args.compare)


if __name__ == "__main__":
    main()
