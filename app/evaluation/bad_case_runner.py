"""Bad Case 回归测试运行器

生命周期：自动采集 → 人工审核 → 回归测试 → 统计报告

核心能力：
    1. load_test_set(): 从 JSONL 文件加载已审核的 bad case 测试集
    2. run_single(): 对单条测试用例重放查询，对比当前输出与期望输出
    3. run_batch(): 批量运行回归测试
    4. generate_report(): 生成统计报告（通过率、失败用例分布、改进趋势）
"""

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage

from app.core.app_logging import get_logger
from app.graph.nodes.nodes import query_rewrite_node

logger = get_logger(__name__)


def _normalize_for_comparison(text: str) -> str:
    """模糊匹配归一化：去除标点、空格差异后比较

    用于对比 actual 和 expected 的重写结果，
    忽略全角/半角标点、空格、末尾标点等差异。
    """
    if not text:
        return ""
    # 去除所有空白字符
    text = re.sub(r"\s+", "", text)
    # 统一全角标点为半角
    text = text.replace("，", ",").replace("。", ".").replace("？", "?").replace("！", "!")
    # 去除首尾标点
    text = text.strip(",.?!;:，。？！；：、")
    # 转小写
    text = text.lower()
    return text


def _fuzzy_match(actual: str, expected: str) -> bool:
    """模糊匹配：归一化后判断是否等价

    判定逻辑：
        1. 归一化后完全相同 → 通过
        2. expected 是 actual 的子串（重写更完整也算通过）→ 通过
        3. actual 包含 expected 的核心实体 → 通过
    """
    norm_actual = _normalize_for_comparison(actual)
    norm_expected = _normalize_for_comparison(expected)

    if not norm_expected:
        return False

    # 完全匹配
    if norm_actual == norm_expected:
        return True

    # 子串匹配：expected 是 actual 的子串（重写更完整也算通过）
    if norm_expected in norm_actual:
        return True

    # 核心实体匹配：expected 中的连续中文/英文片段在 actual 中出现
    core_segments = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", norm_expected)
    if core_segments and all(seg in norm_actual for seg in core_segments):
        return True

    return False


def _build_messages_from_history(history_summary: str) -> List[Any]:
    """将 history_summary 文本转为 HumanMessage/AIMessage 交替的对话历史

    history_summary 格式示例（来自 export_bad_cases.py 输出）：
        "用户头痛，建议布洛芬"
        或
        "用户：我头痛\n助手：建议服用布洛芬"

    处理策略：
        - 含"用户："/"助手："分隔 → 按分隔拆分
        - 无分隔 → 将整段作为 HumanMessage（用户描述症状）
    """
    if not history_summary or not history_summary.strip():
        return []

    text = history_summary.strip()

    # 尝试按"用户："/"助手："分隔
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

    # 无分隔 → 简单拆分为一条 Human + 一条 AI
    # 将摘要作为用户消息，生成一条模拟的 AI 回复
    return [
        HumanMessage(content=text),
        AIMessage(content="好的，我了解了。"),
    ]


class BadCaseRegressionRunner:
    """Bad Case 回归测试运行器

    生命周期：自动采集 → 人工审核 → 回归测试 → 统计报告

    核心能力：
    1. load_test_set(): 从 JSONL 文件加载已审核的 bad case 测试集
    2. run_single(): 对单条测试用例重放查询，对比当前输出与期望输出
    3. run_batch(): 批量运行回归测试
    4. generate_report(): 生成统计报告（通过率、失败用例分布、改进趋势）
    """

    def __init__(self, test_set_path: str = "tests/data/bad_cases_test_set.jsonl"):
        self.test_set_path = test_set_path
        self._test_cases: List[Dict] = []

    def load_test_set(self, path: str = None) -> List[Dict]:
        """加载 JSONL 格式的测试集（已人工审核的 bad case）

        Args:
            path: JSONL 文件路径，默认使用初始化时指定的路径

        Returns:
            加载的测试用例列表

        Raises:
            FileNotFoundError: 测试集文件不存在
        """
        file_path = Path(path or self.test_set_path)

        if not file_path.exists():
            logger.error(f"测试集文件不存在：{file_path}")
            raise FileNotFoundError(f"测试集文件不存在：{file_path}")

        test_cases = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    case = json.loads(line)
                    test_cases.append(case)
                except json.JSONDecodeError as e:
                    logger.warning(f"跳过无效 JSON 行 {line_no}：{e}")

        self._test_cases = test_cases
        logger.info(f"已加载 {len(test_cases)} 条测试用例：{file_path}")
        return test_cases

    def run_single(self, test_case: Dict) -> Dict:
        """对单条用例重放

        流程：
        1. 用 original_query 构建最小化 state
        2. 走 query_rewrite_node 获取重写结果
        3. 对比 final_question 与 expected_rewrite
        4. 返回 {pass: bool, actual: str, expected: str, diff: str}

        Args:
            test_case: 单条测试用例字典，需包含 original_query、expected_rewrite 等字段

        Returns:
            测试结果字典，包含 pass、actual、expected、diff、case_id、case_type 等字段
        """
        case_id = test_case.get("id", test_case.get("case_id", "unknown"))
        case_type = test_case.get("case_type", "unknown")
        original_query = test_case.get("original_query", "")
        expected_rewrite = test_case.get("expected_rewrite", "")
        history_summary = test_case.get("history_summary", "")

        # 如果 expected_rewrite 为空（未人工审核），跳过该用例
        if not expected_rewrite or not expected_rewrite.strip():
            logger.warning(f"跳过未审核用例 {case_id}：expected_rewrite 为空")
            return {
                "pass": None,
                "actual": "",
                "expected": "",
                "diff": "跳过：expected_rewrite 为空，未人工审核",
                "case_id": case_id,
                "case_type": case_type,
                "skipped": True,
            }

        # 构建 messages（从 history_summary 转换）
        messages = _build_messages_from_history(history_summary)

        # 构建最小化 state dict，包含 query_rewrite_node 所需字段
        state = {
            "question": original_query,
            "messages": messages,
            "user_id": test_case.get("user_id", "regression_test"),
            "thread_id": test_case.get("thread_id", f"regression_{uuid.uuid4().hex[:8]}"),
            "image_base64": None,
            "final_answer": None,
            "warnings": [],
            "sources": [],
            "retrieved_docs": None,
            "symptoms": None,
            "error": None,
            "user_profile": None,
            "rewritten_query": None,
            "final_question": None,
            "hyde_answer": None,
            "retrieval_attempts": 0,
            "clinical_checkpoint": None,
        }

        # 执行 query_rewrite_node
        try:
            rewrite_result = query_rewrite_node(state)
            actual_final = rewrite_result.get("final_question", original_query) if rewrite_result else original_query
        except Exception as e:
            logger.error(f"重写节点执行失败 {case_id}：{e}")
            return {
                "pass": False,
                "actual": "",
                "expected": expected_rewrite,
                "diff": f"执行异常：{e}",
                "case_id": case_id,
                "case_type": case_type,
                "error": str(e),
                "skipped": False,
            }

        # 模糊匹配对比
        passed = _fuzzy_match(actual_final, expected_rewrite)

        # 生成差异描述
        if passed:
            diff = ""
        else:
            diff = _describe_diff(actual_final, expected_rewrite)

        result = {
            "pass": passed,
            "actual": actual_final,
            "expected": expected_rewrite,
            "diff": diff,
            "case_id": case_id,
            "case_type": case_type,
            "original_query": original_query,
            "skipped": False,
        }

        status = "PASS" if passed else "FAIL"
        logger.info(f"[{status}] {case_type} | 原问：{original_query[:30]} | 期望：{expected_rewrite[:30]} | 实际：{actual_final[:30]}")

        return result

    def run_batch(self, test_cases: List[Dict] = None) -> List[Dict]:
        """批量回归测试

        Args:
            test_cases: 测试用例列表，默认使用已加载的测试集

        Returns:
            所有测试用例的结果列表
        """
        cases = test_cases or self._test_cases

        if not cases:
            logger.warning("无测试用例可执行，请先 load_test_set()")
            return []

        logger.info(f"开始批量回归测试，共 {len(cases)} 条用例")
        results = []

        for i, case in enumerate(cases, 1):
            logger.info(f"执行用例 {i}/{len(cases)}：{case.get('id', case.get('case_id', 'unknown'))}")
            result = self.run_single(case)
            results.append(result)

        # 统计概要
        total = len(results)
        passed = sum(1 for r in results if r.get("pass") is True)
        failed = sum(1 for r in results if r.get("pass") is False)
        skipped = sum(1 for r in results if r.get("skipped"))

        logger.info(f"批量回归测试完成：总计 {total}，通过 {passed}，失败 {failed}，跳过 {skipped}")

        return results

    def generate_report(self, results: List[Dict]) -> Dict:
        """生成回归测试报告

        Args:
            results: run_batch() 返回的测试结果列表

        Returns:
            报告字典，包含 total、passed、failed、pass_rate、by_case_type、failed_cases、timestamp
        """
        # 过滤掉跳过的用例
        evaluated = [r for r in results if not r.get("skipped")]
        skipped_count = sum(1 for r in results if r.get("skipped"))

        total = len(evaluated)
        passed = sum(1 for r in evaluated if r.get("pass") is True)
        failed = sum(1 for r in evaluated if r.get("pass") is False)
        pass_rate = round(passed / total, 4) if total > 0 else 0.0

        # 按 case_type 分组统计
        by_case_type: Dict[str, Dict[str, Any]] = {}
        for r in evaluated:
            ct = r.get("case_type", "unknown")
            if ct not in by_case_type:
                by_case_type[ct] = {"total": 0, "passed": 0, "failed": 0}
            by_case_type[ct]["total"] += 1
            if r.get("pass") is True:
                by_case_type[ct]["passed"] += 1
            else:
                by_case_type[ct]["failed"] += 1

        # 计算各类型通过率
        for ct in by_case_type:
            ct_total = by_case_type[ct]["total"]
            by_case_type[ct]["pass_rate"] = round(by_case_type[ct]["passed"] / ct_total, 4) if ct_total > 0 else 0.0

        # 失败用例摘要
        failed_cases = [
            {
                "case_id": r.get("case_id"),
                "case_type": r.get("case_type"),
                "original_query": r.get("original_query", ""),
                "actual": r.get("actual", ""),
                "expected": r.get("expected", ""),
                "diff": r.get("diff", ""),
            }
            for r in evaluated if r.get("pass") is False
        ]

        report = {
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped_count,
            "pass_rate": pass_rate,
            "by_case_type": by_case_type,
            "failed_cases": failed_cases,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"回归测试报告：通过率 {pass_rate:.1%}（{passed}/{total}），"
            f"失败 {failed}，跳过 {skipped_count}"
        )

        return report

    def save_report(self, report: Dict, path: str = "data/evaluation/bad_case_report.json"):
        """保存报告到文件

        Args:
            report: generate_report() 返回的报告字典
            path: 输出文件路径（相对于项目根目录）
        """
        file_path = Path(path)

        # 如果是相对路径，基于项目根目录解析
        if not file_path.is_absolute():
            project_root = Path(__file__).resolve().parent.parent.parent
            file_path = project_root / path

        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"回归测试报告已保存：{file_path}")


def _describe_diff(actual: str, expected: str) -> str:
    """描述 actual 与 expected 的差异

    简单差异描述，用于报告中的 diff 字段。
    """
    if not actual:
        return "实际输出为空"
    if not expected:
        return "期望输出为空"

    # 找出 expected 中有但 actual 中没有的核心词
    expected_tokens = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", expected.lower()))
    actual_tokens = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", actual.lower()))
    missing = expected_tokens - actual_tokens
    extra = actual_tokens - expected_tokens

    parts = []
    if missing:
        parts.append(f"缺失：{'、'.join(sorted(missing)[:5])}")
    if extra:
        parts.append(f"多余：{'、'.join(sorted(extra)[:5])}")

    if not parts:
        parts.append("语义差异（归一化后不匹配）")

    return "；".join(parts)
