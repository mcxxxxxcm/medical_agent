"""RAGAS 评估模块 (v2)

四维评估指标：
    - Faithfulness（忠实度）：答案是否基于检索上下文
    - Answer Relevance（答案相关性）：答案是否切题
    - Context Precision（上下文精确度）：检索结果中相关文档的比例
    - Context Relevance（上下文相关性）：检索结果与问题的相关程度

改进：
    1. 使用项目配置的 LLM（get_llm()），而非硬编码 gpt-4o
    2. 简化 RAGAS 版本兼容（仅支持 ragas>=0.1）
    3. 通过检索器+生成器端到端重放，获取完整检索+生成结果
    4. 支持从 bad case 测试集自动生成评估数据
    5. 评估结果版本化管理（支持 A/B 对比）
    6. 增量评估：已评估的样本跳过
"""
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from app.core.app_logging import get_logger
from app.core.llm import get_llm
from app.rag.hybrid_retriever import get_hybrid_retriever

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# RAGAS 可用性检测
# ---------------------------------------------------------------------------
_RAGAS_AVAILABLE = False
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        answer_relevance,
        context_precision,
        context_relevance,
        faithfulness,
    )

    _RAGAS_AVAILABLE = True
except ImportError:
    logger.info("ragas 未安装，将使用规则引擎简易评估")


# ===========================================================================
# 规则引擎简易评估（ragas 未安装时的降级方案）
# ===========================================================================

def _tokenize_chinese(text: str) -> List[str]:
    """简易中文分词：按字符 + 常见标点切分，过滤短词"""
    # 移除标点
    cleaned = re.sub(r"[，。！？、；：""''（）\[\]{}【】《》\s\d]", " ", text)
    tokens = [t for t in cleaned.split() if len(t) >= 2]
    return tokens


def rule_based_faithfulness(answer: str, contexts: List[str]) -> float:
    """规则评估忠实度：答案关键词在上下文中的覆盖率"""
    if not answer or not contexts:
        return 0.0
    answer_tokens = set(_tokenize_chinese(answer))
    if not answer_tokens:
        return 0.0
    context_text = " ".join(contexts)
    context_tokens = set(_tokenize_chinese(context_text))
    if not context_tokens:
        return 0.0
    overlap = answer_tokens & context_tokens
    return min(len(overlap) / len(answer_tokens), 1.0)


def rule_based_relevance(question: str, answer: str) -> float:
    """规则评估答案相关性：问题关键词在答案中的覆盖率"""
    if not question or not answer:
        return 0.0
    question_tokens = set(_tokenize_chinese(question))
    if not question_tokens:
        return 0.0
    answer_tokens = set(_tokenize_chinese(answer))
    if not answer_tokens:
        return 0.0
    overlap = question_tokens & answer_tokens
    return min(len(overlap) / len(question_tokens), 1.0)


def rule_based_context_relevance(question: str, contexts: List[str]) -> float:
    """规则评估上下文相关性：问题关键词在上下文中的覆盖率"""
    if not question or not contexts:
        return 0.0
    question_tokens = set(_tokenize_chinese(question))
    if not question_tokens:
        return 0.0
    context_text = " ".join(contexts)
    context_tokens = set(_tokenize_chinese(context_text))
    if not context_tokens:
        return 0.0
    overlap = question_tokens & context_tokens
    return min(len(overlap) / len(question_tokens), 1.0)


def rule_based_context_precision(question: str, contexts: List[str]) -> float:
    """规则评估上下文精确度：相关文档占总检索文档的比例"""
    if not question or not contexts:
        return 0.0
    question_tokens = set(_tokenize_chinese(question))
    if not question_tokens:
        return 0.0
    relevant_count = 0
    for ctx in contexts:
        ctx_tokens = set(_tokenize_chinese(ctx))
        if question_tokens & ctx_tokens:
            relevant_count += 1
    return relevant_count / len(contexts) if contexts else 0.0


# ===========================================================================
# RAGEvaluator
# ===========================================================================

class RAGEvaluator:
    """RAG 系统评估器 (v2)

    支持：
        1. 加载标准 / bad case 格式测试集
        2. 端到端检索+生成
        3. RAGAS 四维评估（降级为规则引擎）
        4. 增量评估
        5. 版本化结果存储与 A/B 对比
    """

    def __init__(self, output_dir: str = "data/evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"RAG 评估器初始化完成，输出目录: {self.output_dir}")

    # -----------------------------------------------------------------------
    # 测试集加载
    # -----------------------------------------------------------------------

    def load_test_set(self, path: str) -> List[Dict]:
        """加载 JSONL 测试集（支持 bad case 导出格式和标准格式）

        标准格式：
            {"question": "...", "ground_truth": "..."}

        Bad Case 格式：
            {"original_query": "...", "expected_rewrite": "...", ...}
            -> question = original_query, ground_truth = expected_rewrite
        """
        p = Path(path)
        if not p.exists():
            logger.warning(f"测试集文件不存在: {path}，返回空列表")
            return []

        data: List[Dict] = []
        with open(p, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"跳过无效 JSON 行 {line_no}: {line[:80]}")
                    continue

                # 标准格式
                if "question" in item:
                    data.append({
                        "question": item["question"],
                        "ground_truth": item.get("ground_truth", ""),
                        "category": item.get("category", ""),
                    })
                # Bad Case 格式
                elif "original_query" in item:
                    data.append({
                        "question": item["original_query"],
                        "ground_truth": item.get("expected_rewrite", ""),
                        "category": item.get("case_type", "bad_case"),
                    })
                else:
                    logger.warning(f"跳过无法识别的行 {line_no}")

        logger.info(f"加载测试集: {path}，样本数: {len(data)}")
        return data

    # -----------------------------------------------------------------------
    # 检索 & 生成
    # -----------------------------------------------------------------------

    def run_retrieval(self, question: str) -> Dict:
        """执行检索，返回 {contexts: [...], scores: [...], docs: [...]}"""
        retriever = get_hybrid_retriever(k=3, alpha=0.5, use_reranker=True, rerank_top_k=5)
        try:
            docs = retriever.invoke(question, original_query=question)
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return {"contexts": [], "scores": [], "docs": []}

        contexts = [doc.page_content for doc in docs]
        scores = [doc.metadata.get("relevance_score", 0.0) for doc in docs]
        return {"contexts": contexts, "scores": scores, "docs": docs}

    def run_generation(self, question: str, contexts: List[str]) -> str:
        """基于检索结果生成答案"""
        if not contexts:
            return ""

        from app.graph.nodes import answer_generation_node

        docs = [Document(page_content=ctx) for ctx in contexts]
        state = {
            "question": question,
            "retrieved_docs": docs,
        }
        try:
            result = answer_generation_node(state)
            return result.get("final_answer", "")
        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            return ""

    # -----------------------------------------------------------------------
    # 单条评估
    # -----------------------------------------------------------------------

    def evaluate_single(self, sample: Dict) -> Dict:
        """评估单条样本

        流程：
        1. 执行检索
        2. 执行生成
        3. RAGAS / 规则引擎计算四维指标
        4. 返回 {question, answer, contexts, ground_truth, scores: {...}}
        """
        question = sample.get("question", "")
        ground_truth = sample.get("ground_truth", "")

        # 检索
        retrieval_result = self.run_retrieval(question)
        contexts = retrieval_result["contexts"]

        # 生成
        answer = self.run_generation(question, contexts)

        result = {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        }

        if _RAGAS_AVAILABLE:
            result["scores"] = self._ragas_evaluate_single(
                question, answer, contexts, ground_truth
            )
        else:
            result["scores"] = self._rule_evaluate_single(
                question, answer, contexts, ground_truth
            )

        return result

    def _ragas_evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
    ) -> Dict[str, float]:
        """使用 RAGAS 评估单条样本"""
        import pandas as pd
        from datasets import Dataset

        sample_data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        if ground_truth:
            sample_data["ground_truth"] = [ground_truth]

        ds = Dataset.from_dict(sample_data)

        metrics = [faithfulness, answer_relevance, context_relevance, context_precision]

        try:
            llm = get_llm()
            result = ragas_evaluate(dataset=ds, metrics=metrics, llm=llm)
            df = result.to_pandas()
            scores = {}
            for col in df.columns:
                if col in ("question", "answer", "contexts", "ground_truth"):
                    continue
                val = df[col].iloc[0]
                scores[col] = float(val) if val is not None else 0.0
            return scores
        except Exception as e:
            logger.warning(f"RAGAS 评估失败，降级为规则评估: {e}")
            return self._rule_evaluate_single(question, answer, contexts, ground_truth)

    @staticmethod
    def _rule_evaluate_single(
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,  # noqa: ARG004
    ) -> Dict[str, float]:
        """使用规则引擎评估单条样本"""
        return {
            "rule_based_faithfulness": rule_based_faithfulness(answer, contexts),
            "rule_based_relevance": rule_based_relevance(question, answer),
            "rule_based_context_relevance": rule_based_context_relevance(question, contexts),
            "rule_based_context_precision": rule_based_context_precision(question, contexts),
        }

    # -----------------------------------------------------------------------
    # 批量评估
    # -----------------------------------------------------------------------

    def evaluate_batch(
        self,
        test_data: List[Dict],
        incremental: bool = True,
    ) -> Dict:
        """批量评估

        Args:
            test_data: 测试数据
            incremental: 增量评估，跳过已评估的样本（按 question 去重）

        Returns:
            {
                "version": str,
                "total": int,
                "evaluated": int,
                "skipped": int,
                "scores": {metric: avg_score},
                "per_sample": [...],
                "duration_seconds": float,
            }
        """
        start_time = time.time()
        version = time.strftime("%Y%m%d_%H%M%S")

        # 加载已有结果用于增量去重
        evaluated_questions: set = set()
        if incremental:
            evaluated_questions = self._load_evaluated_questions()

        per_sample: List[Dict] = []
        skipped = 0
        evaluated = 0

        for i, sample in enumerate(test_data):
            question = sample.get("question", "")
            if incremental and question in evaluated_questions:
                logger.info(f"跳过已评估样本 [{i + 1}]: {question[:40]}...")
                skipped += 1
                continue

            logger.info(f"评估样本 [{i + 1}/{len(test_data)}]: {question[:40]}...")
            try:
                result = self.evaluate_single(sample)
                per_sample.append(result)
                evaluated += 1
            except Exception as e:
                logger.error(f"评估样本失败 [{i + 1}]: {e}")
                per_sample.append({
                    "question": question,
                    "answer": "",
                    "contexts": [],
                    "ground_truth": sample.get("ground_truth", ""),
                    "scores": {},
                    "error": str(e),
                })
                evaluated += 1

        # 汇总平均分数
        avg_scores = self._compute_avg_scores(per_sample)
        duration = time.time() - start_time

        results = {
            "version": version,
            "total": len(test_data),
            "evaluated": evaluated,
            "skipped": skipped,
            "scores": avg_scores,
            "per_sample": per_sample,
            "duration_seconds": round(duration, 2),
        }

        logger.info(
            f"批量评估完成: total={len(test_data)}, evaluated={evaluated}, "
            f"skipped={skipped}, duration={duration:.1f}s"
        )
        return results

    # -----------------------------------------------------------------------
    # 版本对比
    # -----------------------------------------------------------------------

    def compare_versions(self, v1_path: str, v2_path: str) -> Dict:
        """对比两个版本的评估结果

        返回各指标的 delta 和改进方向
        """
        with open(v1_path, "r", encoding="utf-8") as f:
            v1 = json.load(f)
        with open(v2_path, "r", encoding="utf-8") as f:
            v2 = json.load(f)

        v1_scores = v1.get("scores", {})
        v2_scores = v2.get("scores", {})

        all_metrics = sorted(set(v1_scores) | set(v2_scores))
        comparison = []
        for metric in all_metrics:
            s1 = v1_scores.get(metric, 0.0)
            s2 = v2_scores.get(metric, 0.0)
            delta = round(s2 - s1, 4)
            direction = "improved" if delta > 0 else ("regressed" if delta < 0 else "unchanged")
            comparison.append({
                "metric": metric,
                "v1": s1,
                "v2": s2,
                "delta": delta,
                "direction": direction,
            })

        return {
            "v1_version": v1.get("version", "unknown"),
            "v2_version": v2.get("version", "unknown"),
            "comparison": comparison,
        }

    # -----------------------------------------------------------------------
    # 结果保存
    # -----------------------------------------------------------------------

    def save_results(self, results: Dict, version: str = None):
        """保存评估结果（版本化文件名）

        文件名格式: eval_v{timestamp}.json
        """
        version = version or results.get("version", time.strftime("%Y%m%d_%H%M%S"))
        filename = f"eval_v{version}.json"
        output_path = self.output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"评估结果已保存: {output_path}")

        # 打印摘要
        self._print_summary(results)

    # -----------------------------------------------------------------------
    # 内部工具方法
    # -----------------------------------------------------------------------

    def _load_evaluated_questions(self) -> set:
        """从已有评估结果中加载已评估的 question 集合"""
        questions: set = set()
        for path in self.output_dir.glob("eval_v*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for sample in data.get("per_sample", []):
                    q = sample.get("question", "")
                    if q:
                        questions.add(q)
            except Exception:
                continue
        if questions:
            logger.info(f"增量评估: 已发现 {len(questions)} 条历史评估记录")
        return questions

    @staticmethod
    def _compute_avg_scores(per_sample: List[Dict]) -> Dict[str, float]:
        """计算各指标的平均分"""
        metric_sums: Dict[str, float] = {}
        metric_counts: Dict[str, int] = {}

        for sample in per_sample:
            for metric, value in sample.get("scores", {}).items():
                metric_sums[metric] = metric_sums.get(metric, 0.0) + value
                metric_counts[metric] = metric_counts.get(metric, 0) + 1

        avg_scores = {}
        for metric, total in metric_sums.items():
            count = metric_counts[metric]
            avg_scores[metric] = round(total / count, 4) if count > 0 else 0.0

        return avg_scores

    @staticmethod
    def _print_summary(results: Dict):
        """打印评估结果摘要"""
        print("\n" + "=" * 60)
        print("RAG 评估结果摘要")
        print("=" * 60)

        scores = results.get("scores", {})
        for metric_name, score in scores.items():
            print(f"  {metric_name:40s}: {score:.4f}")

        print("-" * 60)
        print(f"  评估版本:   {results.get('version', 'N/A')}")
        print(f"  总样本数:   {results.get('total', 0)}")
        print(f"  已评估:     {results.get('evaluated', 0)}")
        print(f"  已跳过:     {results.get('skipped', 0)}")
        print(f"  耗时:       {results.get('duration_seconds', 0):.1f}s")
        print("=" * 60 + "\n")


# ===========================================================================
# 便捷工厂函数
# ===========================================================================

def get_evaluator(output_dir: str = "data/evaluation") -> RAGEvaluator:
    """获取 RAG 评估器实例"""
    return RAGEvaluator(output_dir=output_dir)
