"""RAGAS 评估模块
功能描述：
    使用 RAGAS 框架评估 RAG 系统性能
    支持多种评估指标：忠实度、相关性、上下文质量等

设计理念：
    1、模块化：独立评估模块，不影响主流程
    2、可配置：支持自定义评估指标
    3、批量评估：支持批量评估测试集
    4、结果持久化：保存评估结果到文件

评估指标：
    - Faithfulness（忠实度）：答案是否基于检索到的上下文
    - Answer Relevance（答案相关性）：答案是否与问题相关
    - Context Relevance（上下文相关性）：检索的上下文是否与问题相关
    - Context Precision（上下文精确度）：检索结果的精确性
    - Context Recall（上下文召回率）：检索结果的完整性
"""
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_core.documents import Document
from app.core.app_logging import get_logger
from app.rag.hybrid_retriever import get_hybrid_retriever
from app.graph.nodes import answer_generation_node

logger = get_logger(__name__)


class RAGEvaluator:
    """RAG 系统评估器
    
    功能：
        1. 准备评估数据集
        2. 执行 RAG 检索和生成
        3. 使用 RAGAS 计算评估指标
        4. 生成评估报告
    """

    def __init__(
            self,
            retriever=None,
            use_reranker: bool = True,
            k: int = 5,
            output_dir: str = "data/evaluation"
    ):
        """初始化评估器
        
        Args:
            retriever: 检索器实例
            use_reranker: 是否使用 Reranker
            k: 检索文档数量
            output_dir: 评估结果输出目录
        """
        self.retriever = retriever or get_hybrid_retriever(
            k=k,
            use_reranker=use_reranker
        )
        self.use_reranker = use_reranker
        self.k = k
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"RAGAS 评估器初始化完成")
        logger.info(f"  - 检索器: {'Reranker' if use_reranker else '基础检索'}")
        logger.info(f"  - 检索数量: {k}")
        logger.info(f"  - 输出目录: {output_dir}")

    def prepare_evaluation_dataset(
            self,
            test_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """准备评估数据集
        
        Args:
            test_data: 测试数据列表，每个元素包含：
                - question: 用户问题
                - ground_truth: 参考答案（可选）
                - contexts: 参考上下文（可选）
        
        Returns:
            格式化的评估数据集
        """
        dataset = []

        for item in test_data:
            question = item.get("question", "")
            ground_truth = item.get("ground_truth", "")

            # 执行 RAG 检索
            retrieved_docs = self.retriever.invoke(question)

            # 提取检索到的上下文
            contexts = [doc.page_content for doc in retrieved_docs]

            # 执行答案生成
            state = {
                "question": question,
                "retrieved_docs": retrieved_docs,
            }
            result = answer_generation_node(state)
            generated_answer = result.get("final_answer", "")

            # 构建评估样本
            sample = {
                "question": question,
                "answer": generated_answer,
                "contexts": contexts,
                "ground_truth": ground_truth if ground_truth else None,
            }

            dataset.append(sample)

        logger.info(f"准备评估数据集完成，样本数：{len(dataset)}")
        return dataset

    def evaluate(
            self,
            test_data: List[Dict[str, Any]],
            metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """执行 RAGAS 评估
        
        Args:
            test_data: 测试数据
            metrics: 评估指标列表，默认使用核心指标
                - faithfulness: 忠实度（答案是否基于上下文）
                - answer_correctness: 答案正确性（答案是否与 ground truth 一致）
                - context_relevancy: 上下文相关性（检索的上下文是否相关）
                - context_precision: 上下文精确度
                - context_recall: 上下文召回率
                - context_entity_recall: 上下文实体召回率
        
        Returns:
            评估结果字典
        """
        logger.info("开始 RAGAS 评估...")
        start_time = time.time()

        # 准备评估数据集
        dataset = self.prepare_evaluation_dataset(test_data)

        # 🔥 获取项目配置的 LLM，供 RAGAS 使用
        from langchain_openai import ChatOpenAI
        from app.core.config import get_config
        config = get_config()

        evaluator_llm = ChatOpenAI(
            api_key=config.MODEL_API_KEY,
            base_url=config.MODEL_URL,
            model="gpt-4o",
            temperature=0.0  # 评估时温度设为 0，保证结果稳定
        )

        # 转换为 RAGAS 格式（完全兼容版本）
        try:
            from ragas import evaluate

            # 🔥 动态导入可用指标，避免版本兼容问题
            available_metrics = {}

            # 尝试导入核心指标
            try:
                from ragas.metrics import faithfulness
                available_metrics["faithfulness"] = faithfulness
            except ImportError:
                logger.warning("faithfulness 指标不可用")

            try:
                from ragas.metrics import answer_correctness
                available_metrics["answer_correctness"] = answer_correctness
                available_metrics["answer_relevance"] = answer_correctness  # 别名
            except ImportError:
                try:
                    from ragas.metrics import answer_relevance
                    available_metrics["answer_relevance"] = answer_relevance
                    available_metrics["answer_correctness"] = answer_relevance  # 别名
                except ImportError:
                    logger.warning("answer_correctness/answer_relevance 指标不可用")

            try:
                from ragas.metrics import context_relevancy
                available_metrics["context_relevancy"] = context_relevancy
                available_metrics["context_relevance"] = context_relevancy  # 别名
            except ImportError:
                try:
                    from ragas.metrics import context_relevance
                    available_metrics["context_relevance"] = context_relevance
                    available_metrics["context_relevancy"] = context_relevance  # 别名
                except ImportError:
                    logger.warning("context_relevancy/context_relevance 指标不可用")

            try:
                from ragas.metrics import context_precision
                available_metrics["context_precision"] = context_precision
            except ImportError:
                logger.warning("context_precision 指标不可用")

            try:
                from ragas.metrics import context_recall
                available_metrics["context_recall"] = context_recall
            except ImportError:
                logger.warning("context_recall 指标不可用")

            # 选择评估指标
            if metrics is None:
                # 默认使用可用的核心指标
                evaluation_metrics = []
                if "faithfulness" in available_metrics:
                    evaluation_metrics.append(available_metrics["faithfulness"])
                if "answer_correctness" in available_metrics:
                    evaluation_metrics.append(available_metrics["answer_correctness"])
                if "context_relevancy" in available_metrics:
                    evaluation_metrics.append(available_metrics["context_relevancy"])

                if not evaluation_metrics:
                    logger.warning("无可用的评估指标")
                    raise ValueError("RAGAS 评估指标不可用，请检查安装")
            else:
                # 根据配置选择指标
                evaluation_metrics = []
                for m in metrics:
                    if m in available_metrics:
                        evaluation_metrics.append(available_metrics[m])
                    else:
                        logger.warning(f"未知指标或不可用：{m}，已跳过")

            if not evaluation_metrics:
                logger.warning("未选择有效指标，使用默认指标")
                if "faithfulness" in available_metrics:
                    evaluation_metrics.append(available_metrics["faithfulness"])
                if "answer_correctness" in available_metrics:
                    evaluation_metrics.append(available_metrics["answer_correctness"])

            logger.info(f"使用评估指标：{[m.name for m in evaluation_metrics]}")

            # 执行评估
            result = evaluate(
                dataset=dataset,
                metrics=evaluation_metrics,
                llm=evaluator_llm,
            )

            # 转换为字典
            result_dict = result.to_pandas().to_dict(orient='records')

            # 计算平均分数
            scores = {}
            for metric in evaluation_metrics:
                metric_name = metric.name
                scores[metric_name] = result_dict[0].get(metric_name, 0)

            elapsed_time = (time.time() - start_time) / 60
            logger.info(f"RAGAS 评估完成，耗时：{elapsed_time:.2f}分钟")

            return {
                "scores": scores,
                "details": result_dict,
                "dataset_size": len(dataset),
                "evaluation_time_minutes": elapsed_time,
            }

        except ImportError as e:
            logger.error(f"RAGAS 依赖未安装：{e}")
            logger.error("请运行：pip install ragas datasets")
            return {
                "error": str(e),
                "scores": {},
                "details": [],
            }
        except Exception as e:
            logger.error(f"RAGAS 评估失败：{e}")
            return {
                "error": str(e),
                "scores": {},
                "details": [],
            }

    def save_results(
            self,
            results: Dict[str, Any],
            filename: str = "evaluation_results.json"
    ):
        """保存评估结果
        
        Args:
            results: 评估结果
            filename: 输出文件名
        """
        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"评估结果已保存：{output_path}")

        # 打印摘要
        print("\n" + "=" * 60)
        print("RAGAS 评估结果摘要")
        print("=" * 60)

        scores = results.get("scores", {})
        for metric_name, score in scores.items():
            print(f"{metric_name:30s}: {score:.4f}")

        print("-" * 60)
        print(f"评估样本数: {results.get('dataset_size', 0)}")
        print(f"评估耗时: {results.get('evaluation_time_minutes', 0):.2f} 分钟")
        print("=" * 60 + "\n")


# 默认测试数据集
DEFAULT_TEST_DATA = [
    {
        "question": "高血压患者应该如何进行日常护理？",
        "ground_truth": "高血压患者应保持低盐饮食、定期监测血压、适量运动、按时服药、避免情绪激动。"
    },
    {
        "question": "感冒了应该吃什么药？",
        "ground_truth": "感冒用药应根据症状选择，如发热用退烧药，咳嗽用止咳药，但建议在医生指导下用药。"
    },
    {
        "question": "急性支气管炎的家庭护理方法有哪些？",
        "ground_truth": "急性支气管炎应多休息、多喝水、保持室内空气流通、避免烟雾刺激、必要时使用加湿器。"
    },
    {
        "question": "糖尿病的早期症状有哪些？",
        "ground_truth": "糖尿病早期症状包括多饮、多尿、多食、体重下降、疲劳、视力模糊等。"
    },
    {
        "question": "胃炎患者饮食应注意什么？",
        "ground_truth": "胃炎患者应少食多餐、避免辛辣刺激食物、戒烟戒酒、选择易消化的食物。"
    },
]


def get_evaluator(
        use_reranker: bool = True,
        k: int = 5,
        output_dir: str = "data/evaluation"
) -> RAGEvaluator:
    """获取 RAGAS 评估器单例
    
    Args:
        use_reranker: 是否使用 Reranker
        k: 检索文档数量
        output_dir: 评估结果输出目录
    
    Returns:
        RAGEvaluator 实例
    """
    return RAGEvaluator(
        use_reranker=use_reranker,
        k=k,
        output_dir=output_dir
    )
