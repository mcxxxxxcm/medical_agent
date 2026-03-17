"""Reranker 重排序模块 (ONNX轻量版)
优势：
    1、无需pytorch，docker镜像体积减小 ~10g
    2、ONNX Runtime推理速度更快
    3、内存占用低
"""
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from pyexpat import features

from app.core import get_config
from app.core.app_logging import get_logger

logger = get_logger(__name__)


# 默认模型路径逻辑
def get_default_model_path():
    """获取默认模型路径，支持 Docker 和本地环境"""
    config=get_config()
    # 优先使用环境变量
    env_path = config.RERANKER_MODEL_PATH
    if env_path and Path(env_path).exists():
        return env_path

    # 本地开发路径
    local_path = Path(__file__).parent.parent.parent / "bge-reranker-onnx"
    if local_path.exists():
        return str(local_path)

    # 降级：使用 HuggingFace 在线模型
    return "BAAI/bge-reranker-base-onnx-o3-cpu"


RERANKER_MODELS = {
    "bge-v2-m3": get_default_model_path(),
    "bce-base": "maidalun1020/bce-reranker-base_v1",
}
DEFAULT_MODEL = RERANKER_MODELS["bge-v2-m3"]


# 简单的中文清洗规则 (去除多余空白、特殊符号)
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    # 可选：去除过短的无意义片段
    if len(text) < 5:
        return ""
    return text


class Reranker:
    """ONNX 轻量级 Reranker
    使用ONNX Runtime进行推理，无需pytorch
    流程：Query + Doc -> Concat -> BERT -> Sigmoid -> Score
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        self.model_name = model_name
        self._device = device
        self._model = None
        self._tokenizer = None
        self._available = False
        # 延迟加载 (Lazy Load)，避免导入时阻塞
        self._load_model()

    def _load_model(self):
        """加载模型 (带自动降级)"""
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification
            from transformers import AutoTokenizer

            logger.info(f"正在加载 ONNX Reranker 模型：{self.model_name}")

            # 检查是否为本地路径
            model_path=Path(self.model_name)
            if model_path.exists() and model_path.is_dir():
                # 本地模型目录
                self._model = ORTModelForSequenceClassification.from_pretrained(
                    str(model_path),
                    export=False
                )
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            else:
                # HuggingFace 在线模型（自动下载 ONNX 版本）
                self._model = ORTModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    export=False
                )
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            self._available = True
            logger.info(f"ONNX Reranker模型加载成功")

        except ImportError as e:
            logger.error(f"缺少依赖库：{e}。请运行：pip install optimum[onnxruntime] transformers")
            self._available = False
        except Exception as e:
            logger.warning(f"⚠️ Reranker 加载失败：{e}。系统将降级为纯检索模式。")
            self._available = False

    def rerank(
            self,
            query: str,
            documents: List[Document],
            top_k: int = 5,
            score_threshold: float = 0.0
    ) -> List[Document]:
        """
        执行重排序

        Args:
            query: 用户查询
            documents: 待排序文档列表 (通常来自 RRF 融合后的 Top 20-50)
            top_k: 返回前 K 个文档
            score_threshold: 分数阈值 (低于此分的文档即使排名高也被丢弃，防止噪音)

        Returns:
            重排序后的文档列表
        """
        # 1. 边界检查与降级
        if not documents:
            return []

        if not self._available:
            logger.warning("Reranker 不可用，直接返回原始排序结果")
            return documents[:top_k]

        try:
            # 2. 数据预处理 (关键：清洗文本，提升模型效果)
            clean_query = clean_text(query)
            pairs = []
            valid_docs = []

            for doc in documents:
                clean_content = clean_text(doc.page_content)
                if clean_content:
                    pairs.append((clean_query, clean_content))
                    valid_docs.append(doc)

            if not pairs:
                return []

            # Tokenize
            features=self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            # ONNX 推理
            outputs = self._model(**features)
            scores = outputs.logits.squeeze(-1)

            # 转换为列表
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            if not isinstance(scores, list):
                scores = [scores]

            # 排序
            scored_items=list(zip(valid_docs, scores))
            scored_items.sort(key=lambda x: x[1], reverse=True)

            # 阈值过滤 与 截取top-k
            final_docs=[]
            for doc, score in scored_items:
                if score >= score_threshold and len(final_docs) < top_k:
                    doc.metadata["rerank_score"]=float(score)
                    final_docs.append(doc)
                elif len(final_docs) >= top_k:
                    break

            logger.info(f"重排序完成：{len(documents)} -> {len(final_docs)} （最高分：{scored_items[0][1]:.4f}")
            return final_docs

        except Exception as e:
            logger.error(f"❌ 重排序执行出错：{e}，降级返回原始结果")
            return documents[:top_k]


# === 单例管理 ===
_reranker_instance: Optional[Reranker] = None


def get_reranker(model_name: str = DEFAULT_MODEL) -> Reranker:
    """获取全局单例 Reranker"""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Reranker(model_name=model_name)
    return _reranker_instance
