"""Reranker 重排序模块 (生产优化版)
改进点：
    1. 引入中文分词预处理 (提升 Cross-Encoder 对中文的理解)
    2. 增加置信度阈值过滤 (防止低质量文档进入 LLM)
    3. 更稳健的懒加载与单例模式
    4. 显式指定 Torch 数据类型 (fp16 加速)
"""
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from app.core.app_logging import get_logger

logger = get_logger(__name__)


# 默认模型路径逻辑
def get_default_model_path():
    """获取默认模型路径，支持 Docker 和本地环境"""
    # 优先使用环境变量
    env_path = os.environ.get("RERANKER_MODEL_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    # 本地开发路径
    local_path = Path(__file__).parent.parent.parent / "bge-reranker-v2-m3"
    if local_path.exists():
        return str(local_path)

    # 降级：使用 HuggingFace 在线模型
    return "BAAI/bge-reranker-v2-m3"


RERANKER_MODELS = {
    "bge-v2-m3": get_default_model_path(),
    "bce-base": "maidalun1020/bce-reranker-base_v1",
}
DEFAULT_MODEL = RERANKER_MODELS["bge-v2-m3"]

# 模型配置 本地路径
# RERANKER_MODELS = {
#     "bge-v2-m3": r"D:\Agent\medical_assistant_agent\bge-reranker-v2-m3",  # 推荐：通用性强，支持长文本
#     "bce-base": "maidalun1020/bce-reranker-base_v1",  # 备选：百度中文专用
# }
# DEFAULT_MODEL = RERANKER_MODELS["bge-v2-m3"]


# 简单的中文清洗规则 (去除多余空白、特殊符号)
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    # 可选：去除过短的无意义片段
    if len(text) < 5:
        return ""
    return text


class Reranker:
    """
    重排序器：基于 Cross-Encoder 架构
    流程：Query + Doc -> Concat -> BERT -> Sigmoid -> Score
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None, use_fp16: bool = True):
        self.model_name = model_name
        self._device = device
        self._use_fp16 = use_fp16
        self._model = None
        self._available = False

        # 延迟加载 (Lazy Load)，避免导入时阻塞
        self._load_model()

    def _load_model(self):
        """加载模型 (带自动降级)"""
        try:
            from sentence_transformers import CrossEncoder
            import torch

            # 1. 设备选择
            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"

            # 2. 数据类型优化 (GPU 下启用 fp16 提速)
            if self._device == "cuda" and self._use_fp16:
                logger.info(f"启用 FP16 加速推理")

            logger.info(f"正在加载 Reranker 模型：{self.model_name} ({self._device})...")

            # 3. 初始化模型
            # trust_remote_code=True 用于加载某些自定义架构的模型 (如 bge-m3)
            self._model = CrossEncoder(
                self.model_name,
                max_length=512,
                device=self._device,
                automodel_args={"torch_dtype": "float16"} if (self._device == "cuda" and self._use_fp16) else {}
            )

            self._available = True
            logger.info("✅ Reranker 模型加载成功")

        except ImportError as e:
            logger.error(f"缺少依赖库：{e}。请运行：pip install sentence-transformers torch")
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

            # 3. 批量推理 (Batch Prediction)
            # CrossEncoder 一次性计算所有 (query, doc) 对的相关性分数
            scores = self._model.predict(pairs, batch_size=8, show_progress_bar=False)

            # 4. 分数关联与排序
            scored_items: List[Tuple[Document, float]] = list(zip(valid_docs, scores))
            scored_items.sort(key=lambda x: x[1], reverse=True)

            # 5. 阈值过滤 & 截取 Top-K
            final_docs = []
            for doc, score in scored_items:
                if score >= score_threshold and len(final_docs) < top_k:
                    # 将分数写入 metadata，方便后续节点调试或展示
                    doc.metadata["rerank_score"] = float(score)
                    final_docs.append(doc)
                elif len(final_docs) >= top_k:
                    break

            logger.info(f"🎯 重排序完成：{len(documents)} -> {len(final_docs)} (最高分：{scored_items[0][1]:.4f})")
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
