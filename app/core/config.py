# app/core/config.py
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional

# 先确定项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """应用统一配置"""

    # ===== 应用配置 =====
    APP_NAME: str = "Medical Assistant"
    DEBUG: bool = False
    # ===== 安全配置 =====
    SECRET_KEY: str = "dev-secret-key-change-in-production"
    ADMIN_API_KEY: str = "admin-api-key-change-in-production"  # 缓存管理等敏感接口的认证密钥
    CORS_ORIGINS: str = ""  # 允许的跨域来源，逗号分隔；留空则允许所有（开发模式）
    RATE_LIMIT_PER_MINUTE: int = 20  # 每分钟最大请求数（仅限 /api/chat 接口）
    HOST: str = "127.0.0.1"  # 默认仅本地访问，生产环境通过环境变量设置为 0.0.0.0
    PORT: int = 8000

    # ===== LLM 配置 =====
    MODEL_NAME: str = "glm-4-flash"     # 主模型（RAG答案生成），首token 1-3s
    # MODEL_NAME: str = "glm-4.5-air"   # 备选：推理更强但首token 10-20s
    # MODEL_NAME: str = "glm-4"         # 备选：平衡型
    MODEL_URL: Optional[str] = None
    MODEL_API_KEY: Optional[str] = None
    MODEL_TEMPERATURE: float = 0.2
    REWRITE_MODEL_NAME: Optional[str] = None  # 查询重写专用模型（留空则使用MODEL_NAME）
    SYMPTOM_MODEL_NAME: str = "glm-4-flash"  # 症状解析专用模型（快速轻量）
    VISION_MODEL_NAME: str = "glm-4v-plus"  # 多模态视觉模型（图片问诊）

    # ===== 本地模型配置（Ollama） =====
    # 中间节点（查询重写/档案提取/快照更新）使用本地模型，降低延迟
    LOCAL_MODEL_NAME: str = "qwen2.5:1.5b"      # Ollama 模型名（1.5b 适配 4GB VRAM，纯GPU推理）
    LOCAL_MODEL_URL: str = "http://localhost:11434/v1"  # Ollama 默认地址
    LOCAL_MODEL_API_KEY: str = "ollama"         # Ollama 不需要真实 key，填任意值
    LOCAL_MODEL_ENABLED: bool = False            # 是否启用本地模型（需先安装 Ollama 并拉取模型）

    # ===== Embedding 配置 =====
    # EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_MODEL: str = "embedding-3"
    EMBEDDING_DIMENSION: int = 2048

    # ===== 数据库配置 =====
    DATABASE_URL: Optional[str] = None

    # ===== RAG 配置 =====
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    DEFAULT_K: int = 5
    DEFAULT_SEARCH_TYPE: str = "similarity"
    RERANKER_THRESHOLD: float = 0.1  # sigmoid归一化后的阈值，仅过滤极低分文档，排序交给Reranker，过滤交给下游

    # ===== 路径配置 =====
    DOCS_DIR: Path = PROJECT_ROOT / "docs" / "medical"
    DATA_DIR: Path = PROJECT_ROOT / "data"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    PERSIST_DIRECTORY: Path = DATA_DIR / "chroma_db"
    BM25_CACHE_PATH: Path = DATA_DIR / "bm25_index.pkl"

    # ===== 请求限制 =====
    MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024  # 16MB
    MAX_QUESTION_LENGTH: int = 1000

    # ===== 日志配置 =====
    LOG_LEVEL: str = "INFO"

    # ===== 上下文窗口管理（三层架构） =====
    # L1 永久层：Profile（PostgresStore，跨会话）- 姓名、年龄、过敏史
    # L2 会话层：Clinical Snapshot（Checkpointer State，单会话）- 症状、用药、诊断
    # L3 短期窗口：Messages（滑动窗口，最近3轮）
    KEEP_RECENT_MESSAGES: int = 6  # 保留最近的消息数量（3轮=6条）
    SNAPSHOT_TRIGGER: int = 8      # 触发快照更新的消息数量阈值（4轮=8条）

    # ===== 性能优化 =====
    ENABLE_SAFETY_CHECK: bool = False  # 是否启用安全检查（关闭可节省1次LLM调用）
    MERGE_ROUTER_REWRITE: bool = True  # 是否合并router和query_rewrite

    # ===== 缓存配置 =====
    REDIS_URL: str = "redis://localhost:6379/0"  # Redis 连接 URL
    ENABLE_QUERY_CACHE: bool = True  # 是否启用查询缓存
    CACHE_TTL_SECONDS: int = 3600  # 缓存过期时间（秒）
    CACHE_MAX_SIZE: int = 10000  # 最大缓存条目数

    # ===== 语义缓存配置 =====
    ENABLE_SEMANTIC_CACHE: bool = True  # 是否启用语义相似缓存
    SEMANTIC_CACHE_THRESHOLD: float = 0.92  # 语义相似度阈值（医疗场景要求高精度，0.92 = 92% 相似）

    # ===== RERANKER_MODEL本地路径 =====
    # Docker 默认路径为 /app/models/...，本地开发可通过环境变量或 .env 覆盖
    RERANKER_MODEL_PATH: str = str(PROJECT_ROOT / "bge-reranker-onnx")

    class Config:
        extra = "ignore"
        # 使用绝对路径，确保无论从哪个目录运行都能找到 .env 文件
        env_file = str(ENV_FILE_PATH)
        env_file_encoding = "utf-8"


# 全局配置实例
settings = Settings()


def get_config() -> Settings:
    """获取配置实例"""
    return settings


def reload_config() -> dict:
    """热更新配置：重新读取 .env 文件并更新全局 settings。

    Returns:
        dict: {"changed": [...], "reloaded": True/False, "error": "..."}
    """
    import copy
    global settings
    old_values = {k: getattr(settings, k, None) for k in settings.model_fields.keys()}
    try:
        new_settings = Settings()
        settings = new_settings
        changed = [
            k for k in old_values
            if str(old_values[k]) != str(getattr(new_settings, k, None))
        ]
        return {"changed": changed, "reloaded": True}
    except Exception as e:
        settings = Settings.model_validate(old_values)
        return {"changed": [], "reloaded": False, "error": str(e)}
