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
    SECRET_KEY: str = "dev-secret-key-change-in-production"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ===== LLM 配置 =====
    MODEL_NAME: str = "gpt-4-turbo"
    MODEL_URL: Optional[str] = None
    MODEL_API_KEY: Optional[str] = None
    MODEL_TEMPERATURE: float = 0.2

    # ===== Embedding 配置 =====
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536

    # ===== 数据库配置 =====
    DATABASE_URL: Optional[str] = None

    # ===== RAG 配置 =====
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    DEFAULT_K: int = 5
    DEFAULT_SEARCH_TYPE: str = "similarity"
    RERANKER_THRESHOLD: float = 0.3

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

    # ===== 上下文窗口管理 =====
    MAX_MESSAGES: int = 20  # 最大消息数量
    KEEP_RECENT_MESSAGES: int = 6  # 保留最近的消息数量
    SUMMARY_TRIGGER: int = 14  # 触发总结的消息数量阈值

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
    SEMANTIC_CACHE_THRESHOLD: float = 0.80  # 语义相似度阈值（0.92 = 92% 相似）

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
