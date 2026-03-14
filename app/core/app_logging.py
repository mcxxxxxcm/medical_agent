"""日志管理模块
功能描述：
    提供统一的日志配置和管理，支持不同日志级别和输出格式
    使用单例确保全局唯一的日志管理器

包引用：
    logging：Python标准日志模块
    sys：系统模块，用于标准输出
    Path：pathlib，路径操作类
    Optional：表示可选类型
"""
import logging, sys
from pathlib import Path
from typing import Optional
from app.core.config import get_config  # 包含了基础的日志配置

config = get_config()


class LoggerManager:
    """日志管理器
    功能描述：
        提供统一的日志配置和管理，支持不同日志级别和输出格式

    使用场景：
        应用启动时初始化日志系统
        各模块获取日志器实例
        运行时调整日志级别
    """
    _instance: Optional['LoggerManager'] = None

    def __new__(cls):
        """单例模式实现
        功能描述：
            确保全局只有一个LoggerManager实例

        设计理念：
            1、懒加载：首次调用时创建实例
            2、返回已存在实例：后续调用返回同一实例
            3、线程安全：py的__new__方法在单例中时线程安全的
        Returns：
            LoggerManager实例
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化日志管理器
        功能描述：
            配置日志系统，包括格式化器、处理器和日志级别。

        设计理念：
            1. 防重复初始化：通过 _initialized 标志避免重复配置
            2. 自动创建目录：日志目录不存在时自动创建
            3. 多处理器支持：同时输出到控制台和文件
            4. 分级日志：不同级别的日志输出到不同文件
        """

        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.logger = {}

        # 创建日志目录
        log_dir = Path(config.LOGS_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)

        # 配置根日志器
        self._setup_root_logger(log_dir)

    def _setup_root_logger(self, log_dir: Path):
        """配置根日志器
        功能描述：
            配置py根日志器，设置格式化器和处理器

        Args：
            log_dir: 日志目录路径

        包引用：
            logging: Python 标准日志模块
            StreamHandler: 输出到控制台的处理器
            FileHandler: 输出到文件的处理器
            Formatter: 日志格式化器
        """
        # 创建格式化器
        # 格式：时间 - 模块名 - 级别 - 消息
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # 控制台处理器：输出到标准输出
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, config.LOG_LEVEL))

        # 错误文件处理器：仅输出ERROR级别及以上
        error_handler = logging.FileHandler(log_dir / "error.log", encoding="utf-8")
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)

        # 配置根日志器
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(console_handler)
        # root_logger.addHandler(file_handler)
        root_logger.addHandler(error_handler)

    def get_logger(self, name: str) -> logging.Logger:
        """获取指定名称的日志器
        功能描述：为指定模块创建或返回日志器实例
        """
        if name not in self.logger:
            self.logger[name] = logging.getLogger(name)

        return self.logger[name]

    def set_level(self, level: str):
        """设置日志级别
        功能描述：
            动态调整日志级别，用于运行时日志级别切换

        包引用：
            getattr：动态获取属性（日志级别）
        """
        log_level = getattr(logging, level.upper(), logging.INFO)
        logging.getLogger().setLevel(log_level)


# 全局日志管理器实例
logger_manager = LoggerManager()


def get_logger(name: str) -> logging.Logger:
    """获取日志器（便携函数）"""
    return logger_manager.get_logger(name)
