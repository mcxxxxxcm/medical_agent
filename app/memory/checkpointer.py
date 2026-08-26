"""短期记忆检查点保存器模块
功能描述：
    使用langgraph的InMemorySaver管理短期对话记忆
    适用于单会话内的对话历史管理和状态回溯

设计理念：
    1、单例模式：全局共享一个检查点保存期实例
    2、线程安全：InMemorySaver本身是线程安全的
    3、自动管理：langgraph自动保存和恢复状态
    4、内存存储：数据存储在内存中，会话结束即清除
----------------------------------------------------------------------------------
功能描述：
    使用 SQLite/PostgreSQL 持久化存储对话检查点
    支持服务重启后恢复对话状态

缺点：
    基于内存的记忆存储，项目重启之后就没有了

改进：
    使用持久化的记忆存储，如PostgreSQL或SQLite

设计理念：
    1、持久化存储：数据存储在数据库中，服务重启不丢失
    2、异步支持：使用 AsyncSaver 适配 FastAPI 异步架构
    3、单例模式：全局共享一个检查点保存器实例
    4、自动管理：langgraph 自动保存和恢复状态

包引用：
    MemorySaver：langgraph提供的内存检查点保存期
"""
import asyncio
from typing import Optional
from langgraph.checkpoint.memory import InMemorySaver
from app.core.app_logging import get_logger

logger = get_logger(__name__)
_checkpointer = None
_checkpointer_context = None
_init_lock = asyncio.Lock()  # M17：防并发首建各自创建连接池，先建的 context 从不退出导致连接泄漏


async def get_checkpointer():
    """获取持久化检查点保存器（异步）
    使用 PostgreSQL 持久化存储对话检查点
        适合生产环境和分布式部署

    Returns：
        AsyncPostgresSaver：异步 PostgreSQL 检查点保存器实例
    """
    global _checkpointer, _checkpointer_context
    if _checkpointer is None:
        async with _init_lock:
            if _checkpointer is None:
                try:
                    _checkpointer = await _init_postgres_checkpointer()
                except Exception as e:
                    # PG 不可用时降级为内存检查点（与 Redis 降级策略对称），
                    # 保证对话服务不因数据库故障整体不可用；代价是重启后会话状态丢失
                    logger.error(
                        f"PostgreSQL检查点初始化失败，降级为内存检查点：{e}\n"
                        f"降级后对话状态仅在进程内存中保留，服务重启后历史会话将丢失。"
                    )
                    _checkpointer = InMemorySaver()
                    _checkpointer_context = None
    return _checkpointer


async def _init_postgres_checkpointer():
    """初始化 PostgreSQL 检查点（单独封装，便于降级分支复用）"""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from app.core.config import get_config

    global _checkpointer_context

    config = get_config()
    db_url = config.DATABASE_URL
    if not db_url:
        raise ValueError("DATABASE_URL未配置，无法使用PostgreSQL检查点保存器")
    logger.info(f"初始化PostgreSQL检查点保存器")

    _checkpointer_context = AsyncPostgresSaver.from_conn_string(db_url)
    cp = await _checkpointer_context.__aenter__()

    await cp.setup()
    logger.info(f"PostgreSQL检查点保存器初始化完成")
    return cp


async def close_checkpointer():
    """关闭检查点保存器链接
    使用场景：
        应用关闭时调用，确保资源正确释放
    """
    global _checkpointer, _checkpointer_context

    if _checkpointer_context is not None:
        try:
            await _checkpointer_context.__aexit__(None, None, None)
            logger.info(f"检查点保存器连接已关闭")
        except Exception as e:
            logger.warning(f"关闭检查点保存器时出错：{e}")

    _checkpointer = None
    _checkpointer_context = None


def reset_checkpointer() -> None:
    """重置检查点保存器（同步版本，用于测试）"""
    global _checkpointer, _checkpointer_context
    _checkpointer = None
    _checkpointer_context = None
    logger.info(f"检查点保存器已重置")

