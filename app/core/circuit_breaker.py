"""外部服务熔断器（Circuit Breaker）

对 Embedding API / glm-4-flash / Ollama 等外部服务做熔断保护：
- 连续 N 次失败后进入 OPEN 状态（快速失败，不再尝试调用）
- M 秒后进入 HALF_OPEN 状态（放行一次探测请求）
- 探测成功 → CLOSED（正常调用）；探测失败 → OPEN（继续熔断）

用法：
    from app.core.circuit_breaker import get_circuit_breaker

    cb = get_circuit_breaker("embedding_api", failure_threshold=3, recovery_timeout=30)

    if cb.is_open:
        # 快速失败，走降级逻辑
        return fallback_embedding()
    else:
        try:
            result = call_embedding_api()
            cb.record_success()
            return result
        except Exception as e:
            cb.record_failure()
            raise
"""
import threading
import time
from enum import Enum
from typing import Dict, Optional

from app.core.app_logging import get_logger

logger = get_logger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"       # 正常：允许所有请求
    OPEN = "open"           # 熔断：快速失败，不允许请求
    HALF_OPEN = "half_open" # 半开：允许一个探测请求


class CircuitBreaker:
    """单个服务的熔断器"""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                # 检查是否可以进入半开
                if time.time() - self._last_failure_time >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    logger.info(f"熔断器 [{self.name}] 进入半开状态，允许探测请求")
            return self._state

    @property
    def is_open(self) -> bool:
        """是否处于熔断状态（OPEN or HALF_OPEN 不算 open，HALF_OPEN 允许探测）"""
        return self.state == CircuitState.OPEN

    def record_success(self):
        """记录成功调用"""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                logger.info(f"熔断器 [{self.name}] 探测成功，恢复为 CLOSED 状态")
            self._state = CircuitState.CLOSED
            self._failure_count = 0

    def record_failure(self):
        """记录失败调用"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # 半开探测失败 → 重新熔断
                self._state = CircuitState.OPEN
                logger.warning(
                    f"熔断器 [{self.name}] 探测失败，重新熔断 "
                    f"(recovery_timeout={self.recovery_timeout}s)"
                )
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"熔断器 [{self.name}] 进入 OPEN 状态 "
                    f"(连续失败 {self._failure_count} 次 >= {self.failure_threshold})"
                )

    def reset(self):
        """手动重置"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            logger.info(f"熔断器 [{self.name}] 手动重置为 CLOSED")

    def __repr__(self):
        return (
            f"CircuitBreaker(name={self.name}, state={self.state.value}, "
            f"failures={self._failure_count}/{self.failure_threshold})"
        )


class _CircuitBreakerRegistry:
    """熔断器注册表（单例）"""

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
    ) -> CircuitBreaker:
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    name=name,
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                )
            return self._breakers[name]

    def list_all(self) -> Dict[str, CircuitBreaker]:
        return dict(self._breakers)


_registry = _CircuitBreakerRegistry()


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 3,
    recovery_timeout: float = 30.0,
) -> CircuitBreaker:
    """获取指定服务的熔断器（自动创建）

    Args:
        name: 服务名称（如 "embedding_api", "glm4_flash", "ollama"）
        failure_threshold: 连续失败多少次后熔断（默认 3 次）
        recovery_timeout: 熔断后多少秒进入半开探测（默认 30s）
    """
    return _registry.get(name, failure_threshold, recovery_timeout)


def list_circuit_breakers() -> Dict[str, CircuitBreaker]:
    """列出所有熔断器状态"""
    return _registry.list_all()
