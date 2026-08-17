"""自适应阈值模块

v9.2 漏洞4修复：硬编码阈值随数据分布漂移失效

设计理念：
    1. 基于运行时百分位统计动态调整阈值，而非固定值
    2. 冷启动期间使用默认值，积累足够样本后自动校准
    3. 校准结果持久化到 SQLite（metrics.db），重启后恢复
    4. 提供管理员接口手动触发校准

阈值清单：
    - HIGH_CONFIDENCE_THRESHOLD: Dense Top-1 距离 < 此值 → 跳过 Reranker
      风险：知识库扩张后向量空间变密集，0.08 从"极度相似"退化为"有点相关"
    - RERANKER_THRESHOLD: Reranker 评分 < 此值 → 过滤文档
      风险：新领域文档评分普遍偏低，0.02 过于严格可能过滤掉所有文档
    - SEMANTIC_CACHE_THRESHOLD: 语义缓存相似度阈值
      风险：0.92 在密集空间下可能过于宽松
"""
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from app.core.app_logging import get_logger
from app.core.config import get_config

logger = get_logger(__name__)


class AdaptiveThreshold:
    """自适应阈值管理器

    工作原理：
        - 收集运行时观察值（如 dense_top1_distance、reranker_score）
        - 计算百分位数（P5/P10/P25/P50/P75/P90/P95）
        - 将固定阈值替换为基于百分位数的自适应值
        - 例如：HIGH_CONFIDENCE_THRESHOLD 从固定 0.08 → P5 of dense_top1_distance
          含义：当 Top-1 距离优于历史 95% 的查询时，跳过 Reranker

    冷启动：
        - 前 100 个样本使用默认值
        - 之后自动切换为自适应值
        - 每 1000 个样本重新校准一次
    """

    # 最小样本量（低于此数使用默认值）
    MIN_SAMPLES = 100
    # 重新校准间隔
    RECALIBRATE_INTERVAL = 1000
    # 持久化路径
    DB_PATH = None  # 延迟初始化

    def __init__(self):
        self._observations: Dict[str, list] = {}
        self._calibrated: Dict[str, float] = {}
        self._default_values: Dict[str, float] = {}
        self._strategies: Dict[str, dict] = {}
        self._sample_counts: Dict[str, int] = {}
        self._last_recalibrate: Dict[str, int] = {}
        self._db_conn: Optional[sqlite3.Connection] = None

    def register(
            self,
            name: str,
            default: float,
            strategy: str = "percentile",
            percentile: float = 5.0,
            min_value: float = 0.0,
            max_value: float = 1.0,
    ):
        """注册一个自适应阈值

        Args:
            name: 阈值名称
            default: 默认值（冷启动期间使用）
            strategy: 校准策略
                - "percentile": 取观察值的指定百分位数
                - "mean_std": 均值 - N*标准差
            percentile: 百分位数（仅 strategy="percentile" 时有效）
            min_value: 阈值下限
            max_value: 阈值上限
        """
        self._default_values[name] = default
        self._observations[name] = []
        self._sample_counts[name] = 0
        self._last_recalibrate[name] = 0
        self._strategies[name] = {
            "strategy": strategy,
            "percentile": percentile,
            "min_value": min_value,
            "max_value": max_value,
        }

        # 尝试从持久化存储恢复校准值
        calibrated = self._load_calibrated(name)
        if calibrated is not None:
            self._calibrated[name] = calibrated
            logger.info(f"自适应阈值 [{name}] 从持久化恢复：{calibrated:.6f}")
        else:
            logger.info(f"自适应阈值 [{name}] 注册：默认值={default}, 策略={strategy}, 百分位={percentile}")

    def observe(self, name: str, value: float):
        """记录一个观察值

        Args:
            name: 阈值名称
            value: 观察值（如 dense_top1_distance、reranker_score）
        """
        if name not in self._default_values:
            return

        self._observations[name].append(value)
        self._sample_counts[name] = self._sample_counts.get(name, 0) + 1

        # 定期校准
        # M16 修复：首个校准点被 RECALIBRATE_INTERVAL=1000 门控，MIN_SAMPLES=100
        # 形同虚设，冷启动期固定阈值长期不校准。
        # 改为：从未校准（last==0）时达到 MIN_SAMPLES 即校准，此后每间隔 RECALIBRATE_INTERVAL 再校准。
        count = self._sample_counts[name]
        last = self._last_recalibrate.get(name, 0)
        if count >= self.MIN_SAMPLES and (last == 0 or count - last >= self.RECALIBRATE_INTERVAL):
            self._recalibrate(name)

    def get(self, name: str) -> float:
        """获取当前阈值

        Args:
            name: 阈值名称

        Returns:
            如果样本量足够且已校准，返回自适应值；否则返回默认值
        """
        if name not in self._default_values:
            raise KeyError(f"未注册的阈值：{name}")

        if name in self._calibrated:
            return self._calibrated[name]

        return self._default_values[name]

    def force_recalibrate(self, name: Optional[str] = None) -> Dict[str, float]:
        """强制重新校准（管理员接口）

        Args:
            name: 阈值名称（None=全部重新校准）

        Returns:
            {name: new_value}
        """
        names = [name] if name else list(self._default_values.keys())
        results = {}
        for n in names:
            if self._observations.get(n):
                self._recalibrate(n, force=True)
                results[n] = self._calibrated.get(n, self._default_values[n])
            else:
                results[n] = self._default_values.get(n, 0.0)
        return results

    def get_stats(self) -> Dict[str, dict]:
        """获取所有阈值的统计信息"""
        stats = {}
        for name in self._default_values:
            obs = self._observations.get(name, [])
            strat = self._strategies.get(name, {})

            stat = {
                "default": self._default_values[name],
                "current": self.get(name),
                "samples": self._sample_counts.get(name, 0),
                "is_calibrated": name in self._calibrated,
                "strategy": strat.get("strategy", "percentile"),
                "percentile": strat.get("percentile", 5.0),
            }

            if obs:
                sorted_obs = sorted(obs)
                stat["observed_min"] = sorted_obs[0]
                stat["observed_max"] = sorted_obs[-1]
                stat["observed_mean"] = sum(obs) / len(obs)
                # 简单百分位计算
                p5_idx = max(0, int(len(sorted_obs) * 0.05))
                p10_idx = max(0, int(len(sorted_obs) * 0.10))
                p50_idx = max(0, int(len(sorted_obs) * 0.50))
                p90_idx = min(len(sorted_obs) - 1, int(len(sorted_obs) * 0.90))
                stat["p5"] = sorted_obs[p5_idx]
                stat["p10"] = sorted_obs[p10_idx]
                stat["p50"] = sorted_obs[p50_idx]
                stat["p90"] = sorted_obs[p90_idx]

            stats[name] = stat
        return stats

    def _recalibrate(self, name: str, force: bool = False):
        """基于观察值重新校准阈值"""
        obs = self._observations.get(name, [])
        if len(obs) < self.MIN_SAMPLES and not force:
            return

        strat = self._strategies.get(name, {})
        strategy = strat.get("strategy", "percentile")

        sorted_obs = sorted(obs)
        new_value = self._default_values[name]

        if strategy == "percentile":
            percentile = strat.get("percentile", 5.0)
            idx = max(0, min(len(sorted_obs) - 1, int(len(sorted_obs) * percentile / 100)))
            new_value = sorted_obs[idx]
        elif strategy == "mean_std":
            # 均值 - 1*标准差
            mean = sum(obs) / len(obs)
            variance = sum((x - mean) ** 2 for x in obs) / len(obs)
            std = variance ** 0.5
            new_value = mean - std

        # 应用上下限
        new_value = max(strat.get("min_value", 0.0), min(strat.get("max_value", 1.0), new_value))

        old_value = self._calibrated.get(name, self._default_values[name])
        self._calibrated[name] = new_value
        self._last_recalibrate[name] = self._sample_counts.get(name, 0)

        # 持久化校准值
        self._save_calibrated(name, new_value)

        if abs(new_value - old_value) > 0.001:
            logger.info(
                f"自适应阈值 [{name}] 校准：{old_value:.6f} → {new_value:.6f} "
                f"(样本量={len(obs)}, 策略={strategy})"
            )

        # 清理观察值内存（保留最近 5000 个）
        if len(obs) > 5000:
            self._observations[name] = obs[-5000:]

    def _get_db_path(self) -> Path:
        """获取持久化路径"""
        if self.DB_PATH is None:
            config = get_config()
            self.DB_PATH = config.DATA_DIR / "adaptive_thresholds.db"
        return self.DB_PATH

    def _get_db(self) -> sqlite3.Connection:
        """获取 SQLite 连接"""
        if self._db_conn is None:
            db_path = self._get_db_path()
            self._db_conn = sqlite3.connect(str(db_path))
            self._db_conn.execute("""
                CREATE TABLE IF NOT EXISTS calibrated_thresholds (
                    name TEXT PRIMARY KEY,
                    value REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            self._db_conn.commit()
        return self._db_conn

    def _save_calibrated(self, name: str, value: float):
        """持久化校准值"""
        try:
            db = self._get_db()
            db.execute(
                "INSERT OR REPLACE INTO calibrated_thresholds (name, value, updated_at) VALUES (?, ?, ?)",
                (name, value, time.time())
            )
            db.commit()
        except Exception as e:
            logger.warning(f"持久化阈值 [{name}] 失败：{e}")

    def _load_calibrated(self, name: str) -> Optional[float]:
        """从持久化存储加载校准值"""
        try:
            db = self._get_db()
            cursor = db.execute(
                "SELECT value FROM calibrated_thresholds WHERE name = ?",
                (name,)
            )
            row = cursor.fetchone()
            return row[0] if row else None
        except Exception:
            return None

    def close(self):
        """关闭数据库连接"""
        if self._db_conn:
            try:
                self._db_conn.close()
            except Exception:
                pass
            self._db_conn = None


# ===== 全局单例 =====
_adaptive_threshold: Optional[AdaptiveThreshold] = None
_threshold_lock = threading.Lock()


def get_adaptive_threshold() -> AdaptiveThreshold:
    """获取自适应阈值管理器单例（线程安全）"""
    global _adaptive_threshold
    if _adaptive_threshold is None:
        with _threshold_lock:
            # 双重检查锁定
            if _adaptive_threshold is None:
                _adaptive_threshold = AdaptiveThreshold()

                # 注册系统阈值
                # HIGH_CONFIDENCE_THRESHOLD: Dense Top-1 distance 的 P5
                # 含义：Top-1 距离优于历史 95% 查询时，跳过 Reranker
                _adaptive_threshold.register(
                    name="HIGH_CONFIDENCE_THRESHOLD",
                    default=0.08,
                    strategy="percentile",
                    percentile=5.0,
                    min_value=0.01,  # 下限：至少 0.01，避免过度跳过 Reranker
                    max_value=0.20,  # 上限：0.2 以上无论多密集都不应跳过
                )

                # RERANKER_THRESHOLD: Reranker 评分的 P5
                # 含义：评分低于历史 95% 文档的分数才被过滤
                # P2-3 修复：默认值从 config 读取（v9.16 已把 0.02→0.005），
                # 硬编码 0.02 会让 config 的 RERANKER_THRESHOLD 形同虚设
                _adaptive_threshold.register(
                    name="RERANKER_THRESHOLD",
                    default=get_config().RERANKER_THRESHOLD,
                    strategy="percentile",
                    percentile=5.0,
                    min_value=0.005,
                    max_value=0.10,
                )

                # SEMANTIC_CACHE_THRESHOLD: 语义缓存相似度的 P95
                # 含义：相似度高于历史 95% 的查询对才命中缓存
                _adaptive_threshold.register(
                    name="SEMANTIC_CACHE_THRESHOLD",
                    default=0.92,
                    strategy="percentile",
                    percentile=95.0,
                    min_value=0.85,
                    max_value=0.99,
                )

    return _adaptive_threshold
