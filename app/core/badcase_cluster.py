"""Bad Case 聚类去重（L1 自动剔除队列的一部分）
__version__ = 9.54

目标：把"同一问题反复被差评"的 bad case 归并成簇，人工只审每簇的代表件，
其余标记为重复件（dup_of），避免对近似提问重复做医学判断。用 embedding 余弦
相似度归簇（阈值与语义缓存一致 0.92），复用 get_embeddings，整体带进程内
LRU + TTL 缓存，避免反复调用外部 Embedding API。

降级策略：embedding 不可用 / query 为空 / 解析失败一律静默跳过对应 case，
不让聚类失败影响 badcase 后台正常展示。
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Dict, List, Optional

from app.core.app_logging import get_logger

logger = get_logger(__name__)

# 归簇相似度阈值，与语义缓存 SEMANTIC_CACHE_THRESHOLD 对齐（余弦≥此值视为同主题）
DEFAULT_CLUSTER_THRESHOLD = 0.92
# 进程内向量缓存上限（避免对同一 query 反复打 Embedding API）
_VEC_CACHE_MAX = 512
# 聚类结果 TTL（秒）：坏例只在新增时变化，TTL 内多数时点可命中
_CLUSTER_CACHE_TTL = 1800


class _LRUCache:
    def __init__(self, maxsize: int):
        self._maxsize = maxsize
        self._data: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]
            return None

    def put(self, key, value):
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)


_vec_cache = _LRUCache(_VEC_CACHE_MAX)
# 聚类结果缓存：key = (签名, 阈值), value = (时间戳, cluster_map)
_cluster_result = {"sig": None, "threshold": None, "ts": 0.0, "map": None}
_cluster_lock = threading.Lock()


def _to_vec_key(q: str) -> str:
    return q.strip()


def get_query_vec(query: str) -> Optional[List[float]]:
    """取 query 的 embedding 向量（带 LRU 缓存），失败返回 None（调用方跳过）"""
    if not query or not query.strip():
        return None
    key = _to_vec_key(query)
    cached = _vec_cache.get(key)
    if cached is not None:
        return cached
    try:
        from app.core.embeddings import get_embeddings
        vec = get_embeddings().embed_query(key)
        if vec:
            _vec_cache.put(key, vec)
        return vec
    except Exception as e:
        logger.warning(f"获取 query embedding 失败，跳过聚类：{e}")
        return None


def _cosine_sim(a: List[float], b: List[float]) -> Optional[float]:
    if not a or not b or len(a) != len(b):
        return None
    try:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if not na or not nb:
            return None
        return dot / (na * nb)
    except (TypeError, OverflowError):
        return None


def _case_query(c: Dict) -> str:
    return (c.get("original_query") or c.get("final_question") or "").strip()


def build_clusters(cases: List[Dict], threshold: float = DEFAULT_CLUSTER_THRESHOLD) -> Dict[str, Dict]:
    """把 bad case 按 query 相似度归簇。

    Args:
        cases: bad case 列表（须含 case_id）
        threshold: 归簇阈值（余弦相似度）

    Returns:
        {代表件case_id: {"representative": 代表case_id, "members": [case_id...], "count": n}}
    """
    # 只对能取到向量的条目归簇；取不到向量的 case 不进簇（保持独立、必被审）
    clusters: List[Dict] = []  # {"repr", "vec", "members"}
    for c in cases:
        q = _case_query(c)
        if not q:
            continue
        cid = c.get("case_id")
        if not cid:
            continue
        vec = get_query_vec(q)
        if vec is None:
            continue
        placed = False
        for cl in clusters:
            sim = _cosine_sim(vec, cl["vec"])
            if sim is not None and sim >= threshold:
                cl["members"].append(cid)
                placed = True
                break
        if not placed:
            clusters.append({"repr": cid, "vec": vec, "members": [cid]})

    result: Dict[str, Dict] = {}
    for cl in clusters:
        result[cl["repr"]] = {
            "representative": cl["repr"],
            "members": cl["members"],
            "count": len(cl["members"]),
        }
    return result


def _cache_signature(cases: List[Dict]) -> str:
    # 签名 = 排序后的 case_id（去重）。case 集合不变则签名不变，可命中缓存。
    ids = sorted({c.get("case_id") for c in cases if c.get("case_id")})
    return "|".join(ids)


def get_cluster_map(cases: List[Dict], threshold: float = DEFAULT_CLUSTER_THRESHOLD) -> Dict[str, Dict]:
    """带 TTL 缓存地获取聚类结果（避免频繁 embedding 调用）。"""
    sig = _cache_signature(cases)
    now = time.time()
    with _cluster_lock:
        if (
            _cluster_result["sig"] == sig
            and _cluster_result["threshold"] == threshold
            and now - _cluster_result["ts"] < _CLUSTER_CACHE_TTL
            and _cluster_result["map"] is not None
        ):
            return _cluster_result["map"]
    # 未命中/过期 → 在锁外重算（重算慢，避免持锁阻塞），随后原子写入
    cmap = build_clusters(cases, threshold=threshold)
    with _cluster_lock:
        _cluster_result["sig"] = sig
        _cluster_result["threshold"] = threshold
        _cluster_result["ts"] = time.time()
        _cluster_result["map"] = cmap
    return cmap


# ===== 上层便捷：把聚类信息合入每条 bad case =====
def apply_clusters(cases: List[Dict], cluster_map: Optional[Dict[str, Dict]] = None) -> List[Dict]:
    """把 cluster_id / dup_of / rep_count 合入 case 副本，返回新的列表。

    规则：非代表件的 case → dup_of=代表件id、rep_count=簇成员数；
          代表件或未入簇 case → dup_of=None。
    cluster_map 不传时自动调用 get_cluster_map。
    """
    cmap = cluster_map if cluster_map is not None else get_cluster_map(cases)
    # 编号 case_id → 所属簇（all_members 索引建一次，避免外层逐条 scan）
    owner_by_id: Dict[str, Dict] = {}
    for meta in cmap.values():
        for m in meta["members"]:
            owner_by_id[m] = meta
    out: List[Dict] = []
    for c in cases:
        copy = dict(c)
        cid = c.get("case_id")
        owner = owner_by_id.get(cid)
        if owner is not None:
            copy["cluster_id"] = owner["representative"]
            copy["rep_count"] = owner["count"]
            copy["dup_of"] = None if owner["representative"] == cid else owner["representative"]
        else:
            copy["cluster_id"] = None
            copy["rep_count"] = 1
            copy["dup_of"] = None
        out.append(copy)
    return out


def count_duplicates(cases: List[Dict], cluster_map: Optional[Dict[str, Dict]] = None) -> int:
    """统计应被合并的重复件数量（簇成员数 - 1 之和），用于展示"去重后可审减少X条"。"""
    cmap = cluster_map if cluster_map is not None else get_cluster_map(cases)
    if not cmap:
        return 0
    return sum(max(0, (meta["count"] or 1) - 1) for meta in cmap.values())