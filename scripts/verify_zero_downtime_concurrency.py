"""零停机重建 —— 真栈并发验证（真实 Chroma + 隔离临时目录）

验证不变量：
  C1  构建影子集合期间，在线查询线程全程可服务（零中断、有结果）
  C2  共享单例 _vector_store_manager 在"重建置 None"与"在线 get"并发下不崩溃
  C3  校验失败：丢弃影子、活跃指针不变、在线线程不中断、仍命中旧集合
  C4  切换成功后，在线线程自然过渡到新集合（命中新内容）
  C5  全程所有在线查询无异常（ok=True）

隔离：所有 Chroma 文件与 kb_active.json 均写入 data/chroma_db_zdtest，
      运行结束自动清理，不触碰生产 data/chroma_db 与 data/kb_active.json。
"""
import sys
import time
import shutil
import threading
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from langchain_core.documents import Document

# ===== 隔离环境 =====
_ZD_TMP = project_root / "data" / "chroma_db_zdtest"

import app.rag.vector_store as vsmod

_orig_config = vsmod.config
_orig_kb_active_path = vsmod._KB_ACTIVE_CONFIG_PATH
_orig_vsm = vsmod._vector_store_manager
_orig_dcm = vsmod._dual_collection_manager

import types
_mock = types.SimpleNamespace(
    DATA_DIR=_ZD_TMP,
    PERSIST_DIRECTORY=_ZD_TMP / "chroma_db",
    BM25_CACHE_PATH=_ZD_TMP / "bm25_index.pkl",
)
vsmod.config = _mock
vsmod._KB_ACTIVE_CONFIG_PATH = _ZD_TMP / "kb_active.json"
vsmod._vector_store_manager = None
vsmod._dual_collection_manager = None

from app.rag.vector_store import (
    get_vector_store,
    get_dual_collection_manager,
    get_vector_store_manager,
)
from app.rag.vector_store import _DEFAULT_COLLECTION_NAME

dcm = get_dual_collection_manager()
dcm.config_path = vsmod._KB_ACTIVE_CONFIG_PATH
dcm.chroma_base_dir = _ZD_TMP / "chroma_db"

# ===== 集合内容标识 =====
OLD_TAG = "OLD_BASE"
NEW_TAG = "NEW_AFTER_REBUILD"

BASE_DOCS = [
    Document(page_content="布洛芬 OLD_BASE 成人每次200-400mg 饭后服用。",
             metadata={"source": "旧药典.txt", "h2": "布洛芬"}),
    Document(page_content="发烧护理 OLD_BASE 减少衣物散热 温水擦浴。",
             metadata={"source": "旧护理.txt", "h2": "发热"}),
    Document(page_content="高血压注意事项 OLD_BASE 低盐饮食 规律作息。",
             metadata={"source": "旧慢病.txt", "h2": "高血压"}),
]

NEW_DOCS = [
    Document(page_content="布洛芬 NEW_AFTER_REBUILD 成人每次200-400mg 每日3-4次 饭后服用 新版条目。",
             metadata={"source": "新药典.txt", "h2": "布洛芬"}),
    Document(page_content="发烧护理 NEW_AFTER_REBUILD 减少衣物散热 温水擦浴 多饮温水 新版条目。",
             metadata={"source": "新护理.txt", "h2": "发热"}),
    Document(page_content="高血压注意事项 NEW_AFTER_REBUILD 低盐低脂 规律作息 新版条目。",
             metadata={"source": "新慢病.txt", "h2": "高血压"}),
    Document(page_content="咳嗽处理 NEW_AFTER_REBUILD 多喝水 拍背排痰 新内容文档。",
             metadata={"source": "新呼吸.txt", "h2": "咳嗽"}),
]

failures = []
def check(label, ok, detail=""):
    print(f"  {'[PASS]' if ok else '[FAIL]'} {label}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(label)

def _classify(content: str) -> str:
    if NEW_TAG in content:
        return "NEW"
    if OLD_TAG in content:
        return "OLD"
    return "EMPTY"

# ===== 在线查询线程 =====
class OnlineProbe:
    def __init__(self):
        self.results = []   # list of {ok, tag, rc, name, err}
        self.lock = threading.Lock()
        self.stop = False

    def sample(self):
        try:
            vs = get_vector_store()
            name = vs._collection.name
            hits = vs.similarity_search("布洛芬", k=2)
            content = " ".join(h.page_content for h in hits)
            tag = _classify(content)
            with self.lock:
                self.results.append({"ok": True, "tag": tag, "rc": len(hits), "name": name})
        except Exception as e:
            with self.lock:
                self.results.append({"ok": False, "tag": "EXC", "rc": 0, "name": "?", "err": str(e)})

    def run(self):
        iters = 0
        while not self.stop and iters < 400:
            self.sample()
            time.sleep(0.08)
            iters += 1

    def snapshot(self):
        with self.lock:
            return list(self.results)

def _reset():
    vsmod._vector_store_manager = None
    vsmod._kb_version_cache = None

def main():
    print("=" * 72)
    print("零停机重建真栈并发验证（隔离临时目录）")
    print("=" * 72)
    _ZD_TMP.mkdir(parents=True, exist_ok=True)
    print(f"隔离临时目录: {_ZD_TMP}")

    try:
        # ---- 1. 建基线集合（默认 langchain 集合）----
        print("\n[1] 建基线集合（旧内容，模拟线上）")
        VectorStoreManager = vsmod.VectorStoreManager
        base_mgr = VectorStoreManager(persist_directory=_ZD_TMP / "chroma_db")
        base_mgr.create_vector_store(BASE_DOCS, force_rebuild=True)
        print(f"  基线 default 集合：{base_mgr.vector_store._collection.count()} chunks")
        _reset()

        # 指针应为默认哨兵
        active0 = dcm.get_active_collection_name()
        check("基线 active=默认哨兵", active0 == _DEFAULT_COLLECTION_NAME, active0)

        # ---- 在线线程启动 ----
        probe = OnlineProbe()
        t_online = threading.Thread(target=probe.run, daemon=True)
        t_online.start()
        time.sleep(0.4)   # 让在线线程稳定命中旧集
        _reset()

        # ---- 2. 校验失败不中断（空影子 → valid=False）----
        print("\n[2] 校验失败：空影子 → 应丢弃、活跃不变、在线不中断")
        shadow_f = dcm.create_shadow_collection()
        fn, fd = shadow_f
        # 空文档集模拟构建异常/必定校验失败
        shadow_vs_f = dcm.build_shadow_collection([], fn, fd)
        val_f = dcm.validate_shadow_collection(shadow_vs_f, expected_chunk_count=5,
                                               sample_queries=["布洛芬", "发烧", "感冒"])
        print(f"  校验结果 valid={val_f['valid']} errors={val_f['errors']}")
        check("C3 空影子校验失败(valid=False)", not val_f["valid"])
        store_hit_f = get_vector_store()
        check("C3 失败后未切换，仍命中旧内容(OLD)",
              OLD_TAG in " ".join(h.page_content for h in store_hit_f.similarity_search("布洛芬", k=1)))
        # 丢弃影子目录
        shutil.rmtree(fd, ignore_errors=True)
        _reset()

        # ---- 3. 真重建：影子构建→校验→原子切换（并发在线）----
        print("\n[3] 真重建并发：新内容影子 → 校验通过 → 原子切换")
        shadow_name, shadow_dir = dcm.create_shadow_collection()
        print(f"  影子: {shadow_name}")

        def _slow_build():
            # 分批写入并拉长窗口（睡眠），让在线线程充分观察到构建期
            shadow_vs = None
            from app.core.embeddings import get_embeddings
            from langchain_chroma import Chroma
            emb = get_embeddings()
            bs = 2
            for i in range(0, len(NEW_DOCS), bs):
                batch = NEW_DOCS[i:i + bs]
                if shadow_vs is None:
                    shadow_vs = Chroma.from_documents(
                        documents=batch, embedding=emb, collection_name=shadow_name,
                        persist_directory=str(shadow_dir),
                        collection_metadata={"hnsw:space": "cosine"})
                else:
                    shadow_vs.add_documents(batch)
                time.sleep(0.3)
            return shadow_vs

        shadow_vs = _slow_build()
        build_count = shadow_vs._collection.count()
        check("影子集合写入全部新文档", build_count == len(NEW_DOCS), f"{build_count}")

        # 构建期间在线线程应处于 OLD（全在切换前）
        before_switch = probe.snapshot()
        check("C1 构建期间在线全程零异常",
              all(r["ok"] for r in before_switch),
              f"{sum(1 for r in before_switch if not r['ok'])} errs")
        check("C1 构建期间在线命中旧内容(OLD)",
              any(r["ok"] and r["tag"] == "OLD" for r in before_switch),
              f"tags={[r['tag'] for r in before_switch]}")

        val_ok = dcm.validate_shadow_collection(shadow_vs, build_count,
                                                sample_queries=["布洛芬", "发烧", "咳嗽"])
        check("影子校验通过", val_ok["valid"], f"errors={val_ok['errors']}")

        # 原子切换（在线线程继续并发采样 → 被测 C2 单例竞态）
        print("  原子切换中（在线线程并发采样）...")
        dcm.switch_active_collection(shadow_name, str(shadow_dir))
        time.sleep(0.6)   # 给在线线程过渡时间
        cfg = dcm.get_active_config()
        check("C4 切换后指针指向新影子", cfg.get("active_collection") == shadow_name)
        _reset()

        # ---- 4. 汇总分析 ----
        probe.stop = True
        t_online.join(timeout=5)
        results = probe.snapshot()
        print(f"\n[4] 在线线程采样统计")
        print(f"  总采样: {len(results)}")
        errs = [r for r in results if not r["ok"] or r["tag"] == "EXC"]
        old_hits = [r for r in results if r["ok"] and r["tag"] == "OLD"]
        new_hits = [r for r in results if r["ok"] and r["tag"] == "NEW"]
        empties = [r for r in results if r["ok"] and r["tag"] == "EMPTY"]
        print(f"  异常: {len(errs)} | OLD命中: {len(old_hits)} | NEW命中: {len(new_hits)} | 空: {len(empties)}")

        check("C5 全程在线查询零异常", len(errs) == 0,
              errs[:3] if errs else "无异常")
        check("C4 切换后在线命中新内容(NEW)且非空",
              len(new_hits) > 0 and all(r["rc"] > 0 for r in new_hits),
              f"{len(new_hits)} new hits")
        # 切换后应至少有部分采样已过渡到新集合（而非全部仍为 OLD）
        after_idx = None
        for i, r in enumerate(results):
            if r["ok"] and r["tag"] == "NEW":
                after_idx = i
                break
        check("C4 切换后观察到了新集合", after_idx is not None,
              "无 NEW 采样则切换未生效于读路径")

        # 空命中检查：从不返回空结果（除非恰好无任何命中）
        if empties:
            print("  WARN: 部分在线查询返回空（可能过渡瞬间/查询词未命中）")

        print("\n" + "=" * 72)
        if failures:
            print(f"存在失败项 {len(failures)}:")
            for f in failures:
                print(f"  - {f}")
            print("结论：零停机机制存在缺陷")
            return 1
        print("零停机真栈并发验证全部通过")
        print("=" * 72)
        return 0

    finally:
        # ---- 清理：恢复全局 + 删除临时目录 ----
        vsmod.config = _orig_config
        vsmod._KB_ACTIVE_CONFIG_PATH = _orig_kb_active_path
        vsmod._vector_store_manager = _orig_vsm
        vsmod._dual_collection_manager = _orig_dcm
        vsmod._kb_version_cache = None
        import gc
        gc.collect()   # 释放 Chroma 以便删文件
        if _ZD_TMP.exists():
            shutil.rmtree(_ZD_TMP, ignore_errors=True)
            print(f"[cleanup] 已删除隔离目录 {_ZD_TMP}")
        else:
            print("[cleanup] 隔离目录已不存在")


if __name__ == "__main__":
    sys.exit(main())