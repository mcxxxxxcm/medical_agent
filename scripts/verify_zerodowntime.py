"""v9.22 验证：零停机重建（影子集合 → 原子切换 → 回滚）

流程：
1. 基线：get_vector_store() 加载默认集合（langchain，386 chunks）
2. 建 mini 影子集合（3 个测试文档）
3. switch_active_collection → kb_active.json 指向影子
4. get_vector_store() 应加载新集合（3 chunks），能查到测试内容
5. rollback_to_previous → get_vector_store() 回退默认集合（386 chunks）
6. 清理：删 kb_active.json + 影子目录
"""
import sys
import shutil
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from langchain_core.documents import Document

from app.rag.vector_store import (
    get_vector_store,
    get_dual_collection_manager,
    get_vector_store_manager,
)
from app.rag.vector_store import _KB_ACTIVE_CONFIG_PATH

failures = []
def check(label, ok):
    print(f"  {'✅' if ok else '❌'} {label}")
    if not ok:
        failures.append(label)


def main():
    print("=" * 70)
    print("零停机重建验证")
    print("=" * 70)

    # 1. 基线：默认集合
    print("\n[1] 基线：默认集合")
    vs0 = get_vector_store()
    info0 = vs0._collection.count()
    name0 = vs0._collection.name
    print(f"  默认集合名: {name0}, chunks: {info0}")
    check(f"基线为默认集合 langchain", name0 == "langchain" and info0 >= 380)

    # 2. 建 mini 影子集合
    print("\n[2] 建 mini 影子集合")
    dcm = get_dual_collection_manager()
    shadow_name, shadow_dir = dcm.create_shadow_collection()
    print(f"  影子: {shadow_name} @ {shadow_dir}")
    test_docs = [
        Document(page_content="布洛芬用法用量：成人每次200-400mg，每日3-4次，饭后服用。",
                 metadata={"source": "测试药典.txt", "h2": "布洛芬"}),
        Document(page_content="对乙酰氨基酚用法用量：成人每次500mg，每日不超过4次，间隔至少4小时。",
                 metadata={"source": "测试药典.txt", "h2": "对乙酰氨基酚"}),
        Document(page_content="发烧护理：减少衣物散热，温水擦浴，多饮温水，不要捂汗。",
                 metadata={"source": "测试护理指南.txt", "h2": "发热护理"}),
    ]
    shadow_vs = dcm.build_shadow_collection(test_docs, shadow_name, shadow_dir)
    check(f"影子集合写入 {len(test_docs)} 个 chunk", shadow_vs._collection.count() == len(test_docs))

    # 3. 原子切换
    print("\n[3] 原子切换指针")
    dcm.switch_active_collection(shadow_name, str(shadow_dir))
    print(f"  kb_active.json 存在: {_KB_ACTIVE_CONFIG_PATH.exists()}")
    print(f"  活跃集合: {dcm.get_active_collection_name()}")

    # 4. 加载路径应消费指针 → 新集合
    print("\n[4] 切换后 get_vector_store() 应加载新集合")
    # 重置全局 manager，模拟切换后的下一次加载（switch 已重置，此处再确保）
    vs1 = get_vector_store()
    info1 = vs1._collection.count()
    name1 = vs1._collection.name
    print(f"  加载集合名: {name1}, chunks: {info1}")
    check(f"加载到影子集合 {shadow_name}", name1 == shadow_name and info1 == len(test_docs))
    # 新集合可检索到测试内容
    hits = vs1.similarity_search("布洛芬怎么吃", k=1)
    check(f"新集合能检索到布洛芬测试文档", len(hits) > 0 and "布洛芬" in hits[0].page_content)

    # 5. 回滚
    print("\n[5] 回滚到默认集合")
    ok_rollback = dcm.rollback_to_previous()
    check(f"回滚调用成功", ok_rollback)
    vs2 = get_vector_store()
    info2 = vs2._collection.count()
    name2 = vs2._collection.name
    print(f"  回滚后加载集合名: {name2}, chunks: {info2}")
    check(f"回滚后加载默认集合 langchain", name2 == "langchain" and info2 >= 380)

    # 6. 清理
    print("\n[6] 清理测试残留")
    # 回滚后 active 为哨兵名 → 直接删除指针文件，回到完全默认
    if _KB_ACTIVE_CONFIG_PATH.exists():
        _KB_ACTIVE_CONFIG_PATH.unlink()
        print(f"  已删除 {_KB_ACTIVE_CONFIG_PATH.name}")
    # 先释放对影子集合的 Chroma 客户端引用并强制 GC，
    # 否则 Windows 下 HNSW 文件（data_level0.bin）被 mmap 锁住无法删除
    import app.rag.vector_store as vsmod
    import gc
    vsmod._vector_store_manager = None
    shadow_vs = None
    vs1 = None
    gc.collect()
    if Path(shadow_dir).exists():
        try:
            shutil.rmtree(shadow_dir)
            print(f"  已删除影子目录 {shadow_dir}")
        except Exception as e:
            print(f"  影子目录删除失败（进程退出后自动释放）：{e}")

    print("\n" + "=" * 70)
    if failures:
        print(f"存在失败项: {failures}")
        print("=" * 70)
        return 1
    print("零停机重建验证全部通过 ✅")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
