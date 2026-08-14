"""v9.21 验证：症状感知检索词映射 — 各症状召回自己的治疗内容，无跨症状污染"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.graph.nodes.nodes import _enrich_treatment_query
from app.rag.hybrid_retriever import get_hybrid_retriever

retriever = get_hybrid_retriever(k=3, rerank_top_k=5)

# 每个症状：查询 + 期望召回的本症状治疗关键词（用于断言）
CASES = [
    ("发烧怎么办", ["减少衣物", "散热", "温水擦浴", "退热药"]),
    ("头痛怎么办", ["布洛芬", "对乙酰氨基酚", "偏头痛", "紧张型头痛"]),
    ("咳嗽怎么办", ["止咳", "化痰", "祛痰"]),
    ("腹泻怎么办", ["止泻", "补水", "电解质", "补液"]),
    ("流鼻血怎么办", ["止血", "按压", "前倾", "冷敷"]),
]

print("=" * 70)
print("检索单测：各症状各自召回治疗内容（无跨症状污染）")
print("=" * 70)

all_pass = True
for raw_query, expected_keywords in CASES:
    # 1. 症状感知增强（_enrich_treatment_query 已含 发热/发烧 双键，直接对原查询调用）
    enriched = _enrich_treatment_query(raw_query)
    print(f"\n查询: {raw_query}")
    if enriched != raw_query:
        print(f"  增强: {raw_query} → {enriched}")

    docs = retriever.invoke(enriched)
    top3_text = "\n".join(d.page_content for d in docs[:3])
    all_docs_text = "\n".join(d.page_content for d in docs)

    found = [kw for kw in expected_keywords if kw in all_docs_text]
    top_found = [kw for kw in expected_keywords if kw in top3_text]
    status = "✅" if len(top_found) >= 2 or (top_found and len(found) >= 3) else "❌"
    if status == "❌":
        all_pass = False
    print(f"  {status} 命中关键词: {found}")
    print(f"     top3 中命中: {top_found}")
    for i, d in enumerate(docs[:3], 1):
        preview = d.page_content[:70].replace("\n", " ")
        print(f"    [{i}] {preview}")

print("\n" + "=" * 70)
print("跨症状污染检查：")
# 头痛/咳嗽/腹泻不应返回"减少衣物/散热"这类发热治疗作为 top3
for raw_query in ["头痛怎么办", "咳嗽怎么办", "腹泻怎么办"]:
    enriched = _enrich_treatment_query(raw_query)
    docs = retriever.invoke(enriched)
    top3_text = "\n".join(d.page_content for d in docs[:3])
    if "减少衣物" in top3_text and "捂汗" in top3_text:
        print(f"  ❌ {raw_query} top3 含发热治疗内容（减少衣物/捂汗）→ 被发热词劫持")
        all_pass = False
    else:
        print(f"  ✅ {raw_query} top3 无发热治疗内容")

print("\n" + "=" * 70)
print(f"总结果: {'全部通过 ✅' if all_pass else '存在失败 ❌'}")
print("=" * 70)
