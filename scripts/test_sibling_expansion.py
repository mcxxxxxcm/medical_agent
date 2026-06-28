"""验证"头痛怎么办"查询的邻域扩展效果"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.hybrid_retriever import get_hybrid_retriever

retriever = get_hybrid_retriever(k=3, rerank_top_k=5)

query = "头痛怎么办？"
print(f"查询: {query}")
results = retriever.invoke(query)
print(f"\n返回 {len(results)} 个文档:")
for i, doc in enumerate(results, 1):
    source = doc.metadata.get("source", "?")
    doc_id = doc.metadata.get("doc_id", "?")
    section_index = doc.metadata.get("section_index", "?")
    sibling = doc.metadata.get("sibling_expanded", False)
    content_preview = doc.page_content[:80].replace("\n", " ")
    has_ibuprofen = "布洛芬" in doc.page_content
    tag = "📱 扩展" if sibling else "🔍 原始"
    drug_tag = "💊 含布洛芬" if has_ibuprofen else ""
    print(f"  [{i}] {tag} {drug_tag} | source={source} | doc_id={doc_id} | section={section_index}")
    print(f"       {content_preview}...")
