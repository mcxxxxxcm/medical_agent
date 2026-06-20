"""从 ChromaDB 读取所有文档块内容"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.rag.vector_store import get_vector_store

vs = get_vector_store()
if vs is None:
    print("向量库为空")
    sys.exit(0)

# 用 retriever 获取所有文档
retriever = vs.as_retriever(k=20)
results = retriever.invoke("疾病 症状 护理 感冒 湿疹 荨麻疹 糖尿病 高血压 便秘")

output_file = project_root / "scripts" / "chroma_chunks.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"检索到 {len(results)} 个文档块\n\n")
    for i, doc in enumerate(results):
        f.write(f"=== 块 {i+1} (长度:{len(doc.page_content)}) ===\n")
        f.write(f"metadata: {doc.metadata}\n")
        f.write(doc.page_content)
        f.write("\n\n")

    # 关键词搜索
    f.write("\n" + "=" * 60 + "\n")
    f.write("关键词搜索:\n")
    for k in ["荨麻疹", "湿疹", "糖尿病", "感冒", "高血压", "便秘", "头痛", "发烧", "咳嗽", "腹泻", "急性支气管炎", "肠胃炎"]:
        count = sum(1 for d in results if k in d.page_content)
        f.write(f"  {k}: 在 {count} 个块中出现\n")

print(f"Done! {len(results)} chunks written to {output_file}")
