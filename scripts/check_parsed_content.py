"""检查 MinerU 解析结果和切分内容"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.rag.loader import load_medical_documents, split_documents

output_file = project_root / "scripts" / "parsed_content_report.txt"

with open(output_file, "w", encoding="utf-8") as f:
    # 1. 查看原始解析内容
    docs = load_medical_documents()
    f.write("=" * 60 + "\n")
    f.write(f"原始文档数量: {len(docs)}\n")
    for i, doc in enumerate(docs):
        f.write(f"\n--- 文档 {i+1} ---\n")
        f.write(f"长度: {len(doc.page_content)} 字符\n")
        f.write(f"metadata: {doc.metadata}\n")
        f.write(f"内容:\n{doc.page_content}\n")
    f.write("=" * 60 + "\n")

    # 2. 查看切分结果
    chunks = split_documents(docs)
    f.write(f"\n切分后: {len(chunks)} 个块\n")
    for i, chunk in enumerate(chunks):
        f.write(f"\n--- 块 {i+1} (长度:{len(chunk.page_content)}) ---\n")
        f.write(f"metadata: {chunk.metadata}\n")
        f.write(chunk.page_content[:500] + "\n")

    # 3. 搜索关键词
    f.write("\n" + "=" * 60 + "\n")
    f.write("关键词搜索:\n")
    for keyword in ["荨麻疹", "湿疹", "糖尿病", "感冒", "高血压", "便秘"]:
        found = False
        for i, chunk in enumerate(chunks):
            if keyword in chunk.page_content:
                found = True
                f.write(f"  '{keyword}' 在块 {i+1} 中找到\n")
        if not found:
            f.write(f"  '{keyword}' 未找到!\n")

    # 4. 在原始文档中搜索
    f.write("\n原始文档关键词搜索:\n")
    for keyword in ["荨麻疹", "湿疹", "糖尿病", "感冒", "高血压", "便秘"]:
        found = False
        for i, doc in enumerate(docs):
            if keyword in doc.page_content:
                found = True
                f.write(f"  '{keyword}' 在原始文档中找到\n")
        if not found:
            f.write(f"  '{keyword}' 在原始文档中未找到!\n")

print(f"报告已写入: {output_file}")
