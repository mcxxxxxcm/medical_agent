"""用 PyPDFLoader 直接读取 PDF 原始文本内容"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from langchain_community.document_loaders import PyPDFLoader

pdf_path = project_root / "docs" / "medical" / "常见疾病症状与家庭护理指南.webdoc.pdf"
output_file = project_root / "scripts" / "pdf_raw_content.txt"

loader = PyPDFLoader(str(pdf_path))
docs = loader.load()

with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"PDF 页数: {len(docs)}\n\n")
    for i, doc in enumerate(docs):
        f.write(f"=== 第 {i+1} 页 ===\n")
        f.write(doc.page_content)
        f.write("\n\n")

    # 搜索关键词
    f.write("\n" + "=" * 60 + "\n")
    f.write("关键词搜索:\n")
    all_text = "\n".join(d.page_content for d in docs)
    for keyword in ["荨麻疹", "湿疹", "糖尿病", "感冒", "高血压", "便秘", "头痛", "发烧"]:
        count = all_text.count(keyword)
        if count > 0:
            f.write(f"  '{keyword}' 出现 {count} 次\n")
        else:
            f.write(f"  '{keyword}' 未找到!\n")

print(f"报告已写入: {output_file}")
