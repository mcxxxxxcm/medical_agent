"""用 MinerU 解析 PDF 并保存结果到文件"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from langchain_mineru import MinerULoader

pdf_path = project_root / "docs" / "medical" / "常见疾病症状与家庭护理指南.webdoc.pdf"
output_file = project_root / "scripts" / "mineru_parsed.txt"

loader = MinerULoader(
    source=str(pdf_path),
    mode="flash",
    language="ch",
    timeout=300,
)
docs = loader.load()

with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"文档数量: {len(docs)}\n\n")
    for i, doc in enumerate(docs):
        f.write(f"=== 文档 {i+1} (长度:{len(doc.page_content)}) ===\n")
        f.write(f"metadata: {doc.metadata}\n\n")
        f.write(doc.page_content)
        f.write("\n\n")

    # 关键词搜索
    f.write("\n" + "=" * 60 + "\n")
    f.write("关键词搜索:\n")
    t = docs[0].page_content if docs else ""
    for k in ["荨麻疹", "湿疹", "糖尿病", "感冒", "高血压", "便秘", "头痛", "发烧", "咳嗽", "腹泻", "急性支气管炎", "肠胃炎"]:
        f.write(f"  {k}: {t.count(k)}\n")

print(f"Done! Written to {output_file}")
