import os
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.app_logging import get_logger

logger = get_logger(__name__)


def get_project_root() -> Path:
    """项目根目录：此处应该向上返回三级"""
    return Path(__file__).resolve().parent.parent.parent


def load_txt(file_path: Path) -> List[Document]:
    """加载txt文件，尝试多种编码进行读取"""
    from langchain_community.document_loaders import TextLoader

    encodings = ["utf-8", "gbk", "gb2312", "utf-8-sig"]
    for encoding in encodings:
        try:
            loader = TextLoader(str(file_path), encoding=encoding)
            return loader.load()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法解码：{file_path.name}")


def load_pdf(file_path: Path) -> List[Document]:
    """使用 MinerU 加载 PDF 文件，输出结构化 Markdown 文档。

    MinerU 优势（相比 Docling）：
    - 中文解析准确率 96.4%（Docling 89.7%）
    - 布局检测 mAP 97.5%（Docling 93.1%）
    - 表格结构完整保留（Docling 部分丢失）
    - 公式输出 LaTeX（Docling 仅 PNG）
    - 模型国内托管，下载稳定
    """
    try:
        from langchain_mineru import MinerULoader

        loader = MinerULoader(
            source=str(file_path),
            mode="flash",          # 免Token，适合本地离线解析
            language="ch",         # 中文文档
            timeout=300,
        )
        docs = loader.load()
        logger.info(f"MinerU 解析 PDF 成功：{file_path.name}，生成 {len(docs)} 个文档")
        return docs
    except ImportError:
        logger.warning("langchain-mineru 未安装，回退到 PyPDFLoader。安装命令：pip install langchain-mineru")
        return _load_pdf_fallback(file_path)
    except Exception as e:
        logger.warning(f"MinerU 解析失败：{e}，回退到 PyPDFLoader")
        return _load_pdf_fallback(file_path)


def _load_pdf_fallback(file_path: Path) -> List[Document]:
    """PyPDFLoader 回退方案"""
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(str(file_path))
    return loader.load()


def load_docx(file_path: Path) -> List[Document]:
    """使用 MinerU 加载 DOCX 文件"""
    try:
        from langchain_mineru import MinerULoader

        loader = MinerULoader(
            source=str(file_path),
            mode="flash",
            language="ch",
            timeout=300,
        )
        docs = loader.load()
        logger.info(f"MinerU 解析 DOCX 成功：{file_path.name}，生成 {len(docs)} 个文档")
        return docs
    except ImportError:
        raise ValueError("加载 DOCX 需要 langchain-mineru，请安装：pip install langchain-mineru")
    except Exception as e:
        logger.error(f"MinerU 解析 DOCX 失败：{e}")
        raise


# 全局定义字典说明文档加载的策略
LOADERS = {
    ".txt": load_txt,
    ".pdf": load_pdf,
    ".docx": load_docx,
}


def load_single_file(file_path: Path) -> List[Document]:
    """按拓展名选择加载器，加载单个文件。"""
    suffix = file_path.suffix.lower()
    if suffix not in LOADERS:
        raise ValueError(f"不支持读取该文档类型：{suffix}，支持：{list(LOADERS.keys())}")
    return LOADERS[suffix](file_path)


def add_metadata(docs: List[Document], file_path: Path) -> None:
    """为文档列表添加来源等元数据。"""
    for doc in docs:
        doc.metadata["source"] = file_path.name
        doc.metadata["file_path"] = str(file_path)


def load_medical_documents(docs_dir: str | Path = "docs/medical") -> List[Document]:
    """加载目录下所有的txt和pdf文档，相对路径相对于项目根目录"""
    base = Path(docs_dir)
    if not base.is_absolute():
        base = get_project_root() / docs_dir

    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f'目录不存在或者不是目录：{base}')

    all_docs: List[Document] = []
    for ext in LOADERS:
        for path in base.glob(f'*{ext}'):
            try:
                docs = load_single_file(path)
                add_metadata(docs, path)
                all_docs.extend(docs)
            except Exception as e:
                print(f'加载失败{path.name}：{e}')
    return all_docs


def print_docs(docs: List[Document]) -> None:
    """打印已经加载好的文档。"""
    for doc in docs:
        print(doc.metadata)
        preview = doc.page_content[:30] + "..." if len(doc.page_content) > 30 else doc.page_content
        print(preview)


def split_documents(
    docs: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    use_markdown_splitter: bool = True,
) -> List[Document]:
    """将文档切分成小块chunks

    优先使用 Markdown 标题层级切分（Docling 输出的 Markdown），
    回退到 RecursiveCharacterTextSplitter。

    Args:
         docs: 原始文档列表
         chunk_size: 每个块的最大字符数量（默认1000，比之前500更大以保留语义完整性）
         chunk_overlap: 块间重叠字符数（默认200，20%重叠防止边界丢失）
         use_markdown_splitter: 是否使用 Markdown 结构感知切分

    Returns:
        切分后的文档列表，每个doc的metadata会保留
    """
    if not docs:
        return []

    # 判断是否使用 Markdown 切分器
    if use_markdown_splitter and _has_markdown_headers(docs):
        return _split_by_markdown_headers(docs, chunk_size, chunk_overlap)

    # 回退到普通递归切分
    return _split_by_recursive(docs, chunk_size, chunk_overlap)


def _has_markdown_headers(docs: List[Document]) -> bool:
    """检查文档是否包含 Markdown 标题层级"""
    for doc in docs:
        content = doc.page_content
        if any(line.startswith("#") for line in content.split("\n")[:20]):
            return True
    return False


def _split_by_markdown_headers(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """基于 Markdown 标题层级的结构感知切分

    策略：
    1. 先按 Markdown 标题层级切分（H1/H2/H3）
    2. 如果某个章节超过 chunk_size，再用 RecursiveCharacterTextSplitter 二次切分
    3. 保留标题层级信息到 metadata
    """
    try:
        from langchain_text_splitters import MarkdownHeaderTextSplitter
    except ImportError:
        logger.warning("MarkdownHeaderTextSplitter 不可用，回退到递归切分")
        return _split_by_recursive(docs, chunk_size, chunk_overlap)

    all_chunks: List[Document] = []

    # Markdown 标题层级 → 切分后保留到 metadata 的 key
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,  # 保留标题在内容中
    )

    # 用于二次切分超大章节
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "，", ",", " ", ""],
    )

    for doc in docs:
        try:
            # 第一阶段：按 Markdown 标题切分
            md_chunks = markdown_splitter.split_text(doc.page_content)

            # 第二阶段：对超大章节二次切分
            for chunk in md_chunks:
                # 合并原始 metadata 和标题层级 metadata
                merged_metadata = {**doc.metadata, **chunk.metadata}

                if len(chunk.page_content) > chunk_size:
                    # 超大章节，递归切分
                    sub_chunks = recursive_splitter.split_text(chunk.page_content)
                    for sub_text in sub_chunks:
                        sub_doc = Document(
                            page_content=sub_text,
                            metadata={**merged_metadata},
                        )
                        all_chunks.append(sub_doc)
                else:
                    chunk.metadata = merged_metadata
                    all_chunks.append(chunk)

        except Exception as e:
            logger.warning(f"Markdown 切分失败，回退到递归切分：{e}")
            chunks = _split_by_recursive([doc], chunk_size, chunk_overlap)
            all_chunks.extend(chunks)

    logger.info(f"Markdown 结构感知切分完成：{len(docs)} 个文档 → {len(all_chunks)} 个块")
    return all_chunks


def _split_by_recursive(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """普通递归字符切分（回退方案）"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "，", ",", " ", ""],
    )
    chunks = text_splitter.split_documents(docs)
    logger.info(f"递归切分完成：{len(docs)} 个文档 → {len(chunks)} 个块")
    return chunks


def print_chunks(chunks: List[Document]) -> None:
    """测试文档切分的块是否正确"""
    print(f'文档切块数量为：{len(chunks)}')
    for chunk in chunks:
        print(chunk.page_content[:30])


if __name__ == '__main__':
    print_docs(load_medical_documents())
