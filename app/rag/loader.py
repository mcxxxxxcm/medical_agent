from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_project_root() -> Path:
    """项目根目录：此处应该向上返回三级"""
    return Path(__file__).resolve().parent.parent.parent


def load_txt(file_path: Path) -> List[Document]:
    """加载txt文件，尝试多种编码进行读取"""
    encodings = ["utf-8", "gbk", "gb2312", "utf-8-sig"]
    # 循环以上4种常见的编码进行读取txt文件
    for encoding in encodings:
        try:
            # 使用LangChain的TextLoader进行txt文件的读取，配置参数1.文件路径；2.编码方式
            loader = TextLoader(str(file_path), encoding=encoding)
            return loader.load()
        # 如果循环时出现编码错误先不返回错误，而是继续执行查看后续的编码是否能成功
        except UnicodeDecodeError:
            continue
    # 循环结束也就意味着没有返回任何加载好的文件，则报错
    raise ValueError(f"无法解码：{file_path.name}")


def load_pdf(file_path: Path) -> List[Document]:
    """加载.pdf文件，每页一个Document。"""
    # LangChain的PyPDFLoader是基于pypdf包的
    loader = PyPDFLoader(str(file_path))
    return loader.load()


# 可以添加其他格式的文档加载函数
# 全局定义字典说明文档加载的策略
LOADERS = {
    ".txt": load_txt,
    ".pdf": load_pdf,
}


def load_single_file(file_path: Path) -> List[Document]:
    """按拓展名选择加载器，加载单个文件。"""
    suffix = file_path.suffix.lower()
    if suffix not in LOADERS:
        raise ValueError(f"不支持读取该文档类型：{suffix}，支持：{list(LOADERS.keys())}")
    # 如果支持加载该文件，则返回时调用函数load_xxx()，所以函数的返回值依旧是一个List
    return LOADERS[suffix](file_path)


def add_metadata(docs: List[Document], file_path: Path) -> None:
    """为文档列表添加来源等元数据。"""
    for doc in docs:
        doc.metadata["source"] = file_path.name
        doc.metadata["file_path"] = str(file_path)


def load_medical_documents(docs_dir: str | Path = "docs/medical") -> List[Document]:
    """加载目录下所有的txt和pdf文档，相对路径相对于项目根目录"""
    base = Path(docs_dir)
    # 如果没有使用绝对路径，就将项目根目录拼接在base之前。
    if not base.is_absolute():
        base = get_project_root() / docs_dir

    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f'目录不存在或者不是目录：{base}')

    all_docs: List[Document] = []
    # 遍历LOADERS字典的所有文件拓展名
    for ext in LOADERS:
        # 遍历base目录下的文件拓展名并匹配
        for path in base.glob(f'*{ext}'):
            # 加载文档
            try:
                docs = load_single_file(path)
                add_metadata(docs, path)
                # 将加载好的文档添加到所有文档变量all_docs中
                all_docs.extend(docs)
            except Exception as e:
                print(f'加载失败{path.name}：{e}')
    return all_docs


def print_docs(docs: List[Document]) -> None:
    """打印已经加载好的文档。"""
    # 先调用函数加载文档

    for doc in docs:
        # 文件名和文件路径
        print(doc.metadata)
        # 打印文件预览
        preview = doc.page_content[:30] + "..." if len(doc.page_content) > 30 else doc.page_content
        print(preview)


def split_documents(docs: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """将文档切分成小块chunks

    Args:
         docs: 原始文档列表
         chunk_size: 每个块的最大字符数量
         chunk_overlap: 块间重叠字符数

    Returns:
        切分后的文档列表，每个doc的metadata会保留
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "，", ",", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    return chunks


def print_chunks(chunks: List[Document]) -> None:
    """测试文档切分的块是否正确"""
    # 打印块数量
    print(f'文档切块数量为：{len(chunks)}')
    for chunk in chunks:
        # 打印前20个字符
        print(chunk.page_content[:30])


if __name__ == '__main__':
    # print(get_project_root())
    # print_docs(load_medical_documents())
    # print_chunks(split_documents(load_medical_documents()))
    print_docs(load_medical_documents())
