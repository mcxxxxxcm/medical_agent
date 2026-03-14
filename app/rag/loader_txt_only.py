# 从docs/medical/目录加载所有的知识库文件
# 技术：使用langchain-community.document_loader.TextLoader进行加载文本文件
import os
from pathlib import Path
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document


def load_medical_documents(docs_dir: str = 'docs/medical') -> List[Document]:
    """
    加载医疗文档目录下的所有文本文件

    Args:
        docs_dir: 文档目录路径，默认为"docs/medical"

    Returns:
        List[Document]: 文档列表，每个Document包含页面内容和元数据

    Raises:
        FileNotFoundError: 如果文档目录不存在
    """
    # 1.检查目录是否存在
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f'文件目录不存在：{docs_dir}')

    # 检查是否是目录（而不是文件）
    if not docs_path.is_dir():
        raise ValueError(f'路径不是目录：{docs_dir}')

    # 2.初始化文档列表
    all_documents = []  # 用于存储所有加载的文档

    # 3.遍历目录下所有的.txt文件，使用glob("*.txt")可以匹配目录下所有的txt文档
    for file_path in docs_path.glob("*.txt"):
        try:
            # 4.使用多种编码格式进行加载文档，防止编码不匹配导致乱码或解码错误
            documents = None  # 用于存储成功加载的文档
            encodings = ["utf-8", "gbk", "gb2312", "utf-8-sig"]

            for encoding in encodings:
                try:
                    # TextLoader是LangChain提供的文档加载器，可以读取文件内容，并创建一个Document对象
                    loader = TextLoader(str(file_path), encoding=encoding)
                    documents = loader.load()  # load()返回Document列表
                    break
                except UnicodeDecodeError:
                    # 如果当前编码无法解码，就尝试下一个编码
                    continue

            # 如果所有编码都失败，抛出异常
            if documents is None:
                raise ValueError(f"无法解码文件，尝试其他编码：{file_path.name}")

            # 5.为每个文档添加元数据：元数据用于记录文档来源
            # 方便后续追溯答案来源；用于过滤、排序等操作
            for doc in documents:
                # 添加文件名到元数据
                doc.metadata["source"] = file_path.name
                # 添加完整路径到元数据
                doc.metadata["file_path"] = str(file_path)
                # 其他的元数据...

            # 6.将文档添加到总列表
            all_documents.extend(documents)  # extend将列表中的元素逐个添加

            # 打印加载成功的提示信息
            print(f'已加载文档：{file_path.name}({len(documents)}页)')
        except Exception as e:
            print(f'加载文档失败{file_path.name}:{e}')
            # 多个文档加载时，如果出现某个文档加载失败，可以选择使用continue或raise（中断）
            continue
    # 7.返回所有文档
    # 打印总结信息
    if all_documents:
        print(f'\n总共加载了{len(all_documents)}个文档页面')
    else:
        print(f'\n警告：没有加载到任何文件。')
    return all_documents

if __name__ == '__main__':
    load_medical_documents()