import hashlib
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

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

        # 扫描件检测：MinerU 解析出的文档几乎无文本 → 可能是扫描件
        total_chars = sum(len(doc.page_content) for doc in docs)
        if total_chars < 50 and docs:
            logger.info(f"MinerU 解析文本过少（{total_chars} 字符），疑似扫描件，尝试 OCR")
            try:
                return load_scanned_pdf(file_path)
            except Exception as ocr_err:
                logger.warning(f"扫描件 OCR 失败：{ocr_err}，返回 MinerU 原始结果")

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


# ===== 表格文档加载器 =====

def _dataframe_to_markdown_table(df, max_rows: int = 200, fill_down: bool = True) -> str:
    """将 pandas DataFrame 转为 Markdown 表格字符串

    Args:
        df: pandas DataFrame
        max_rows: 最大行数，超过截断并标注
        fill_down: 合并单元格继承（fill-down），将空值填充为上一个非空值
                   如"解热镇痛药"行下的多行都继承该分类
    """
    if df.empty:
        return ""

    # 截断过长表格
    truncated = len(df) > max_rows
    if truncated:
        df = df.head(max_rows)

    # 合并单元格继承（fill-down）
    if fill_down:
        for col in df.columns:
            prev_val = ""
            for idx in df.index:
                val = str(df.at[idx, col]).strip()
                if val and val != "nan" and val != "NaN" and val != "None":
                    prev_val = val
                elif prev_val:
                    df.at[idx, col] = prev_val

    # 填充 NaN
    df = df.fillna("")

    # 构建 Markdown 表格
    headers = [str(col).strip() for col in df.columns]
    lines = []
    # 表头行
    lines.append("| " + " | ".join(headers) + " |")
    # 分隔行
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    # 数据行
    for _, row in df.iterrows():
        cells = [str(val).strip().replace("\n", " ") for val in row]
        lines.append("| " + " | ".join(cells) + " |")

    if truncated:
        lines.append(f"\n> 表格已截断，仅展示前 {max_rows} 行（共 {len(df)} 行）")

    return "\n".join(lines)


def load_xlsx(file_path: Path) -> List[Document]:
    """加载 Excel 文件，每个 Sheet 转为一个 Document（Markdown 表格格式）

    流程：
        1. openpyxl 读取 Excel（无需 Excel 运行时）
        2. 每个 Sheet → DataFrame → Markdown 表格
        3. 添加表格元数据（sheet_name, row_count, col_count, is_table）
        4. 空行/空列自动跳过

    支持：.xlsx, .xls（xlrd 降级）
    """
    try:
        import pandas as pd
    except ImportError:
        raise ValueError("加载 Excel 需要 pandas + openpyxl，请安装：pip install pandas openpyxl")

    docs: List[Document] = []
    try:
        # 读取所有 Sheet
        xls = pd.ExcelFile(str(file_path), engine="openpyxl")
    except Exception:
        # 降级尝试 xlrd（.xls 格式）
        try:
            xls = pd.ExcelFile(str(file_path), engine="xlrd")
        except Exception as e:
            raise ValueError(f"无法读取 Excel 文件：{e}，请安装 openpyxl 或 xlrd")

    for sheet_name in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            if df.empty:
                logger.debug(f"跳过空 Sheet：{sheet_name}")
                continue

            # 去除全为 NaN 的列
            df = df.dropna(axis=1, how="all")
            if df.empty:
                continue

            md_table = _dataframe_to_markdown_table(df)
            if not md_table:
                continue

            # 构建表头摘要（用于检索增强）
            headers = [str(col).strip() for col in df.columns]
            header_summary = f"表格列：{', '.join(headers)}"
            row_count = len(df)
            col_count = len(df.columns)

            doc = Document(
                page_content=md_table,
                metadata={
                    "sheet_name": sheet_name,
                    "is_table": True,
                    "table_headers": headers,
                    "table_row_count": row_count,
                    "table_col_count": col_count,
                    "table_header_summary": header_summary,
                },
            )
            docs.append(doc)
        except Exception as e:
            logger.warning(f"读取 Sheet '{sheet_name}' 失败：{e}")
            continue

    logger.info(f"Excel 加载完成：{file_path.name}，{len(docs)} 个 Sheet")
    return docs


def load_csv(file_path: Path) -> List[Document]:
    """加载 CSV 文件，转为 Markdown 表格 Document

    支持：.csv（自动检测编码和分隔符）
    """
    try:
        import pandas as pd
    except ImportError:
        raise ValueError("加载 CSV 需要 pandas，请安装：pip install pandas")

    # 多编码尝试
    encodings = ["utf-8", "gbk", "gb2312", "utf-8-sig", "latin-1"]
    df = None
    for encoding in encodings:
        try:
            df = pd.read_csv(str(file_path), encoding=encoding)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue

    if df is None:
        raise ValueError(f"无法解码 CSV：{file_path.name}")

    if df.empty:
        return []

    df = df.dropna(axis=1, how="all")
    md_table = _dataframe_to_markdown_table(df)
    if not md_table:
        return []

    headers = [str(col).strip() for col in df.columns]
    header_summary = f"表格列：{', '.join(headers)}"

    doc = Document(
        page_content=md_table,
        metadata={
            "is_table": True,
            "table_headers": headers,
            "table_row_count": len(df),
            "table_col_count": len(df.columns),
            "table_header_summary": header_summary,
        },
    )
    logger.info(f"CSV 加载完成：{file_path.name}，{len(df)} 行 × {len(df.columns)} 列")
    return [doc]


def load_md(file_path: Path) -> List[Document]:
    """加载 Markdown 文件，保留原始 Markdown 格式（含表格）"""
    encodings = ["utf-8", "gbk", "utf-8-sig"]
    for encoding in encodings:
        try:
            content = file_path.read_text(encoding=encoding)
            return [Document(page_content=content)]
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法解码 Markdown：{file_path.name}")


# ===== 扫描件 OCR 加载器 =====

# 支持的图片扩展名
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}

# PaddleOCR 版面结构标签 → Markdown 映射
_LAYOUT_LABEL_MAP = {
    "title": "heading",
    "text": "text",
    "table": "table",
    "figure": "figure",
    "figure_caption": "text",
    "table_caption": "text",
    "header": "header",
    "footer": "footer",
    "reference": "text",
    "equation": "text",
}


def _ocr_image_to_markdown(
    image_path: str,
    use_table_rec: bool = True,
    lang: str = "ch",
) -> str:
    """使用 PaddleOCR PP-Structure 对图片/扫描页进行版面分析 + OCR + 表格识别

    流程（参考日志1：先还原结构，再输出结构化内容）：
        1. PP-Structure 版面分析 → 检测 text/table/title/figure 区域
        2. 文本区域 → OCR → 纯文本
        3. 表格区域 → 表格识别 → HTML → Markdown 表格
        4. 标题区域 → OCR → Markdown 标题层级（# / ## / ###）
        5. 按阅读顺序组装 Markdown

    Args:
        image_path: 图片路径
        use_table_rec: 是否启用表格识别（PP-Structure table）
        lang: OCR 语言（ch=中英文混合）

    Returns:
        Markdown 格式文本
    """
    try:
        from paddleocr import PPStructure
    except ImportError:
        raise ValueError(
            "扫描件 OCR 需要 PaddleOCR，请安装：\n"
            "  pip install paddleocr paddlepaddle\n"
            "  或 GPU 版：pip install paddleocr paddlepaddle-gpu"
        )

    # 初始化 PP-Structure（版面分析 + OCR + 表格识别）
    engine = PPStructure(
        image_dir=image_path,
        show_log=False,
        layout=True,          # 版面分析
        table=use_table_rec,  # 表格识别
        ocr=True,             # OCR
        lang=lang,
        recovery=True,        # 版面还原模式
    )

    # 执行分析
    result = engine(image_path)

    # 组装 Markdown
    md_lines = []
    for region in result:
        label = region.get("type", "text")
        bbox = region.get("bbox", [0, 0, 0, 0])
        page_number = region.get("page_no", -1)

        mapped_label = _LAYOUT_LABEL_MAP.get(label, "text")

        if mapped_label == "heading":
            # 标题区域 → Markdown 标题
            text = region.get("res", [])
            if isinstance(text, list):
                text = " ".join([item.get("text", "") for item in text])
            if text.strip():
                # 根据字号推断层级（简化：用 bbox 高度启发式）
                h = bbox[3] - bbox[1] if len(bbox) >= 4 else 0
                level = "##" if h < 30 else "#"  # 大字→H1，中字→H2
                md_lines.append(f"{level} {text.strip()}\n")

        elif mapped_label == "table":
            # 表格区域
            if use_table_rec and "res" in region:
                table_res = region["res"]
                if isinstance(table_res, dict) and "html" in table_res:
                    # PaddleOCR 表格识别输出 HTML → 转 Markdown
                    html_table = table_res["html"]
                    md_table = _html_table_to_markdown(html_table)
                    md_lines.append(md_table + "\n")
                elif isinstance(table_res, list):
                    # 降级：OCR 文本列表
                    text = " ".join([item.get("text", "") for item in table_res])
                    if text.strip():
                        md_lines.append(text.strip() + "\n")
            else:
                # 未启用表格识别，OCR 当文本
                text = region.get("res", [])
                if isinstance(text, list):
                    text = " ".join([item.get("text", "") for item in text])
                if text.strip():
                    md_lines.append(text.strip() + "\n")

        elif mapped_label == "text":
            # 文本区域
            text = region.get("res", [])
            if isinstance(text, list):
                text = " ".join([item.get("text", "") for item in text])
            if text.strip():
                md_lines.append(text.strip() + "\n")

        elif mapped_label == "header":
            # 页眉（忽略，不进入正文）
            pass
        elif mapped_label == "footer":
            # 页脚（忽略）
            pass
        elif mapped_label == "figure":
            # 图片区域（记录占位，OCR 无法提取图中文字）
            md_lines.append("<!-- 图片区域 -->\n")

    return "\n".join(md_lines)


def _html_table_to_markdown(html: str) -> str:
    """将 PaddleOCR 输出的 HTML 表格转为 Markdown 表格

    处理：
        1. 解析 <table><tr><td> 结构
        2. 合并单元格（colspan/rowspan）→ fill-down 继承
        3. 输出 Markdown 表格
    """
    try:
        from html.parser import HTMLParser

        class TableParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.rows = []
                self.current_row = []
                self.current_cell = ""
                self.in_cell = False

            def handle_starttag(self, tag, attrs):
                if tag == "tr":
                    self.current_row = []
                elif tag in ("td", "th"):
                    self.in_cell = True
                    self.current_cell = ""

            def handle_endtag(self, tag):
                if tag in ("td", "th"):
                    self.in_cell = False
                    self.current_row.append(self.current_cell.strip())
                elif tag == "tr" and self.current_row:
                    self.rows.append(self.current_row)

            def handle_data(self, data):
                if self.in_cell:
                    self.current_cell += data

        parser = TableParser()
        parser.feed(html)

        if not parser.rows:
            return ""

        # 构建 Markdown 表格
        # 第一行作为表头
        headers = parser.rows[0]
        # 对齐列数
        max_cols = max(len(row) for row in parser.rows)
        headers = headers + [""] * (max_cols - len(headers))

        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * max_cols) + " |")

        for row in parser.rows[1:]:
            row = row + [""] * (max_cols - len(row))
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)
    except Exception:
        return ""


def load_scanned_image(file_path: Path) -> List[Document]:
    """加载扫描件图片，使用 PaddleOCR 进行版面分析 + OCR + 表格识别

    支持：.png, .jpg, .jpeg, .tiff, .bmp, .webp

    流程：
        图片 → PaddleOCR PP-Structure → 版面分析 → OCR/表格识别 → Markdown → Document
    """
    md_content = _ocr_image_to_markdown(str(file_path))
    if not md_content.strip():
        logger.warning(f"OCR 未识别出内容：{file_path.name}")
        return []

    doc = Document(
        page_content=md_content,
        metadata={
            "is_scanned": True,
            "ocr_engine": "paddleocr",
        },
    )
    logger.info(f"扫描件 OCR 完成：{file_path.name}，识别 {len(md_content)} 字符")
    return [doc]


def load_scanned_pdf(file_path: Path) -> List[Document]:
    """加载扫描件 PDF（每页转图片后 OCR）

    流程：
        1. pdf2image 将 PDF 每页转为图片
        2. 每页图片 → PaddleOCR PP-Structure → Markdown
        3. 按页组装，标注页码元数据
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ValueError("扫描件 PDF 需要 pdf2image + poppler，请安装：pip install pdf2image")

    try:
        from PIL import Image
    except ImportError:
        raise ValueError("需要 Pillow：pip install Pillow")

    # PDF 每页转图片
    try:
        images = convert_from_path(str(file_path), dpi=300)
    except Exception as e:
        raise ValueError(f"PDF 转图片失败（需安装 poppler）：{e}")

    if not images:
        return []

    docs = []
    for page_num, image in enumerate(images, start=1):
        # 临时保存图片
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name, "PNG")
            tmp_path = tmp.name

        try:
            md_content = _ocr_image_to_markdown(tmp_path)
        finally:
            os.unlink(tmp_path)

        if not md_content.strip():
            continue

        doc = Document(
            page_content=md_content,
            metadata={
                "is_scanned": True,
                "ocr_engine": "paddleocr",
                "page_number": page_num,
                "total_pages": len(images),
            },
        )
        docs.append(doc)

    logger.info(f"扫描件 PDF OCR 完成：{file_path.name}，{len(docs)}/{len(images)} 页有内容")
    return docs


# 全局定义字典说明文档加载的策略
LOADERS = {
    ".txt": load_txt,
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".md": load_md,
    ".xlsx": load_xlsx,
    ".xls": load_xlsx,
    ".csv": load_csv,
    # 扫描件图片
    ".png": load_scanned_image,
    ".jpg": load_scanned_image,
    ".jpeg": load_scanned_image,
    ".tiff": load_scanned_image,
    ".tif": load_scanned_image,
    ".bmp": load_scanned_image,
    ".webp": load_scanned_image,
}


def load_single_file(file_path: Path) -> List[Document]:
    """按拓展名选择加载器，加载单个文件。"""
    suffix = file_path.suffix.lower()
    if suffix not in LOADERS:
        raise ValueError(f"不支持读取该文档类型：{suffix}，支持：{list(LOADERS.keys())}")
    return LOADERS[suffix](file_path)


def add_metadata(
    docs: List[Document],
    file_path: Path,
    extract_doc_meta: bool = True,
) -> Optional[Dict[str, Any]]:
    """为文档列表添加来源等元数据（溯源必备）。

    溯源元数据清单（参考日志1：入库时除正文外还要保存的元数据）：
        - source: 文档文件名（如"发热诊断与家庭护理指南.txt"）
        - file_path: 文档完整路径
        - file_type: 文档类型（txt/pdf/docx/xlsx/csv/md/image）
        - file_size: 文件大小（字节）
        - doc_hash: 文档内容 MD5 前 8 位（防篡改校验）
        - doc_version / doc_effective_date / doc_authority_level 等（多源交叉校验自动提取）

    Args:
        docs: 文档列表
        file_path: 文件路径
        extract_doc_meta: 是否执行多源元数据自动提取（默认True）

    Returns:
        提取的元数据报告（含置信度和待审核字段），或None
    """
    file_type = file_path.suffix.lstrip(".").lower()
    file_size = file_path.stat().st_size if file_path.exists() else 0

    # 文档内容哈希（防篡改校验）
    doc_hash = ""
    try:
        content_bytes = file_path.read_bytes()
        doc_hash = hashlib.md5(content_bytes).hexdigest()[:8]
    except Exception:
        pass

    for doc in docs:
        doc.metadata["source"] = file_path.name
        doc.metadata["file_path"] = str(file_path)
        doc.metadata["file_type"] = file_type
        doc.metadata["file_size"] = file_size
        doc.metadata["doc_hash"] = doc_hash

    # 多源元数据自动提取（版本/日期/权威等级等）
    meta_report = None
    if extract_doc_meta and docs:
        try:
            from app.rag.metadata_extractor import (
                extract_document_metadata,
                apply_metadata_to_documents,
            )
            meta_report = extract_document_metadata(file_path, documents=docs)
            apply_metadata_to_documents(docs, meta_report)
            logger.info(
                f"文档元数据自动提取：{file_path.name}, "
                f"overall_confidence={meta_report.get('overall_confidence')}, "
                f"needs_review={meta_report.get('needs_manual_review')}"
            )
        except Exception as e:
            logger.warning(f"文档元数据自动提取失败（不影响入库）：{e}")

    return meta_report


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

    表格感知：检测 Markdown 表格，保持表格原子性（不拆分跨行），
    并添加表格上下文元数据（表头摘要、列名、行数）。

    Args:
         docs: 原始文档列表
         chunk_size: 每个块的最大字符数量（默认500）
         chunk_overlap: 块间重叠字符数（默认50）
         use_markdown_splitter: 是否使用 Markdown 结构感知切分

    Returns:
        切分后的文档列表，每个doc的metadata会保留
    """
    if not docs:
        return []

    # 表格感知预处理：提取文档中的表格，添加元数据
    docs = _enrich_table_metadata(docs)

    # 判断是否使用 Markdown 切分器
    if use_markdown_splitter and _has_markdown_headers(docs):
        return _split_by_markdown_headers(docs, chunk_size, chunk_overlap)

    # 回退到普通递归切分
    return _split_by_recursive(docs, chunk_size, chunk_overlap)


# ===== 表格感知处理 =====

# Markdown 表格行匹配：| xxx | yyy |
_MD_TABLE_ROW = re.compile(r"^\s*\|.+\|\s*$")
# Markdown 表格分隔行：| --- | --- |
_MD_TABLE_SEP = re.compile(r"^\s*\|[\s\-:]+\|[\s\-:|]*\|\s*$")


def _detect_markdown_tables(content: str) -> List[Tuple[int, int, List[str]]]:
    """检测 Markdown 内容中的所有表格区域

    Returns:
        List of (start_line, end_line, headers) for each table
        headers: 表头列名列表（用于上下文增强）
    """
    lines = content.split("\n")
    tables = []
    i = 0
    while i < len(lines):
        # 查找表头行：| xxx | yyy | 格式
        if _MD_TABLE_ROW.match(lines[i]) and i + 1 < len(lines) and _MD_TABLE_SEP.match(lines[i + 1]):
            start = i
            # 提取表头
            header_line = lines[i].strip()
            headers = [cell.strip() for cell in header_line.strip("|").split("|")]
            # 跳过分隔行，找数据行
            i += 2
            while i < len(lines) and _MD_TABLE_ROW.match(lines[i]):
                i += 1
            tables.append((start, i, headers))
        else:
            i += 1
    return tables


def _extract_table_context(content: str, table_start: int, headers: List[str]) -> str:
    """提取表格的上下文信息

    策略：
        1. 向上查找最近的 Markdown 标题（## xxx）作为表格标题
        2. 组合标题+表头作为表格上下文摘要

    Returns:
        表格上下文摘要字符串
    """
    lines = content.split("\n")

    # 向上查找最近的标题（最多 10 行）
    table_title = ""
    for j in range(table_start - 1, max(table_start - 10, -1), -1):
        if j < 0:
            break
        line = lines[j].strip()
        if line.startswith("#"):
            table_title = line.lstrip("#").strip()
            break

    # 构建上下文摘要
    parts = []
    if table_title:
        parts.append(f"表格标题：{table_title}")
    if headers:
        parts.append(f"表格列：{', '.join(headers)}")
    return "；".join(parts) if parts else ""


def _enrich_table_metadata(docs: List[Document]) -> List[Document]:
    """表格感知预处理：检测文档中的 Markdown 表格，添加元数据

    对每个文档：
        1. 检测所有 Markdown 表格区域
        2. 为整个文档标注 is_table / table_count
        3. 收集所有表格的表头（用于检索增强）
    """
    for doc in docs:
        # 已有 is_table 元数据的文档（Excel/CSV 加载器设置），跳过
        if doc.metadata.get("is_table"):
            continue

        content = doc.page_content
        tables = _detect_markdown_tables(content)

        if not tables:
            continue

        # 标注文档含表格
        doc.metadata["is_table"] = True
        doc.metadata["table_count"] = len(tables)

        # 收集所有表格的表头（用于检索增强）
        all_headers = []
        for _, _, headers in tables:
            all_headers.extend(headers)
        if all_headers:
            doc.metadata["table_headers"] = list(dict.fromkeys(all_headers))  # 去重保序

    return docs


def _extract_table_title(lines: List[str], table_start: int) -> str:
    """向上查找最近的 Markdown 标题作为表格标题"""
    for j in range(table_start - 1, max(table_start - 10, -1), -1):
        if j < 0:
            break
        line = lines[j].strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return ""


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
                    # 超大章节，检查是否含表格
                    has_table = _detect_markdown_tables(chunk.page_content)
                    if has_table:
                        # 含表格：行级切片（每行独立 chunk + 概览 chunk）
                        sub_docs = _split_table_aware(chunk.page_content, chunk_size, merged_metadata)
                        all_chunks.extend(sub_docs)
                    else:
                        # 无表格：普通递归切分
                        sub_chunks = recursive_splitter.split_text(chunk.page_content)
                        for sub_text in sub_chunks:
                            sub_doc = Document(
                                page_content=sub_text,
                                metadata={**merged_metadata},
                            )
                            all_chunks.append(sub_doc)
                else:
                    # 小章节，但可能含表格→仍需行级切片
                    has_table = _detect_markdown_tables(chunk.page_content)
                    if has_table:
                        sub_docs = _split_table_aware(chunk.page_content, chunk_size, merged_metadata)
                        all_chunks.extend(sub_docs)
                    else:
                        chunk.metadata = merged_metadata
                        all_chunks.append(chunk)

        except Exception as e:
            logger.warning(f"Markdown 切分失败，回退到递归切分：{e}")
            chunks = _split_by_recursive([doc], chunk_size, chunk_overlap)
            all_chunks.extend(chunks)

    logger.info(f"Markdown 结构感知切分完成：{len(docs)} 个文档 → {len(all_chunks)} 个块")
    return all_chunks


def _split_table_aware(
    content: str,
    chunk_size: int,
    metadata: dict,
) -> List[Document]:
    """表格感知切分：行级切片 + 双格式（概览 chunk + 行级 chunk）

    设计原则（参考日志1：复杂表格入库最佳实践）：
        1. 每个表格行独立成 chunk，携带完整上下文（表头+表格标题+章节）
        2. 每个行 chunk 附带自然语言摘要（用于 Embedding/BM25 检索增强）
        3. 额外生成一个概览 chunk（表头 + 前 3 行），支持对比类查询
        4. 非表格文本正常递归切分

    Args:
        content: 章节文本内容
        chunk_size: 最大 chunk 字符数
        metadata: 继承的元数据

    Returns:
        切分后的 Document 列表
    """
    lines = content.split("\n")
    tables = _detect_markdown_tables(content)

    if not tables:
        # 无表格，直接递归切分
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "，", ",", " ", ""],
        )
        chunks = splitter.split_text(content)
        return [Document(page_content=t, metadata={**metadata}) for t in chunks]

    # 标记表格行
    table_line_set = set()
    for start, end, _ in tables:
        for line_idx in range(start, end):
            table_line_set.add(line_idx)

    result_docs = []

    # 分段处理：表格区域 vs 非表格区域
    segments = _segment_by_table(lines, table_line_set)

    for seg_text, is_table, seg_start_line in segments:
        if not seg_text.strip():
            continue

        if is_table:
            # 找到对应的表格信息
            table_info = None
            for t_start, t_end, t_headers in tables:
                if t_start <= seg_start_line < t_end:
                    table_info = (t_start, t_end, t_headers)
                    break

            if table_info:
                t_start, t_end, headers = table_info
                table_title = _extract_table_title(lines, t_start)
                # 行级切片 + 概览 chunk
                table_docs = _generate_row_chunks(seg_text, headers, table_title, metadata)
                result_docs.extend(table_docs)
            else:
                # 降级：整表保留
                doc = Document(page_content=seg_text, metadata={**metadata, "is_table": True})
                result_docs.append(doc)
        else:
            # 非表格区域：递归切分
            if len(seg_text) <= chunk_size:
                doc = Document(page_content=seg_text, metadata={**metadata})
                result_docs.append(doc)
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=50,
                    separators=["\n\n", "\n", "。", "，", ",", " ", ""],
                )
                sub_chunks = splitter.split_text(seg_text)
                for sub_text in sub_chunks:
                    doc = Document(page_content=sub_text, metadata={**metadata})
                    result_docs.append(doc)

    return result_docs


def _segment_by_table(lines: List[str], table_line_set: set) -> List[Tuple[str, bool, int]]:
    """将文档行按表格/非表格区域分段

    Returns:
        List of (text, is_table, start_line_index)
    """
    segments = []
    current_lines = []
    current_is_table = False
    current_start = 0

    for i, line in enumerate(lines):
        is_table_line = i in table_line_set
        if is_table_line != current_is_table and current_lines:
            segments.append(("\n".join(current_lines), current_is_table, current_start))
            current_lines = []
            current_start = i
        if not current_lines:
            current_start = i
        current_lines.append(line)
        current_is_table = is_table_line

    if current_lines:
        segments.append(("\n".join(current_lines), current_is_table, current_start))

    return segments


def _generate_row_chunks(
    table_text: str,
    headers: List[str],
    table_title: str,
    metadata: dict,
) -> List[Document]:
    """行级切片：每行独立 chunk + 自然语言摘要 + 概览 chunk

    双格式策略：
        1. 概览 chunk（1个）：表头 + 前 3 行 → 支持对比类查询（"布洛芬和对乙酰氨基酚哪个好？"）
        2. 行级 chunk（N个）：每行 1 个 → 支持精确查询（"布洛芬的每日最大量？"）

    每个行级 chunk 内容格式：
        ---
        【表格上下文】
        文档：{source}
        表格：{table_title}
        字段：{headers}
        ---
        | header1 | header2 | ... |
        |---------|---------|-----|
        | val1    | val2    | ... |
        ---
        摘要：{natural_language_summary}
        ---
    """
    table_lines = table_text.split("\n")

    # 解析表头行和分隔行
    header_line = ""
    sep_line = ""
    data_lines = []
    for line in table_lines:
        if _MD_TABLE_SEP.match(line):
            sep_line = line
            continue
        if _MD_TABLE_ROW.match(line) and not sep_line:
            header_line = line
            continue
        if _MD_TABLE_ROW.match(line) and sep_line:
            data_lines.append(line)

    if not header_line or not data_lines:
        # 无法解析，降级整表保留
        return [Document(page_content=table_text, metadata={**metadata, "is_table": True})]

    # 解析表头列名
    header_cols = [cell.strip() for cell in header_line.strip("|").split("|")]

    # 构建上下文前缀
    source = metadata.get("source", "未知文档")
    context_parts = []
    context_parts.append(f"文档：{source}")
    if table_title:
        context_parts.append(f"表格：{table_title}")
    if header_cols:
        context_parts.append(f"字段：{', '.join(header_cols)}")
    context_str = "\n".join(context_parts)

    result_docs = []

    # === 概览 chunk ===
    overview_rows = data_lines[:3]
    overview_text = "\n".join([header_line, sep_line] + overview_rows)
    if len(data_lines) > 3:
        overview_text += f"\n... (共 {len(data_lines)} 行数据)"
    overview_content = f"【表格概览】\n{context_str}\n---\n{overview_text}"
    overview_doc = Document(
        page_content=overview_content,
        metadata={
            **metadata,
            "is_table": True,
            "chunk_type": "table_overview",
            "table_title": table_title,
            "table_headers": header_cols,
            "table_row_count": len(data_lines),
            # 溯源字段
            "source_trace": f"{source} | {table_title} | 概览（前3行）",
        },
    )
    result_docs.append(overview_doc)

    # === 行级 chunk ===
    for row_idx, data_line in enumerate(data_lines):
        # 解析行数据
        row_cells = [cell.strip() for cell in data_line.strip("|").split("|")]

        # 单行 Markdown 表格
        row_table = f"{header_line}\n{sep_line}\n{data_line}"

        # 自然语言摘要
        summary = _generate_row_summary(header_cols, row_cells, table_title)

        # 构建完整 chunk 内容
        row_content = f"【表格上下文】\n{context_str}\n---\n{row_table}\n---\n摘要：{summary}"

        # 行主键：第一列的值（如"布洛芬"、"退热效果"等）
        row_primary_key = row_cells[0] if row_cells else ""

        row_doc = Document(
            page_content=row_content,
            metadata={
                **metadata,
                "is_table": True,
                "chunk_type": "table_row",
                "table_title": table_title,
                "table_headers": header_cols,
                "row_index": row_idx,
                "row_primary_key": row_primary_key,
                "row_summary": summary,
                # 溯源字段
                "source_trace": f"{source} | {table_title} | 行{row_idx}: {row_primary_key}",
            },
        )
        result_docs.append(row_doc)

    return result_docs


def _generate_row_summary(
    headers: List[str],
    cells: List[str],
    table_title: str,
) -> str:
    """生成单行数据的自然语言摘要

    策略：
        1. 第一列作为主维度（如"退热效果"、"布洛芬"）
        2. 其余列按"字段：值"拼接
        3. 附加表格标题作为上下文

    示例：
        headers=["项目", "对乙酰氨基酚", "布洛芬"]
        cells=["每日最大量", "2000mg", "1200mg"]
        → "在退热药物对比中，每日最大量：对乙酰氨基酚为2000mg，布洛芬为1200mg"
    """
    if not headers or not cells:
        return ""

    # 对齐 headers 和 cells（可能不等长）
    min_len = min(len(headers), len(cells))
    headers = headers[:min_len]
    cells = cells[:min_len]

    primary_key = cells[0] if cells else ""

    # 判断表格类型：第一列是"维度"还是"实体"
    # 如果第一列的 header 是"项目"/"症状"/"指标"等→维度表（对比表）
    # 如果第一列的 header 是"药物"/"名称"/"项目"等→实体表（参数表）
    _DIMENSION_HEADERS = {"项目", "症状", "指标", "参数", "类型", "特征", "区别", "方面"}
    is_dimension_table = headers[0] in _DIMENSION_HEADERS if headers else False

    parts = []
    if is_dimension_table:
        # 对比表：每行是一个维度，列是不同实体
        # 如"每日最大量：对乙酰氨基酚为2000mg，布洛芬为1200mg"
        for i in range(1, min_len):
            if cells[i]:
                parts.append(f"{headers[i]}为{cells[i]}")
        summary = f"{primary_key}：" + "，".join(parts) if parts else primary_key
    else:
        # 实体表：每行是一个实体，列是不同属性
        # 如"布洛芬：退热效果良好，止痛效果中重度，抗炎作用有"
        for i in range(1, min_len):
            if cells[i]:
                parts.append(f"{headers[i]}{cells[i]}")
        summary = f"{primary_key}：" + "，".join(parts) if parts else primary_key

    # 附加表格标题
    if table_title:
        summary = f"在{table_title}中，{summary}"

    return summary


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
