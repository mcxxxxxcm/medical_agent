"""文档元数据自动提取模块（多源交叉校验）

解决的核心问题：
    文档入库时缺少版本、生效日期、权威等级等元数据，
    导致检索时无法做版本冲突裁决。

设计原则：
    1. 多源提取：文件名解析 + PDF/DOCX属性 + LLM正文提取 + 文件系统时间
    2. 交叉校验：多个独立来源的结果互相验证
    3. 准确性保障：只有来源一致时才自动确认，不一致则标记待人工审核
    4. 置信度分级：high（自动确认）/ mid（需人工确认）/ low（缺失需填写）

提取流程：
    文件名解析 → PDF/DOCX属性 → LLM正文提取 → 文件系统时间
         ↓              ↓              ↓              ↓
    交叉校验引擎 → 一致性评分 → 自动确认 / 标记待审核
"""
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.documents import Document

from app.core.app_logging import get_logger

logger = get_logger(__name__)


# ===== 常量 =====

# 权威等级优先级（数值越高越权威）
AUTHORITY_PRIORITY = {
    "national_guideline": 5,     # 国家指南
    "society_consensus": 4,      # 协会共识
    "institutional_protocol": 3, # 机构规程
    "textbook": 3,               # 教材
    "expert_opinion": 2,         # 专家意见
    "training_material": 1,      # 培训材料
    "faq": 0,                    # FAQ
    "unknown": -1,               # 未知
}

# 权威等级关键词映射（用于LLM提取和文件名解析）
AUTHORITY_KEYWORDS = {
    "national_guideline": ["国家指南", "国家标准", "国家卫健委", "国家药监局", "国标"],
    "society_consensus": ["共识", "学会", "协会", "中华医学"],
    "institutional_protocol": ["规程", "规范", "制度", "医院", "诊疗规范", "临床路径"],
    "textbook": ["教材", "教科书", "内科学", "外科学", "儿科学", "妇产科学"],
    "expert_opinion": ["专家", "意见", "建议", "解读"],
    "training_material": ["培训", "讲义", "课件", "教案"],
    "faq": ["问答", "FAQ", "常见问题", "百问"],
}

# 医学体系关键词
MEDICAL_SYSTEM_KEYWORDS = {
    "western": ["西医", "现代医学", "临床", "循证", "指南", "共识"],
    "tcm": ["中医", "中药", "针灸", "推拿", "辨证", "经方", "方剂"],
    "integrated": ["中西医结合", "中西医保建"],
}

# 适用人群关键词
POPULATION_KEYWORDS = {
    "adult": ["成人", "成年人", "老年", "中老年"],
    "pediatric": ["儿童", "小儿", "新生儿", "婴幼儿", "儿科"],
    "maternal": ["孕妇", "产妇", "妊娠", "哺乳期"],
    "general": ["通用", "全年龄", "大众"],
}

# 文件名命名规范正则
# 支持格式：{文档名}_{版本号}_{日期}_{权威等级}.{ext}
# 或：{文档名}_{版本号}_{日期}.{ext}
_FILENAME_PATTERN_V1 = re.compile(
    r"^(.+?)_v(\d+(?:\.\d+)*)_(\d{4}[-]?\d{2}[-]?\d{2})_(\w+)$"
)
_FILENAME_PATTERN_V2 = re.compile(
    r"^(.+?)_v(\d+(?:\.\d+)*)_(\d{4}[-]?\d{2}[-]?\d{2})$"
)
_FILENAME_PATTERN_V3 = re.compile(
    r"^(.+?)_(\d{4}[-]?\d{2}[-]?\d{2})$"
)
# 日期格式：20250301 或 2025-03-01
_FILENAME_DATE_PATTERN = re.compile(r"(\d{4})[-]?(\d{2})[-]?(\d{2})")


# ===== 源1：文件名解析 =====

def extract_from_filename(file_path: Path) -> Dict[str, Any]:
    """从文件名中解析元数据

    支持的命名规范：
        - {文档名}_v{版本号}_{日期}_{权威等级}.{ext}
        - {文档名}_v{版本号}_{日期}.{ext}
        - {文档名}_{日期}.{ext}

    示例：
        - 发热诊断指南_v2_20250301_national.pdf
        - 高血压用药共识_v1.1_2025-06-01_society.pdf
        - 内科培训手册_20250101.docx
    """
    result = {"metadata_source": "filename"}
    stem = file_path.stem
    matched = False

    # 格式1：完整四段式
    m = _FILENAME_PATTERN_V1.match(stem)
    if m:
        result["title"] = m.group(1).strip()
        result["version"] = m.group(2)
        result["effective_date"] = _normalize_date(m.group(3))
        result["authority_level"] = _normalize_authority(m.group(4))
        matched = True

    # 格式2：三段式（无权威等级）
    if not matched:
        m = _FILENAME_PATTERN_V2.match(stem)
        if m:
            result["title"] = m.group(1).strip()
            result["version"] = m.group(2)
            result["effective_date"] = _normalize_date(m.group(3))
            matched = True

    # 格式3：两段式（仅标题+日期）
    if not matched:
        m = _FILENAME_PATTERN_V3.match(stem)
        if m:
            result["title"] = m.group(1).strip()
            result["effective_date"] = _normalize_date(m.group(2))
            matched = True

    if not matched:
        result["title"] = stem

    # 从标题中提取权威等级（即使文件名不含权威等级字段）
    if "authority_level" not in result:
        for level, keywords in AUTHORITY_KEYWORDS.items():
            for kw in keywords:
                if kw in stem:
                    result["authority_level"] = level
                    break
            if "authority_level" in result:
                break

    # 从标题中提取医学体系
    for system, keywords in MEDICAL_SYSTEM_KEYWORDS.items():
        for kw in keywords:
            if kw in stem:
                result["medical_system"] = system
                break
        if "medical_system" in result:
            break

    # 从标题中提取适用人群
    for pop, keywords in POPULATION_KEYWORDS.items():
        for kw in keywords:
            if kw in stem:
                result["applicable_population"] = pop
                break
        if "applicable_population" in result:
            break

    return result


def _normalize_date(date_str: str) -> str:
    """将各种日期格式标准化为 YYYY-MM-DD"""
    m = _FILENAME_DATE_PATTERN.match(date_str)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return date_str


def _normalize_authority(authority_str: str) -> str:
    """将权威等级字符串标准化"""
    authority_str = authority_str.lower().strip()
    # 直接匹配枚举值
    if authority_str in AUTHORITY_PRIORITY:
        return authority_str
    # 关键词匹配
    for level, keywords in AUTHORITY_KEYWORDS.items():
        if authority_str in keywords or authority_str == level:
            return level
    return "unknown"


# ===== 源2：PDF/DOCX 内嵌属性 =====

def extract_from_file_properties(file_path: Path) -> Dict[str, Any]:
    """从 PDF/DOCX 文件的内嵌属性中提取元数据

    PDF：使用 PyMuPDF（fitz）提取 creationDate/modDate/author/title/subject
    DOCX：使用 python-docx 提取 core_properties
    """
    result = {"metadata_source": "file_properties"}
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        result.update(_extract_pdf_properties(file_path))
    elif suffix == ".docx":
        result.update(_extract_docx_properties(file_path))
    else:
        # 其他格式无法提取内嵌属性
        pass

    return result


def _extract_pdf_properties(file_path: Path) -> Dict[str, Any]:
    """提取 PDF 内嵌元数据"""
    props = {}
    try:
        import fitz
        doc = fitz.open(str(file_path))
        meta = doc.metadata or {}
        doc.close()

        # 标题
        title = meta.get("title", "").strip()
        if title and title != "Untitled" and len(title) > 2:
            props["title"] = title

        # 作者/发布机构
        author = meta.get("author", "").strip()
        if author and author != "Unknown":
            props["issuing_body"] = author

        # 创建日期
        creation_date = meta.get("creationDate", "").strip()
        if creation_date:
            parsed = _parse_pdf_date(creation_date)
            if parsed:
                props["creation_date"] = parsed

        # 修改日期
        mod_date = meta.get("modDate", "").strip()
        if mod_date:
            parsed = _parse_pdf_date(mod_date)
            if parsed:
                props["modification_date"] = parsed

        # 主题/关键词
        subject = meta.get("subject", "").strip()
        if subject:
            props["subject"] = subject

        keywords = meta.get("keywords", "").strip()
        if keywords:
            props["keywords"] = keywords

        # 从标题和作者中推断权威等级
        text_for_inference = f"{title} {author} {subject}"
        for level, keywords_list in AUTHORITY_KEYWORDS.items():
            for kw in keywords_list:
                if kw in text_for_inference:
                    props["authority_level"] = level
                    break
            if "authority_level" in props:
                break

    except ImportError:
        logger.debug("PyMuPDF(fitz) 未安装，跳过PDF属性提取")
    except Exception as e:
        logger.warning(f"PDF属性提取失败：{e}")

    return props


def _extract_docx_properties(file_path: Path) -> Dict[str, Any]:
    """提取 DOCX 内嵌元数据"""
    props = {}
    try:
        from docx import Document as DocxDocument

        doc = DocxDocument(str(file_path))
        cp = doc.core_properties

        if cp.title and cp.title.strip():
            props["title"] = cp.title.strip()

        if cp.author and cp.author.strip():
            props["issuing_body"] = cp.author.strip()

        if cp.created:
            props["creation_date"] = cp.created.strftime("%Y-%m-%d")

        if cp.modified:
            props["modification_date"] = cp.modified.strftime("%Y-%m-%d")

        if cp.subject and cp.subject.strip():
            props["subject"] = cp.subject.strip()

        if cp.keywords and cp.keywords.strip():
            props["keywords"] = cp.keywords.strip()

        # 从属性中推断权威等级
        text_for_inference = f"{cp.title or ''} {cp.author or ''} {cp.subject or ''}"
        for level, keywords_list in AUTHORITY_KEYWORDS.items():
            for kw in keywords_list:
                if kw in text_for_inference:
                    props["authority_level"] = level
                    break
            if "authority_level" in props:
                break

    except ImportError:
        logger.debug("python-docx 未安装，跳过DOCX属性提取")
    except Exception as e:
        logger.warning(f"DOCX属性提取失败：{e}")

    return props


def _parse_pdf_date(date_str: str) -> Optional[str]:
    """解析PDF日期格式

    PDF日期格式：D:YYYYMMDDHHmmSSOHH'mm'
    示例：D:20250301120000+08'00'
    """
    if not date_str:
        return None
    # 去掉 D: 前缀
    clean = date_str.replace("D:", "")
    # 提取前8位：YYYYMMDD
    digits = re.sub(r"[^0-9]", "", clean)
    if len(digits) >= 8:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    return None


# ===== 源3：LLM 正文提取 =====

def extract_from_content(documents: List[Document], file_path: Path) -> Dict[str, Any]:
    """从文档正文前500字中用LLM提取元数据

    适用场景：文件名不规范且PDF/DOCX属性为空时的兜底方案。
    医疗文档的开头通常包含版本、日期、发布机构等信息。
    """
    result = {"metadata_source": "llm_content"}

    # 拼接前500字
    head_text = ""
    for doc in documents:
        head_text += doc.page_content + "\n"
        if len(head_text) >= 500:
            break
    head_text = head_text[:500].strip()

    if not head_text or len(head_text) < 30:
        return result

    try:
        from app.core.llm import get_local_llm
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = get_local_llm()
        messages = [
            SystemMessage(content=_META_EXTRACT_SYSTEM_PROMPT),
            HumanMessage(content=f"文档标题：{file_path.stem}\n\n文档内容：\n{head_text}"),
        ]
        response = llm.invoke(messages)
        parsed = _parse_llm_meta_response(response.content)
        result.update(parsed)

    except Exception as e:
        logger.warning(f"LLM元数据提取失败：{e}")

    return result


_META_EXTRACT_SYSTEM_PROMPT = """你是一个文档元数据提取助手。从文档内容中提取以下元数据字段。

【输出规则】
1. 只输出JSON格式，不要输出其他内容
2. 无法从文档中确定的信息，对应字段留空字符串""
3. 日期格式统一为 YYYY-MM-DD
4. authority_level 只能是以下值之一：national_guideline / society_consensus / institutional_protocol / textbook / expert_opinion / training_material / faq / unknown
5. medical_system 只能是以下值之一：western / tcm / integrated / unknown
6. applicable_population 只能是以下值之一：adult / pediatric / maternal / general / unknown

【输出格式】
{"version": "", "effective_date": "", "authority_level": "", "issuing_body": "", "medical_system": "", "applicable_population": ""}

【提取规则】
- version：文档中明确标注的版本号，如"第2版"、"v1.3"、"2025年版"中的版本部分
- effective_date：文档中标注的发布日期或生效日期，不是文件创建时间
- authority_level：根据发布机构推断，如"国家卫健委"→national_guideline，"中华医学会"→society_consensus
- issuing_body：发布机构名称
- medical_system：西医/中医/中西医结合
- applicable_population：文档明确针对的人群"""


def _parse_llm_meta_response(response_text: str) -> Dict[str, Any]:
    """解析LLM返回的元数据JSON"""
    result = {}
    try:
        # 提取JSON部分
        json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
        if json_match:
            import json
            data = json.loads(json_match.group())

            # 只取非空值
            for key in ["version", "effective_date", "authority_level",
                        "issuing_body", "medical_system", "applicable_population"]:
                val = data.get(key, "").strip()
                if val and val != "unknown":
                    result[key] = val

    except Exception as e:
        logger.warning(f"解析LLM元数据响应失败：{e}")

    return result


# ===== 源4：文件系统时间 =====

def extract_from_filesystem(file_path: Path) -> Dict[str, Any]:
    """从文件系统获取时间元数据（最不可靠，仅作回退）

    注意：mtime可能是复制/下载时间，非文档原始时间
    """
    result = {"metadata_source": "filesystem"}

    try:
        stat = file_path.stat()
        # 创建时间（Windows为出生时间，Linux为元数据变更时间）
        ctime = datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d")
        # 修改时间
        mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d")

        result["file_created_date"] = ctime
        result["file_modified_date"] = mtime

        # 作为effective_date的低置信度回退
        if mtime:
            result["effective_date"] = mtime
            result["effective_date_source"] = "filesystem_mtime"

    except Exception as e:
        logger.warning(f"文件系统元数据提取失败：{e}")

    return result


# ===== 交叉校验引擎 =====

def cross_validate_metadata(
    filename_meta: Dict[str, Any],
    properties_meta: Dict[str, Any],
    llm_meta: Dict[str, Any],
    filesystem_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """多源交叉校验，生成最终元数据 + 置信度

    校验规则：
        1. 多个来源的值一致 → 取该值，confidence=high
        2. 多数来源一致 → 取多数值，confidence=mid，标记需人工确认
        3. 来源冲突 → 取优先级最高的来源，confidence=mid，标记需人工确认
        4. 仅单一来源 → 取该来源，confidence=low
        5. 全部缺失 → 标记 unknown，confidence=none

    来源优先级（同字段冲突时）：
        文件名 > PDF/DOCX属性 > LLM提取 > 文件系统时间
    """
    # 合并所有来源，按字段聚合
    field_sources = _aggregate_sources(filename_meta, properties_meta, llm_meta, filesystem_meta)

    result = {}
    pending_review = []  # 需人工确认的字段

    for field, sources in field_sources.items():
        # 跳过元数据来源标记字段
        if field in ("metadata_source", "effective_date_source", "file_created_date", "file_modified_date"):
            continue

        resolved = _resolve_field(field, sources)
        result[field] = resolved["value"]
        result[f"{field}_confidence"] = resolved["confidence"]
        result[f"{field}_source"] = resolved["source"]

        if resolved["confidence"] in ("mid", "low", "none"):
            pending_review.append({
                "field": field,
                "value": resolved["value"],
                "confidence": resolved["confidence"],
                "sources": sources,
                "reason": resolved["reason"],
            })

    # 总体置信度
    high_count = sum(1 for k, v in result.items() if k.endswith("_confidence") and v == "high")
    total_fields = sum(1 for k in result if k.endswith("_confidence"))
    if total_fields > 0 and high_count == total_fields:
        result["overall_confidence"] = "high"
    elif total_fields > 0 and high_count >= total_fields * 0.6:
        result["overall_confidence"] = "mid"
    else:
        result["overall_confidence"] = "low"

    result["pending_review"] = pending_review
    result["needs_manual_review"] = len(pending_review) > 0

    return result


def _aggregate_sources(*source_dicts: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """将多个来源的元数据按字段聚合

    Returns:
        {field: [{"value": ..., "source": "filename"}, ...]}
    """
    source_names = ["filename", "file_properties", "llm_content", "filesystem"]
    aggregated = {}

    for source_dict, source_name in zip(source_dicts, source_names):
        for key, value in source_dict.items():
            if key in ("metadata_source", "effective_date_source",
                       "file_created_date", "file_modified_date"):
                continue
            if not value or value == "unknown":
                continue
            if key not in aggregated:
                aggregated[key] = []
            aggregated[key].append({"value": str(value), "source": source_name})

    return aggregated


def _resolve_field(field: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """解析单个字段的最终值和置信度

    优先级：filename > file_properties > llm_content > filesystem
    """
    if not sources:
        return {
            "value": "unknown",
            "confidence": "none",
            "source": "none",
            "reason": "所有来源均无此字段",
        }

    # 按来源优先级排序
    source_priority = {"filename": 4, "file_properties": 3, "llm_content": 2, "filesystem": 1}

    # 单一来源无法交叉验证 → low（否则 filesystem mtime 等粗粒度来源
    # 会被当成 high 直接写入 doc_effective_date，污染元数据）
    if len(sources) == 1:
        only = sources[0]
        return {
            "value": only["value"],
            "confidence": "low",
            "source": only["source"],
            "reason": f"仅 {only['source']} 单一来源，需人工复核",
        }

    # 值一致性检查
    unique_values = set(s["value"] for s in sources)

    if len(unique_values) == 1:
        # 多来源一致
        best_source = max(sources, key=lambda s: source_priority.get(s["source"], 0))
        return {
            "value": sources[0]["value"],
            "confidence": "high",
            "source": best_source["source"],
            "reason": f"{len(sources)}个来源一致",
        }

    # 多数一致
    from collections import Counter
    value_counts = Counter(s["value"] for s in sources)
    most_common_value, most_common_count = value_counts.most_common(1)[0]

    if most_common_count > len(sources) / 2:
        return {
            "value": most_common_value,
            "confidence": "mid",
            "source": "majority_vote",
            "reason": f"多数来源一致({most_common_count}/{len(sources)})，但有冲突值",
        }

    # 无多数，按来源优先级取
    sorted_sources = sorted(sources, key=lambda s: source_priority.get(s["source"], 0), reverse=True)
    return {
        "value": sorted_sources[0]["value"],
        "confidence": "mid",
        "source": sorted_sources[0]["source"] + "(priority)",
        "reason": f"来源冲突，取优先级最高的来源：{sorted_sources[0]['source']}",
    }


# ===== 主入口 =====

def extract_document_metadata(
    file_path: Path,
    documents: Optional[List[Document]] = None,
) -> Dict[str, Any]:
    """文档元数据自动提取主入口

    执行四源提取 + 交叉校验，返回带置信度的元数据。

    Args:
        file_path: 文档文件路径
        documents: 已加载的Document列表（用于LLM正文提取）

    Returns:
        {
            "version": "v2", "version_confidence": "high", ...
            "overall_confidence": "high",
            "pending_review": [...],
            "needs_manual_review": False,
        }
    """
    logger.info(f"开始提取文档元数据：{file_path.name}")

    # 源1：文件名解析
    filename_meta = extract_from_filename(file_path)
    logger.info(f"  [源1-文件名] 提取到 {len([k for k in filename_meta if k != 'metadata_source'])} 个字段")

    # 源2：PDF/DOCX内嵌属性
    properties_meta = extract_from_file_properties(file_path)
    logger.info(f"  [源2-文件属性] 提取到 {len([k for k in properties_meta if k != 'metadata_source'])} 个字段")

    # 源3：LLM正文提取（仅当源1和源2信息不足时触发）
    llm_meta = {"metadata_source": "llm_content"}
    _has_sufficient_meta = (
        filename_meta.get("version") or filename_meta.get("effective_date")
    ) or (
        properties_meta.get("version") or properties_meta.get("effective_date")
    )
    if not _has_sufficient_meta and documents:
        llm_meta = extract_from_content(documents, file_path)
        logger.info(f"  [源3-LLM正文] 提取到 {len([k for k in llm_meta if k != 'metadata_source'])} 个字段")
    else:
        logger.info("  [源3-LLM正文] 跳过（源1/2已有足够信息）")

    # 源4：文件系统时间
    filesystem_meta = extract_from_filesystem(file_path)
    logger.info(f"  [源4-文件系统] 提取到 {len([k for k in filesystem_meta if k != 'metadata_source'])} 个字段")

    # 交叉校验
    result = cross_validate_metadata(filename_meta, properties_meta, llm_meta, filesystem_meta)

    logger.info(
        f"元数据提取完成：overall_confidence={result.get('overall_confidence')}, "
        f"needs_manual_review={result.get('needs_manual_review')}, "
        f"pending_fields={len(result.get('pending_review', []))}"
    )

    return result


def apply_metadata_to_documents(
    documents: List[Document],
    extracted_meta: Dict[str, Any],
) -> List[Document]:
    """将提取的元数据应用到文档列表

    只写入置信度 >= mid 的字段，low/none 的字段不写入（避免错误元数据污染检索）
    """
    # 可写入的字段列表（排除置信度、来源等辅助字段）
    writable_fields = [
        "version", "effective_date", "authority_level",
        "issuing_body", "medical_system", "applicable_population",
        "applicable_region", "title",
    ]

    # 过期日期：effective_date + 3年（默认有效期为3年，可配置）
    effective_date = extracted_meta.get("effective_date")
    expire_date = None
    if effective_date and effective_date != "unknown":
        try:
            from datetime import timedelta
            dt = datetime.strptime(effective_date, "%Y-%m-%d")
            expire_date = (dt + timedelta(days=365 * 3)).strftime("%Y-%m-%d")
        except Exception:
            pass

    for doc in documents:
        for field in writable_fields:
            value = extracted_meta.get(field)
            confidence = extracted_meta.get(f"{field}_confidence", "none")

            if value and value != "unknown" and confidence in ("high", "mid"):
                doc.metadata[f"doc_{field}"] = value
            elif confidence == "low":
                # 低置信度字段写入但标记为待确认
                doc.metadata[f"doc_{field}_pending"] = value or "unknown"

        # 写入有效期
        if expire_date:
            doc.metadata["doc_expire_date"] = expire_date

        # 写入元数据整体置信度
        doc.metadata["doc_meta_confidence"] = extracted_meta.get("overall_confidence", "unknown")
        doc.metadata["doc_needs_meta_review"] = extracted_meta.get("needs_manual_review", False)

    return documents
