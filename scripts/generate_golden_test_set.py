"""黄金测试集自动生成脚本

从 docs/medical/ 目录的文档中提取结构化信息，半自动生成 RAG 评估测试用例。

功能：
    1. 解析 Markdown 文档结构（章节、药物、症状、禁忌、剂量）
    2. 按模板生成测试查询（用药安全、禁忌症、症状分诊、剂量查询、急救、慢性病管理）
    3. 提取 key_facts 用于确定性校验
    4. 输出 JSONL 格式（兼容 RAGEvaluator.load_test_set）

用法：
    # 生成测试用例并输出到 stdout
    python scripts/generate_golden_test_set.py

    # 追加到黄金测试集文件
    python scripts/generate_golden_test_set.py --append

    # 仅分析文档结构（不生成测试用例）
    python scripts/generate_golden_test_set.py --analyze

    # 指定输出文件
    python scripts/generate_golden_test_set.py -o tests/data/golden_test_set_auto.jsonl

注意：
    - 自动生成的用例需要人工审核 ground_truth 和 key_facts
    - 建议作为增量扩充的手段，核心场景仍应人工编写
"""
import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# 文档结构解析
# ---------------------------------------------------------------------------

@dataclass
class DocSection:
    """文档章节"""
    title: str
    level: int  # ## = 2, ### = 3
    content: str
    children: List["DocSection"] = field(default_factory=list)

@dataclass
class ParsedDoc:
    """解析后的文档"""
    filename: str
    title: str
    sections: List[DocSection] = field(default_factory=list)
    drugs: Set[str] = field(default_factory=set)
    symptoms: Set[str] = field(default_factory=set)
    contraindications: List[str] = field(default_factory=list)
    dosages: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def parse_document(filepath: Path) -> ParsedDoc:
    """解析 Markdown 文档，提取结构化信息"""
    text = filepath.read_text(encoding="utf-8")
    lines = text.split("\n")

    doc = ParsedDoc(
        filename=filepath.name,
        title=filepath.stem,
    )

    current_section = None
    current_content_lines: List[str] = []

    for line in lines:
        # 检测标题
        header_match = re.match(r"^(#{1,4})\s+(.+)$", line)
        if header_match:
            # 保存上一个章节
            if current_section:
                current_section.content = "\n".join(current_content_lines).strip()
                _extract_entities(current_section.content, doc)
                doc.sections.append(current_section)

            level = len(header_match.group(1))
            title = header_match.group(2).strip()
            current_section = DocSection(title=title, level=level, content="")
            current_content_lines = []
        else:
            current_content_lines.append(line)

    # 保存最后一个章节
    if current_section:
        current_section.content = "\n".join(current_content_lines).strip()
        _extract_entities(current_section.content, doc)
        doc.sections.append(current_section)

    return doc


# 常见药物名模式
_DRUG_PATTERNS = [
    r"布洛芬", r"对乙酰氨基酚", r"阿司匹林", r"双氯芬酸", r"奥司他韦",
    r"右美沙芬", r"氨溴索", r"溴己新", r"奥美拉唑", r"多潘立酮",
    r"蒙脱石散", r"氯雷他定", r"西替利嗪", r"依巴斯汀", r"红霉素",
    r"炉甘石洗剂", r"氨氯地平", r"缬沙坦", r"硝苯地平", r"二甲双胍",
    r"格列美脲", r"伪麻黄碱", r"氯苯那敏", r"舒马曲坦", r"甲氧氯普胺",
    r"普萘洛尔", r"氟桂利嗪", r"丙戊酸钠", r"苯海拉明", r"昂丹司琼",
    r"沙丁胺醇", r"秋水仙碱", r"依托考昔", r"别嘌醇", r"非布司他",
    r"苯溴马隆", r"阿仑膦酸钠", r"塞来昔布", r"甲钴胺", r"玻璃酸钠",
    r"氨基葡萄糖", r"达格列净", r"西格列汀", r"利拉鲁肽", r"胰岛素",
    r"硝酸甘油", r"他汀类", r"阿昔洛韦", r"伐昔洛韦", r"加巴喷丁",
    r"普瑞巴林", r"莫匹罗星", r"阿达帕林", r"过氧化苯甲酰", r"克林霉素",
    r"多西环素", r"米诺环素", r"异维A酸", r"特比萘芬", r"酮康唑",
    r"伊曲康唑", r"泼尼松", r"氢化可的松", r"糠酸莫米松", r"曲安奈德",
    r"卤米松", r"他克莫司", r"吡美莫司", r"法莫替丁", r"硫糖铝",
    r"枸橼酸铋钾", r"铝碳酸镁", r"匹维溴铵", r"乳果糖", r"聚乙二醇",
    r"洛哌丁胺", r"口服补液盐", r"地舒单抗", r"特立帕肽",
]

# 禁忌相关关键词
_CONTRA_KEYWORDS = [
    "禁用", "禁忌", "禁", "不可", "不得", "严禁", "不应", "不建议",
    "慎用", "避免", "不可使用", "不能使用",
]

# 剂量相关关键词
_DOSAGE_KEYWORDS = [
    "每次", "每日", "mg", "ml", "g", "剂量", "用量", "用法",
    "不超过", "最大", "起始", "维持",
]

# 安全警告关键词
_WARNING_KEYWORDS = [
    "立即就医", "拨打120", "急诊", "危险", "严重", "致命", "危及生命",
    "红旗征", "警示", "警告", "注意",
]


def _extract_entities(content: str, doc: ParsedDoc):
    """从章节内容中提取实体"""
    # 提取药物名
    for pattern in _DRUG_PATTERNS:
        if re.search(pattern, content):
            doc.drugs.add(pattern)

    # 提取禁忌信息
    for kw in _CONTRA_KEYWORDS:
        for sentence in content.split("。"):
            sentence = sentence.strip()
            if kw in sentence and len(sentence) > 5:
                # 提取包含禁忌的句子（简化，取前100字）
                doc.contraindications.append(sentence[:100])

    # 提取剂量信息
    for kw in _DOSAGE_KEYWORDS:
        for sentence in content.split("。"):
            sentence = sentence.strip()
            if kw in sentence and re.search(r"\d+", sentence):
                doc.dosages.append(sentence[:100])

    # 提取安全警告
    for kw in _WARNING_KEYWORDS:
        for sentence in content.split("。"):
            sentence = sentence.strip()
            if kw in sentence and len(sentence) > 5:
                doc.warnings.append(sentence[:100])


# ---------------------------------------------------------------------------
# 测试用例生成模板
# ---------------------------------------------------------------------------

# 按文档分类的查询模板
_QUERY_TEMPLATES = {
    "常见药物使用指南": {
        "用药安全": [
            "{drug}的禁忌症有哪些？",
            "{drug}和{drug2}能一起吃吗？",
            "孕妇能用{drug}吗？",
            "儿童能用{drug}吗？",
        ],
        "剂量查询": [
            "{drug}成人常用剂量是多少？",
            "{drug}每天最多吃多少？",
        ],
    },
    "神经系统症状鉴别指南": {
        "症状分诊": [
            "头痛什么情况下需要立即就医？",
            "偏头痛应该吃什么药？",
            "头晕和眩晕有什么区别？",
        ],
    },
    "慢性病管理与用药指南": {
        "慢性病管理": [
            "高血压的诊断标准是什么？",
            "2型糖尿病一线用药是什么？",
            "低血糖怎么紧急处理？",
            "脑卒中怎么快速识别？",
        ],
        "剂量查询": [
            "{drug}的用法用量是什么？",
        ],
    },
    "呼吸系统疾病诊疗指南": {
        "用药安全": [
            "感冒需要吃抗生素吗？",
            "流感什么时候用奥司他韦最好？",
        ],
        "症状分诊": [
            "感冒和流感怎么区分？",
            "哮喘急性发作怎么处理？",
        ],
    },
    "消化系统疾病诊疗指南": {
        "知识问答": [
            "便秘怎么治疗？",
            "腹泻最重要的是什么？",
            "胃食管反流病怎么治疗？",
        ],
    },
    "发热诊断与家庭护理指南": {
        "用药安全": [
            "儿童发烧能用阿司匹林吗？",
            "物理降温有哪些错误做法？",
        ],
        "症状分诊": [
            "发烧多少度需要吃退烧药？",
            "发烧伴皮疹可能是什么病？",
        ],
    },
    "儿童常见疾病护理指南": {
        "用药安全": [
            "6岁以下儿童能用复方感冒药吗？",
            "3个月以下婴儿发烧怎么办？",
        ],
        "急救处理": [
            "热性惊厥怎么处理？",
            "孩子腹泻脱水怎么判断？",
        ],
    },
    "外伤急救与处理指南": {
        "急救处理": [
            "扭伤后应该冷敷还是热敷？",
            "烧烫伤怎么急救处理？",
            "被狗咬伤了怎么处理？",
        ],
    },
    "急诊常见症状识别与处理指南": {
        "症状分诊": [
            "胸痛可能是什么严重疾病？",
            "急性腹痛能吃止痛药吗？",
        ],
        "急救处理": [
            "过敏性休克怎么处理？",
            "一氧化碳中毒怎么急救？",
            "什么情况需要立即拨打120？",
        ],
    },
    "皮肤疾病诊疗指南": {
        "症状分诊": [
            "荨麻疹什么情况下需要急诊？",
        ],
        "用药安全": [
            "面部湿疹能用什么药膏？",
            "带状疱疹什么时候开始抗病毒治疗最好？",
        ],
    },
    "骨科常见疾病诊疗指南": {
        "知识问答": [
            "颈椎病有哪些分型？",
            "腰椎间盘突出怎么保守治疗？",
            "骨质疏松怎么诊断？",
        ],
        "慢性病管理": [
            "痛风急性发作怎么治疗？",
            "痛风患者饮食要注意什么？",
        ],
    },
}


def generate_test_cases(doc: ParsedDoc) -> List[Dict]:
    """从文档生成测试用例模板"""
    cases = []
    doc_name = doc.title

    # 获取文档对应的查询模板
    templates = _QUERY_TEMPLATES.get(doc_name, {})
    if not templates:
        return cases

    drugs_list = sorted(doc.drugs)
    for category, queries in templates.items():
        for query_tmpl in queries:
            # 替换模板变量
            if "{drug}" in query_tmpl and not drugs_list:
                continue

            if "{drug}" in query_tmpl and "{drug2}" in query_tmpl:
                # 需要两个药物
                if len(drugs_list) < 2:
                    continue
                for i, d1 in enumerate(drugs_list[:3]):
                    for d2 in drugs_list[i+1:i+2]:
                        query = query_tmpl.format(drug=d1, drug2=d2)
                        cases.append({
                            "query": query,
                            "ground_truth": "",  # 需人工填写
                            "key_facts": [],
                            "category": category,
                            "difficulty": "medium",
                            "source_doc": doc.filename,
                            "auto_generated": True,
                        })
            elif "{drug}" in query_tmpl:
                for drug in drugs_list[:3]:  # 每个模板最多3个药物变体
                    query = query_tmpl.format(drug=drug)
                    cases.append({
                        "query": query,
                        "ground_truth": "",  # 需人工填写
                        "key_facts": [],
                        "category": category,
                        "difficulty": "medium",
                        "source_doc": doc.filename,
                        "auto_generated": True,
                    })
            else:
                # 无需药物变量的查询
                cases.append({
                    "query": query_tmpl,
                    "ground_truth": "",  # 需人工填写
                    "key_facts": [],
                    "category": category,
                    "difficulty": "medium",
                    "source_doc": doc.filename,
                    "auto_generated": True,
                })

    return cases


# ---------------------------------------------------------------------------
# 禁忌症专项测试生成
# ---------------------------------------------------------------------------

def generate_contraindication_cases(doc: ParsedDoc) -> List[Dict]:
    """从禁忌信息中生成专项安全测试用例"""
    cases = []
    seen_queries: Set[str] = set()

    for contra in doc.contraindications:
        # 尝试提取药物名和禁忌对象
        for drug in doc.drugs:
            if drug in contra:
                # 查找禁忌人群
                for people in ["孕妇", "儿童", "哺乳期", "肝功能不全", "肾功能不全",
                              "老年人", "妊娠", "12岁以下"]:
                    if people in contra:
                        query = f"{people}能用{drug}吗？"
                        if query not in seen_queries:
                            seen_queries.add(query)
                            cases.append({
                                "query": query,
                                "ground_truth": "",  # 需人工填写
                                "key_facts": [drug, people, "禁用"],
                                "category": "用药安全",
                                "difficulty": "hard",
                                "source_doc": doc.filename,
                                "auto_generated": True,
                                "safety_critical": True,
                            })

    return cases


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="黄金测试集自动生成")
    parser.add_argument("--docs-dir", default="docs/medical",
                       help="医疗文档目录")
    parser.add_argument("-o", "--output", default=None,
                       help="输出文件路径（默认 stdout）")
    parser.add_argument("--append", action="store_true",
                       help="追加到黄金测试集文件")
    parser.add_argument("--analyze", action="store_true",
                       help="仅分析文档结构，不生成测试用例")
    args = parser.parse_args()

    # 查找文档
    docs_dir = Path(args.docs_dir)
    if not docs_dir.exists():
        # 尝试项目根目录下的路径
        docs_dir = Path("d:/Agent/medical_assistant_agent") / args.docs_dir

    doc_files = sorted(docs_dir.glob("*.txt"))
    if not doc_files:
        print(f"错误：在 {docs_dir} 中未找到 .txt 文档", file=sys.stderr)
        sys.exit(1)

    print(f"找到 {len(doc_files)} 个医疗文档")

    # 解析文档
    all_docs: List[ParsedDoc] = []
    for f in doc_files:
        doc = parse_document(f)
        all_docs.append(doc)

    # 分析模式
    if args.analyze:
        for doc in all_docs:
            print(f"\n{'='*60}")
            print(f"文档: {doc.filename}")
            print(f"  章节数: {len(doc.sections)}")
            print(f"  药物数: {len(doc.drugs)}")
            if doc.drugs:
                print(f"  药物: {', '.join(sorted(doc.drugs)[:10])}{'...' if len(doc.drugs)>10 else ''}")
            print(f"  禁忌信息: {len(doc.contraindications)} 条")
            print(f"  剂量信息: {len(doc.dosages)} 条")
            print(f"  安全警告: {len(doc.warnings)} 条")
            for section in doc.sections:
                indent = "  " * (section.level - 1)
                print(f"  {indent}· {section.title} ({len(section.content)} 字)")
        return

    # 生成测试用例
    all_cases: List[Dict] = []
    for doc in all_docs:
        # 按模板生成
        cases = generate_test_cases(doc)
        all_cases.extend(cases)

        # 禁忌专项
        contra_cases = generate_contraindication_cases(doc)
        all_cases.extend(contra_cases)

    # 去重（按 query）
    seen: Set[str] = set()
    unique_cases: List[Dict] = []
    for case in all_cases:
        if case["query"] not in seen:
            seen.add(case["query"])
            unique_cases.append(case)

    # 按类别排序
    category_order = {
        "用药安全": 0, "症状分诊": 1, "急救处理": 2,
        "慢性病管理": 3, "剂量查询": 4, "知识问答": 5,
    }
    unique_cases.sort(key=lambda c: (category_order.get(c["category"], 9), c["query"]))

    # 输出
    output_path = None
    if args.append:
        output_path = Path("tests/data/golden_test_set.jsonl")
        if not output_path.exists():
            output_path = Path("d:/Agent/medical_assistant_agent/tests/data/golden_test_set.jsonl")
    elif args.output:
        output_path = Path(args.output)

    if output_path:
        mode = "a" if args.append else "w"
        with open(output_path, mode, encoding="utf-8") as f:
            for case in unique_cases:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")
        print(f"已写入 {len(unique_cases)} 条到 {output_path}")
    else:
        for case in unique_cases:
            print(json.dumps(case, ensure_ascii=False))

    # 统计
    print(f"\n--- 生成统计 ---", file=sys.stderr)
    print(f"文档数: {len(all_docs)}", file=sys.stderr)
    print(f"总用例数: {len(unique_cases)}", file=sys.stderr)

    # 按类别统计
    category_counts: Dict[str, int] = {}
    for case in unique_cases:
        cat = case["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
    print(f"\n按类别:", file=sys.stderr)
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}", file=sys.stderr)

    # 安全关键用例
    safety_cases = [c for c in unique_cases if c.get("safety_critical")]
    print(f"\n安全关键用例: {len(safety_cases)}", file=sys.stderr)
    print(f"需人工填写 ground_truth: {len([c for c in unique_cases if not c['ground_truth']])}", file=sys.stderr)


if __name__ == "__main__":
    main()
