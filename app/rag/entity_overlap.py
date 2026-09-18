"""实体词抽取 + 查询-文档实体重叠判定

把 jieba 实体抽取抽成共享模块，供 graph 的启发式过滤（filter_relevant_docs）
与 hybrid_retriever 的精排实体兜底补全两处复用，避免 hybrid_retriever 反向 import
nodes.py（会循环导入）。

旧实现用 re.findall 把中文整句当单个 token，"布洛芬每天最大剂量是多少"无法与任何
文档匹配，启发式过滤形同虚设。本实现用 jieba 切出 query 的实质实体词（药名/症状/
疾病名，如"布洛芬"），再判断文档是否含其中任意一个。

v9.65：纯字面子串匹配对同义词（用户"发烧"↔文档"发热"、用户"流鼻血"↔文档"鼻出血"、
用户"宝宝"↔文档"婴儿"）判成不相关，导致相关文档被整个过滤/无法补回。改为在匹配前
先把实体词做同义词扩展（expand_term），term 的任一等价形式命中即算重叠，双向覆盖。
"""
from typing import Dict, List

import jieba

_GENERIC_TERMS = frozenset("""
怎么办 怎么样 怎么治 怎么处理 怎么缓解 怎么吃 吃什么 如何 该如何 需要 应该 可以 能不能
多少 最大 剂量 每天 每次 一次 几次 时间 多久 什么时候 是什么 哪些 有什么 会不会 是不是
还是 哪个 治疗 用药 用法 用量 注意 事项 医院 医生 立即 严重 症状 处理 方法 建议 现在 目前
请问 咨询 想看 想 问 一下 帮 帮忙 麻烦 正常 是否 有没有 得了 出现 引起 导致 常见 主要
""".split())


# 口语/书面变体 → 规范词（方向对齐 keyword_matcher 标准词，见 nodes.py v9.52 注释）
# 只收录医疗上确切的同义/近义对，避免把语义不同的词合并；只收多字（≥2）词。
# 来源：
#   - nodes.py `symptom_alias`（方向推断用）原 9 条
#   - nodes.py `_SYMPTOM_CANONICAL`（答案清洗用）中不冲突的条目
#   - v9.65 实测缺口：流鼻血→鼻出血、宝宝→婴儿
SYMPTOM_SYNONYMS: Dict[str, str] = {
    # 抽取自 nodes.py `symptom_alias`（方向不变）
    "咽痛": "嗓子疼", "咽喉痛": "嗓子疼", "喉咙痛": "嗓子疼",
    "咽喉炎": "咽炎", "头昏": "头晕", "流涕": "流鼻涕",
    "发烧": "发热", "咽喉": "嗓子", "腹疼": "腹痛",
    # 抽取自 nodes.py `_SYMPTOM_CANONICAL`（"喉咙痛"排除，避免与上面 1561 方向冲突）
    "头疼": "头痛", "肚子疼": "腹痛", "肚痛": "腹痛",
    "拉肚子": "腹泻", "眩晕": "头晕", "恶心想吐": "恶心", "量体温": "发热",
    # v9.65 实测缺口
    "流鼻血": "鼻出血", "宝宝": "婴儿",
    # v9.69 实测缺口（黄金诊断 狗咬伤 0 召回、退烧/剂量、呕血、拨打120 共用要点漏）：
    # 过滤层 has_query_overlap / 锚定检索 normalize_term 双向命中都靠这张表，方向对准文档用词
    "狗咬伤": "犬咬伤", "狗咬": "犬咬",
    "退烧药": "解热镇痛", "退热药": "解热镇痛", "退烧": "解热镇痛", "退热": "解热镇痛",
    "吐血": "呕血",
    "打120": "拨打120",
}

# 由 SYMPTOM_SYNONYMS 反推：规范词 → 其全部变体（含自身）。用于等价类扩展命中。
_CANONICAL_VARIANTS: Dict[str, List[str]] = {}
for _variant, _canonical in SYMPTOM_SYNONYMS.items():
    _lst = _CANONICAL_VARIANTS.setdefault(_canonical, [_canonical])
    if _variant not in _lst:
        _lst.append(_variant)


def normalize_term(term: str) -> str:
    """变体 → 规范词；无映射原样返回。"""
    return SYMPTOM_SYNONYMS.get(term, term)


def expand_term(term: str) -> List[str]:
    """返回 term 的整个等价类（含自身与所有变体），用于双向命中判定。"""
    canonical = normalize_term(term)
    return _CANONICAL_VARIANTS.get(canonical) or [term]


# 实体锚定检索用的过泛词（hybrid_retriever._entity_anchored_search 跳过这些锚词）。
# 注意："怎么"能通过 extract_entity_terms（不在 _GENERIC_TERMS），但作为锚词过泛，
# 必须在此拦下，避免用问句泛词做定向检索污染结果。
ANCHOR_SKIP_TERMS = frozenset("""
什么 怎么 如何 为什么 哪个 哪些 多少 多久 什么时候 是不是 会不会 应该 需要 可以
得了 出现 引起 常见 正常 检查 治疗 用药 处理 症状 方法 建议 主要 目前 现在
""".split())


def extract_entity_terms(question: str) -> List[str]:
    """用 jieba 切 question，去掉问句泛词/虚词，返回可作主题实体的候选词"""
    if not question:
        return []
    terms: List[str] = []
    seen = set()
    for w in (ww.strip() for ww in jieba.cut(question)):
        if not w or len(w) < 2 or w in _GENERIC_TERMS:
            continue
        if w not in seen:
            seen.add(w)
            terms.append(w)
    return terms


def has_query_overlap(question: str, doc_content: str) -> bool:
    """基于实体词重叠的轻量相关性判断

    无实体词可判（纯泛化问句）时返回 True，不下相关性断言，避免把合法知识问句的召回全误杀。
    v9.65：匹配前做同义词扩展，term 任一等价形式命中即算重叠（双向覆盖"发烧/发热"等）。
    """
    terms = extract_entity_terms(question)
    if not terms:
        return True
    content = (doc_content or "").lower()
    for t in terms:
        for v in expand_term(t):
            if v and v in content:
                return True
    return False