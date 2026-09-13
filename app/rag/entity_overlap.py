"""实体词抽取 + 查询-文档实体重叠判定

把 jieba 实体抽取抽成共享模块，供 graph 的启发式过滤（filter_relevant_docs）
与 hybrid_retriever 的精排实体兜底补全两处复用，避免 hybrid_retriever 反向 import
nodes.py（会循环导入）。

旧实现用 re.findall 把中文整句当单个 token，"布洛芬每天最大剂量是多少"无法与任何
文档匹配，启发式过滤形同虚设。本实现用 jieba 切出 query 的实质实体词（药名/症状/
疾病名，如"布洛芬"），再判断文档是否含其中任意一个。
"""
from typing import List

import jieba

_GENERIC_TERMS = frozenset("""
怎么办 怎么样 怎么治 怎么处理 怎么缓解 怎么吃 吃什么 如何 该如何 需要 应该 可以 能不能
多少 最大 剂量 每天 每次 一次 几次 时间 多久 什么时候 是什么 哪些 有什么 会不会 是不是
还是 哪个 治疗 用药 用法 用量 注意 事项 医院 医生 立即 严重 症状 处理 方法 建议 现在 目前
请问 咨询 想看 想 问 一下 帮 帮忙 麻烦 正常 是否 有没有 得了 出现 引起 导致 常见 主要
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
    """
    terms = extract_entity_terms(question)
    if not terms:
        return True
    content = (doc_content or "").lower()
    return any(t in content for t in terms)