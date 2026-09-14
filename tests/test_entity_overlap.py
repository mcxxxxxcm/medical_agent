"""测试 entity_overlap 同义词对齐（v9.65 修复 A）。

覆盖：同义词双向命中、expand/normalize 等价类、无别名词条零变化、
无实体词兜底不动、方向推断共享表替换无回归。全部离线，无网络/LLM。
"""

import pytest

from app.rag.entity_overlap import (
    SYMPTOM_SYNONYMS,
    expand_term,
    has_query_overlap,
    normalize_term,
)
from app.graph.nodes.nodes import _infer_disease_direction


class TestHasQueryOverlapSynonym:
    """同义词（口语↔文档规范写法）应双向命中"""

    def test_faver_shou_fare(self):
        assert has_query_overlap("发烧了怎么处理", "发热是儿童最常见的急症")

    def test_fare_shou_faver(self):
        assert has_query_overlap("发热了怎么处理", "发烧时应注意补水")

    def test_liubixue_nasang(self):
        assert has_query_overlap("流鼻血怎么处理", "鼻出血时不要仰头")

    def test_baobao_yinger(self):
        assert has_query_overlap("宝宝发烧怎么办", "婴儿发热时需立即就医")


class TestHasQueryOverlapNoAliasUnchanged:
    """无同义词别名的词条，行为与改动前逐字一致"""

    def test_no_alias_still_hits(self):
        assert has_query_overlap("高血压急症怎么识别", "高血压患者血压急剧升高时需警惕")

    def test_irrelevant_doc_misses(self):
        assert not has_query_overlap("高血压急症怎么识别", "荨麻疹通常是过敏引起")


class TestHasQueryOverlapGenericFallback:
    """无实体可判（纯泛化问句）时返回 True，不误杀"""

    def test_pure_question_words(self):
        # "怎么办" 全部落在 _GENERIC_TERMS → 无实体词 → 返回 True
        assert has_query_overlap("怎么办", "随便什么内容")


class TestExpandTerm:
    def test_fever_eq_class(self):
        assert set(expand_term("发烧")) >= {"发烧", "发热"}

    def test_symmetric_farer(self):
        # 双向：从规范词也能展开出变体
        assert "发烧" in expand_term("发热")

    def test_liubixue(self):
        assert set(expand_term("流鼻血")) >= {"流鼻血", "鼻出血"}

    def test_no_mapping(self):
        assert expand_term("高血压") == ["高血压"]


class TestNormalizeTerm:
    def test_liubixue(self):
        assert normalize_term("流鼻血") == "鼻出血"

    def test_baobao(self):
        assert normalize_term("宝宝") == "婴儿"

    def test_no_mapping(self):
        assert normalize_term("高血压") == "高血压"


class TestSharedTableInferDirectionUnchanged:
    """nodes.py 共享表替换后，方向推断输出与改前一致（防回归）"""

    def test_fever_direction(self):
        assert "上呼吸道感染" in _infer_disease_direction("发烧了怎么办", {"symptoms": ["发烧"]})

    def test_soret_direction(self):
        assert "上呼吸道感染" in _infer_disease_direction("嗓子疼怎么办", {"symptoms": ["嗓子疼"]})

    def test_diarrhea_direction(self):
        assert "急性肠胃炎" in _infer_disease_direction("又吐又拉怎么办", {"symptoms": ["呕吐", "腹泻"]})

    def test_empty(self):
        assert _infer_disease_direction("", {"symptoms": []}) == []


class TestSynonymTableSanity:
    def test_only_multi_char_entries(self):
        # 只收多字词，避免单字词过度归一
        for variant in SYMPTOM_SYNONYMS:
            assert len(variant) >= 2