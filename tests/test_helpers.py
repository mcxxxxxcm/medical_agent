"""测试辅助工具函数（extract_json_block, _coerce_list_fields 等）"""

import pytest
from app.graph.nodes.helpers import extract_json_block, _coerce_list_fields


class TestExtractJsonBlock:
    """extract_json_block 的 5 层回退解析测试"""

    def test_direct_json(self):
        """直接 JSON 解析"""
        result = extract_json_block('{"key": "value"}')
        assert result == {"key": "value"}

    def test_markdown_fenced_json(self):
        """Markdown 代码块中的 JSON"""
        result = extract_json_block('```json\n{"symptoms": ["头痛"]}\n```')
        assert result == {"symptoms": ["头痛"]}

    def test_nested_json(self):
        """嵌套 JSON 对象"""
        result = extract_json_block('{"user": {"name": "张三", "age": 30}}')
        assert result == {"user": {"name": "张三", "age": 30}}

    def test_brace_extraction(self):
        """从文本中提取最大的 {...} 块"""
        result = extract_json_block('结果是 {"question_type": "symptom"}，请参考')
        assert result == {"question_type": "symptom"}

    def test_empty_input(self):
        """空字符串返回 None"""
        assert extract_json_block("") is None
        assert extract_json_block(None) is None

    def test_list_field_coercion(self):
        """字符串列表字段自动转为列表"""
        data = {"symptoms": "头痛,发烧"}
        _coerce_list_fields(data)
        assert data["symptoms"] == ["头痛", "发烧"]

    def test_list_field_already_list(self):
        """已是列表不改变"""
        data = {"symptoms": ["头痛"]}
        _coerce_list_fields(data)
        assert data["symptoms"] == ["头痛"]

    def test_list_field_none(self):
        """None 转为空列表"""
        data = {"symptoms": None}
        _coerce_list_fields(data)
        assert data["symptoms"] == []

    def test_nested_dict_flattening(self):
        """嵌套字典元素自动展平"""
        data = {"red_flags": [{"symptom": "头痛加重"}, "发烧"]}
        _coerce_list_fields(data)
        assert data["red_flags"] == ["头痛加重", "发烧"]

    def test_chinese_comma_split(self):
        """中文逗号分割"""
        data = {"symptoms": "头痛，发烧，咳嗽"}
        _coerce_list_fields(data)
        assert data["symptoms"] == ["头痛", "发烧", "咳嗽"]


class TestDrugKeywords:
    """药物关键词常量测试"""

    def test_keywords_not_empty(self):
        from app.graph.nodes.helpers import _DRUG_KEYWORDS, _DRUG_INTENT_KEYWORDS
        assert len(_DRUG_KEYWORDS) > 0
        assert len(_DRUG_INTENT_KEYWORDS) > 0
        assert "布洛芬" in _DRUG_KEYWORDS
        assert "怎么吃" in _DRUG_INTENT_KEYWORDS
