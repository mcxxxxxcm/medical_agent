"""测试路由和症状解析节点"""

import pytest
from unittest.mock import patch, MagicMock

from app.graph.nodes.nodes import (
    detect_rule_based_route,
    normalize_router_label,
    parse_router_output,
    _extract_symptoms_by_rules,
    is_same_query,
)


class TestDetectRuleBasedRoute:
    """规则路由测试"""

    def test_symptom_route_headache(self):
        assert detect_rule_based_route("我头痛怎么办") == "symptom"

    def test_symptom_route_fever(self):
        assert detect_rule_based_route("发烧了吃什么药") == "symptom"

    def test_symptom_route_pain(self):
        assert detect_rule_based_route("肚子疼") == "symptom"

    def test_knowledge_route(self):
        assert detect_rule_based_route("高血压是什么原因") == "knowledge"

    def test_general_greeting(self):
        assert detect_rule_based_route("你好") == "general"

    def test_general_who_are_you(self):
        assert detect_rule_based_route("你是谁") == "general"

    def test_empty_input(self):
        assert detect_rule_based_route("") == "general"

    def test_unknown_fallback(self):
        """未匹配任何规则时返回 None，交给 LLM"""
        assert detect_rule_based_route("今天天气不错") is None

    def test_symptom_priority_over_general(self):
        """症状关键词优先级高于 general（'你好我是王艺涵发烧了'）"""
        result = detect_rule_based_route("你好我是王艺涵发烧了")
        assert result == "symptom"


class TestNormalizeRouterLabel:
    """路由标签规范化测试"""

    def test_valid_labels(self):
        assert normalize_router_label("symptom") == "symptom"
        assert normalize_router_label("knowledge") == "knowledge"
        assert normalize_router_label("general") == "general"

    def test_chinese_labels(self):
        assert normalize_router_label("症状") == "symptom"
        assert normalize_router_label("知识") == "knowledge"

    def test_fallback_default(self):
        assert normalize_router_label("unknown") == "general"
        assert normalize_router_label("") == "general"


class TestParseRouterOutput:
    """路由输出解析测试"""

    def test_json_output(self):
        result = parse_router_output('{"question_type": "symptom"}')
        assert result == "symptom"

    def test_plain_text_output(self):
        result = parse_router_output("symptom")
        assert result == "symptom"

    def test_empty_output(self):
        assert parse_router_output("") is None
        assert parse_router_output(None) is None


class TestExtractSymptomsByRules:
    """基于规则的症状提取测试"""

    def test_basic_symptom(self):
        result = _extract_symptoms_by_rules("我头痛")
        assert result is not None
        assert "头痛" in result["symptoms"]

    def test_multiple_symptoms(self):
        result = _extract_symptoms_by_rules("发烧咳嗽流鼻涕")
        assert result is not None
        symptoms = result["symptoms"]
        assert "发烧" in symptoms
        assert "咳嗽" in symptoms

    def test_severity_detection(self):
        result = _extract_symptoms_by_rules("我肚子非常疼，很严重")
        assert result is not None
        assert result["severity"] == "严重"

    def test_body_part_detection(self):
        result = _extract_symptoms_by_rules("我胸口疼")
        assert result is not None
        assert result["body_parts"] is not None
        assert any("胸" in p for p in result["body_parts"])

    def test_duration_detection(self):
        result = _extract_symptoms_by_rules("头痛3天了")
        assert result is not None
        assert result["duration"] is not None

    def test_no_symptom_returns_none(self):
        result = _extract_symptoms_by_rules("今天天气真好")
        assert result is None

    def test_deduplication(self):
        result = _extract_symptoms_by_rules("头痛头疼头痛")
        assert result is not None
        assert result["symptoms"].count("头痛") == 1

    def test_pain_pattern_fallback(self):
        """'X疼' 通用模式兜底"""
        result = _extract_symptoms_by_rules("手腕疼")
        assert result is not None
        assert any("手腕" in s for s in result["symptoms"])


class TestIsSameQuery:
    """查询相似性判断测试"""

    def test_identical(self):
        assert is_same_query("头痛怎么办", "头痛怎么办") is True

    def test_different(self):
        assert is_same_query("头痛怎么办", "咳嗽吃什么药") is False

    def test_none_handling(self):
        assert is_same_query(None, "test") is False
        assert is_same_query("test", None) is False
        assert is_same_query(None, None) is False


class TestGradeDocuments:
    """文档评分节点测试（需要 mock 状态）"""

    def test_oscillation_detection_no_improvement(self):
        """重试后分数无改善时应跳过重复重试"""
        from app.graph.nodes.nodes import grade_documents_node

        # 模拟重试场景：前次分数 0.1，当前分数 0.12（delta < 0.05）
        mock_doc = MagicMock()
        mock_doc.metadata = {"rerank_score": 0.12, "source": "test.txt"}
        mock_doc.page_content = "测试内容"

        state = {
            "question": "头痛怎么办",
            "retrieved_docs": [mock_doc],
            "retrieval_attempts": 1,
            "rewritten_query": "头痛如何处理",
            "_prev_max_score": 0.1,
            "_prev_relevant_count": 0,
        }

        from unittest.mock import patch
        with patch("app.graph.nodes.nodes.filter_relevant_docs", return_value=[]):
            result = grade_documents_node(state)
            # 由于 score_delta < 0.05 且 doc_delta < 1，应该跳过重试
            assert result.goto == "answer_generation"
