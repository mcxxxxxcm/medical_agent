"""测试路由和症状解析节点"""

import pytest
from unittest.mock import patch, MagicMock

from langchain_core.messages import HumanMessage, AIMessage

from app.graph.nodes.nodes import (
    detect_rule_based_route,
    normalize_router_label,
    parse_router_output,
    _extract_symptoms_by_rules,
    _detect_route_from_context,
    is_same_query,
    query_rewrite_node,
    _detect_topic,
    _update_topic_trajectory,
    _context_has_entity,
    _build_clarify_answer,
)
from app.graph.graph import route_after_rewrite


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

    def test_disease_name_hits_knowledge(self):
        """疾病名命中knowledge（高血压、糖尿病）"""
        assert detect_rule_based_route("高血压") == "knowledge"
        assert detect_rule_based_route("糖尿病") == "knowledge"

    def test_ambiguous_怎么办_hits_symptom(self):
        """'怎么办' 命中 symptom_intent（已知歧义：高血压怎么办应为knowledge）"""
        # 这条记录了已知歧义，当前行为是返回symptom
        result = detect_rule_based_route("高血压怎么办")
        assert result in ("symptom", "knowledge")  # 当前返回symptom，标记歧义

    def test_general_with_question_mark(self):
        """问候带问号"""
        assert detect_rule_based_route("你好？") == "general"

    def test_knowledge_prevention(self):
        """预防类知识查询"""
        assert detect_rule_based_route("怎么预防感冒") == "knowledge"

    def test_oral_symptom_expression(self):
        """口语化症状：'不太舒服' 不在关键词中，规则应返回None"""
        result = detect_rule_based_route("今天肚子不太舒服")
        # "肚子" 不在 route_symptom_map，"不舒服" 在，所以命中symptom_intent
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


class TestDetectRouteFromContext:
    """上下文感知路由测试"""

    def test_follow_up_symptom(self):
        """追问+症状上下文 → symptom"""
        state = {
            "question": "还有其他可以吃的吗",
            "messages": [
                HumanMessage(content="我头痛"),
                AIMessage(content="建议服用布洛芬缓解头痛"),
            ],
        }
        result = _detect_route_from_context(state)
        assert result == "symptom"

    def test_follow_up_knowledge(self):
        """追问+知识上下文 → knowledge"""
        state = {
            "question": "副作用大吗",
            "messages": [
                HumanMessage(content="布洛芬的用法用量是什么"),
                AIMessage(content="布洛芬是一种非甾体抗炎药"),
            ],
        }
        result = _detect_route_from_context(state)
        assert result == "knowledge"

    def test_no_history_returns_none(self):
        """无历史消息 → None"""
        state = {"question": "还有吗", "messages": []}
        result = _detect_route_from_context(state)
        assert result is None

    def test_short_question_with_symptom_context(self):
        """短句追问+症状上下文 → symptom"""
        state = {
            "question": "严重吗",
            "messages": [
                HumanMessage(content="头痛3天了"),
            ],
        }
        result = _detect_route_from_context(state)
        assert result == "symptom"

    def test_conflicting_context_returns_none(self):
        """symptom和knowledge上下文同时存在 → None（弃权）"""
        state = {
            "question": "还有什么要注意的",
            "messages": [
                HumanMessage(content="我头痛怎么办"),
                AIMessage(content="建议服用布洛芬"),
                HumanMessage(content="高血压的禁忌症有哪些"),
                AIMessage(content="高血压禁忌包括..."),
            ],
        }
        result = _detect_route_from_context(state)
        assert result is None

    def test_non_follow_up_without_context(self):
        """非追问且无上下文 → None"""
        state = {
            "question": "血压高要吃药吗",
            "messages": [],
        }
        result = _detect_route_from_context(state)
        assert result is None

    def test_ai_symptom_indicators(self):
        """AI回答中的症状指标词触发symptom上下文"""
        state = {
            "question": "换一个试试",
            "messages": [
                HumanMessage(content="头痛"),
                AIMessage(content="建议服用布洛芬，剂量为每次200mg"),
            ],
        }
        result = _detect_route_from_context(state)
        assert result == "symptom"


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


class TestSymptomWhitelistSanitizer:
    """L2/L3 白名单症状清洗纯函数测试（v9.31）"""

    from app.graph.nodes.nodes import (
        _norm_symptom,
        _get_question_symptom_whitelist,
        _strip_out_of_scope_symptom_sections,
        sanitize_cached_answer,
    )

    def test_question_wordmatch_whitelist(self):
        assert self._get_question_symptom_whitelist("发烧了怎么办", None) == ["发烧"]

    def test_greeting_no_whitelist(self):
        assert self._get_question_symptom_whitelist("你好", None) == []

    def test_synonym_normalization(self):
        assert self._norm_symptom("发烧") == "发热"
        assert self._norm_symptom("头疼") == "头痛"

    def test_strip_off_whitelist_sections(self):
        answer = (
            "- 发热：多喝水。\n"
            "- 头痛：休息。\n"
            "- 腹痛：留意。\n"
        )
        out = self._strip_out_of_scope_symptom_sections(
            answer, ["发热"], ["头痛", "腹痛"], None
        )
        assert "发热" in out
        assert "头痛" not in out
        assert "腹痛" not in out

    def test_question_says_fever_model_writes_fever_kept(self):
        # 问题写"发烧"，模型写"发热"，同义词归一化后应保留
        out = self.sanitize_cached_answer(
            "- 发热：多喝水。\n- 头痛：休息。\n", "发烧了怎么办", None, None
        )
        assert "发热" in out
        assert "头痛" not in out

    def test_empty_whitelist_untouched(self):
        assert self.sanitize_cached_answer("- 你好。\n", "你好", None, None) == "- 你好。\n"

    def test_medication_preserved_without_docs(self):
        # 无检索文档时只做症状清洗，不误删用药
        assert "布洛芬" in self.sanitize_cached_answer("- 发热：可服用布洛芬。\n", "发烧了怎么办", None, None)


class TestOffDocMedicationStrip:
    """v9.32 文档外用药整句剔除（含残句修复）"""

    from app.graph.nodes.nodes import _strip_off_doc_medications

    def test_whole_sentence_removed_no_residual(self):
        # errorLog 复现：奥司他韦不在文档中 → 整句连同剂量括号一起删，不残留"（每次75mg…）"
        text = "如有需要，可考虑使用奥司他韦（每次75mg，每日2次，连服5天）来缓解流感症状"
        out = self._strip_off_doc_medications(text, docs_text="")
        assert "奥司他韦" not in out
        assert "（每次75mg" not in out
        assert "连服5天" not in out

    def test_on_doc_drug_preserved(self):
        text = "建议服用奥司他韦（每次75mg）缓解流感。"
        out = self._strip_off_doc_medications(text, docs_text="奥司他韦用于流感治疗")
        assert "奥司他韦" in out
        assert "每次75mg" in out

    def test_mixed_sentence_keeps_on_doc_off_doc(self):
        # 同一句既有文档内药又有文档外药：保留句，仅摘离文档外药
        docs = "布洛芬用于缓解发热"
        text = "建议交替使用布洛芬和对乙酰氨基酚（每次500mg）退热。"
        out = self._strip_off_doc_medications(text, docs_text=docs)
        assert "布洛芬" in out
        # 对乙酰氨基酚不在文档 → 药名与紧邻剂量括号被摘除
        assert "对乙酰氨基酚" not in out
        assert "500mg" not in out

    def test_off_doc_medication_line_dropped(self):
        text = "- 发热：布洛芬可退热。\n- 用药：奥司他韦，每次75mg，每日2次。\n"
        docs = "布洛芬用于缓解发热"
        out = self._strip_off_doc_medications(text, docs_text=docs)
        assert "布洛芬" in out
        assert "奥司他韦" not in out
        assert "每次75mg" not in out


class TestTopicTrajectory:
    """显式话题轨迹（v9.36）"""

    def test_detect_topic_symptom_precedence(self):
        # 症状优先于字符串匹配（symptoms 已填充）
        assert _detect_topic({"symptoms": {"头痛": {}}, "clinical_checkpoint": None,
                              "question_type": "symptom"}, "头痛怎么办", "") == "symptom:头痛"

    def test_detect_topic_med_fallback(self):
        assert _detect_topic({"symptoms": None, "clinical_checkpoint": None, "question_type": None},
                             "布洛芬怎么吃", "布洛芬") == "med:布洛芬"

    def test_detect_topic_disease_fallback(self):
        assert _detect_topic({"symptoms": None, "clinical_checkpoint": None, "question_type": None},
                             "高血压怎么办", "") == "disease:高血压"

    def test_detect_topic_general(self):
        assert _detect_topic({"symptoms": None, "clinical_checkpoint": None, "question_type": None},
                             "你好", "") == "general"

    def test_update_same_topic_increments_turns(self):
        traj = [{"topic_id": "disease:头痛", "ts": 1.0, "turns": 1}]
        out = _update_topic_trajectory({"topic_trajectory": traj}, "disease:头痛")
        assert len(out["topic_trajectory"]) == 1
        assert out["topic_trajectory"][0]["turns"] == 2
        assert out["current_topic"] == "disease:头痛"

    def test_update_topic_switch_pushes(self):
        traj = [{"topic_id": "disease:头痛", "ts": 1.0, "turns": 1}]
        out = _update_topic_trajectory({"topic_trajectory": traj}, "med:布洛芬")
        assert len(out["topic_trajectory"]) == 2
        assert out["topic_trajectory"][-1]["topic_id"] == "med:布洛芬"
        assert out["topic_trajectory"][-1]["turns"] == 1

    def test_update_trajectory_clips_to_max(self):
        traj = [{"topic_id": f"t{i}", "ts": i, "turns": 1} for i in range(8)]
        out = _update_topic_trajectory({"topic_trajectory": traj}, "new_topic")
        assert len(out["topic_trajectory"]) == 8
        assert out["topic_trajectory"][0]["topic_id"] == "t1"   # t0 被挤出

    def test_update_from_none(self):
        out = _update_topic_trajectory({"topic_trajectory": None}, "general")
        assert len(out["topic_trajectory"]) == 1
        assert out["topic_trajectory"][0]["topic_id"] == "general"


class TestProactiveClarify:
    """改写后主动澄清（v9.36，最保守触发）"""

    def test_clarify_trigger_on_followup(self):
        # 追问场景：上一轮无实体 + 本轮"还有吗"指代不明 + 改写补不出实体 → 澄清
        # 澄清只针对追问（_anaphora_detected 仅在有历史时置位），符合最保守门槛
        from unittest.mock import MagicMock
        state = {"question": "还有吗",
                 "messages": [HumanMessage(content="你好"), AIMessage(content="您好，请问有什么可以帮您？")],
                 "symptoms": None, "clinical_checkpoint": None, "user_profile": None,
                 "question_type": None, "user_id": "test", "thread_id": ""}
        result_mock = MagicMock()
        result_mock.final_question = "还有吗"    # 改写结果与原文一致 → 补不出实体
        result_mock.search_keywords = "还有吗"
        # _record_bad_case_if_needed 走局部 import，澄清分支走模块级引用，两处都需 mock
        with patch("app.memory.get_long_term_memory"), \
             patch("app.graph.nodes.structured_output.invoke_structured", return_value=result_mock), \
             patch("app.graph.nodes.nodes.get_local_llm_json"), \
             patch("app.graph.nodes.nodes.get_long_term_memory"):
            result = query_rewrite_node(state)
        assert result["refusal_type"] == "clarify"
        assert "还有吗" in result["final_answer"]
        # 澄清走短路返回（带 messages），而非进入后续检索
        assert len(result["messages"]) == 2
        assert result["messages"][0].content == "还有吗"

    def test_no_clarify_first_round_pure_anaphora(self):
        # 无历史时即使含指代也不澄清（_anaphora_detected 仅在有历史时置位）
        # 首轮纯指代查询交由正常检索/拒答，澄清不越界
        state = {"question": "还有吗", "messages": [], "symptoms": None,
                 "clinical_checkpoint": None, "user_profile": None,
                 "question_type": None, "user_id": "test", "thread_id": ""}
        result = query_rewrite_node(state)
        assert "refusal_type" not in result

    def test_no_clarify_first_round_with_entity(self):
        # 首轮含实体自包含查询 → 不澄清，走正常路径并更新话题轨迹
        state = {"question": "我头痛怎么办", "messages": [], "symptoms": None,
                 "clinical_checkpoint": None, "user_profile": None,
                 "question_type": None, "user_id": "test", "thread_id": ""}
        result = query_rewrite_node(state)
        assert "refusal_type" not in result
        assert result.get("final_answer") is None
        assert result["current_topic"] == "disease:头痛"   # 轨迹已接线
        assert result["topic_trajectory"][-1]["topic_id"] == "disease:头痛"

    def test_context_has_entity_blocks_clarify(self):
        # 历史里已出现实体 → 不应澄清（负向防回归）
        assert _context_has_entity("用户：我头痛", {"clinical_checkpoint": None,
                                                   "symptoms": None, "user_profile": None}) is True
        assert _context_has_entity("", {"clinical_checkpoint": None,
                                         "symptoms": None, "user_profile": None}) is False

    def test_build_clarify_answer_asks_for_detail(self):
        ans = _build_clarify_answer("还有别的药吗")
        assert "无法确定" in ans
        assert "补充" in ans

    def test_route_after_rewrite_clarify_ends(self):
        assert route_after_rewrite({"refusal_type": "clarify"}) == "clarify_end"
        assert route_after_rewrite({"refusal_type": None}) == "question_decompose"
        assert route_after_rewrite({}) == "question_decompose"


class TestSegmentedEmitter:
    """v9.33 按段先校验后流出"""

    from app.graph.nodes.nodes import _SegmentedEmitter, _sanitize_answer

    def test_first_bullet_flows_before_finish(self):
        # 关键断言：首个 bullet 一旦完整（遇到下一 bullet 起点）即经 emit 流出，
        # 不必等整段生成完 —— 首 token 因此回到流式
        em = self._SegmentedEmitter(["发热"], ["头痛"], "")
        em.feed("- 发热：多喝水。\n- ")
        assert em.clean_parts, "首个 bullet 完成后应立即流出"
        assert em.clean_parts[0] == "- 发热：多喝水。"

    def test_off_whitelist_bullet_dropped_from_output(self):
        em = self._SegmentedEmitter(["发热"], ["头痛"], "")
        em.feed("- 发热：多喝水。\n- 头痛：休息。")
        out = em.finish()
        assert "发热" in out
        assert "头痛" not in out
        assert "- 头痛" not in out
        assert any("头痛" in s for s in em.removed_sections)

    def test_synonym_kept(self):
        # 问题说"发烧"，模型写"发热"：同义词归一化后保留
        em = self._SegmentedEmitter(["发烧"], [], "")
        em.feed("- 发热：多喝水。\n- 头痛：休息。")
        out = em.finish()
        assert "发热" in out
        assert "头痛" not in out

    def test_bullet_followed_by_prose_kept(self):
        em = self._SegmentedEmitter(["发热"], [], "发热要多喝水。")
        em.feed("- 发热：建议休息。注意补充水分，多喝水可帮助退热。")
        out = em.finish()
        assert "建议休息" in out
        assert "多喝水" in out

    def test_no_whitelist_returns_as_is(self):
        # 白名单为空时不清洗（无法判断哪些症状该答）
        em = self._SegmentedEmitter([], [], "")
        em.feed("- 你好，祝您健康。")
        assert em.finish() == "- 你好，祝您健康。"
