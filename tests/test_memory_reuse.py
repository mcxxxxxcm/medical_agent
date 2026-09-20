"""回归测试：长期记忆复用（P0 用药史注入 / P1 症状趋势 / P2 跨会话澄清）

覆盖新增的"读了长期记忆"路径（此前长期记忆"重写轻读"，用药史/症状趋势/查询历史
只写不读）：
    1. check_medication_history_interactions — 新建议药 vs 历史用药相互作用核对
    2. run_medication_guide_review 端到端（注入 current_medications）
    3. _build_memory_context_section — 用药史 + 症状趋势注入 prompt
    4. _build_clarify_answer 跨会话提示（P2）
"""

import pytest


# ===== 1. P0 历史用药相互作用核对 =====
def test_history_interaction_both_directions():
    from app.skills.medication_guide_engine import check_medication_history_interactions
    # 回答建议布洛芬，用户历史在服阿司匹林 → 命中相互作用
    r = check_medication_history_interactions(["布洛芬"], ["阿司匹林"])
    assert r["has_interaction"] is True
    assert r["involved_history_drugs"] == ["布洛芬"]
    assert any(i["drug_a"] == "布洛芬" and i["drug_b"] == "阿司匹林" for i in r["interactions"])

    # 反向：回答建议阿司匹林，历史在服布洛芬 → 也命中
    r2 = check_medication_history_interactions(["阿司匹林"], ["布洛芬"])
    assert r2["has_interaction"] is True


def test_history_interaction_no_conflict():
    from app.skills.medication_guide_engine import check_medication_history_interactions
    r = check_medication_history_interactions(["布洛芬"], ["对乙酰氨基酚"])
    assert r["has_interaction"] is False


def test_history_interaction_edge_cases():
    from app.skills.medication_guide_engine import check_medication_history_interactions
    # 无历史药
    assert check_medication_history_interactions(["布洛芬"], None)["has_interaction"] is False
    assert check_medication_history_interactions(["布洛芬"], [])["has_interaction"] is False
    # 同药不算相互作用
    assert check_medication_history_interactions(["布洛芬"], ["布洛芬"])["has_interaction"] is False


# ===== 2. P0 端到端注入 =====
def test_run_medication_guide_review_injects_history_warning():
    from app.skills.medication_guide_engine import run_medication_guide_review
    res = run_medication_guide_review(
        "头痛可以服用布洛芬，每次1片。",
        clinical_checkpoint=None,
        user_profile=None,
        current_medications=["阿司匹林"],
    )
    assert res["status"] == "revise"
    assert "medication_history_interaction" in res["risk_tags"]
    assert "正在服用的药物" in res["revised_answer"]


def test_run_medication_guide_review_no_history_is_pass_or_other():
    from app.skills.medication_guide_engine import run_medication_guide_review
    # 无历史用药时不触发 history_interaction 风险
    res = run_medication_guide_review(
        "头痛可以服用布洛芬，每次1片。",
        clinical_checkpoint=None,
        user_profile=None,
        current_medications=None,
    )
    assert "medication_history_interaction" not in res["risk_tags"]
    # 无药物回答不进入用药审查
    res2 = run_medication_guide_review("今天天气不错。", None, None, None)
    assert res2["status"] == "pass"


# ===== 3. P1/P0 prompt 注入段 =====
def test_build_memory_context_section_meds_and_trends():
    from app.graph.nodes.nodes import _build_memory_context_section
    state = {
        "current_medications": ["阿司匹林", "布洛芬"],
        "symptom_trends": {
            "头痛": {"count": 3, "first_ts": 1, "last_ts": 3},
            "发烧": {"count": 1, "first_ts": 2, "last_ts": 2},
        },
    }
    text = _build_memory_context_section(state)
    assert "阿司匹林" in text and "布洛芬" in text
    # 只注入复发>=2次的症状趋势
    assert "头痛" in text and "已出现3次" in text
    assert "发烧" not in text
    # 明确标注仅供核对、非建议来源
    assert "仅供用药安全核对参考" in text


def test_build_memory_context_section_empty_meds():
    from app.graph.nodes.nodes import _build_memory_context_section
    assert _build_memory_context_section({}) == ""
    state = {"current_medications": None, "symptom_trends": None}
    assert _build_memory_context_section(state) == ""


# ===== 4. P2 跨会话澄清提示 =====
def test_clarify_answer_no_user_id_does_not_hit_db():
    from app.graph.nodes.nodes import _build_clarify_answer
    ans = _build_clarify_answer("还有别的药吗")
    assert "抱歉" in ans


def test_clarify_answer_cross_session_hint(monkeypatch):
    from app.graph.nodes import nodes

    class FakeMemory:
        def get_query_history(self, user_id, limit=10):
            return [{"question": "阿司匹林可以长期吃吗"}, {"question": "你好"}]

    # 注入假记忆，命中历史咨询话题
    monkeypatch.setattr(nodes, "get_long_term_memory", lambda: FakeMemory())
    ans = nodes._build_clarify_answer("这个药还能吃吗", user_id="u1")
    assert "阿司匹林" in ans or "之前咨询过" in ans

    # 记忆缺失（抛异常）→ 静默回落为普通澄清，不炸
    def boom():
        raise RuntimeError("no db")
    monkeypatch.setattr(nodes, "get_long_term_memory", boom)
    ans2 = nodes._build_clarify_answer("这个药还能吃吗", user_id="u2")
    assert "抱歉" in ans2