"""工具注册表 + 审计日志测试"""
import tempfile

import pytest

from app.tools import (
    TOOL_REGISTRY,
    enabled_tools,
    get_tool,
    invoke_tool,
    register_tool,
    tool_descriptions,
)


# ===================================================================
# 注册完备性
# ===================================================================

def test_core_tools_registered():
    names = set(TOOL_REGISTRY)
    expected = {"safety_review", "symptom_triage", "medication_guide", "retrieval"}
    assert expected.issubset(names)


def test_core_tools_enabled_by_default():
    enabled = set(enabled_tools())
    assert {"safety_review", "symptom_triage", "medication_guide"}.issubset(enabled)


def test_metadata_nonempty():
    for name in ("safety_review", "symptom_triage", "medication_guide"):
        tool = TOOL_REGISTRY[name]
        assert tool.name
        assert isinstance(tool.description, str) and tool.description
        assert callable(tool.handler)
        assert tool.kind == "rule_engine"


def test_retrieval_tool_disabled_by_default():
    tool = get_tool("retrieval")
    assert tool is not None
    assert tool.enabled is False
    assert tool.kind == "retrieval"


def test_tool_descriptions_contains_tool_names():
    text = tool_descriptions()
    assert text
    for name in ("safety_review", "symptom_triage", "medication_guide"):
        assert name in text


# ===================================================================
# invoke_tool 基本行为 + 审计
# ===================================================================

@pytest.fixture
def temp_audit_collector(monkeypatch):
    """把 MetricsCollector 指到临时库，并开启审计。"""
    from app.core import metrics
    from app.core.metrics import MetricsCollector
    from app.core import config

    tmp = tempfile.mktemp(suffix="_audit.db")
    collector = MetricsCollector(db_path=tmp)
    monkeypatch.setattr(metrics, "_metrics_collector", collector)
    # 强制开审计
    monkeypatch.setattr(config.settings, "ENABLE_TOOL_AUDIT", True)
    return collector


def test_invoke_tool_returns_result_and_audits(temp_audit_collector):
    # 明确含确诊断言的句子 → 触发 revise
    result = invoke_tool(
        "safety_review",
        request_id="r_1", thread_id="t_1",
        answer="你这个情况就是肺癌，必须马上住院。",
    )
    assert result["status"] in ("pass", "revise")

    rows = temp_audit_collector.get_tool_audit(hours=1)
    safety_rows = [r for r in rows if r["tool_name"] == "safety_review"]
    assert safety_rows, "应当写入一条 safety_review 审计"
    row = safety_rows[0]
    assert row["request_id"] == "r_1"
    assert row["thread_id"] == "t_1"
    assert row["status"] == "ok"
    assert row["duration_ms"] >= 0


def test_invoke_tool_audit_disabled_writes_nothing(monkeypatch):
    from app.core import metrics
    from app.core.metrics import MetricsCollector
    from app.core import config

    tmp = tempfile.mktemp(suffix="_audit_off.db")
    collector = MetricsCollector(db_path=tmp)
    monkeypatch.setattr(metrics, "_metrics_collector", collector)
    monkeypatch.setattr(config.settings, "ENABLE_TOOL_AUDIT", False)

    invoke_tool("safety_review", request_id="r_off", answer="你好。")
    assert collector.get_tool_audit(hours=1) == []


def test_invoke_tool_unknown_tool_raises():
    with pytest.raises(KeyError):
        invoke_tool("does_not_exist", answer="x")


def test_invoke_tool_error_is_audited(monkeypatch):
    from app.core import metrics
    from app.core.metrics import MetricsCollector
    from app.core import config

    tmp = tempfile.mktemp(suffix="_audit_err.db")
    collector = MetricsCollector(db_path=tmp)
    monkeypatch.setattr(metrics, "_metrics_collector", collector)
    monkeypatch.setattr(config.settings, "ENABLE_TOOL_AUDIT", True)


def test_register_tool_and_get():
    def _handler(x):
        return {"status": "pass", "value": x}

    register_tool("__test_tool__", "测试工具", _handler, enabled=True, kind="rule_engine")
    tool = get_tool("__test_tool__")
    assert tool is not None
    assert tool.name == "__test_tool__"
    # 清理注册，避免影响其他测试
    TOOL_REGISTRY.pop("__test_tool__", None)

    # 重新注册同名应覆盖不抛错
    register_tool("__test_tool__", "测试工具2", _handler)
    assert get_tool("__test_tool__").description == "测试工具2"
    TOOL_REGISTRY.pop("__test_tool__", None)


# ===================================================================
# 回归：领域引擎本体仍可直调（改造只是加统一入口，未动引擎函数）
# ===================================================================

def test_engines_still_callable_directly():
    from app.skills.safety_review_engine import run_rule_based_review
    from app.skills.medication_guide_engine import run_medication_guide_review
    from app.skills.symptom_triage_engine import run_symptom_triage

    assert callable(run_rule_based_review)
    assert callable(run_medication_guide_review)
    assert callable(run_symptom_triage)