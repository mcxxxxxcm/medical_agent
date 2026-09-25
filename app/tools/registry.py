"""医疗工具注册表

将领域规则引擎（安全审查 / 症状分诊 / 用药指导）统一注册为标准化「工具」，
并提供受审计的统一调用入口 invoke_tool，以及供 LLM 感知能力的 tool_descriptions。

设计原则（医疗安全底线）：
    - 调度权始终在图流程（graph）手里，注册表只收口「调用入口 + 元数据 + 审计」
    - 审计写库失败只告警不阻断主流程，与 app/core/metrics.py 的容错一致
    - 检索工具默认不注册（ENABLE_RETRIEVAL_TOOL=False），防 LLM 绕过确定性路由/检索管线

用法：
    from app.tools import TOOL_REGISTRY, invoke_tool, tool_descriptions

    result = invoke_tool("safety_review", request_id=req, thread_id=tid,
                          answer=answer, clinical_checkpoint=cp)
"""
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from app.core.app_logging import get_logger
from app.core.config import get_config
from app.core.metrics import get_metrics_collector

logger = get_logger(__name__)


@dataclass
class MedicalTool:
    """一个可被注册表管理的医疗工具。

    name: 唯一标识（与 TOOL_REGISTRY 键一致）
    description: 供 LLM 感知的能力说明（含触发条件/边界）
    handler: 实际执行函数（包装对应的 *_engine 纯函数）
    enabled: 是否可用（False 则 invoke_tool 拒绝调用）
    kind: "rule_engine"（确定性规则引擎）| "retrieval"（检索引擎，默认禁用）
    """

    name: str
    description: str
    handler: Callable[..., Dict[str, Any]]
    enabled: bool = True
    kind: str = "rule_engine"
    registered_at: float = field(default_factory=time.time)

    def to_meta(self) -> Dict[str, str]:
        """序列化为可注入 prompt 的元信息。"""
        return {
            "name": self.name,
            "kind": self.kind,
            "enabled": str(self.enabled).lower(),
            "description": self.description,
        }


# 注册表：name -> MedicalTool
TOOL_REGISTRY: Dict[str, MedicalTool] = {}


def register_tool(
    name: str,
    description: str,
    handler: Callable[..., Dict[str, Any]],
    enabled: bool = True,
    kind: str = "rule_engine",
) -> MedicalTool:
    """注册一个工具。重复注册同名工具时覆盖（带上警告日志）。"""
    if name in TOOL_REGISTRY:
        logger.warning(f"工具 {name} 重复注册，将被覆盖")
    tool = MedicalTool(
        name=name, description=description, handler=handler,
        enabled=enabled, kind=kind,
    )
    TOOL_REGISTRY[name] = tool
    return tool


def get_tool(name: str) -> Optional[MedicalTool]:
    return TOOL_REGISTRY.get(name)


def enabled_tools() -> List[str]:
    """返回所有 enabled 工具名列表（供调试/校验）。"""
    return [n for n, t in TOOL_REGISTRY.items() if t.enabled]


def tool_descriptions() -> str:
    """生成「系统可用工具」说明文本，供 LLM 作为只读规范约束注入。

    不修正调度权——仅告知 LLM 系统具备哪些能力及触发条件。
    """
    if not TOOL_REGISTRY:
        return "（当前系统没有可用工具）"
    lines = ["系统具备以下领域工具（由确定性流程调度，模型不得自行绕过默认流程）："]
    for name in sorted(TOOL_REGISTRY):
        t = TOOL_REGISTRY[name]
        mark = "可用" if t.enabled else "禁用"
        lines.append(f"- {name} [{t.kind}/{mark}]：{t.description}")
    return "\n".join(lines)


def _summarize(value: Any, limit: int = 200) -> str:
    """把参数/结果压缩为审计摘要（截断，避免审计库被长文本撑大）。"""
    if value is None:
        return ""
    text = str(value)
    if not text.strip():
        return ""
    return text[:limit]


def invoke_tool(
    name: str,
    *,
    request_id: str = "",
    thread_id: str = "",
    **kwargs,
) -> Dict[str, Any]:
    """受审计的统一调用入口。

    执行 handler(**kwargs) 并记录调用审计（受 settings.ENABLE_TOOL_AUDIT 控制）。
    异常时记录 status=error 后 re-raise，由调用方（graph 节点）既有 try/except 处理。

    Raises:
        KeyError: 工具未注册
        ValueError: 工具已注册但被禁用
        (原异常): handler 抛出的异常原样向上传递
    """
    tool = get_tool(name)
    if tool is None:
        raise KeyError(f"未知工具：{name}")

    summary_in = _summarize(kwargs)
    start = time.time()
    try:
        result = tool.handler(**kwargs)
        error = ""
        status = "ok"
    except Exception as e:
        result = {}
        error = f"{type(e).__name__}: {e}"
        status = "error"
        logger.warning(f"工具 {name} 执行失败：{error}")
        if _should_audit():
            _record(name, request_id, thread_id, status, start, summary_in, "", error)
        raise

    if _should_audit():
        summary_out = _summarize({k: result.get(k) for k in ("status", "risk_tags")})
        _record(name, request_id, thread_id, status, start, summary_in, summary_out, "")
    return result


def _should_audit() -> bool:
    try:
        return bool(get_config().ENABLE_TOOL_AUDIT)
    except Exception:
        return False


def _record(name, request_id, thread_id, status, start, summary_in, summary_out, error):
    try:
        get_metrics_collector().record_tool_call(
            tool_name=name,
            request_id=request_id,
            thread_id=thread_id,
            status=status,
            duration_ms=(time.time() - start) * 1000,
            input_summary=summary_in,
            output_summary=summary_out,
            error=error,
        )
    except Exception as e:
        logger.warning(f"工具 {name} 审计记录失败：{e}")


# ===================================================================
# 领域工具注册
# ===================================================================

def _safe_review_handler(answer: str, clinical_checkpoint=None):
    from app.skills.safety_review_engine import run_rule_based_review
    return run_rule_based_review(answer, clinical_checkpoint)


def _symptom_triage_handler(symptoms, severity=None, duration_hours=None, clinical_checkpoint=None):
    from app.skills.symptom_triage_engine import run_symptom_triage
    return run_symptom_triage(
        symptoms=symptoms, severity=severity,
        duration_hours=duration_hours, clinical_checkpoint=clinical_checkpoint,
    )


def _medication_guide_handler(answer, clinical_checkpoint=None, user_profile=None, current_medications=None):
    from app.skills.medication_guide_engine import run_medication_guide_review
    return run_medication_guide_review(
        answer, clinical_checkpoint,
        user_profile=user_profile, current_medications=current_medications,
    )


def _retrieval_handler(query, k=5, categories=None, **kwargs):
    from app.rag.hybrid_retriever import get_hybrid_retriever
    retriever = get_hybrid_retriever(k=k, categories=categories)
    docs = retriever.invoke(query)
    return {"status": "ok", "docs": [d.page_content[:300] for d in docs]}


def _register_core_tools():
    register_tool(
        name="safety_review",
        description=(
            "医疗回答合规与安全审查：诊断性断言检测、紧急风险拦截、免责声明注入。"
            "在生成最终回答后由安全流程强制调用，用于确保输出合规、不遗漏紧急提醒。"
        ),
        handler=_safe_review_handler,
        enabled=True,
        kind="rule_engine",
    )
    register_tool(
        name="symptom_triage",
        description=(
            "症状分诊与紧急度评估：对症状组合做紧急度分级（红/黄/绿）、危险组合检测、"
            "就诊时限建议。触发条件：用户主诉含多症状、严重程度或持续时间信息。"
            "分诊≠诊断，禁止输出确定性诊断。"
        ),
        handler=_symptom_triage_handler,
        enabled=True,
        kind="rule_engine",
    )
    register_tool(
        name="medication_guide",
        description=(
            "用药指导规则核查：药物实体识别、禁忌人群交叉检查、用量安全范围、相互作用初筛、"
            "5 字段规范性校验。触发条件：回答中出现药物名。"
        ),
        handler=_medication_guide_handler,
        enabled=True,
        kind="rule_engine",
    )

    # 检索引擎：默认禁用，防止 LLM 绕过确定性路由/检索管线
    retrieval_enabled = False
    try:
        retrieval_enabled = bool(get_config().ENABLE_RETRIEVAL_TOOL)
    except Exception:
        pass
    register_tool(
        name="retrieval",
        description=(
            "知识库混合检索（dense+sparse+rerank）。默认禁用；仅当显式开启"
            " ENABLE_RETRIEVAL_TOOL 时由确定性流程按需启用，避免绕过系统路由。"
        ),
        handler=_retrieval_handler,
        enabled=retrieval_enabled,
        kind="retrieval",
    )


_register_core_tools()


# 供 __init__.py / 调试使用的便捷导出
__all__ = [
    "MedicalTool",
    "TOOL_REGISTRY",
    "register_tool",
    "get_tool",
    "enabled_tools",
    "tool_descriptions",
    "invoke_tool",
]