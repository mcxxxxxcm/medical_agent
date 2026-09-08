"""v9.49 Bad Case 自动标注脚本（方案A：离线 draft，供人工肉眼审阅）

从 PostgresStore 读取已采集的 bad case（默认仅未审核），用一个 LLM 对每条做
自动标注，核心第一步是**判断这是不是"真 badcase"**——重点排除用户误点/乱点
导致好答案被误传到后台的情况。

对每条输出：
    1. is_valid_badcase: 是否真 badcase（False 表示误点/乱点/答案其实没问题）
    2. validity_reason:  判真伪的依据（引用户反馈原因 + 答案与问题相关性）
    3. root_cause:       根因归类（复用项目既有 case_type 九类 + 新增误点类）
    4. draft_expected_answer: 期望正确回答草稿（对应后续黄金集 ground_truth，
                             医疗关键事实不确定时标 [需人工核实]，不臆造剂量禁忌）
    5. confidence:       自动标注置信度 high/low（low 者强制人工重点审）

产出一份 JSONL（默认 tests/data/bad_cases_annotated_draft.jsonl），
每行 = 原 bad case 字段 + 自动标注字段，供肉眼评估标注质量。本脚本**不写回**数据库，
人工定稿后才入库。

用法：
    python scripts/auto_annotate_bad_cases.py [--limit 500] [--output ...]
        [--model glm-4.5-air] [--samples 5] [--case-type ...]
"""
__version__ = "9.49"
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

# 将项目根目录加入 sys.path（与 export_bad_cases.py 一致）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 全量 case_type（与 long_term_memory.append_bad_case 的枚举一致）
CASE_TYPES = [
    "rewrite_missed_anaphora",
    "rewrite_lost_entity",
    "rewrite_same_as_original",
    "low_score_no_clarify",
    "hallucination_suspected",
    "retrieval_miss",
    "route_misclassification",
    "user_negative_feedback",
    "manual_flag",
]
# 判真伪为"非真 badcase"时使用的两个特殊类
INVALID_CASE_TYPES = ["user_misclick", "acceptable_answer"]


def _trim(text: str, limit: int) -> str:
    """截断长文本，避免撑爆 prompt"""
    text = (text or "").strip()
    if len(text) > limit:
        return text[:limit] + "...[截断]"
    return text


def _user_feedback(case: dict) -> str:
    """从 bad case 字段里尽量找用户反馈的原因/补充说明"""
    meta = case.get("metadata") or {}
    parts = []
    reason = case.get("reason") or meta.get("reason") or meta.get("rating_reason") or ""
    note = case.get("note") or meta.get("note") or meta.get("feedback_note") or ""
    if reason:
        parts.append(f"用户选择的差评原因: {reason}")
    if note:
        parts.append(f"用户补充说明: {note}")
    return "；".join(parts) if parts else "无（用户未附具体原因，需重点判断是否误点）"


def build_prompt(case: dict) -> str:
    """构造单条 bad case 的标注 prompt"""
    feedback = _user_feedback(case)
    meta = (case.get("metadata") or {})
    extra = ""
    if meta:
        try:
            extra = "\n".join(
                "  - %s: %s" % (k, _trim(str(v), 200))
                for k, v in meta.items()
                if k.lower() not in ("reason", "note") and str(v).strip()
            )
        except TypeError:
            extra = ""
    extra_block = ("系统附加元数据显示:\n" + extra + "\n") if extra else ""

    return """你是一名医疗助手系统的 bad case 质量管理助手。你负责判断一条用户差评记录是否构成一个"真实有效的 bad case"，并给出标注。你的判断必须严谨、有依据，宁可存疑（标 low 置信度）也不要武断。

【任务背景】系统自动采集到一条用户差评，可能来自：检索错误、答案幻觉、答非所问、遗漏信息等。但**用户也可能误点/乱点**，把一条本不错的答案误判成差评。你的第一职责是把"误点/乱点导致好答案被误传"的情况识别出来。

【bad case 记录】
- case_id: %(case_id)s
- 自动类型(case_type): %(case_type)s
- 用户原始问题(original_query): %(original_query)s
- 会话历史摘要(history_summary): %(history_summary)s
- 系统给的最终问题(final_question): %(final_question)s
- 系统回答预览(answer_preview): %(answer_preview)s
- 重写结果(rewritten_query): %(rewritten_query)s
- 期望重写(expected_rewrite): %(expected_rewrite)s
- 检索最高分(top_doc_score): %(top_doc_score)s
- 评分结果(grade_result): %(grade_result)s
- 用户反馈: %(feedback)s
%(extra_block)s
【请严格输出一个 JSON 对象，不要任何其他文字，字段如下】
{
  "is_valid_badcase": true或false,  // 判断是否是"真 badcase"。判断优先级：
      //   1) 若答案与用户问题高度相关、内容基本正确，而用户未附具体差评原因或原因缺乏说服力 → 大概率误点 → false
      //   2) 若答案存在答非所问、明显遗漏核心信息、幻觉/包含文档内不存在的错误事实、检索完全跑偏 → true
      //   3) 无法定论时，本着医疗安全与低误杀原则：凡是信息不足的都设 true 但 confidence 用 low，绝不因不确定而把真 badcase 误判为误点
  "validity_reason": "一句话说明判真伪的核心依据，引用答案与问题的关系",
  "root_cause": "仅在 is_valid_badcase=true 时给出根因归类（%(case_types)s），若是误点则填 %(misclick)s，若答案其实可接受仅用户误评则填 %(acceptable)s",
  "draft_expected_answer": "仅在 is_valid_badcase=true 时：给出期望的正确回答草稿（供后续黄金测试集 ground_truth 参考）。注意:①只基于『用户问题』本身，不要照抄原答案里的错误内容;②不得臆造药物剂量/禁忌症等医疗关键事实；但凡你不确定的地方，写 [需人工核实:具体疑问]，绝不编造;③控制在 2-5 句话",
  "confidence": "high 或 low"  // 你对自己本次判断的把握;root_cause 分不清或答案涉及专业医学判断时用 low
}""" % {
        "case_id": case.get('case_id', ''),
        "case_type": case.get('case_type', ''),
        "original_query": _trim(case.get('original_query', ''), 300),
        "history_summary": _trim(case.get('history_summary', ''), 400) or '（无）',
        "final_question": _trim(case.get('final_question', ''), 300),
        "answer_preview": _trim(case.get('answer_preview', ''), 500) or '（无）',
        "rewritten_query": _trim(case.get('rewritten_query', ''), 300),
        "expected_rewrite": _trim(case.get('expected_rewrite', ''), 300) or '（未标注）',
        "top_doc_score": case.get('top_doc_score', ''),
        "grade_result": _trim(case.get('grade_result', ''), 300) or '（无）',
        "feedback": feedback,
        "extra_block": extra_block,
        "case_types": "、".join(CASE_TYPES),
        "misclick": INVALID_CASE_TYPES[0],
        "acceptable": INVALID_CASE_TYPES[1],
    }


def _maybe_extract_json(text: str) -> dict:
    """容错解析 LLM 输出的 JSON 对象"""
    text = (text or "").strip()
    # 去掉可能的 markdown 代码围栏
    text = re.sub(r"^```(?:json)?\s*", "", text).rstrip("`").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # 尝试提取最外层 { ... } 块
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    raise ValueError(f"无法解析模型输出为 JSON: {_trim(text, 200)}")


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "是", "有效", "真")
    return bool(value)


def annotate_case(case: dict, model_name: Optional[str]) -> dict:
    """对单条 bad case 调用 LLM 标注，返回 {原字段 + 标注字段}"""
    try:
        from langchain_core.messages import HumanMessage
        from app.core.llm import get_llm
    except Exception as e:
        print(f"导入依赖失败: {e}", file=sys.stderr)
        raise

    prompt = build_prompt(case)

    llm = get_llm(model_name=model_name) if model_name else get_llm()
    # 标注需要一定的判断力，用非流式、低温度（get_llm 默认温度 0.2）
    resp = llm.invoke([HumanMessage(content=prompt)])

    raw = resp.content if isinstance(resp.content, str) else str(resp.content or "")
    try:
        ann = _maybe_extract_json(raw)
    except ValueError as e:
        ann = {
            "is_valid_badcase": True,
            "validity_reason": f"模型输出解析失败，按有效处理待人工复核({e})",
            "root_cause": case.get("case_type") or "user_negative_feedback",
            "draft_expected_answer": "",
            "confidence": "low",
            "_parse_error": str(e),
        }

    root_cause = str(ann.get("root_cause") or case.get("case_type") or "user_negative_feedback")
    # 归一化 root_cause：非法值回退
    if root_cause not in CASE_TYPES and root_cause not in INVALID_CASE_TYPES:
        root_cause = "user_negative_feedback"

    result = {
        # 原字段透传，便于回溯
        "case_id": case.get("case_id", ""),
        "case_type": case.get("case_type", ""),
        "original_query": case.get("original_query", ""),
        "history_summary": case.get("history_summary", ""),
        "answer_preview": case.get("answer_preview", ""),
        "created_at": case.get("created_at", ""),
        # 自动标注字段
        "is_valid_badcase": _coerce_bool(ann.get("is_valid_badcase")),
        "validity_reason": str(ann.get("validity_reason") or ""),
        "root_cause": root_cause,
        "draft_expected_answer": str(ann.get("draft_expected_answer") or ""),
        "confidence": "high" if str(ann.get("confidence", "")).lower() == "high" else "low",
    }
    if ann.get("_parse_error"):
        result["parse_error"] = ann["_parse_error"]
    return result


def run(
    limit: int = 500,
    output: str = "tests/data/bad_cases_annotated_draft.jsonl",
    model_name: Optional[str] = None,
    samples: Optional[int] = None,
    case_type: Optional[str] = None,
):
    from app.memory import get_long_term_memory

    memory = get_long_term_memory()
    cases = memory.get_bad_cases(
        user_id="system",
        case_type=case_type,
        reviewed=False,  # 只处理未人工审核的，不覆盖已审
        limit=limit,
    )

    if not cases:
        print("没有找到未审核的 bad case")
        return

    if samples:
        cases = cases[:samples]
        print(f"【快速试跑模式】仅处理前 {len(cases)} 条")

    print(f"待标注 {len(cases)} 条 bad case，开始自动标注...\n")

    output_path = PROJECT_ROOT / output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] case_id={case.get('case_id', '')} ...")
        try:
            ann = annotate_case(case, model_name)
        except Exception as e:
            print(f"  标注失败，按有效/低置信度占位跳过: {e}", file=sys.stderr)
            ann = {
                "case_id": case.get("case_id", ""),
                "case_type": case.get("case_type", ""),
                "original_query": case.get("original_query", ""),
                "history_summary": case.get("history_summary", ""),
                "answer_preview": case.get("answer_preview", ""),
                "created_at": case.get("created_at", ""),
                "is_valid_badcase": True,
                "validity_reason": f"标注调用异常: {e}",
                "root_cause": case.get("case_type") or "user_negative_feedback",
                "draft_expected_answer": "",
                "confidence": "low",
            }
        results.append(ann)
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ann, ensure_ascii=False) + "\n")

    # 统计
    n = len(results)
    valid = sum(1 for r in results if r.get("is_valid_badcase"))
    invalid = n - valid
    low_conf = sum(1 for r in results if r.get("confidence") == "low")
    parse_err = sum(1 for r in results if "parse_error" in r)

    print("\n" + "=" * 60)
    print(f"标注完成: 共 {n} 条 → {output_path}")
    print(f"  有效 badcase: {valid} ({valid / n * 100:.1f}%)")
    print(f"  判定误点/可接受(非真 badcase): {invalid} ({invalid / n * 100:.1f}%)")
    print(f"  低置信度(需人工重点审核): {low_conf}")
    print(f"  解析失败: {parse_err}")

    print("\n根因分布（有效 badcase）:")
    from collections import Counter
    counts = Counter(r.get("root_cause") for r in results if r.get("is_valid_badcase"))
    for cause, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cause}: {cnt}")

    print("\n提示: 请人工审阅该 draft 中 is_valid_badcase 的判断和 draft_expected_answer 的医学正确性，确认后再入库为黄金测试集。")


def main():
    parser = argparse.ArgumentParser(description="Bad case 自动标注（方案A：离线draft）")
    parser.add_argument("--limit", type=int, default=500, help="最多处理条数")
    parser.add_argument("--output", default="tests/data/bad_cases_annotated_draft.jsonl",
                        help="输出 JSONL 路径（相对项目根目录）")
    parser.add_argument("--model", default=None,
                        help="指定标注模型（默认用配置 MODEL_NAME=glm-4-flash；建议标注用更强模型如 glm-4.5-air）")
    parser.add_argument("--samples", type=int, default=None, help="快速试跑：只处理前 N 条")
    parser.add_argument("--case-type", default=None,
                        choices=CASE_TYPES,
                        help="只标注指定类型")
    args = parser.parse_args()

    run(
        limit=args.limit,
        output=args.output,
        model_name=args.model,
        samples=args.samples,
        case_type=args.case_type,
    )


if __name__ == "__main__":
    main()