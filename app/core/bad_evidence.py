"""Bad Case 证据式标注（L3 半自动标注）
__version__ = 9.56

把审核从"在错误答案上改"降维成"对着证据判句子"：

    retrieve_evidence(case) → 用知识库混合检索命中与该 case 相关的 Top-K 文档片段，
                               每段带 frag_id + 内容 + 来源，供人工"勾选/核对引用"。
    generate_draft(...)      → 基于『人工勾选的片段』生成 ground_truth 草稿，且要求
                               每个断言标注来源片段 ID（可溯源）。草稿只起提示作用，
                               最终医学正确性仍由人工把关注册 source。

设计准则（医疗严谨）：
    1. 绝不把原坏例的错误答案当草稿——只基于用户问题 + 勾选片段。
    2. 不得臆造剂量/禁忌/治疗；模型不确定的地方输出 [需人工核实]。
    3. 任何一步失败静默降级（fragments=[] / draft=""），不阻塞坏例后台。
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage

from app.core.app_logging import get_logger

logger = get_logger(__name__)

EVIDENCE_TOP_K = 5           # 检索返回片段上限
FRAG_MAX_CHARS = 350         # 单片段内容上限（控 prompt token）
DRAFT_SENTENCE_MIN = 2       # 草稿最少句数（提示用）
DRAFT_SENTENCE_MAX = 5


def _case_query(case: Dict) -> str:
    return (case.get("final_question") or case.get("original_query") or "").strip()


def _case_intent_meta(case: Dict) -> Dict:
    """对该 badcase 的查询做全链路意图分类（复用 intent_clarify，判别与线上同源）"""
    query = _case_query(case)
    if not query:
        return {
            "intent": "general",
            "intent_label": "一般交流",
            "missing_slots": [],
            "clarify_needed": False,
            "hint": "",
        }
    try:
        from app.core.intent_clarify import classify_intent
        return classify_intent(query)
    except Exception as e:
        logger.warning(f"badcase 意图分类失败：{e}")
        return {
            "intent": "general",
            "intent_label": "一般交流",
            "missing_slots": [],
            "clarify_needed": False,
            "hint": "",
        }


def retrieve_evidence(case: Dict, k: int = EVIDENCE_TOP_K) -> List[Dict]:
    """混合检索返回与 case 相关的文档片段列表。失败返回 []。

    每项: {frag_id:"#i", content, source}
    """
    query = _case_query(case)
    if not query:
        return []
    try:
        from app.rag.hybrid_retriever import get_cached_hybrid_retriever
        retriever = get_cached_hybrid_retriever(
            k=k, alpha=0.5, use_reranker=True, rerank_top_k=k * 2,
        )
        docs = retriever.invoke(query)
        out: List[Dict] = []
        for i, d in enumerate(docs[:k], 1):
            content = (d.page_content or "").strip()
            if not content:
                continue
            out.append({
                "frag_id": f"#{i}",
                "content": content[:FRAG_MAX_CHARS],
                "source": d.metadata.get("source") or d.metadata.get("file_path") or "unknown",
            })
        return out
    except Exception as e:
        logger.warning(f"badcase 证据检索失败：{e}")
        return []


def _filter_selected(fragments: List[Dict], selected: Optional[List[str]]) -> List[Dict]:
    """按人工勾选的 frag_id 过滤片段；未传/为空视为全选前 n 片段。"""
    if not selected:
        return fragments[:3]
    wanted = {str(s).strip() for s in selected if str(s).strip()}
    picks = [f for f in fragments if f.get("frag_id") in wanted]
    return picks if picks else fragments[:3]


def build_draft_prompt(question: str, fragments: List[Dict]) -> str:
    frag_block = "\n".join(
        f"[片段 {f['frag_id']}] 来源:{f.get('source')}\n{f.get('content')}" for f in fragments
    )
    return f"""你是一名医疗问答质量标注助手。请为一个『用户问题』据此下的知识库片段，起草一份期望正确回答（ground_truth）。

【严格要求】
1. 只基于下列【知识库片段】回答，不得依据片段之外的记忆补充臆测；凡片段未覆盖的关键事实（尤其药物剂量/禁忌人群/治疗方案），一律写 [需人工核实:具体疑问]，绝不编造。
2. 全文 2-5 句话，直接给可入库的期望答案文本，不要疑问句、不要自我说明。
3. 每个断言尽量用括号标注其来源片段ID（如 (片段#1)），便于人工核对引用是否对应。

【用户问题】
{question}

【知识库片段】
{frag_block}

【请严格输出一个 JSON 对象，不要任何其他文字】
{{
  "draft": "期望正确回答文本",
  "notes": ["给审核者的提示，如不确定点依赖哪个片段", ...]
}}"""


def _extract_obj(text: str) -> Dict:
    text = (text or "").strip()
    fenced = re.sub(r"^```(?:json)?\s*", "", text).rstrip("`").strip()
    try:
        obj = json.loads(fenced)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = fenced.find("{")
    end = fenced.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(fenced[start:end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return {}


def generate_draft(question: str, fragments: List[Dict], selected: Optional[List[str]] = None) -> Dict:
    """基于勾选片段生成可溯源草稿。返回 {draft, notes, drafts_total, used_frag_ids}。

    失败降级返回 {draft:"", notes:["草稿生成失败，请人工撰写"], ...}，不抛异常。
    意图感知：问题缺关键槽位（如"吃了三粒怎么办"缺药名）时直接拒绝成稿——
    问题本就不可回答，绝不基于弱片段伪造剂量/期望答案（医疗红线，见模块 docstring）。
    """
    try:
        from app.core.intent_clarify import classify_intent
        meta = classify_intent(question or "")
    except Exception:
        meta = None
    if meta and meta.get("clarify_needed"):
        hint = meta.get("hint") or "此问题缺关键信息"
        return {
            "draft": "",
            "notes": [hint, "线上会对该问题触发澄清追问；不建议基于弱检索结果生成内容草稿，请先补全关键信息再判断。"],
            "used_frag_ids": [],
        }

    picks = _filter_selected(fragments, selected)
    if not question or not picks:
        return {"draft": "", "notes": ["无可用片段，请人工撰写期望回答"], "used_frag_ids": []}
    try:
        from app.core.llm import get_llm
        prompt = build_draft_prompt(question, picks)
        llm = get_llm()
        resp = llm.invoke([HumanMessage(content=prompt)])
        raw = resp.content if isinstance(resp.content, str) else str(resp.content or "")
        obj = _extract_obj(raw)
        draft = str(obj.get("draft") or "").strip()
        notes = obj.get("notes") if isinstance(obj.get("notes"), list) else []
        notes = [str(n) for n in notes if str(n).strip()][:6]
        return {
            "draft": draft,
            "notes": notes,
            "used_frag_ids": [f.get("frag_id") for f in picks],
        }
    except Exception as e:
        logger.warning(f"badcase 草稿生成失败：{e}")
        return {"draft": "", "notes": [f"草稿生成失败：{e}，请人工撰写"], "used_frag_ids": []}


def evidence_payload(case: Dict, selected: Optional[List[str]] = None) -> Dict:
    """组合返回证据面板所需完整载荷，供路由直接输出。"""
    query = _case_query(case)
    intent = _case_intent_meta(case)
    fragments = retrieve_evidence(case)
    draft_res = generate_draft(query, fragments, selected) if query else {
        "draft": "", "notes": ["问题为空，无法检索"], "used_frag_ids": [],
    }
    return {
        "case_id": case.get("case_id", ""),
        "query": query,
        "intent": intent.get("intent", "general"),
        "intent_label": intent.get("intent_label", "一般交流"),
        "missing_slots": intent.get("missing_slots", []),
        "clarify_needed": intent.get("clarify_needed", False),
        "clarify_hint": intent.get("hint", ""),
        "fragments": fragments,
        "frag_total": len(fragments),
        "draft": draft_res["draft"],
        "notes": draft_res["notes"],
        "used_frag_ids": draft_res["used_frag_ids"],
    }