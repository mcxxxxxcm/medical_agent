"""黄金测试集召回诊断（Stage 2 前置，只读） — LLM 语义版

对 `--only` 匹配的样本跑真实检索（复用 RAGEvaluator.run_retrieval 完整链路），
判定每条 key_fact 是否出现在**任意召回 chunk** 里（两级：先字面定位、未命中交 LLM 语义复审）：

- FOUND in chunk N           = 该要点已被检索召回 → 答不出属「生成覆盖不全」（B 类）
- FOUND (LLM 语义复审)       = 字面未逐字对上、但召回上下文覆盖其语义（假阴性被挽回）
- MISSING                     = 召回上下文中完全找不到该要点 → 属「检索召回缺失」（A 类）

与 v9.67 之前的旧版不同：旧版用纯字面子串匹配，"换说法/更具体化"会全判缺失，
A/B 分类大量失真（记忆 feedback_golden_eval_semantic）。本版把
「要点是否被召回到」的判定升级为 LLM 语义判定，输出可信的**检索层要点召回率**
（= 语义命中的 key_fact / 该 query 的 key_fact 总数），据此把失败样本精确归为 A / B 类。

用法：
    /d/Agent/software/envs/my_medical_env/python.exe scripts/diag_recall.py --only "布洛芬|流鼻血"
    /d/Agent/software/envs/my_medical_env/python.exe scripts/diag_recall.py          # 全部样本
    /d/Agent/software/envs/my_medical_env/python.exe scripts/diag_recall.py --no-semantic   # 退回字面
"""
import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DEFAULT_GOLDEN = PROJECT_ROOT / "tests/data/golden_test_set.jsonl"


def _normalize(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[，。！？、；：”“‘’（）\[\]{}【】《》\s\-_=+|\\<>\"'~`]", "", text)
    return text.lower()


def _extract_json(text: str):
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    candidates = [text]
    if not (text.startswith("[") and text.endswith("]")):
        m = re.search(r"\[.*\]", text, re.S)
        if m:
            candidates.insert(0, m.group(0))
    for c in candidates:
        try:
            return json.loads(c)
        except Exception:
            continue
    return None


def _llm_judge_recall(context_dump: str, facts):
    """用 LLM 判定召回上下文是否覆盖每个 to-be-judged 要点的语义（一次调用判一批）。

    返回 {fact: {'hit': bool, 'reason': str}}。
    判据与生成覆盖一致：允许换说法/更具体化/同义表述，含义一致即算覆盖。
    """
    from app.core.llm import get_llm
    from langchain_core.messages import HumanMessage

    if not facts or not context_dump or not context_dump.strip():
        return {f: {"hit": False, "reason": ""} for f in facts}

    numbered = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(facts))
    prompt = (
        "你是检索质量评测员。下面给出某条医疗查询的【检索召回上下文】（可能含多段）。\n"
        "请判断该【召回上下文】是否【明确到达/出现】每条要点的语义。\n"
        "判断标准：允许换一种说法、更具体化、引用同义表述；只要含义一致就算到达(hit=1)。\n"
        "召回上下文里完全没提、或含义相反/冲突、或仅部分相关不算到达(hit=0)。\n"
        "只输出 JSON 数组（不要任何其他文字），每项格式：{\"idx\": 要点编号, \"hit\": 0或1, \"reason\": \"一句话理由\"}。\n"
        "\n【召回上下文】\n"
        f"{context_dump}"
        "\n【要点】\n"
        f"{numbered}"
    )
    try:
        llm = get_llm(streaming=False)
        resp = llm.invoke([HumanMessage(content=prompt)])
        arr = _extract_json(resp.content)
        result = {}
        if isinstance(arr, list):
            for item in arr:
                if not isinstance(item, dict):
                    continue
                idx = item.get("idx")
                hit = bool(item.get("hit"))
                reason = str(item.get("reason") or "")
                if isinstance(idx, int) and 1 <= idx <= len(facts):
                    result[facts[idx - 1]] = {"hit_v": hit, "reason": reason}
        # 用单独字段返回，避免与字面判定混排
        return {f: {"hit": result.get(f, {}).get("hit_v", False),
                    "reason": result.get(f, {}).get("reason", "")} for f in facts}
    except Exception as e:
        return {f: {"hit": False, "reason": f"LLM判定失败: {e}"} for f in facts}


def load_golden(path: Path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            q = item.get("question") or item.get("query") or item.get("original_query")
            if q:
                data.append({
                    "question": q,
                    "category": item.get("category", ""),
                    "key_facts": item.get("key_facts", []),
                })
    return data


def _judge_sample(contexts, key_facts, use_semantic):
    """两级判定：先字面定位段落；未命中的要点批量交 LLM 语义复审。

    返回依次为：seen(逐要点判定列表)、hits、total。
    • 判定条目含 loc 段落号列表；未字面定位而 LLM 确认的 loc 为空。
    """
    norm_all = _normalize("\n".join(contexts))
    seen = []
    llm_candidates = []
    for kf in key_facts:
        nkf = _normalize(kf)
        if not nkf:
            seen.append({"fact": kf, "hit": True, "loc": [], "how": "空要点按命中"})
            continue
        if contexts and nkf in norm_all:
            loc = [i for i, c in enumerate(contexts) if nkf in _normalize(c)]
            seen.append({"fact": kf, "hit": True, "loc": loc, "how": "substr"})
        else:
            llm_candidates.append(kf)

    if use_semantic and llm_candidates:
        context_dump = "\n---\n".join(
            f"[第{i + 1}段] {c}" for i, c in enumerate(contexts)
        )
        judged = _llm_judge_recall(context_dump, llm_candidates)
        for kf in llm_candidates:
            j = judged.get(kf, {})
            hit = bool(j.get("hit"))
            seen.append({
                "fact": kf,
                "hit": hit,
                "loc": [],
                "how": "llm" if hit else "missing",
                "reason": j.get("reason", ""),
            })
    else:
        for kf in llm_candidates:
            seen.append({"fact": kf, "hit": False, "loc": [], "how": "missing", "reason": ""})

    hits = sum(1 for s in seen if s["hit"])
    return seen, hits, len(seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="只诊断 question 含该子串的样本，| 分隔")
    ap.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    ap.add_argument("--no-semantic", action="store_true", help="只用字面定位，关闭 LLM 语义复审")
    args = ap.parse_args()

    # 固定采样温度 0.0，使语义复审可复现（仅本进程，不影响生产）
    from app.core.config import get_config
    get_config().MODEL_TEMPERATURE = 0.0

    from app.rag.evaluation import RAGEvaluator

    evaluator = RAGEvaluator()
    samples = [s for s in load_golden(Path(args.golden)) if s["key_facts"]]
    if args.only:
        pats = [p for p in args.only.split("|") if p]
        samples = [s for s in samples if any(p in s["question"] for p in pats)]

    use_semantic = not args.no_semantic
    out = []
    total_hits, total_facts = 0, 0
    per_sample_ratios = []
    empty_recall_count = 0

    for s in samples:
        out.append("=" * 60)
        out.append(f"Q: {s['question']}   [category={s['category']}]")
        res = evaluator.run_retrieval(s["question"])
        contexts = res.get("contexts", [])
        out.append(f"召回 {len(contexts)} 段:")
        for i, c in enumerate(contexts, 1):
            first_line = next((ln for ln in c.splitlines() if ln.strip()), "").strip()
            out.append(f"  #{i}: {first_line[:90]}")

        seen, hits, total = _judge_sample(contexts, s["key_facts"], use_semantic)
        total_hits += hits
        total_facts += total
        per_sample_ratios.append(hits / total if total else 1.0)
        if not contexts:
            empty_recall_count += 1

        for item in seen:
            kf, hit = item["fact"], item["hit"]
            how = item["how"]
            if hit and how == "substr":
                where = "、".join(f"#{i}" for i in item["loc"])
                out.append(f"  [命中] {kf}  (出现在 {where})")
            elif hit and how == "llm":
                out.append(f"  [命中·LLM语义] {kf}  (recall 覆盖其语义: {item.get('reason', '')})")
            else:
                out.append(f"  [缺失] {kf}  (recall 中找不到{((' · ' + item.get('reason', '')) if item.get('reason') else '')})")
        out.append(f"  → 检索层要点召回率: {hits}/{total}")
        out.append("")

    out.append("=" * 60)
    out.append(f"检索层要点召回率汇总（{len(samples)} 条，{total_facts} 个要点"
               + ("，LLM 语义判定" if use_semantic else "，仅字面") + ")")
    avg = (total_hits / total_facts) if total_facts else 0.0
    per = sum(per_sample_ratios) / len(per_sample_ratios) if per_sample_ratios else 0.0
    out.append(f"  总体: {total_hits}/{total_facts} = {avg:.1%}")
    out.append(f"  平均每条(未按要点数加权): {per:.1%}")
    out.append(f"  每条均全命中(召回率=100%)的样本数: {sum(1 for r in per_sample_ratios if r >= 1.0)}/{len(per_sample_ratios)}")
    out.append(f"  A 类候选(检索 0 召回段)样本数: {empty_recall_count}/{len(samples)}")

    report = "\n".join(out)
    dst = PROJECT_ROOT / "data/diag_recall_report.txt"
    dst.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n已写 {dst}")


if __name__ == "__main__":
    main()