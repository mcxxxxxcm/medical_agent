"""黄金测试集召回诊断（Stage 2 前置，只读）

对 `--only` 匹配的样本跑真实检索（复用 RAGEvaluator.run_retrieval 完整链路），
打印每条 key_fact 是否出现在**任意召回 chunk** 里：

- FOUND in <chunk N>   = 该要点已被检索召回 → 答不出属「生成覆盖不全」（B 类）
- MISSING              = 召回上下文中完全找不到该要点 → 属「检索召回缺失」（A 类）

用于把失败样本精确归类，指导检索层修复。合成 UTF-8 报告便于阅读。
用法：
    /d/Agent/software/envs/my_medical_env/python.exe scripts/diag_recall.py --only "胃食管反流|一氧化碳|对乙酰氨基酚"
    ENABLE_SEMANTIC_CACHE=false <python> scripts/diag_recall.py --only "布洛芬|流鼻血"
"""
import argparse
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


def load_golden(path: Path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json_loads(line)
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


def json_loads(s):
    import json
    return json.loads(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="只诊断 question 含该子串的样本，| 分隔")
    ap.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    args = ap.parse_args()

    from app.rag.evaluation import RAGEvaluator

    evaluator = RAGEvaluator()
    samples = [s for s in load_golden(Path(args.golden)) if s["key_facts"]]
    if args.only:
        pats = [p for p in args.only.split("|") if p]
        samples = [s for s in samples if any(p in s["question"] for p in pats)]

    out = []
    for s in samples:
        out.append("=" * 60)
        out.append(f"Q: {s['question']}   [category={s['category']}]")
        res = evaluator.run_retrieval(s["question"])
        contexts = res.get("contexts", [])
        out.append(f"召回 {len(contexts)} 段:")
        for i, c in enumerate(contexts, 1):
            # 只回显首行定位来源，避免报告过长
            first_line = next((ln for ln in c.splitlines() if ln.strip()), "").strip()
            out.append(f"  #{i}: {first_line[:90]}")
        nm = _normalize("\n".join(contexts))
        for kf in s["key_facts"]:
            if _normalize(kf) and _normalize(kf) in nm:
                # 定位命中的 chunk
                where = []
                for i, c in enumerate(contexts, 1):
                    if _normalize(kf) in _normalize(c):
                        where.append(f"#{i}")
                out.append(f"  [命中] {kf}  (出现在 {'、'.join(where)})")
            else:
                out.append(f"  [缺失] {kf}  (召回中找不到)")
        out.append("")

    report = "\n".join(out)
    dst = PROJECT_ROOT / "data/diag_recall_report.txt"
    dst.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n已写 {dst}")


if __name__ == "__main__":
    main()