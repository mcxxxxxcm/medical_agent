"""黄金测试集答案匹配判定脚本（支持并行 + LLM 语义复审）

对黄金测试集（tests/data/golden_test_set.jsonl）逐条执行「检索 → 生成」，
然后用每条 ground_truth 里预标注的 key_facts（金标要点）判定系统回答是否命中要点，
输出「✅通过 / ❌未通过」清单、平均要点命中率，并保存 golden_match_report.json。

为什么需要 LLM 语义复审：
    单纯的字面子串匹配有严重假阴性——系统答对但换了同义词/语序/括号引用/更具体化，
    就全判"未命中"（早期实测 55 条只过 3 条，绝大部分是这种假阴性，而非真答错）。
    本脚本两级判定：
        1. 先做归一化后字面子串匹配（快、省调用）；
        2. 对"字面未命中"的要点，用 LLM 判 answer 是否覆盖其语义（同义/更具体/换说法都算命中）。
    报告除了 pass/fail，还会统计"被 LLM 语义复审挽回的假阴性数"，用于衡量评测的苛刻程度。

并行说明：
    瓶颈是网络 IO（云端 embedding 检索 + 云端 LLM 生成 + LLM 语义判定），天然适合并行。
    本脚本用 ThreadPoolExecutor 让每个样本在独立 worker 线程里跑，各自并发调云端 API。
    检索器/LLM 是 repo 的 lru_cache 单例（线程安全场景下共享复用，避免重复加载 reranker 模型）。
    并发越高越可能触发云端限流，默认 3，可用 --concurrent 调整，遇失败会指数退避重试。

用法：
    python scripts/evaluate_golden_match.py                   # 默认并发 3 跑全部 55 条
    python scripts/evaluate_golden_match.py --concurrent 4    # 提高并行度
    python scripts/evaluate_golden_match.py --limit 3         # 只跑前 3 条，快速试跑
    python scripts/evaluate_golden_match.py --threshold 0.7   # 通过阈值（默认 0.7）
    python scripts/evaluate_golden_match.py --no-semantic     # 关闭 LLM 语义复审，只用字面子串
"""
import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows 控制台默认 GBK，无法打印 ✅/❌，强制 UTF-8 输出
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from app.rag.evaluation import RAGEvaluator
from app.core.app_logging import get_logger
from app.core.config import get_config

logger = get_logger(__name__)

DEFAULT_GOLDEN_SET = str(PROJECT_ROOT / "tests/data/golden_test_set.jsonl")
DEFAULT_OUTPUT = str(PROJECT_ROOT / "data/golden_match_report.json")

# 空答案重试次数（空答案基本意味着请求失败/被限流，静默失败会产假 ❌）
_EMPTY_ANSWER_RETRIES = 2


def _normalize(text: str) -> str:
    """归一化：去空白、全半角空格、常见标点，并转小写，用于稳健的子串匹配"""
    if not text:
        return ""
    text = re.sub(r"[，。！？、；：”“‘’（）\[\]{}【】《》\s\-_=+|\\<>\"'~`]", "", text)
    return text.lower()


def _extract_json(text: str):
    """从 LLM 回复里解析 JSON（数组优先）；解析失败返回 None"""
    if not text:
        return None
    text = text.strip()
    # 去掉可能的 ```json 代码块围栏
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    candidates = [text]
    # 若非纯 JSON，截取第一个 [ 到最后一个 ]
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


def _llm_judge_facts(answer: str, facts: List[str]) -> Dict[str, Dict]:
    """用 LLM 判定 answer 是否覆盖每个 fact 的语义（一次调用判一批）。

    返回 {fact: {'hit': bool, 'reason': str}}。
    prompt 用编号而非事实原文做键，避免 LLM 改写原文导致映射失配。
    """
    from app.core.llm import get_llm
    from langchain_core.messages import HumanMessage

    if not facts or not answer or not answer.strip():
        return {f: {"hit": False, "reason": ""} for f in facts}

    numbered = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(facts))
    prompt = (
        "你是医疗问答质检员。下面给出【系统回答】和若干条该问题应覆盖的【要点】。\n"
        "请判断【系统回答】是否明确覆盖了每条要点的【语义】。判断标准：\n"
        "- 允许换一种说法、更具体化、引用同义表述；只要含义一致就算覆盖(hit=1)。\n"
        "- 系统回答没提、或含义相反/冲突、或仅部分相关不算覆盖(hit=0)。\n"
        "只输出 JSON 数组（不要任何其他文字），每项格式：{\"idx\": 要点编号, \"hit\": 0或1, \"reason\": \"一句话理由\"}。\n"
        "\n【系统回答】\n"
        f"{answer}\n"
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
                    result[facts[idx - 1]] = {"hit": hit, "reason": reason}
        for f in facts:
            if f not in result:
                result[f] = {"hit": False, "reason": "（LLM 未返回该项，按未命中）"}
        return result
    except Exception as e:
        logger.error(f"LLM 语义判定失败，按字面未命中处理: {e}")
        return {f: {"hit": False, "reason": f"LLM 判定失败: {e}"} for f in facts}


def match_key_facts(answer: str, key_facts: List[str], use_semantic: bool = True) -> List[Dict]:
    """两级判定：先字面子串，再对未命中要点做 LLM 语义复审。

    返回每项 {'fact','hit','reason','judged_by': 'substr'|'llm'|'none'}。
    """
    norm_answer = _normalize(answer)
    results: List[Dict] = []
    llm_candidates: List[str] = []

    for fact in key_facts:
        if fact and norm_answer and _normalize(fact) in norm_answer:
            results.append({"fact": fact, "hit": True, "reason": "字面命中", "judged_by": "substr"})
        else:
            llm_candidates.append(fact)

    if use_semantic and llm_candidates:
        judged = _llm_judge_facts(answer, llm_candidates)
        for fact in llm_candidates:
            j = judged.get(fact, {})
            results.append({
                "fact": fact,
                "hit": bool(j.get("hit")),
                "reason": j.get("reason", ""),
                "judged_by": "llm",
            })
    else:
        for fact in llm_candidates:
            results.append({"fact": fact, "hit": False, "reason": "", "judged_by": "none"})

    return results


def run_one(evaluator: RAGEvaluator, sample: Dict, threshold: float,
            use_semantic: bool, store_contexts: bool = False) -> Dict:
    """对单条样本执行检索+生成+匹配判定（含空答案重试）"""
    question = sample.get("question", "")
    ground_truth = sample.get("ground_truth", "")
    key_facts = sample.get("key_facts", []) or []

    answer = ""
    contexts = []
    retries = 0
    source = "success"
    while retries <= _EMPTY_ANSWER_RETRIES:
        try:
            retrieval_result = evaluator.run_retrieval(question)
            contexts = retrieval_result["contexts"]
            candidate = evaluator.run_generation(question, retrieval_result["contexts"])
            if candidate and candidate.strip():
                answer = candidate
                source = "success"
                break
            wait = retries * 3 + random.uniform(0, 1)
            retries += 1
            if retries <= _EMPTY_ANSWER_RETRIES:
                logger.warning(f"空答案，{wait:.1f}s 后重试 [{retries}/{_EMPTY_ANSWER_RETRIES}]: {question[:40]}...")
                time.sleep(wait)
            else:
                source = "empty_answer"
        except Exception as e:
            wait = retries * 3 + random.uniform(0, 1)
            retries += 1
            if retries <= _EMPTY_ANSWER_RETRIES:
                logger.error(f"执行失败({e})，{wait:.1f}s 后重试 [{retries}/{_EMPTY_ANSWER_RETRIES}]: {question[:40]}...")
                time.sleep(wait)
            else:
                source = f"error: {e}"

    fact_matches = match_key_facts(answer, key_facts, use_semantic)
    hit_count = sum(1 for fm in fact_matches if fm["hit"])
    total = len(fact_matches)
    hit_ratio = round(hit_count / total, 4) if total else 0.0

    # 被 LLM 语义复审挽回的假阴性数（子串未中、LLM 判定语义命中）
    llm_recovered = sum(1 for fm in fact_matches if fm["judged_by"] == "llm" and fm["hit"])

    return {
        "question": question,
        "answer": answer,
        "ground_truth": ground_truth,
        "category": sample.get("category", ""),
        "difficulty": sample.get("difficulty", ""),
        "key_facts": key_facts,
        "fact_matches": fact_matches,
        "hit_count": hit_count,
        "fact_total": total,
        "hit_ratio": hit_ratio,
        "passed": total > 0 and hit_ratio >= threshold,
        "run_status": source,
        "llm_recovered": llm_recovered,
        "contexts": contexts[:3] if store_contexts else [],
    }


def _print_report(results, passed, failed, avg_hit, llm_recovered_total,
                  facts_llm_judged, total, threshold, duration, skipped=0) -> None:
    print("\n" + "=" * 60)
    print("黄金测试集匹配判定报告")
    print("=" * 60)
    print(f"  总样本:      {total}" + (f"（另有 {skipped} 条因缺 key_facts 跳过统计）" if skipped else ""))
    print(f"  通过:        {len(passed)}   ❌未通过: {len(failed)}")
    print(f"  通过率:      {len(passed) / total:.1%}" if total else "  通过率:      -")
    print(f"  平均要点命中率: {avg_hit:.1%}")
    print(f"  阈值:        {threshold:.0%}")
    if facts_llm_judged:
        print(f"  语义复审:    {facts_llm_judged} 条字面未中要点交 LLM 判定，"
              f"其中 {llm_recovered_total} 条确认语义命中（被挽回的假阴性）")
    print(f"  耗时:        {duration:.1f}s")
    print("=" * 60)

    for idx, r in enumerate(results, 1):
        icon = "✅" if r.get("passed") else "❌"
        miss = [fm for fm in r.get("fact_matches", []) if not fm["hit"]]
        hit_llm = [fm for fm in r.get("fact_matches", []) if fm["hit"] and fm["judged_by"] == "llm"]
        print(f"\n[{icon}] #{idx} [{r.get('difficulty', '')}] [{r.get('category', '')}] "
              f"要点命中 {r.get('hit_count', 0)}/{r.get('fact_total', 0)} "
              f"({r.get('hit_ratio', 0.0):.0%})")
        print(f"    问 : {r.get('question', '')}")
        if hit_llm:
            print(f"    LLM语义命中: {', '.join(fm['fact'] for fm in hit_llm)}")
        if miss:
            desc = " / ".join(
                (f"{fm['fact']}" + (f"〔{fm['reason']}〕" if fm.get("reason") else ""))
                for fm in miss
            )
            print(f"    未命中要点: {desc}")
        if r.get("answer"):
            print(f"    答 : {r['answer'][:200]}")
        if r.get("ground_truth"):
            print(f"    标 : {r['ground_truth'][:200]}")


def main():
    parser = argparse.ArgumentParser(description="黄金测试集答案匹配判定（可并行 + LLM 语义复审）")
    parser.add_argument("--golden-set", type=str, default=DEFAULT_GOLDEN_SET, help="黄金测试集 JSONL 路径")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="报告输出路径")
    parser.add_argument("--threshold", type=float, default=0.7, help="通过阈值（0~1，默认 0.7）")
    parser.add_argument("--limit", type=int, default=0, help="只评估前 N 条（0=全部）")
    parser.add_argument("--concurrent", type=int, default=3, help="并行 worker 数（默认 3；云端并发越高越易限流）")
    parser.add_argument("--no-semantic", action="store_true", help="关闭 LLM 语义复审，只用字面子串匹配")
    parser.add_argument("--store-contexts", dest="store_contexts", action="store_true", default=True,
                        help="把每条检索召回的前3段写进报告（诊断检索错位用，默认开启）")
    parser.add_argument("--no-contexts", dest="store_contexts", action="store_false",
                        help="不把检索召回写进报告（省体积）")
    parser.add_argument("--only", type=str, default="", metavar="子串",
                        help="只评估 question 包含该子串的样本（如 --only 布洛芬），多个用 | 分隔")
    args = parser.parse_args()

    # v9.67: 评测可复现标尺——固定 LLM 采样温度为 0。
    # 答案生成 + 语义判定共用 get_llm（lru_cache 懒加载：首个样本生成时才建实例），
    # 因此在构建 evaluator 前改写 config.MODEL_TEMPERATURE（默认 0.2）即可让整轮以确定性
    # 采样，消除云端 LLM 随机性带来的 pass 波动，使全量通过率成为可复现、可比对的标尺。
    # 说明：只影响评测进程内后续新建的 LLM 实例，不改变生产配置。
    _cfg = get_config()
    _cfg.MODEL_TEMPERATURE = 0.0
    logger.info("评测确定性模式：MODEL_TEMPERATURE 已固定为 0.0")

    if args.concurrent < 1:
        logger.error("--concurrent 必须 >= 1")
        sys.exit(1)

    use_semantic = not args.no_semantic

    evaluator = RAGEvaluator()
    test_data = evaluator.load_test_set(args.golden_set)
    if not test_data:
        logger.error(f"黄金测试集为空或不存在: {args.golden_set}")
        sys.exit(1)

    if args.only:
        keys = [k for k in args.only.split("|") if k]
        test_data = [s for s in test_data if any(k in s.get("question", "") for k in keys)]
        if not test_data:
            logger.error(f"没有命中 --only 过滤条件的样本: {args.only}")
            sys.exit(1)
        logger.info(f"--only 过滤后：{len(test_data)} 条")

    if args.limit > 0:
        test_data = test_data[: args.limit]
        logger.info(f"限制模式：仅评估前 {len(test_data)} 条")

    results: List[Dict] = [None] * len(test_data)
    start = time.time()
    total = len(test_data)
    done = 0

    def _worker(sample: Dict) -> Dict:
        return run_one(evaluator, sample, args.threshold, use_semantic, args.store_contexts)

    logger.info(f"并行度 {args.concurrent}，共 {total} 条开始评估" + ("，启用 LLM 语义复审" if use_semantic else "，仅字面子串"))
    if args.concurrent == 1:
        for i, sample in enumerate(test_data):
            results[i] = _worker(sample)
            done += 1
            logger.info(f"进度 [{done}/{total}]")
    else:
        with ThreadPoolExecutor(max_workers=args.concurrent) as ex:
            future_to_idx = {ex.submit(_worker, s): i for i, s in enumerate(test_data)}
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    logger.error(f"样本最终失败 [{idx}]: {e}")
                    results[idx] = {
                        "question": test_data[idx].get("question", ""),
                        "answer": "",
                        "ground_truth": test_data[idx].get("ground_truth", ""),
                        "category": test_data[idx].get("category", ""),
                        "difficulty": test_data[idx].get("difficulty", ""),
                        "key_facts": test_data[idx].get("key_facts", []),
                        "fact_matches": [],
                        "hit_count": 0,
                        "fact_total": 0,
                        "hit_ratio": 0.0,
                        "passed": False,
                        "run_status": f"error: {e}",
                        "llm_recovered": 0,
                    }
                done += 1
                logger.info(f"进度 [{done}/{total}] 完成: {results[idx].get('question', '')[:30]}...")

    # 过滤掉无要点（fact_total=0）的样本：0/0 必然 failed，是数据缺 key_facts 而非系统能力，
    # 不让它进统计、也不算失败，单独计入 skipped。
    skipped = [r for r in results if r.get("fact_total", 0) == 0]
    results = [r for r in results if r.get("fact_total", 0) > 0]

    passed = [r for r in results if r.get("passed")]
    failed = [r for r in results if not r.get("passed")]
    total = len(results)
    avg_hit = round(
        sum(r.get("hit_ratio", 0.0) for r in results) / len(results), 4
    ) if results else 0.0
    llm_recovered_total = sum(r.get("llm_recovered", 0) for r in results)
    facts_llm_judged = sum(
        1 for r in results for fm in r.get("fact_matches", []) if fm.get("judged_by") == "llm"
    )

    _print_report(results, passed, failed, avg_hit, llm_recovered_total,
                  facts_llm_judged, total, args.threshold, time.time() - start, skipped=len(skipped))

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": total,
        "passed": len(passed),
        "failed": len(failed),
        "pass_rate": round(len(passed) / total, 4) if total else 0.0,
        "avg_hit_ratio": avg_hit,
        "threshold": args.threshold,
        "concurrency": args.concurrent,
        "semantic_judge": use_semantic,
        "facts_llm_judged": facts_llm_judged,
        "llm_recovered": llm_recovered_total,
        "skipped_no_facts": len(skipped),
        "per_sample": results,
        "skipped_samples": [{"question": s.get("question", ""),
                             "reason": f"key_facts空缺({s.get('fact_total', 0)}个要点)"} for s in skipped],
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {out_path}")


if __name__ == "__main__":
    main()