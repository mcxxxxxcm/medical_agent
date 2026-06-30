"""HyDE A/B 测试：对比有/无 HyDE 的检索效果

测试维度：
1. Recall：召回文档与人工标注相关文档的覆盖率
2. Relevance：Reranker 最高分
3. Latency：检索耗时
4. 文档重叠度：两种模式召回的文档重叠情况

测试集：10 条典型医疗查询（覆盖症状/药物/追问/模糊/精确等类型）
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.app_logging import get_logger
logger = get_logger(__name__)

# ===== 测试集 =====
# 每条查询标注了"期望命中的文档关键词"（用于计算 Recall）
TEST_QUERIES = [
    {
        "query": "头痛怎么办？",
        "type": "症状-自包含",
        "expected_keywords": ["头痛", "布洛芬", "对乙酰氨基酚"],  # 期望召回含这些关键词的文档
    },
    {
        "query": "布洛芬的用法用量",
        "type": "药物-精确",
        "expected_keywords": ["布洛芬", "用法", "用量"],
    },
    {
        "query": "感冒发烧怎么处理？",
        "type": "症状-复合",
        "expected_keywords": ["感冒", "发热", "退热"],
    },
    {
        "query": "高血压患者饮食注意什么",
        "type": "疾病-追问",
        "expected_keywords": ["高血压", "饮食", "钠盐"],
    },
    {
        "query": "小孩拉肚子怎么办",
        "type": "症状-口语化",
        "expected_keywords": ["腹泻", "儿童", "脱水"],
    },
    {
        "query": "糖尿病日常管理",
        "type": "疾病-管理",
        "expected_keywords": ["糖尿病", "血糖", "随访"],
    },
    {
        "query": "流鼻血了怎么办？",
        "type": "症状-自包含",
        "expected_keywords": ["鼻出血", "止血"],
    },
    {
        "query": "皮肤过敏用什么药",
        "type": "症状+药物",
        "expected_keywords": ["过敏", "抗组胺", "氯雷他定"],
    },
    {
        "query": "便秘怎么办？",
        "type": "症状-自包含",
        "expected_keywords": ["便秘", "膳食纤维"],
    },
    {
        "query": "胸闷喘不上气",
        "type": "症状-口语化",
        "expected_keywords": ["胸闷", "呼吸困难"],
    },
]


def generate_hyde(query: str) -> str:
    """用本地模型生成 HyDE 假想答案"""
    try:
        from app.core.llm import get_local_llm
        llm = get_local_llm()
        prompt = f"""请针对以下医学问题，写一段简短的假想性医学回答（2-3句话）。
不需要完全正确，但要使用医学专业术语，使其在语义上接近医学文献。

问题：{query}

假想回答："""
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.error(f"HyDE 生成失败：{e}")
        return ""


def retrieve_with_hyde(query: str, hyde_answer: str):
    """用 HyDE 假想答案做 Dense 检索"""
    from app.rag.hybrid_retriever import get_hybrid_retriever
    retriever = get_hybrid_retriever(k=3, rerank_top_k=5)
    # HyDE 模式：Dense 用假想答案，Sparse 用原始查询
    # 通过 invoke 传入，retriever 内部会根据 hyde_answer 使用
    start = time.time()
    docs = retriever.invoke(hyde_answer or query)  # Dense 用 HyDE
    elapsed = (time.time() - start) * 1000
    return docs, elapsed


def retrieve_without_hyde(query: str):
    """直接用原始查询检索"""
    from app.rag.hybrid_retriever import get_hybrid_retriever
    retriever = get_hybrid_retriever(k=3, rerank_top_k=5)
    start = time.time()
    docs = retriever.invoke(query)
    elapsed = (time.time() - start) * 1000
    return docs, elapsed


def calc_keyword_recall(docs, expected_keywords: list) -> float:
    """计算关键词覆盖率：期望关键词在召回文档中出现的比例"""
    if not expected_keywords:
        return 0.0
    all_text = " ".join(d.page_content for d in docs)
    hits = sum(1 for kw in expected_keywords if kw in all_text)
    return hits / len(expected_keywords)


def calc_doc_overlap(docs_a, docs_b) -> float:
    """计算两个文档列表的重叠度（Jaccard）"""
    set_a = set(d.page_content[:80] for d in docs_a)
    set_b = set(d.page_content[:80] for d in docs_b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def main():
    print("=" * 80)
    print("HyDE A/B 测试")
    print("=" * 80)

    results = []

    for i, test_case in enumerate(TEST_QUERIES):
        query = test_case["query"]
        q_type = test_case["type"]
        expected = test_case["expected_keywords"]

        print(f"\n[{i+1}/{len(TEST_QUERIES)}] 查询: {query} (类型: {q_type})")

        # ===== 无 HyDE =====
        docs_no_hyde, latency_no = retrieve_without_hyde(query)
        recall_no = calc_keyword_recall(docs_no_hyde, expected)
        rerank_no = max((d.metadata.get("rerank_score", 0) for d in docs_no_hyde), default=0)

        # ===== 有 HyDE =====
        hyde_start = time.time()
        hyde_answer = generate_hyde(query)
        hyde_latency = (time.time() - hyde_start) * 1000

        docs_with_hyde, latency_with = retrieve_with_hyde(query, hyde_answer)
        recall_with = calc_keyword_recall(docs_with_hyde, expected)
        rerank_with = max((d.metadata.get("rerank_score", 0) for d in docs_with_hyde), default=0)

        # 文档重叠度
        overlap = calc_doc_overlap(docs_no_hyde, docs_with_hyde)

        # 总耗时（HyDE 生成 + 检索）
        total_with = hyde_latency + latency_with
        total_without = latency_no

        result = {
            "query": query,
            "type": q_type,
            "recall_no": recall_no,
            "recall_with": recall_with,
            "recall_delta": recall_with - recall_no,
            "rerank_no": rerank_no,
            "rerank_with": rerank_with,
            "latency_no": total_without,
            "latency_with": total_with,
            "latency_delta": total_with - total_without,
            "hyde_latency": hyde_latency,
            "overlap": overlap,
            "hyde_answer": hyde_answer[:80],
            "docs_no_count": len(docs_no_hyde),
            "docs_with_count": len(docs_with_hyde),
        }
        results.append(result)

        # 打印单条结果
        delta_recall = result["recall_delta"]
        delta_tag = "✅ 提升" if delta_recall > 0.01 else ("❌ 下降" if delta_recall < -0.01 else "➡️ 持平")
        print(f"  无HyDE: Recall={recall_no:.1%}, Rerank={rerank_no:.4f}, 耗时={total_without:.0f}ms")
        print(f"  有HyDE: Recall={recall_with:.1%}, Rerank={rerank_with:.4f}, 耗时={total_with:.0f}ms (HyDE生成={hyde_latency:.0f}ms)")
        print(f"  差异: Recall {delta_tag} {delta_recall:+.1%}, 耗时 +{result['latency_delta']:.0f}ms, 文档重叠={overlap:.0%}")
        print(f"  HyDE: {hyde_answer[:60]}...")

    # ===== 汇总 =====
    print("\n" + "=" * 80)
    print("汇总报告")
    print("=" * 80)

    avg_recall_no = sum(r["recall_no"] for r in results) / len(results)
    avg_recall_with = sum(r["recall_with"] for r in results) / len(results)
    avg_delta = avg_recall_with - avg_recall_no
    avg_latency_no = sum(r["latency_no"] for r in results) / len(results)
    avg_latency_with = sum(r["latency_with"] for r in results) / len(results)
    avg_overlap = sum(r["overlap"] for r in results) / len(results)
    avg_hyde_ms = sum(r["hyde_latency"] for r in results) / len(results)

    # 统计 HyDE 正向/负向/持平
    positive = sum(1 for r in results if r["recall_delta"] > 0.01)
    negative = sum(1 for r in results if r["recall_delta"] < -0.01)
    neutral = len(results) - positive - negative

    print(f"\n{'指标':<25} {'无HyDE':<15} {'有HyDE':<15} {'差异':<15}")
    print("-" * 70)
    print(f"{'平均 Recall':<25} {avg_recall_no:<15.1%} {avg_recall_with:<15.1%} {avg_delta:+<15.1%}")
    print(f"{'平均耗时':<25} {avg_latency_no:<15.0f}ms {avg_latency_with:<15.0f}ms +{avg_latency_with-avg_latency_no:<14.0f}ms")
    print(f"{'HyDE 生成平均耗时':<25} {'—':<15} {avg_hyde_ms:<15.0f}ms")
    print(f"{'文档重叠度':<25} {avg_overlap:<15.0%}")
    print(f"\nHyDE Recall 正向: {positive}条, 负向: {negative}条, 持平: {neutral}条")

    # 决策建议
    print("\n" + "=" * 80)
    print("决策建议")
    print("=" * 80)
    if avg_delta < 0.02 and avg_delta >= -0.02:
        print("📊 Recall 提升 < 2%：HyDE 在当前架构下无显著正向收益")
        print("💡 建议：默认关闭 HyDE，保留开关按需触发")
    elif avg_delta < -0.02:
        print("📊 Recall 反而下降：HyDE 生成的噪声在污染检索")
        print("💡 建议：立即移除 HyDE")
    elif avg_delta >= 0.02 and positive > negative:
        print("📊 Recall 有正向提升，但需权衡延迟代价")
        print("💡 建议：仅对追问/模糊查询按需触发 HyDE")
    else:
        print("📊 结果不确定，需更多测试数据")

    print(f"\n延迟代价：平均 +{avg_latency_with-avg_latency_no:.0f}ms (主要为 HyDE 生成耗时 {avg_hyde_ms:.0f}ms)")
    print(f"收益：平均 Recall 提升 {avg_delta:+.1%}")


if __name__ == "__main__":
    main()
