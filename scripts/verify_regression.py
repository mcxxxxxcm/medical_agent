"""v9.21 回归：greeting/knowledge 类型不受症状增强影响，_enrich_treatment_query 不误改非治疗类查询"""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.graph.nodes.nodes import _enrich_treatment_query

# 1. 单元级：非治疗意图查询不应被增强
print("=" * 60)
print("回归 1：_enrich_treatment_query 非治疗类查询不应被改动")
print("=" * 60)
unit_cases = [
    "你好",
    "谢谢",
    "你是谁",
    "什么是高血压",
    "糖尿病的症状有哪些",
    "布洛芬的副作用",
    "我今年30岁",
]
unit_pass = True
for q in unit_cases:
    out = _enrich_treatment_query(q)
    ok = (out == q)
    if not ok:
        unit_pass = False
    print(f"  {'✅' if ok else '❌'} '{q}' → '{out}'")

# 2. 端到端：greeting 走 direct_answer，返回问候（不进 RAG，不被增强）
print("\n" + "=" * 60)
print("回归 2：端到端'你好'应走 direct_answer 直接回复")
print("=" * 60)
from langgraph.checkpoint.memory import MemorySaver
from app.graph.graph import build_graph


async def run(question: str, thread_id: str):
    builder = build_graph()
    try:
        from app.memory.long_term_memory import get_long_term_memory
        store = get_long_term_memory().store
    except Exception:
        store = None
    graph = builder.compile(checkpointer=MemorySaver(), store=store)
    config = {"configurable": {"thread_id": thread_id, "user_id": "reg", "store": store}}
    nodes = []
    final_answer = None
    async for event in graph.astream({"question": question, "user_id": "reg"}, config, stream_mode="updates"):
        for node, update in event.items():
            nodes.append(node)
            if isinstance(update, dict) and update.get("final_answer"):
                final_answer = update["final_answer"]
    return nodes, final_answer


nodes, answer = asyncio.run(run("你好", "reg_greeting"))
print(f"  经过节点: {' → '.join(dict.fromkeys(nodes))}")
print(f"  回复: {(answer or '')[:80]}")
went_direct = "direct_answer" in nodes and "knowledge_retrieval" not in nodes
print(f"  {'✅' if went_direct else '❌'} greeting 走 direct_answer，未进 RAG 检索")

print(f"\n回归总结果: {'全部通过 ✅' if (unit_pass and went_direct) else '存在失败 ❌'}")
sys.exit(0 if (unit_pass and went_direct) else 1)
