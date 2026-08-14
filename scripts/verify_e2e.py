"""v9.21 验证：端到端跑图"发烧怎么办"，确认护理措施为"减少衣物/散热"，不再出现"增加衣物保暖"幻觉"""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from langgraph.checkpoint.memory import MemorySaver

from app.graph.graph import build_graph
from app.core.app_logging import get_logger

logger = get_logger(__name__)


async def run(question: str, thread_id: str = "e2e_test") -> dict:
    builder = build_graph()
    checkpointer = MemorySaver()
    try:
        from app.memory.long_term_memory import get_long_term_memory
        store = get_long_term_memory().store
    except Exception:
        store = None

    graph = builder.compile(checkpointer=checkpointer, store=store)
    config = {"configurable": {"thread_id": thread_id, "user_id": "e2e", "store": store}}

    input_state = {"question": question, "user_id": "e2e"}
    final_answer = None
    node_updates = []
    async for event in graph.astream(input_state, config, stream_mode="updates"):
        for node, update in event.items():
            node_updates.append(node)
            if isinstance(update, dict) and update.get("final_answer"):
                final_answer = update["final_answer"]
    return {"final_answer": final_answer, "nodes": node_updates}


def main():
    q = "发烧怎么办"
    print(f"=" * 70)
    print(f"端到端测试：{q}")
    print(f"=" * 70)
    result = asyncio.run(run(q))
    answer = result["final_answer"] or "(无 final_answer)"
    print(f"经过节点: {' → '.join(dict.fromkeys(result['nodes']))}")
    print(f"\n【答案】\n{answer}\n")

    # 每项 (label, passed)：passed=True 表示该项通过
    cooling_ok = ("减少衣物" in answer) or ("散热" in answer)
    checks = [
        ("✅ 包含正确散热护理（减少衣物/散热）", cooling_ok),
        ("✅ 不应出现'增加衣物'（旧幻觉）", "增加衣物" not in answer),
        ("✅ 不应出现'注意保暖'（旧幻觉）", "注意保暖" not in answer),
        # 捂汗只能作为"禁忌"出现（文档原文"禁忌：…捂汗"），绝不能作为建议
        ("✅ '捂汗'仅作禁忌引用（不构成错误建议）", ("捂汗" not in answer) or ("禁忌" in answer and "捂汗" in answer)),
        ("✅ 含免责声明", "仅供参考" in answer),
    ]
    all_pass = True
    for label, passed in checks:
        print(f"{label}")
        if not passed:
            all_pass = False
    print(f"\n端到端结果: {'通过 ✅' if all_pass else '存在失败 ❌'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
