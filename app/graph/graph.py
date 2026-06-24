"""LangGraph工作流构建模块
功能描述：
    构建医疗助手的完整工作流图
    整合State、Nodes、Checkpointer、Store

设计理念：
    1、单一入口：router_node 作为入口节点
    2、Command路由：使用Command.goto控制流程
    3、条件分支：symptom路径和knowledge路径
    4、统一出口：所有路径汇聚到answer_generation

工作流：
    START -> router -> [symptom_analysis | knowledge_retrieval]
          -> answer_generation -> safety_check -> END
"""
import sys
import asyncio

# Windows 平台需要设置事件循环策略
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uuid
from typing import List, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from app.graph.state import (
    MedicalAssistantState,
    InputSchema,
    OutputSchema,
)
from app.graph.nodes import (
    router_node,
    symptom_analysis_node,
    knowledge_retrieval_node,
    answer_generation_node,
    safety_check_node, memory_load_node,
    profile_extraction_node,
    direct_answer_node,
    vision_analysis_node,
    query_rewrite_node,
    grade_documents_node,
    should_update_snapshot,
    update_clinical_snapshot_node
)
from app.memory import get_long_term_memory
from app.memory.checkpointer import get_checkpointer
from app.core.app_logging import get_logger

logger = get_logger(__name__)


def build_graph() -> StateGraph:
    """构建工作流图

    Returns:
        StateGraph: 未编译的工作流图构建器

    ⚠️ 同步提醒：
        流式接口 (app/api/routes.py) 手动编排节点调用顺序。
        修改此图的节点或边时，必须同步更新 routes.py 中的流式处理逻辑。
        启动时会自动运行 validate_streaming_sync() 检测不一致。
    """
    builder = StateGraph(
        MedicalAssistantState,
        input_schema=InputSchema,
        output_schema=OutputSchema,
    )

    builder.add_node("memory_load", memory_load_node)
    builder.add_node("profile_extraction", profile_extraction_node)
    builder.add_node("router", router_node)
    builder.add_node("symptom_analysis", symptom_analysis_node)
    builder.add_node("knowledge_retrieval", knowledge_retrieval_node)
    builder.add_node("grade_documents", grade_documents_node)
    builder.add_node("direct_answer", direct_answer_node)
    builder.add_node("vision_analysis", vision_analysis_node)
    builder.add_node("answer_generation", answer_generation_node)
    builder.add_node("safety_check", safety_check_node)
    builder.add_node("query_rewrite", query_rewrite_node)
    # L2会话层：临床状态快照（滑动窗口触发）
    builder.add_node("update_snapshot", update_clinical_snapshot_node)
    builder.add_edge("update_snapshot", END)

    # ===== 添加边 =====

    # 入口路径
    builder.add_edge(START, "memory_load")
    builder.add_edge("memory_load", "profile_extraction")
    builder.add_edge("profile_extraction", "router")

    # symptom 路径：router -> symptom_analysis -> query_rewrite -> knowledge_retrieval
    builder.add_edge("symptom_analysis", "query_rewrite")
    builder.add_edge("query_rewrite", "knowledge_retrieval")

    # knowledge 路径：router -> query_rewrite -> knowledge_retrieval

    # general 路径：router -> direct_answer

    # 检索后先评分，grade_documents 使用 Command.goto 决定去向：
    #   相关 -> answer_generation
    #   不相关 -> query_rewrite（自纠正循环）
    builder.add_edge("knowledge_retrieval", "grade_documents")

    # 根据配置决定是否启用 safety_check
    from app.core.config import get_config
    config = get_config()

    if getattr(config, 'ENABLE_SAFETY_CHECK', True):
        # 启用 safety_check
        builder.add_edge("direct_answer", "safety_check")
        builder.add_edge("vision_analysis", "safety_check")
        builder.add_edge("answer_generation", "safety_check")

        builder.add_conditional_edges(
            "safety_check",
            should_update_snapshot,
            {
                "update_snapshot": "update_snapshot",
                END: END
            }
        )
    else:
        # 关闭 safety_check，直接跳到快照判断
        builder.add_conditional_edges(
            "direct_answer",
            should_update_snapshot,
            {
                "update_snapshot": "update_snapshot",
                END: END
            }
        )
        builder.add_conditional_edges(
            "vision_analysis",
            should_update_snapshot,
            {
                "update_snapshot": "update_snapshot",
                END: END
            }
        )
        builder.add_conditional_edges(
            "answer_generation",
            should_update_snapshot,
            {
                "update_snapshot": "update_snapshot",
                END: END
            }
        )

    logger.info("工作流图构建完成")
    return builder


async def compile_graph():
    """编译工作流图

    Returns:
        CompiledGraph: 可执行的工作流图
    """
    builder = build_graph()

    # 异步获取checkpointer
    from app.memory.checkpointer import get_checkpointer
    checkpointer = await get_checkpointer()
    try:
        memory = get_long_term_memory()
        store = memory.store
        logger.info(f"长期记忆store已加载")
    except Exception as e:
        logger.warning(f"长期记忆Store加载失败：{e}")
        store = None

    graph = builder.compile(
        checkpointer=checkpointer,
        store=store
    )

    logger.info("工作流图编译完成")

    return graph


_graph = None


async def get_graph():
    """获取工作流图实例（单例）

    Returns:
        CompiledGraph: 可执行的工作流图
    """
    global _graph
    if _graph is None:
        _graph = await compile_graph()
    return _graph


def reset_graph() -> None:
    """重置工作流图实例

    使用场景：
        单元测试中重置状态
    """
    global _graph
    _graph = None
    logger.info("工作流图已重置")


# 流式接口必须编排的节点名称集合（不含 memory_load/profile_extraction，它们在流式接口中单独处理）
_STREAMING_REQUIRED_NODES = {
    "router", "symptom_analysis", "query_rewrite",
    "knowledge_retrieval", "grade_documents",
    "answer_generation", "direct_answer", "vision_analysis",
    "update_snapshot",
}


def validate_streaming_sync() -> List[str]:
    """验证流式接口与 Graph 定义是否同步

    检查 Graph 中定义的节点是否都在流式接口中被编排。
    启动时自动调用，发现不一致时输出警告。

    Returns:
        不一致的节点名称列表（空列表表示完全同步）
    """
    builder = build_graph()
    graph_nodes = set(builder.nodes.keys())

    # 检查流式接口是否遗漏了 Graph 中的节点
    missing_in_streaming = graph_nodes - _STREAMING_REQUIRED_NODES - {"memory_load", "profile_extraction", "safety_check"}

    # 检查流式接口是否引用了 Graph 中不存在的节点
    extra_in_streaming = _STREAMING_REQUIRED_NODES - graph_nodes

    issues = []
    if missing_in_streaming:
        issues.append(f"Graph 中存在但流式接口未编排的节点：{missing_in_streaming}")
    if extra_in_streaming:
        issues.append(f"流式接口引用了 Graph 中不存在的节点：{extra_in_streaming}")

    for issue in issues:
        logger.warning(f"⚠️ 流式接口与 Graph 不同步：{issue}，请检查 app/api/routes.py")

    if not issues:
        logger.info("✅ 流式接口与 Graph 节点定义同步验证通过")

    return issues


async def run_graph(
        question: str,
        user_id: str = None,
        thread_id: str = None,
) -> dict:
    """运行工作流

    Args:
        question: 用户问题
        user_id: 用户ID，可选
        thread_id: 会话线程ID，可选

    Returns:
        输出状态字典
    """
    graph = await get_graph()
    try:
        memory = get_long_term_memory()
        store = memory.store
    except Exception as e:
        store = None

    config = {
        "configurable": {
            "thread_id": thread_id or f"thread_{user_id or 'default'}",
            "user_id": user_id,
            "store": store
        }
    }

    input_state = {
        "question": question,
        "user_id": user_id,
    }

    result = await graph.ainvoke(input_state, config)

    if user_id:
        try:
            memory = get_long_term_memory()
            memory.save_query_record(
                user_id=user_id,
                query_id=str(uuid.uuid4()),
                query_data={
                    "question": question,
                    "answer": result.get("final_answer"),
                    "sources": result.get("sources", []),
                }
            )
        except Exception as e:
            logger.warning(f"保存查询记录失败：{e}")

    return result


if __name__ == "__main__":
    async def test():
        result = await run_graph(
            question="你好，我刚刚问了什么问题？",
            user_id="test_user3",
            thread_id="test_thread_003",
        )
        print(f"最终答案: {result.get('final_answer')}")
        print(f"警告信息: {result.get('warnings')}")
        print(f"来源信息: {result.get('sources')}")

    import asyncio
    asyncio.run(test())
