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

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

# Windows 平台需要设置事件循环策略
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uuid
from typing import Literal

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
    query_rewrite_node,
    grade_documents_node,
    should_summarize,
    summarize_conversation_node
)
from app.memory import get_long_term_memory
from app.memory.checkpointer import get_checkpointer
from app.core.app_logging import get_logger

logger = get_logger(__name__)


# 使用create_agent创建的agent可以以节点的方式直接加入workflow, 但这里的逻辑有误，子agent无法总结主agent的messages
# def create_summarization_subagent():
#     """创建消息总结子agent"""
#     return create_agent(
#         model="gpt-4o-mini",
#         tools=[],
#         middleware=[
#             SummarizationMiddleware(
#                 model="gpt-4o-mini",
#                 trigger={"tokens": 4000, "messages": 10},  # tokens > 4000 AND messages > 10时触发
#                 keep={"messages": 6}  # 保留最近6条消息
#             ),
#         ]
#     )


def build_graph() -> StateGraph:
    """构建工作流图

    Returns:
        StateGraph: 未编译的工作流图构建器
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
    builder.add_node("answer_generation", answer_generation_node)
    builder.add_node("safety_check", safety_check_node)
    builder.add_node("query_rewrite", query_rewrite_node)
    # 自定义消息总结
    builder.add_node("summarize_conversation", summarize_conversation_node)

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
        builder.add_edge("answer_generation", "safety_check")

        builder.add_conditional_edges(
            "safety_check",
            should_summarize,
            {
                "summarize": "summarize_conversation",
                END: END
            }
        )
    else:
        # 关闭 safety_check，直接跳到总结判断
        builder.add_conditional_edges(
            "direct_answer",
            should_summarize,
            {
                "summarize": "summarize_conversation",
                END: END
            }
        )
        builder.add_conditional_edges(
            "answer_generation",
            should_summarize,
            {
                "summarize": "summarize_conversation",
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
    import asyncio


    async def test():
        # print("【保存用户档案】")
        # user_id = "test_user"
        # try:
        #     memory = get_long_term_memory()
        #     memory.save_user_profile(
        #         user_id=user_id,
        #         profile={
        #             "name": "张三",
        #             "age": 30,
        #             "gender": "男",
        #             "allergies": ["青霉素", "海鲜"],
        #         }
        #     )
        #     print(f"  用户档案已保存：user_id={user_id}")
        # except Exception as e:
        #     print(f"  保存失败：{e}")

        print("=== 测试工作流 ===\n")

        result = await run_graph(
            question="你好,我刚刚问了什么问题？",
            user_id="test_user3",
            thread_id="test_thread_003",
        )

        print(f"最终答案: {result.get('final_answer')}")
        print(f"警告信息: {result.get('warnings')}")
        print(f"来源信息: {result.get('sources')}")


    asyncio.run(test())
