"""RAG问答链模块
功能描述：
    实现基于检索增强生成(RAG)的问答链，结合向量检索和大语言模型生成准确答案
    支持流式输出、来源追踪、多轮对话等企业级功能

设计理念：
    1、链式组合：使用LangChain的链式组合模式，将检索、生成、后处理串联
    2、可配置性：支持自定义提示词、检索参数、生成参数
    3、来源追踪：返回答案的同时提供文档来源信息，增强可信度
    4、流式输出：支持流式响应，提升用户体验
    5、错误处理：完善的异常处理机制，保证系统稳定性

包引用：
    typing：Python类型提示模块
    List：列表类型，用于存储多个相同类型的元素
    Dict：字典类型，用于存储键值对
    Any：任意类型，用于不确定类型的场景
    Optional：可选类型，表示字段可以为None
    AsyncIterator：异步迭代器类型，用于流式输出
    langchain_core.prompts：LangChain提示词模块
    ChatPromptTemplate：聊天提示词模板，用于构建结构化提示
    langchain_core.runnables：LangChain可运行对象模块
    RunnablePassthrough：直通组件，用于数据传递
    RunnableParallel：并行组件，用于并行执行多个链
    langchain_core.output_parsers：LangChain输出解析器模块
    StrOutputParser：字符串输出解析器，用于解析LLM输出
    langchain_core.documents：LangChain文档模块
    Document：文档类，表示一个文档对象，包含内容和元数据
    app.core.llm：自定义LLM模块，提供大语言模型实例
    get_llm：获取LLM实例的便携函数
    app.rag.vector_store：向量存储模块，提供文档检索功能
    get_retriever：获取检索器的便携函数
    app.core.app_logging：日志模块，提供日志记录功能
    get_logger：获取日志记录器的便携函数
"""
from typing import List, Dict, Any, Optional, AsyncIterator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from app.core.llm import get_llm
from app.rag.vector_store import get_retriever
from app.core.app_logging import get_logger

logger = get_logger(__name__)


class QAChain:
    """RAG问答链类
    功能描述：
        封装完整的RAG问答流程，包括文档检索、答案生成、来源追踪等
        这是RAG系统的核心组件，负责将检索和生成串联起来

    使用场景：
        医疗问答系统
        知识库问答
        文档智能检索
        客服问答系统

    属性说明：
        retriever：文档检索器实例，用于检索相关文档
        llm：大语言模型实例，用于生成答案
        return_sources：是否返回文档来源信息
        prompt_template：提示词模板字符串
        chain：LangChain链对象，封装了完整的问答流程
    """

    def __init__(
            self,
            retriever=None,
            llm=None,
            prompt_template: Optional[str] = None,
            return_sources: bool = True
    ):
        """初始化RAG问答链
        功能描述：
            初始化问答链的所有组件，包括检索器、LLM、提示词模板等

        Args:
            retriever：文档检索器实例，如果为None则使用默认检索器
            llm：大语言模型实例，如果为None则使用默认模型
            prompt_template：自定义提示词模板，如果为None则使用默认模板
            return_sources：是否返回文档来源信息，默认为True

        包引用：
            get_retriever：获取向量检索器实例
            get_llm：获取大语言模型实例
            get_logger：获取日志记录器
        """
        self.retriever = retriever or get_retriever()
        self.llm = llm or get_llm()
        self.return_sources = return_sources
        self.prompt_template = prompt_template or self._get_default_prompt()

        self.chain = self._build_chain()
        logger.info("RAG问答链初始化完成")

    def _get_default_prompt(self) -> str:
        """获取默认提示词模板
        功能描述：
            定义医疗领域的专业提示词，引导LLM生成准确、安全的医疗建议

        Returns:
            str：提示词模板字符串

        设计理念：
            1、角色定位：明确AI助手的专业身份
            2、安全提醒：强调非医生建议，建议就医
            3、来源引用：要求基于检索到的文档回答
            4、结构化输出：引导生成清晰的答案结构
            5、用户友好：生成易于理解的答案

        提示词结构：
            - 重要提醒：安全免责声明
            - 参考文档：检索到的文档内容
            - 用户问题：用户的具体问题
            - 回答要求：答案生成的指导原则
        """
        return """你是一位专业的医疗健康助手，基于提供的医疗文档内容回答用户问题。
        【重要提醒】
        1. 你的回答仅供参考，不能替代专业医生的诊断和治疗建议
        2. 如果问题涉及紧急医疗情况，请立即建议用户就医
        3. 回答时要基于提供的文档内容，不要编造信息

        【参考文档】
        {context}

        【用户问题】
        {question}

        【回答要求】
        1. 基于参考文档内容，准确回答用户问题
        2. 如果文档中没有相关信息，请明确说明
        3. 回答要清晰易懂，避免过于专业的术语
        4. 必要时提供实用的家庭护理建议
        5. 结尾加上安全提醒："如有疑问，请及时就医"

        请用中文回答：
        """

    def _build_chain(self):
        """构建RAG问答链
        功能描述：
            使用LangChain的链式组合模式，构建完整的问答流程
            流程：输入问题 -> 检索相关文档 -> 构建提示词 -> LLM生成答案 -> 解析输出

        Returns:
            Runnable：可执行的链对象

        设计理念：
            1、链式组合：使用LangChain的链式API组合各个组件
            2、并行执行：同时执行检索和问题传递
            3、格式化处理：将检索结果格式化为提示词
            4、输出解析：解析LLM的输出为字符串

        链式结构：
            RunnablePassthrough.assign(context=...) -> prompt -> llm -> StrOutputParser()

        包引用：
            ChatPromptTemplate：聊天提示词模板
            RunnablePassthrough：直通组件，传递原始问题
            RunnableParallel：并行组件，同时执行检索和问题传递
            StrOutputParser：字符串输出解析器
        """
        prompt = ChatPromptTemplate.from_template(self.prompt_template)

        def format_docs(docs: List[Document]) -> str:
            """格式化检索到的文档
            功能描述：
                将检索到的文档列表格式化为字符串，便于插入提示词

            Args:
                docs：文档列表

            Returns:
                str：格式化后的文档字符串

            设计理念：
                1、来源标注：每段内容标注来源文件
                2、内容截取：每段内容限制长度，避免提示词过长
                3、清晰分隔：使用分隔符区分不同文档片段
                4、可读性：格式化的文档易于LLM理解和生成答案
            """
            formatted = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "未知来源")
                content = doc.page_content
                formatted.append(f"[文档{i} 来源：{source}]\n{content}")
            return "\n\n".join(formatted)

        chain = (
                RunnablePassthrough.assign(
                    context=lambda x: format_docs(self.retriever.invoke(x["question"]))
                )
                | prompt
                | self.llm
                | StrOutputParser()
        )

        return chain

    def invoke(self, question: str, return_sources: Optional[bool] = None) -> Dict[str, Any]:
        """同步调用问答链
        功能描述：
            执行完整的问答流程，返回答案和来源信息
            这是主要的调用接口，用于同步问答

        Args:
            question：用户问题，字符串类型
            return_sources：是否返回来源信息，如果为None则使用实例默认值

        Returns:
            Dict[str, Any]：包含答案和来源信息的字典
                - answer：生成的答案
                - sources：来源信息列表（如果return_sources为True）
                - question：原始问题

        设计理念：
            1、完整流程：执行检索、生成、解析的完整流程
            2、来源追踪：返回文档来源，增强答案可信度
            3、错误处理：完善的异常处理机制
            4、日志记录：记录执行过程，便于调试

        包引用：
            retriever.invoke：检索相关文档
            chain.invoke：执行问答链
            get_logger：获取日志记录器
        """
        logger.info(f"开始处理问题：{question}")

        try:
            should_return_sources = return_sources if return_sources is not None else self.return_sources

            docs = self.retriever.invoke(question)
            logger.info(f"检索到 {len(docs)} 个相关文档")

            answer = self.chain.invoke({"question": question})
            logger.info(f"生成答案完成，长度：{len(answer)} 字符")

            result = {
                "answer": answer,
                "question": question
            }

            if should_return_sources:
                sources = [
                    {
                        "source": doc.metadata.get("source", "未知来源"),
                        "file_path": doc.metadata.get("file_path", ""),
                        "content": doc.page_content[:200]
                    }
                    for doc in docs
                ]
                result["sources"] = sources

            return result

        except Exception as e:
            logger.error(f"问答链执行失败：{str(e)}")
            raise

    async def astream(self, question: str) -> AsyncIterator[str]:
        """异步流式调用问答链
        功能描述：
            以流式方式生成答案，支持实时输出，提升用户体验
            适用于需要实时显示答案的场景

        Args:
            question：用户问题，字符串类型

        Yields:
            str：生成的答案片段

        设计理念：
            1、流式输出：实时生成答案片段，提升用户体验
            2、异步执行：不阻塞主线程，提高系统性能
            3、实时反馈：用户可以实时看到答案生成过程
            4、错误处理：完善的异常处理机制

        包引用：
            chain.astream：异步流式执行问答链
            get_logger：获取日志记录器
        """
        logger.info(f"开始流式处理问题：{question}")

        try:
            async for chunk in self.chain.astream({"question": question}):
                if chunk:
                    yield chunk

        except Exception as e:
            logger.error(f"流式问答执行失败：{str(e)}")
            raise

    def stream(self, question: str):
        """同步流式调用问答链
        功能描述：
            以流式方式生成答案，支持实时输出
            适用于需要实时显示答案的场景

        Args:
            question：用户问题，字符串类型

        Yields:
            str：生成的答案片段

        设计理念：
            1、流式输出：实时生成答案片段，提升用户体验
            2、同步执行：简单的流式接口，易于使用
            3、实时反馈：用户可以实时看到答案生成过程
            4、错误处理：完善的异常处理机制

        包引用：
            chain.stream：流式执行问答链
            get_logger：获取日志记录器
        """
        logger.info(f"开始流式处理问题：{question}")

        try:
            for chunk in self.chain.stream({"question": question}):
                if chunk:
                    yield chunk

        except Exception as e:
            logger.error(f"流式问答执行失败：{str(e)}")
            raise


def get_qa_chain(
        retriever=None,
        llm=None,
        prompt_template: Optional[str] = None,
        return_sources: bool = True
) -> QAChain:
    """获取RAG问答链实例（便携函数）
    功能描述：
        便捷函数，用于快速获取问答链实例
        封装了QAChain的初始化逻辑，提供更简洁的接口

    Args:
        retriever：文档检索器实例，可选
        llm：大语言模型实例，可选
        prompt_template：自定义提示词模板，可选
        return_sources：是否返回文档来源信息，默认为True

    Returns:
        QAChain：问答链实例

    设计理念：
        1、便捷接口：提供简洁的获取方式
        2、默认配置：使用合理的默认配置
        3、可定制：支持自定义各个组件
        4、单例模式：可以扩展为单例模式

    使用场景：
        快速创建问答链实例
        在服务层或API层使用
        测试和调试

    包引用：
        QAChain：问答链类
    """
    return QAChain(
        retriever=retriever,
        llm=llm,
        prompt_template=prompt_template,
        return_sources=return_sources
    )


if __name__ == "__main__":
    # 测试问答链
    qa_chain = get_qa_chain()
    result = qa_chain.invoke("告诉我母崇徐的六级成绩？")
    print("答案：", result["answer"])
    if "sources" in result:
        print("\n来源：")
        for source in result["sources"]:
            print(f"- {source['source']}")
