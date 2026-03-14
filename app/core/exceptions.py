"""异常处理模块

"""
from typing import Optional, Any


class MedicalAssistantException(Exception):
    """医疗助手基础异常类
    功能描述：
        所有自定义异常类的基类，提供统一的异常处理机制
    属性：
        message：错误消息（继承自Exception）
        error_code:错误码，用于错误分类和追踪
        details：详细信息字典，包含额外的上下文信息
    """

    def __init__(self, message: str, error_code: Optional[str], details: Optional[dict]):
        """初始化异常

        Args:
            message: 错误消息，描述发生了什么错误
            error_code: 错误码，用于错误分类和追踪
            details: 详细信息字典，例如：{'file_path': '...', 'query': '...'}
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}

        # 调用父类初始化，设置异常消息
        super().__init__(self.message)

    def to_dict(self) -> dict:
        """将异常转化成字典格式

        功能描述：
            提供统一的错误输出格式，便于API响应和日志记录

        Returns:
            {
                'error': '错误信息'
                'error_code': '错误码'
                'details': {'额外信息': '值'}
            }
        """
        return {
            'error': self.message,
            'error_code': self.error_code,
            'details': self.details
        }


class VectorStoreException(MedicalAssistantException):
    """向量库异常

    功能描述：
        用于向量库相关错误的异常类，例如向量库创建、加载、检索操作中的错误。

    错误码：VECTOR_STORE_ERROR
    """

    def __init__(self, message: str, details: Optional[dict] = None):
        """初始化向量库异常"""
        super().__init__(
            message=message,
            error_code="VECTOR_STORE_ERROR",
            details=details
        )


class RetrievalError(MedicalAssistantException):
    """检索异常

    功能描述：
        用于文档检索相关错误的异常类，如检索器初始化失败、检索操作失败、检索结果为空、检索参数无效

    错误码：
        RETRIEVAL_ERROR
    """

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(
            message=message,
            error_code="RETRIEVAL_ERROR",
            details=details
        )


class ToolExecutionError(MedicalAssistantException):
    """工具执行异常
    功能描述：
        用于工具执行相关错误的异常类

    错误码：
        TOOL_EXECUTION_ERROR
    """

    def __init__(self, message: str, tool_name: str, details: Optional[dict] = None):
        # 合并工具名称到详细信息中
        tool_details = {**(details or {}), "tool_name": tool_name}
        super().__init__(
            message=message,
            error_code="TOOL_EXECUTION_ERROR",
            details=tool_details
        )


class AgentError(MedicalAssistantException):
    """Agent异常
    功能描述：
        用于Agent操作相关错误的异常类

    错误码：
        AGENT_ERROR
    """

    def __init__(self, message: str, agent_name: str, details: Optional[dict] = None):
        # 合并Agent名称到详细信息中
        agent_details = {**(details or {}), "agent_name": agent_name}
        super().__init__(
            message=message,
            error_code="AGENT_ERROR",
            details=agent_details
        )


class ValidationError(MedicalAssistantException):
    """验证异常
    功能描述：
        用于数据验证相关错误的异常类，例如：请求参数验证失败、数据格式验证失败、业务规则验证失败、字段长度/类型验证失败

    错误码：
        VALIDATION_ERROR
    """

    def __init__(self, message: str, field: str, details: Optional[dict] = None):
        """初始化验证异常
        Args：
            field：字段名称，标识哪个字段验证失败
        """
        # 合并字段名称到详细信息中
        validation_details = {**(details or {}), "field": field}
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=validation_details
        )


class LLMError(MedicalAssistantException):
    """LLM调用异常类"""

    def __init__(self, message: str, model: str, details: Optional[dict] = None):
        """初始化LLM异常
        Args:
            model: LLM模型名称，标识哪个模型出错
        """
        # 合并模型名称到详细信息
        llm_details = {**(details or {}), "model": model}
        super().__init__(
            message=message,
            error_code="LLM_ERROR",
            details=llm_details
        )