"""数据模型模块
功能描述：
    定义项目中所有Pydantic数据模型，用于请求验证、响应格式化、数据结构定义等。使用Pydantic提供自动验证和类型转换。

设计理念：
    1、使用Pydantic BaseModel：自动数据验证
    2、清晰的字段描述：提供详细的字段说明

包引用：
    pydantic：数据验证和设置管理库
    BaseModel：Pydantic基础模型类
    Field：Pydantic字段定义类，用于配置字段属性
    filed_validator：Pydantic验证器装饰器
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any


class QuestionRequest(BaseModel):
    """问答请求模型
    功能描述：
        定义用户回答问题的数据结构，包含验证规则。

    字段说明：
        question：用户问题（必须）
        return_sources：是否返回来源（可选）
        k：返回的文档数量（可选）
        stream：是否使用流式输出（可选）
    """
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="用户问题，长度在1-1000字符之间"
    )

    return_sources: bool = Field(
        default=True,
        description="是否返回消息来源，默认为True"
    )

    k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="返回的文档数量，范围在1-10之间"
    )

    stream: bool = Field(
        default=True,
        description="是否使用流式输出，默认为True"
    )

    @field_validator('question')
    def question_must_not_contain_sensitive_words(cls, value):
        """自定义验证器：检查问题是否包含敏感词
        功能描述：
            在字段级别添加自定义验证逻辑

        Args:
            cls：模型类
            value：字段值

        包引用：
            filed_validator：验证器装饰器
        """
        sensitive_keywords = ["自杀", "自残", "杀人"]
        if any(keyword in value for keyword in sensitive_keywords):
            raise ValueError("问题包含敏感内容，请注意用词。")
        return value


class QuestionResponse(BaseModel):
    """问答响应模型
    功能描述：
        定义问答相应的数据结构，统一API返回格式
        提供标准化的成功/失败响应格式

    字段说明：
        success：是否成功（必须）
        answer：生成的答案（可选）
        sources：来源信息列表（可选）
        error：错误信息（可选）
        warnings：警告信息列表（可选）
    """
    success: bool = Field(
        ...,
        description="是否成功处理请求"
    )

    answer: Optional[str] = Field(
        default=None,
        description="生成的答案内容"
    )

    sources: Optional[List[str]] = Field(
        default_factory=list,
        description="来源信息列表，包含文档来源和内容预览"
    )

    error: Optional[str] = Field(
        default=None,
        description="错误信息"
    )

    warning: Optional[str] = Field(
        default=None,
        description="警告信息列表"
    )


class SourceInfo(BaseModel):
    """来源信息模型
    功能描述：
        定义文档来源的数据结构，用于返回检索到的文档信息

    使用场景：
        RAG检索结果格式化
        来源信息展示
        答案引用

    字段说明：
        source：文档来源名称（必须）
        file_path：文档路径（必须）
        content：内容预览（必须）
    """

    source: str = Field(
        ...,
        description="文档来源名称，如文件名"
    )

    file_path: str = Field(
        ...,
        description="文档的完整路径"
    )

    content: str = Field(
        ...,
        description="文档的内容预览"
    )


class ToolCallRequest(BaseModel):
    """工具调用请求模型
    功能描述：
        定义工具调用的请求数据结构
        用于Agent或服务层调用工具时的参数验证

    使用场景：
        Agent工具调用
        API工具接口
        工具参数验证

    字段说明：
        tool_name：工具名称（必须）
        parameters：工具参数（可选）
    """

    tool_name: str = Field(
        ...,
        description="工具名称，标识要调用的具体工具"
    )

    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="工具参数字典"
    )


class ToolCallResponse(BaseModel):
    """工具调用响应模型
    功能描述：
        定义工具调用时的响应数据结构
        统一工具调用执行结果的返回格式

    使用场景：
        Agent工具调用结果
        API工具接口响应
        工具执行监控

    字段说明：
        success：是否成功（必须）
        result：工具执行结果（可选）
        error：错误信息（可选）
    """

    success: bool = Field(
        ...,
        description="工具是否成功执行"
    )

    result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="工具执行结果数据，结果取决于具体工具"
    )

    error: Optional[str] = Field(
        default=None,
        description="工具执行错误信息"
    )


class HealthCheckResponse(BaseModel):
    """健康检查响应模型
    功能描述：
        定义健康检查接口的响应数据机构
        用于监控和负载均衡器的健康检查

    字段说明：
        status：服务状态（必须）
        service：服务名称（必须）
        version：版本号（可选）
        dependencies：依赖服务列表（可选）
    """

    status: str = Field(
        ...,
        description="服务状态，如：healthy、unhealthy",
    )

    service: str = Field(
        ...,
        description="服务名称，如：medical-assistant"
    )

    version: str = Field(
        default="1.0.0",
        description="应用版本号"
    )

    dependencies: List[str] = Field(
        default_factory=list,
        description="以来服务列表，如：vector-store、llm-api"
    )


class ErrorResponse(BaseModel):
    """错误响应模型
    功能描述：
        定义统一的错误响应数据结构
        所有API错误都使用次格式返回

    字段说明：
        error：错误信息（必须）
        error_code：错误码（可选）
        details：详细信息（可选）
    """

    error: str = Field(
        ...,
        description="错误信息"
    )

    error_code: str = Field(
        default=None,
        description="错误码"
    )

    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="详细信息字典，包含额外的错误上下文"
    )
