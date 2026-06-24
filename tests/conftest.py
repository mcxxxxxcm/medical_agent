"""pytest 配置文件"""

import pytest


@pytest.fixture
def mock_llm():
    """Mock LLM 实例，避免测试中调用真实 API"""
    from unittest.mock import MagicMock
    mock = MagicMock()
    mock.invoke.return_value.content = "test response"
    return mock


@pytest.fixture
def mock_embeddings():
    """Mock Embedding 实例"""
    from unittest.mock import MagicMock
    mock = MagicMock()
    mock.embed_query.return_value = [0.1] * 2048
    return mock


@pytest.fixture
def base_state():
    """基础测试状态"""
    return {
        "question": "测试问题",
        "user_id": "test_user",
        "messages": [],
        "final_answer": None,
        "warnings": [],
        "sources": [],
        "retrieved_docs": None,
        "symptoms": None,
        "error": None,
        "user_profile": None,
        "rewritten_query": None,
        "hyde_answer": None,
        "retrieval_attempts": 0,
        "clinical_checkpoint": None,
    }
