# 医疗助手智能问答系统

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.129.0-green)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0.10-orange)](https://langchain-ai.github.io/langgraph/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue)](https://www.docker.com/)

基于 RAG（检索增强生成）技术的医疗领域智能问答系统，支持多轮对话、知识检索、图片识别等功能。

## ✨ 核心特性

### 🤖 智能问答
- **多轮对话**：支持上下文感知的连续对话
- **流式响应**：SSE 实时推送，无需等待完整生成
- **智能路由**：根据问题类型自动选择处理路径（症状/知识/一般问题）
- **查询重写**：HyDE 技术优化检索质量

### 🔍 混合检索
- **Dense + Sparse**：向量检索 + BM25 混合
- **RRF 融合**：Reciprocal Rank Fusion 算法融合结果
- **Reranker 重排序**：bge-reranker 提升相关性
- **三层缓存**：L1 精确匹配、L2 语义缓存、L3 热点缓存

### 💾 持久化存储
- **PostgreSQL**：对话检查点持久化，支持断点续聊
- **Redis**：查询结果缓存，加速响应
- **ChromaDB**：向量数据库存储医疗文档

### 🖼️ 图片识别（新增）
- **OCR 预处理**：PaddleOCR 提取报告文字
- **多模态分析**：支持 GPT-4o 等视觉模型
- **报告解析**：自动分析检查单、化验单

### 🐳 容器化部署
- **Docker Compose**：一键启动所有服务
- **健康检查**：自动监控服务状态
- **数据持久化**：容器重启数据不丢失

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        前端层                                │
│              Web UI (HTML + JavaScript)                     │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      API 网关层                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ /api/chat   │  │/api/chat/   │  │ /api/upload/analyze │  │
│  │   同步聊天   │  │   stream    │  │     图片分析        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph 工作流                          │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │  Memory │───→│  Router │───→│Retrieve │───→│ Generate│  │
│  │  Load   │    │  路由   │    │  检索   │    │  生成   │  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      RAG 检索层                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Vector Store│  │   BM25      │  │    Reranker         │  │
│  │  (ChromaDB) │  │ (稀疏检索)   │  │  (bge-reranker-onnx)│  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      存储层                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  PostgreSQL │  │    Redis    │  │    ChromaDB         │  │
│  │ (Checkpoints│  │  (L1 Cache) │  │   (Vector Store)    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 环境要求

- Python 3.11+
- Docker & Docker Compose
- 8GB+ 内存
- 10GB+ 磁盘空间

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd medical_assistant_agent
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```bash
# LLM 配置（支持 OpenAI、智谱等）
MODEL_NAME=gpt-4o
MODEL_URL=https://api.openai.com/v1
MODEL_API_KEY=your-api-key
MODEL_TEMPERATURE=0.2

# Embedding 配置
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536

# 数据库配置
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/medical_assistant_db
REDIS_URL=redis://localhost:6379/0

# 缓存配置
ENABLE_QUERY_CACHE=true
CACHE_TTL_SECONDS=3600
ENABLE_SEMANTIC_CACHE=true
SEMANTIC_CACHE_THRESHOLD=0.75
```

### 3. Docker 部署（推荐）

```bash
# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f app

# 停止服务
docker-compose down
```

### 4. 本地开发

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 启动服务
python app/api/routes.py
```

## 📖 API 文档

### 同步聊天

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "高血压应该如何护理？",
    "user_id": "user_001"
  }'
```

**响应**：
```json
{
  "answer": "高血压患者应保持低盐饮食...",
  "sources": [
    {
      "source": "高血压护理指南.pdf",
      "file_path": "docs/medical/高血压护理指南.pdf",
      "content": "..."
    }
  ],
  "warnings": ["本