# 医疗助手智能问答系统

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.129.0-green)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0.10-orange)](https://langchain-ai.github.io/langgraph/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue)](https://www.docker.com/)

基于 RAG（检索增强生成）技术的医疗领域智能问答系统，支持多轮对话、知识检索与流式响应等能力。

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
- **两级缓存**：L1 精确匹配、L2 语义缓存

### 💾 持久化存储
- **PostgreSQL**：对话检查点持久化，支持断点续聊
- **Redis**：查询结果缓存，加速响应
- **ChromaDB**：向量数据库存储医疗文档

### 🖼️ 图片识别（规划中）
- 当前主服务 `app/api/routes.py` 未暴露图片分析接口
- 如需保留该能力，建议以独立模块或实验性接口形式补充并在文档中单独标注

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
│  ┌─────────────┐  ┌─────────────┐                           │
│  │ /api/chat   │  │/api/chat/   │                           │
│  │   同步聊天   │  │   stream    │                           │
│  └─────────────┘  └─────────────┘                           │
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

# 启动服务（开发模式）
uvicorn app.api.routes:app --host 0.0.0.0 --port 8000 --reload
```

### 5. 生产启动建议

```bash
uvicorn app.api.routes:app --host 0.0.0.0 --port 8000 --workers 2
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
  "warnings": ["本回答仅供参考，不能替代专业医生的诊断和治疗建议"]
}
```

### 流式聊天

```bash
curl -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "question": "糖尿病的早期症状有哪些？",
    "user_id": "user_001"
  }'
```

**响应**：SSE 流式输出
```
data: "糖尿病的早期症状包括..."
data: "多饮、多尿、多食..."
data: [DONE]
```

### 图片分析

```bash
curl -X POST http://localhost:8000/api/upload/analyze \
  -F "file=@report.jpg" \
  -F "question=这份报告有什么问题？" \
  -F "use_ocr=true"
```

**响应**：
```json
{
  "ocr_text": "血常规检查...",
  "analysis": "根据报告分析...",
  "suggestions": ["建议复查..."],
  "warnings": ["本分析仅供参考..."]
}
```

### 健康检查

```bash
curl http://localhost:8000/api/health
```

**响应示例**：
```json
{
  "status": "healthy",
  "database": "healthy",
  "vector_store": "healthy",
  "cache": "healthy",
  "reranker": "healthy"
}
```

## 🧪 评估测试

### RAGAS 评估

```bash
# 使用默认测试集
python scripts/evaluate_rag.py

# 使用自定义测试数据
python scripts/evaluate_rag.py \
  --test-data data/evaluation/test_data.json \
  --metrics faithfulness,answer_correctness
```

### 性能测试

```bash
# 测试检索性能
python scripts/test_vector_store.py

# 测试 LLM 连接
python scripts/test_llm.py
```

## ⚙️ 配置说明

### 核心配置项

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `MODEL_NAME` | LLM 模型名称 | gpt-4o |
| `MODEL_URL` | LLM API 地址 | - |
| `MODEL_API_KEY` | LLM API 密钥 | - |
| `EMBEDDING_MODEL` | Embedding 模型 | text-embedding-3-small |
| `RERANKER_THRESHOLD` | Reranker 阈值 | 0.0 |
| `ENABLE_SEMANTIC_CACHE` | 启用语义缓存 | true |
| `SEMANTIC_CACHE_THRESHOLD` | 语义相似度阈值 | 0.75 |

### 路径配置

| 路径 | 说明 |
|------|------|
| `docs/medical/` | 医疗文档目录 |
| `data/chroma_db/` | 向量数据库 |
| `data/uploads/` | 上传图片存储 |
| `logs/` | 日志文件 |

## 📊 性能指标

### 响应时间

| 场景 | 优化前 | 优化后 | 优化措施 |
|------|--------|--------|----------|
| 缓存命中 | ~8秒 | ~3秒 | Embedding 复用、查询重写跳过 |
| 缓存未命中 | ~21秒 | ~11秒 | Jieba 预加载、条件化 LLM 调用 |
| 首次响应 | ~3秒 | ~1.5秒 | Reranker 预加载、异步处理 |

### 缓存命中率

| 缓存层级 | 命中率 | 作用 |
|----------|--------|------|
| L1 (Redis) | ~30% | 完全相同的查询 |
| L2 (Semantic) | ~20% | 语义相似的查询 |
| 合计 | ~50% | - |

## 🔧 常见问题

### Q: Docker 启动失败？

**A**: 检查端口占用：
```bash
# 检查端口
netstat -ano | findstr :8000
netstat -ano | findstr :5432
netstat -ano | findstr :6379

# 清理旧容器
docker-compose down -v
docker-compose up -d
```

### Q: Reranker 返回空结果？

**A**: 降低阈值：
```python
# app/core/config.py
RERANKER_THRESHOLD = 0.0  # 从 0.3 改为 0.0
```

### Q: PostgreSQL 连接断开？

**A**: 添加连接池配置：
```python
# 在连接字符串中添加
DATABASE_URL=postgresql://...?pool_size=5&max_overflow=10&pool_recycle=1800
```

### Q: 如何添加新的医疗文档？

**A**: 
```bash
# 1. 放入文档目录
cp new_document.pdf docs/medical/

# 2. 重建向量库
python scripts/rebuild_vector_store.py
```

## 📁 项目结构

```
medical_assistant_agent/
├── app/
│   ├── api/              # API 路由
│   ├── cache/            # 缓存模块（Redis、语义缓存）
│   ├── core/             # 核心配置（LLM、Embedding）
│   ├── graph/            # LangGraph 工作流
│   ├── memory/           # 记忆管理（PostgreSQL）
│   ├── rag/              # RAG 检索（向量库、BM25、Reranker）
│   ├── vision/           # 视觉识别（OCR、图片分析）
│   └── static/           # 静态文件
├── data/                 # 数据存储
├── docs/medical/         # 医疗文档
├── scripts/              # 工具脚本
├── tests/                # 测试用例
├── docker-compose.yml    # Docker 编排
├── Dockerfile            # 容器镜像
└── requirements.txt      # Python 依赖
```

## 🛣️ 路线图

- [x] 基础 RAG 问答
- [x] 多轮对话支持
- [x] 流式响应
- [x] 混合检索（Dense + Sparse）
- [x] Reranker 重排序
- [x] 三层缓存架构
- [x] Docker 容器化
- [ ] 图片识别功能（主服务未开放）
- [ ] RAGAS 自动评估
- [ ] 多语言支持
- [ ] 语音输入输出
- [ ] 移动端适配

## 🤝 贡献指南

1. Fork 项目
2. 创建分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📄 许可证

[MIT License](LICENSE)

## 🙏 致谢

- [LangChain](https://github.com/langchain-ai/langchain) - LLM 应用框架
- [LangGraph](https://github.com/langchain-ai/langgraph) - 工作流编排
- [FastAPI](https://github.com/tiangolo/fastapi) - Web 框架
- [ChromaDB](https://github.com/chroma-core/chroma) - 向量数据库
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - OCR 引擎

---

**⚠️ 免责声明**：本系统提供的医疗建议仅供参考，不能替代专业医生的诊断和治疗。如有健康问题，请及时就医。
