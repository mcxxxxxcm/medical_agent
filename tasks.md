# 医疗助手智能问答系统 — 任务清单

## 元信息

- 关联规范：README.md, docs/architecture_topology.html
- 任务总数：10
- 预计总执行时间：8-12 小时（含测试）
- 执行策略：按依赖关系顺序执行，独立任务可并行

---

## Task 1: 创建项目结构和基础配置

**描述**：初始化项目目录结构，创建统一配置层（.env + app/core/config.py），安装核心依赖，配置测试环境。

**输入**：README.md（模块划分）, requirements.txt（依赖清单）

**输出**：项目根目录结构 + .env + app/core/config.py + requirements.txt + pytest.ini

**依赖**：无

**验收标准**：
- [ ] 目录结构符合 README 的模块划分（app/, scripts/, tests/, docs/, data/, logs/）
- [ ] requirements.txt 包含 fastapi, langgraph, langchain-core, chromadb, redis, rank-bm25, httpx, pydantic-settings
- [ ] config.py 基于 pydantic-settings 读取 .env，配置缺失必填字段时抛出明确错误
- [ ] settings 实例可全局获取，热更新 reload_config() 后清空 LLM 缓存
- [ ] `pytest` 在 conda env my_medical_env 下可正常收集用例

---

## Task 2: 实现 LLM 与 Embedding 基础层

**描述**：实现主模型、本地模型、视觉模型、向量化与 token 计数的统一封装。

**输入**：README.md（模型清单）; .env（MODEL_NAME / EMBEDDING_MODEL 等）

**输出**：app/core/llm.py, app/core/embeddings.py, app/core/token_tracker.py, app/core/exceptions.py + 对应单元测试

**依赖**：Task 1

**验收标准**：
- [ ] llm.py 封装 glm-4-flash（主模型）与 qwen2.5:1.5b（本地 Ollama），可配置 LOCAL_MODEL_ENABLED 开关
- [ ] 主模型目标首 token 1-3s，备选模型切换只改配置不改代码
- [ ] embeddings.py 调用 embedding-3（维度 2048），语义缓存与向量库复用同一实例
- [ ] token_tracker.py 能统计每次调用的 token 消耗并记录指标
- [ ] 所有 LLM 调用带超时与重试，失败抛出异常并被上游捕获
- [ ] 使用 mock 请求的单元测试全部通过

---

## Task 3: 实现 RAG 检索流水线（混合检索）

**描述**：实现 Dense（ChromaDB）+ Sparse（BM25）双路召回、RRF 融合、Reranker 重排序、启发式文档过滤的完整检索链。

**输入**：README.md §混合检索 RAG 流水线; config.py（DEFAULT_K / RERANKER_THRESHOLD 等）

**输出**：app/rag/hybrid_retriever.py, app/rag/vector_store.py, app/rag/reranker.py + 单元测试

**依赖**：Task 2

**验收标准**：
- [ ] Dense 检索基于 Embedding 余弦相似度查询 ChromaDB，Sparse 检索基于 BM25 关键词匹配
- [ ] RRF 融合正确（α=0.5, k=60），双路结果合并去重
- [ ] Reranker 用 bge-reranker-onnx 本地推理，sigmoid 归一化到 [0,1] 并按 RERANKER_THRESHOLD 过滤
- [ ] 文档过滤保证前 2 名无条件保留，避免误过滤高质量文档
- [ ] 检索最终送入 LLM 的文档按 ANSWER_CONTEXT_CHAR_BUDGET 截断（默认 15000 字符）
- [ ] 使用 mock 向量库的单元测试全部通过

---

## Task 4: 实现三层上下文管理与记忆

**描述**：实现 L1 永久层（用户档案）、L2 会话层（临床快照）、L3 短期窗口（滑动窗口）及增量快照更新机制。

**输入**：README.md §三层上下文管理; config.py（SNAPSHOT_TRIGGER / KEEP_RECENT_MESSAGES 等）

**输出**：app/graph/state.py, app/memory/checkpointer.py, app/memory/long_term_memory.py, app/memory/fallback_buffer.py + 单元测试

**依赖**：Task 2

**验收标准**：
- [ ] L1 档案存 PostgresStore，独立注入 System Prompt，永不参与压缩
- [ ] L2 快照为结构化 JSON（ClinicalCheckpointOutput），增量更新而非全量重建
- [ ] messages 数量达 SNAPSHOT_TRIGGER（8）时触发快照提取，滑动窗口后保留最近 6 条
- [ ] 快照更新在后台异步执行，不阻塞响应；per-thread asyncio.Lock 保证并发安全
- [ ] 锁内重新读取最新状态并二次检查阈值，避免重复处理
- [ ] 记忆复用场景的单元测试全部通过

---

## Task 5: 实现 LangGraph 工作流与规则路由

**描述**：实现对话主流程（LangGraph 图）、规则路由（symptom/knowledge/general/vision）与查询重写、结构化输出。

**输入**：README.md 架构说明; state.py（状态 schema）

**输出**：app/graph/graph.py, app/graph/nodes/nodes.py, app/graph/nodes/prompts.py, app/graph/nodes/structured_output.py, app/core/keyword_matcher.py + 单元测试

**依赖**：Task 3, Task 4

**验收标准**：
- [ ] graph 支持多轮对话状态累积（Reducer）与 Command.goto 路由，不污染 State
- [ ] 规则路由按问题类型分发，MERGE_ROUTER_REWRITE 开启时合并 router 与 query_rewrite
- [ ] 查询重写支持关键词跳过与上下文感知重写（本地模型执行）
- [ ] 生成节点执行 RAG 文档去重 + 相关度排序 + 字符预算压缩注入
- [ ] 幻觉检测 / grade_documents 节点对无关文档降级兜底
- [ ] 使用 mock 的图遍历单元测试全部通过

---

## Task 6: 实现技能引擎与工具注册

**描述**：实现症状分诊、用药指导、用药安全审查三个技能引擎，以及统一工具注册表与审计日志。

**输入**：app/skills/ 业务规则; config.py（ENABLE_TOOL_AUDIT / ENABLE_RETRIEVAL_TOOL）

**输出**：app/skills/symptom_triage_engine.py, app/skills/medication_guide_engine.py, app/skills/safety_review_engine.py, app/tools/registry.py + 单元测试

**依赖**：Task 2

**验收标准**：
- [ ] 症状分诊技能正确解析主诉、输出分诊等级与就医建议
- [ ] 用药指导技能按知识库文档返回剂量、频次、注意事项
- [ ] 安全审查技能对处方/用药执行禁忌与相互作用核查（ENABLE_SAFETY_CHECK 控制是否调用）
- [ ] 工具注册表统一暴露技能与检索工具入口，审计日志记录每次调用
- [ ] 检索工具默认关闭（ENABLE_RETRIEVAL_TOOL=False），防 LLM 绕过确定性路由
- [ ] 技能与注册表单元测试全部通过

---

## Task 7: 实现多级缓存体系

**描述**：实现 L0 答案缓存与 L2 语义缓存，以及带 kb_version / prompt_version 的缓存失效策略。

**输入**：README.md §多级缓存体系; config.py（ENABLE_SEMANTIC_CACHE / SEMANTIC_CACHE_THRESHOLD 等）

**输出**：app/cache/redis_cache.py, app/cache/semantic_cache.py + 单元测试

**依赖**：Task 3

**验收标准**：
- [ ] L0 精确匹配缓存（无历史对话时）TTL 30min
- [ ] L2 语义缓存按余弦相似度 ≥ 0.90 命中，TTL 1h
- [ ] 缓存 key 绑定 kb_version 与 prompt_version，知识库或 Prompt 变更后自动失效
- [ ] 命中缓存时跳过检索与 LLM 调用，直接返回，但检索结果缓存命中时答案仍重新生成
- [ ] 有 thread 历史时跳过 L0/L2 复用，防跨会话串用
- [ ] 检索/过滤类改动评测需以 ENABLE_SEMANTIC_CACHE=false 验证（防 L2 缓存掩盖改动效果）

---

## Task 8: 实现 API 层与流式响应

**描述**：实现 FastAPI 路由、鉴权、限流、SSE 流式响应、健康检查与配置热更新接口。

**输入**：app/models/schemas.py; config.py（RATE_LIMIT_PER_MINUTE / CORS_ORIGINS 等）

**输出**：app/api/routes.py, app/api/check.py, app/graph/streaming.py + 接口测试

**依赖**：Task 5, Task 6

**验收标准**：
- [ ] `/api/chat` 接受多轮对话输入并返回最终答案 + 警告 + 来源（sources）
- [ ] 流式响应按 SSE 推送文档截断（300 字）与来源元数据，首个 token 快速返回
- [ ] 每分钟请求数受限流保护（RATE_LIMIT_PER_MINUTE=20）
- [ ] 图片问诊走 VLM（glm-4v-plus）结构化提取后进入主流程
- [ ] 健康检查 / 管理接口受 ADMIN_API_KEY 保护（如缓存清理）
- [ ] 配置热更新接口 reload_config 生效且清空 LLM 缓存

---

## Task 9: 实现评测与知识库管理

**描述**：实现黄金测试集生成、RAG 检索评测、Badcase 归类回归、知识库构建与零停机重建工具。

**输入**：README.md 评测方法; docs/medical/ 知识库文档

**输出**：scripts/ 下评测与构建脚本 + 知识库构建工具

**依赖**：Task 3

**验收标准**：
- [ ] `generate_golden_test_set.py` 生成黄金测试集，key_facts 语义判定而非字面子串匹配
- [ ] `evaluate_rag.py` 产出检索召回率 / 答案正确性指标，evaluate_golden_match.py 做 LLM 语义复审
- [ ] `run_bad_case_regression.py` 对 badcase 归类并回归，防止修复回档
- [ ] `rebuild_vector_store.py` 支持零停机重建向量库，双副本热切换
- [ ] `verify_e2e.py` 跑通端到端查询验证检索与回答
- [ ] 逻辑分类过滤开关默认关闭，文档达 1000+ 篇或来源>3 类时必须物理分库

---

## Task 10: 集成测试与端到端验证

**描述**：编写端到端集成测试，验证多轮对话、流式响应、记忆持久化与降级场景的完整行为。

**输入**：全部模块 + scripts/verify_*.py

**输出**：tests/test_nodes.py, tests/test_intent_clarify.py, tests/test_memory_reuse.py, tests/test_hybrid_backfill.py 等

**依赖**：Task 8, Task 9

**验收标准**：
- [ ] 正常场景：多轮对话经路由→检索→生成→流式返回，答案带来源
- [ ] 记忆场景：跨轮对话后临床快照增量更新，用户档案跨会话保留
- [ ] 降级场景：Reranker 全部过滤时触发兜底，检索为空时 LLM 直接回答而非报错
- [ ] 并发场景：同一会话快照更新并发安全，不重复处理
- [ ] 缓存场景：关语义缓存时检索/过滤改动可被评测正确观察到
- [ ] 全部用例在 conda env my_medical_env 下通过

---

## 执行顺序

推荐执行顺序（考虑依赖关系和并行机会）：

| 阶段 | 任务 | 并行度 | 说明 |
|------|------|--------|------|
| 1 | T1 | 1 | 项目初始化与配置层 |
| 2 | T2 | 1 | LLM/Embedding 基础层（所有模块前置依赖） |
| 3 | T3 + T4 + T6 | 3 | 检索流水线 / 上下文记忆 / 技能引擎可并行 |
| 4 | T5 + T7 | 2 | LangGraph 工作流 / 多级缓存（依赖 T3/T4） |
| 5 | T8 | 1 | API 层与流式响应（依赖 T5/T6） |
| 6 | T9 | 1 | 评测与知识库管理（依赖 T3） |
| 7 | T10 | 1 | 集成测试与端到端验证（依赖 T8/T9） |

最大并行度：3（阶段 3）

总阶段数：7（利用并行，10 个任务压缩为 7 个阶段）

---