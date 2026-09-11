# 医疗助手智能问答系统

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.129.0-green)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0.10-orange)](https://langchain-ai.github.io/langgraph/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue)](https://www.docker.com/)

基于 RAG（检索增强生成）技术的医疗领域智能问答系统，采用三层上下文管理架构与混合检索流水线，支持多轮对话、知识检索与流式响应。

## 核心技术架构

### 三层上下文管理

针对医疗场景中"关键信息不可丢失"的刚性需求，设计了分层上下文管理策略，将信息按生命周期和重要性分为三层：

```
┌─────────────────────────────────────────────────────────────┐
│  L1 永久层（Persistent Context Anchoring）                    │
│  存储：PostgresStore（跨会话持久化）                           │
│  内容：用户档案 — 姓名、年龄、性别、过敏史、既往病史            │
│  特性：独立注入 System Prompt，永不参与压缩，杜绝"摘要的摘要"  │
│        导致过敏信息丢失                                       │
├─────────────────────────────────────────────────────────────┤
│  L2 会话层（Incremental State Checkpointing）                 │
│  存储：Checkpointer State（单会话，PostgreSQL）               │
│  内容：临床状态快照 — 主诉、症状时间线、用药记录、              │
│        高危症状、已确认/排除诊断                               │
│  特性：结构化 JSON 输出（ClinicalCheckpointOutput），          │
│        增量更新而非全量重建，滑动窗口触发时自动提取              │
├─────────────────────────────────────────────────────────────┤
│  L3 短期窗口（Sliding Window）                               │
│  存储：Messages（Checkpointer，滑动窗口）                     │
│  内容：最近 3 轮对话（6 条消息）                               │
│  特性：messages > SNAPSHOT_TRIGGER 时触发 L2 快照更新，       │
│        早期消息提取为快照后删除，后台异步执行不阻塞响应          │
└─────────────────────────────────────────────────────────────┘
```

**滑动窗口机制**：

```
第1-3轮: messages ≤ 6 → 不触发，全部保留
第4轮:   messages = 8 → 触发快照更新
         → LLM 从 H1-A2 提取临床快照（JSON）
         → 删除 H1,A1,H2,A2 → 剩余 4 条
第5轮:   messages = 6 → 不触发
第6轮:   messages = 8 → 再次触发
         → 增量更新快照（合并 H3-A4 的新信息）
         → 删除 H3,A3,H4,A4 → 剩余 4 条
```

**并发安全**：per-thread asyncio.Lock 保证同一会话不会并发执行快照更新；锁内重新读取最新状态并二次检查阈值，避免重复处理。

### 混合检索 RAG 流水线

```
用户问题
   │
   ├─ 规则路由（symptom / knowledge / general / vision）
   │
   ├─ 查询重写（关键词跳过 + 上下文感知重写）
   │
   └─ 混合检索
        │
        ├─ Dense 检索（向量相似度，ChromaDB）
        ├─ Sparse 检索（BM25 关键词匹配）
        │
        ├─ RRF 融合（Reciprocal Rank Fusion，α=0.5, k=60）
        │
        ├─ Reranker 重排序（bge-reranker-onnx，本地推理）
        │   ├─ sigmoid 归一化（logits → [0,1]）
        │   ├─ 阈值过滤（RERANKER_THRESHOLD）
        │   └─ top_k 截断
        │
        ├─ 启发式文档过滤（关键词重叠 + rerank 分数）
        │
        └─ LLM 生成（文档截断 300 字 + 来源元数据 SSE 推送）
```

**关键设计**：

| 环节 | 策略 | 目的 |
|------|------|------|
| Dense + Sparse | 双路召回 + RRF 融合 | 互补语义匹配与关键词匹配 |
| Reranker | ONNX 本地推理，sigmoid 归一化 | 纠正 RRF 融合的低质量排序 |
| 文档过滤 | 前2名无条件保留 + 关键词重叠 | 避免误过滤高质量文档 |
| 文档截断 | 每片 300 字 | 控制 prompt token，降低 TTFT |
| 语义上下文压缩 | RAG 文档原文存 Redis（TTL=2h），历史 AI 消息替换为 doc_id | 防止上下文被旧文献塞满 |

### 多级缓存体系

| 层级 | 类型 | 存储 | 命中条件 | TTL |
|------|------|------|----------|-----|
| L0 | 答案缓存 | Redis | 精确匹配（无历史对话时） | 30min |
| L2 | 语义缓存 | Redis + Embedding | 余弦相似度 ≥ 0.92 | 1h |

- 缓存 key 绑定 `kb_version` 与 `prompt_version`，知识库或 Prompt 变更后旧缓存自动失效
- 命中时跳过检索和 LLM 调用，直接返回缓存答案（有 thread 历史时跳过 L0/L2 复用，防跨会话串用）

### 记忆管理

| 类型 | 存储 | 生命周期 | 管理方式 |
|------|------|----------|----------|
| 短期记忆（Messages） | PostgreSQL Checkpointer | 单会话 | 滑动窗口，保留最近 3 轮 |
| 临床快照（Snapshot） | PostgreSQL Checkpointer | 单会话 | 增量更新，结构化 JSON |
| 用户档案（Profile） | PostgreSQL PostgresStore | 跨会话 | LLM 提取，永久保留 |
| RAG 文档缓存 | Redis | 2 小时 | doc_id 引用，按需恢复 |

## 其他特性

### 模型清单

| 模型 | 用途 | 部署方式 | 配置项 |
|------|------|----------|--------|
| glm-4-flash | RAG 答案生成、直接回答 | 云端 API（智谱） | `MODEL_NAME` |
| qwen2.5:1.5b | 查询重写、路由、症状解析、档案提取、快照更新、安全审查 | 本地部署（Ollama） | `LOCAL_MODEL_NAME` / `LOCAL_MODEL_ENABLED` |
| glm-4v-plus | 图片问诊 VLM 结构化提取 | 云端 API（智谱） | `VISION_MODEL_NAME` |
| embedding-3 | 文档向量化、语义缓存相似度计算 | 云端 API（智谱） | `EMBEDDING_MODEL` |
| bge-reranker-onnx | 检索结果重排序 | 本地 ONNX 推理 | `RERANKER_MODEL_PATH` |
| BM25 (rank-bm25) | 稀疏检索（关键词匹配） | 本地内存 | - |

**模型分工策略**：最终答案生成调用云端 API（保证质量），中间节点（重写/解析/提取/审查）调用本地 1.5B 模型（降低延迟与成本，`LOCAL_MODEL_ENABLED=False` 时自动降级回云端 API），Reranker 使用 ONNX 本地推理（避免 GPU 依赖）。

### 智能问答
- **流式响应**：SSE 实时推送，无需等待完整生成
- **智能路由**：规则优先 + LLM 降级的多级路由（symptom > knowledge > general）
- **查询重写**：轻量模型 + 关键词跳过，优化检索质量
- **问候直达**：简单问候/寒暄直接返回预设回复，零延迟

### 性能优化
- **规则优先路由**：避免误判，减少无效 LLM 调用
- **规则优先症状提取**：关键词匹配直接提取，跳过 LLM 调用
- **档案提取后置**：用户档案提取移至回答生成之后，不阻塞首 token
- **Redis 超时保护**：连接/读写超时 2 秒，故障自动降级
- **Embedding/LLM 超时保护**：request_timeout=10s, max_retries=1

### 持久化存储
- **PostgreSQL**：对话检查点 + 用户档案持久化
- **Redis**：查询缓存 + RAG 文档缓存（自动重连，故障降级为内存缓存）
- **ChromaDB**：向量数据库存储医疗文档

### 安全防护
- **CORS 限制**：生产环境通过 `CORS_ORIGINS` 配置允许的来源，`allow_origins=*` 时自动禁用 `credentials`
- **接口认证**：`X-Admin-API-Key` 认证覆盖全部管理面——缓存管理（`/api/cache/*`）、知识库管理（`/api/admin/kb/*`）、拒答日志（`/api/admin/refusal/*`）、评估指标（`/api/metrics/*`）；未配置密钥时仅允许本地访问
- **输入限制**：`question` 最大 1000 字符，`image_base64` 最大 10MB，知识库上传文件名净化防路径穿越
- **异常脱敏**：生产环境（`DEBUG=false`）全局异常处理器返回通用消息，不泄露内部实现细节
- **Redis 自动重连**：连接断开后每 30 秒尝试重连，恢复后自动切回 Redis，避免永久降级

### 🖼️ 图片问诊（VLM + OCR）
- 已通过聊天接口 `image_base64` 字段实现，同步（`/api/chat`）与流式（`/api/chat/stream`）均支持，上限 10MB
- 处理流程（`graph/nodes/nodes.py` 的 `vision_analysis_node`）：
  1. **VLM 结构化提取**：识别图片类型（检验单/报告/处方等）、置信度、可能方向与数值
  2. **OCR 数值校准**：数据类图片追加 PaddleOCR 提取结果，`_merge_ocr_into_vision` 覆盖 VLM 数值偏差
  3. **追问闭环**：识别不确定（`needs_followup`）或低置信度时反问用户，追问内容写入 checkpointer，下一轮上下文衔接
  4. **RAG 续查**：由图片信息构建检索查询，走正常检索/生成流水线
- 追问/低置信度/异常分支均安全收尾（`_vision_fallback_goto`），不会卡死图流程

### 🗂️ 知识库管理与零停机重建

基于版本化（`version_id`）+ 软删除（`status`）机制，支持增量更新与双集合原子切换：

| 能力 | 说明 |
|------|------|
| 增量更新双缓冲 | 上传后新 chunk 先入 `pending` → 校验 → 激活 `active` → 仅废弃旧版本中已删除的块；0 窗口期 |
| 版本管理 | 同一文档每次修改 `version_id+1`，`keep_hashes` 保留未变块，避免"改几行整篇丢失" |
| 零停机全量重建 | 影子集合构建 → 校验 → `kb_active.json` 指针原子切换 → 保留上一代+active 供回滚，延迟只清理更旧世代（Windows 句柄占用时带重试、失败留待重启清理） |
| 软删除 / 恢复 | `status=deprecated` 而非物理删除，支持 `restore` 恢复，BM25 索引同步过滤 |
| 回滚 | `/api/admin/kb/rollback` 秒级切回旧集合 |
| 一致性校验 | `run_reconciliation` 比对文档目录与向量库差异，`stale-detect` 探测过期数据 |
| 审计日志 | 每次上传/删除/重建/回滚写入审计记录，可查询 |

**管理接口**（全部需 `X-Admin-API-Key` 认证）：

| 端点 | 功能 |
|------|------|
| `/api/admin/kb/status` | 文档列表、向量库、版本、Embedding 一致性 |
| `/api/admin/kb/upload` | 上传文档（增量双缓冲 + 版本化） |
| `/api/admin/kb/documents/{filename}` | 删除文档（软删除） |
| `/api/admin/kb/restore/{filename}` | 恢复已删除文档 |
| `/api/admin/kb/rebuild` | 全量重建（零停机） |
| `/api/admin/kb/rollback` | 回滚到旧集合 |
| `/api/admin/kb/audit-log` | 审计日志查询 |
| `/api/admin/kb/reconcile` / `stale-detect` | 一致性校验 / 过期探测 |
| `/api/admin/kb/collection-info` | 集合信息 |

文件名经 `_sanitize_kb_filename` 净化（拒绝 `/`、`\`、盘符、`..`），防路径穿越。

### 🗂️ 知识库逻辑分类与物理分库（重要政策）

医疗知识库不能一股脑塞一个向量库：药品说明书、疾病诊疗、检查报告、护理指南、急救处置的
语义空间与权威源天然不同，混在一个 collection 检索会互相污染，也无法按来源做合规隔离。
但直接物理分库会让"意图/类型判错 → 零召回"的成本从"排序问题"升级成"干脆查不到"，且需
维护 N 套独立索引。因此**分两阶段落地**：

**阶段一·逻辑分类（当前，默认关闭）**
- 文档摄入时按文件名打 `category` 标签（`app/core/doc_category.py`，零 LLM、确定性）：
  `disease 疾病诊疗 / drug 药品说明 / report 检查报告 / nurse 护理指南 / emergency 急救处置 /
  guide 就医引导 / general 通用`。落库为 chunk 的 `category` 元数据。
- 检索时按意图做**软过滤**（`app/rag/hybrid_retriever.py`，`ENABLE_INTENT_KB_FILTER` 开关，
  默认 `False`）：`drug→drug 药库 / exam→report 检查报告 / symptom→disease 疾病库`。
- 软过滤**从不缩水、零命中兜底**：类别候选不足时保留原始全库候选，把"意图误判"的代价
  控制在排序以内，绝不退回零召回。`knowledge/general` 意图不过滤（全库）。
- **启用前提**：必须先重建知识库索引（`scripts/rebuild_vector_store.py`）让 `category` 落到
  向量库，并经黄金测试集（RAGAS）验证路由准确率后再开。

**阶段二·物理分库（硬性政策，务必记住）**
> ⚠️ 当**文档数达到 1000+ 篇 或 来源 > 3 类**时，**必须执行物理分库**，不得继续用一个库。

理由：文档多了之后，类别过滤只是把噪声推到堆外，检索的延迟与错检仍会随库内文档增长而劣化；
且单一 collection 无法做来源级合规隔离。分库后按 `collection`（`disease_kb / drug_kb /
report_kb / nurse_kb / emergency_kb / guide_kb / general_kb`）各建独立索引，检索路由层零改动——
物理库名已在 `app/core/doc_category.py` 的 `PHYSICAL_COLLECTION_NAMES` 预埋，届时直接按
category 拆 collection 即可。触发该政策时同步更新 `CHANGELOG.md` 并记录。

### 🛡️ 安全检查引擎

回答生成后由 `safety_check_node` 执行多层核查（`app/skills/`），不依赖单一 LLM 判断：

| 引擎 | 核查内容 |
|------|----------|
| `medication_guide_engine` | ① 剂量上限（统一 g/mg/μg → mg，累计每日总剂量）② 禁忌人群交叉（儿童/老年按年龄数值判定）③ 重复用药 ④ 药物相互作用初筛（对称去重）⑤ 用药建议 5 字段完整性（适应症/用法用量/注意事项/禁忌/如症状持续请就医） |
| `symptom_triage_engine` | 从临床快照提取症状，响应**危险症状组合**（如头痛+发热+颈僵→脑膜炎高风险）与 🟡 建议就诊信号，注入紧凑警告 |
| `safety_review_engine` | 紧急信号检测（回答含紧急症状未给就医指引时追加提示）+ LLM 深度审查 + 拒答（多数据加权置信度） |

- 触发风险标签后进入 LLM 深度审查路径；`ENABLE_SAFETY_CHECK` 控制开关，流式与非流式路径均已接线
- 拒答日志/统计接口（`/api/admin/refusal/*`）需管理员鉴权

### 🔁 Bad Case 利用与迭代闭环

本系统的核心竞争力之一——把每一次「用户差评」都转化为可消费、可回归、可驱动持续优化的数据资产，形成**采集 → 自动标注 → 人工审核 → 入黄金测试集 → 回归复测**的完整闭环，而非放任差评流失。

```
用户差评/负反馈
   │
   ├─ 自动采集（append_bad_case）
   │    └─ 按 user_id 分命名空间持久化，跨用户聚合查询
   │
   ├─ ① AI 预筛判真伪（scripts/auto_annotate_bad_cases.py）
   │    ├─ 判定"真 badcase" vs "误点/乱点"
   │    │     └─ 答案相关但正确+用户无具体原因 → 误点剔除，防垃圾进黄金集
   │    ├─ 根因归类（细粒度 case_type）
   │    └─ 期望回答草稿（防臆造剂量/禁忌，不确定标 [需人工核实]）
   │
   ├─ ② 失败大类归因（badcase_categories.py）
   │     检索失败 / 知识缺失 / 生成失败 / 其他
   │     └─ case_type 自动预选大类，管理后台可人工指定
   │
   ├─ ③ 管理后台人工审核（/admin/badcases）
   │     累计/待审核/类型分布统计 + 三大类筛选 + 弹窗定稿
   │
   └─ ④ 审核通过 → 自动并入黄金测试集（golden_test_set.jsonl）
        ├─ 仅「三大失败类 + 期望回答非空」才写入，含失败归因与来源 case_id
        ├─ 按 query 判重，防重复采集/重复入库
        └─ 回流 RAGAS 评估与 Bad Case 回归测试，精准复现并修复历史失败
```

**亮点**：

- **闭环不被中断**：差评自动落库 → AI 预筛 → 人工定稿 → 自动进黄金集，各节点显式 `reason` 透出，杜绝静默失败
- **低误杀原则**：判真伪时信息不足的一律按有效 `low` 置信度处理，绝不因存疑错杀真 badcase；期望回答草稿严格要求忠于问题、不臆造医疗事实，标注 `[需人工核实]`
- **失败归因视角**：检索失败 / 知识缺失 / 生成失败三大类定位「错在哪一环」，为定向优化（检索召回 / 知识库补档 / 生成约束）提供依据
- **黄金测试集自动生长**：人工审核沉淀的真实失败样本自动回流测试集，让评估集随系统使用持续扩大、更贴近真实用户问题
- **跨用户聚合**：管理后台汇总所有用户命名空间的 bad case，避免漏审

### 🧪 评估与迭代

- **RAGAS 四维指标**：Faithfulness / Answer Relevancy / Context Precision / Context Recall
- **版本化评估**：结果按版本归档，支持 A/B 对比
- **Bad Case 回归**：失败样本导出并回归复测，防止修复回退
- **黄金测试集**：Bad Case 审核通过的失败样本自动并入 `tests/data/golden_test_set.jsonl`，作为评估与回归的持续扩充数据源
- 接口：`/api/feedback`（反馈）、`/api/admin/badcases/*`（Bad Case 列表/统计/审核）、`/api/metrics/*`（节点/请求/token 指标，需鉴权）

### 🐳 容器化部署
- **Docker Compose**：一键启动所有服务
- **健康检查**：自动监控服务状态
- **数据持久化**：容器重启数据不丢失

## 系统架构

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
│                                                              │
│  ┌──────────┐   规则优先路由                                  │
│  │  Memory  │   symptom > knowledge > general               │
│  │  Load    │        │                                      │
│  │ (L1+L2)  │        ├─→ direct_answer (问候直达/LLM)        │
│  └──────────┘        ├─→ symptom_analysis (规则/LLM)        │
│                      └─→ query_rewrite → retrieve → generate│
│                              │                              │
│                    ┌─────────┴──────────┐                    │
│                    │  update_snapshot   │                    │
│                    │  (L2 快照更新)      │                    │
│                    │  后台异步执行       │                    │
│                    └────────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      RAG 检索层                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Vector Store│  │   BM25      │  │    Reranker         │  │
│  │  (ChromaDB) │  │ (稀疏检索)   │  │  (bge-reranker-onnx)│  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│         ↓ RRF 融合 → sigmoid 归一化 → 阈值过滤 → 缓存写入    │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    上下文管理层                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ L1 Profile  │  │ L2 Snapshot │  │ L3 Messages         │  │
│  │ (Postgres   │  │ (Checkpointer│  │ (滑动窗口            │  │
│  │  Store)     │  │  State)     │  │  最近3轮)            │  │
│  │ 跨会话持久   │  │ 单会话JSON  │  │ 后台异步截断         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      存储层                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  PostgreSQL │  │    Redis    │  │    ChromaDB         │  │
│  │ (Checkpoint │  │ (L0+L2 Cache│  │   (Vector Store)    │  │
│  │  + Profile) │  │  + Doc Cache)│  │                     │  │
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

# 查询重写专用模型（可选，留空则使用 MODEL_NAME）
REWRITE_MODEL_NAME=

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
SEMANTIC_CACHE_THRESHOLD=0.92

# Reranker 配置
RERANKER_THRESHOLD=0.0
RERANKER_MODEL_PATH=/app/models/bge-reranker-onnx
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

### 图片问诊（多模态）

通过聊天接口传 `image_base64` 触发 VLM 分析，无需独立上传端点：

```bash
curl -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "question": "这份检查报告有什么问题？",
    "user_id": "user_001",
    "image_base64": "<图片base64编码>"
  }'
```

**流程**：VLM 结构化提取 → 数据类图片追加 OCR 数值校准 → 不确定时追问用户 → 构建 RAG 查询继续检索生成。

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
| `MODEL_NAME` | LLM 模型名称 | glm-4-flash |
| `MODEL_URL` | LLM API 地址 | - |
| `MODEL_API_KEY` | LLM API 密钥 | - |
| `REWRITE_MODEL_NAME` | 查询重写专用模型 | 空（使用 MODEL_NAME） |
| `EMBEDDING_MODEL` | Embedding 模型 | embedding-3 |
| `RERANKER_THRESHOLD` | Reranker 阈值 | 0.1 |
| `RERANKER_MODEL_PATH` | Reranker 模型路径 | /app/models/bge-reranker-onnx |
| `ENABLE_SEMANTIC_CACHE` | 启用语义缓存 | true |
| `SEMANTIC_CACHE_THRESHOLD` | 语义相似度阈值 | 0.92 |
| `REDIS_URL` | Redis 连接地址 | redis://localhost:6379/0 |
| `CORS_ORIGINS` | 允许的跨域来源（逗号分隔） | 空（允许所有） |
| `ADMIN_API_KEY` | 缓存管理接口认证密钥 | admin-api-key-change-in-production |
| `DEBUG` | 调试模式（影响异常信息详细程度） | false |

### 路径配置

| 路径 | 说明 |
|------|------|
| `docs/medical/` | 医疗文档目录 |
| `data/chroma_db/` | 向量数据库 |
| `data/uploads/` | 上传图片存储 |
| `logs/` | 日志文件 |

## 📊 性能指标

### 三种典型请求场景

| 场景 | 请求示例 | 处理路径 | 优化前 | 优化后 | 优化措施 |
|------|----------|----------|--------|--------|----------|
| 简单问候 | "你好" | 问候直达 | ~8s | **<100ms** | 预设回复，跳过 LLM |
| 自我介绍 | "你好，我是王艺涵" | direct_answer + 档案后置 | ~18s | **~3s** | 问候检测 + 档案提取后置 |
| 症状咨询 | "我是王艺涵，我有芒果过敏" | symptom(规则) → RAG | ~49s | **~15s** | 规则提取 + Reranker跳过 + 重写跳过 |

### 响应时间

| 场景 | 优化前 | 优化后 | 优化措施 |
|------|--------|--------|----------|
| 缓存命中 | ~8秒 | ~3秒 | Embedding 复用、查询重写跳过 |
| 缓存未命中 | ~21秒 | ~11秒 | Jieba 预加载、条件化 LLM 调用 |
| 首次响应 | ~3秒 | ~1.5秒 | Reranker 预加载、异步处理 |

### 缓存命中率

| 缓存层级 | 命中率 | 作用 |
|----------|--------|------|
| L0 (答案缓存) | ~30% | 完全相同的查询直接返回答案（1ms） |
| L2 (语义缓存) | ~20% | 语义相似的查询返回缓存文档 |
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

**A**: ONNX Reranker 分数范围与原生模型不同，阈值应设为 0.0：
```python
# app/core/config.py
RERANKER_THRESHOLD = 0.0
```

### Q: Redis 连接超时导致响应慢？

**A**: 已添加连接超时保护（2秒），Redis 不可用时自动降级为内存缓存。

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

### Q: 如何配置查询重写专用模型？

**A**: 在 `.env` 中设置 `REWRITE_MODEL_NAME`，留空则使用主模型：
```bash
# 使用更轻量的模型加速查询重写
REWRITE_MODEL_NAME=glm-4-flash
```

## 📁 项目结构

```
medical_assistant_agent/
├── app/
│   ├── api/              # API 路由（聊天、KB 管理、metrics、admin 鉴权）
│   ├── cache/            # 缓存模块（Redis 答案缓存、语义缓存）
│   ├── core/             # 核心配置（LLM、Embedding、限流、熔断、指标、自适应阈值）
│   ├── evaluation/       # 评估（RAGAS 四维指标、Bad Case 回归）
│   ├── graph/            # LangGraph 工作流（nodes、streaming、state、prompts，含 vision 节点）
│   ├── memory/           # 记忆管理（PostgreSQL checkpointer、长期记忆、fallback 缓冲）
│   ├── models/           # Pydantic 数据模型
│   ├── rag/              # RAG 检索（向量库、BM25、Reranker、KB 版本更新、元数据提取）
│   ├── skills/           # 三大安全引擎（用药指南、症状分诊、安全审查）
│   └── static/           # 静态文件
├── data/                 # 数据存储（chroma_db、uploads、metrics）
├── docs/medical/         # 医疗文档
├── scripts/              # 工具脚本（重建/评估/验证/审计脚本）
├── tests/                # 测试用例
├── docker-compose.yml    # Docker 编排
├── Dockerfile            # 容器镜像
└── requirements.txt      # Python 依赖
```

## 路线图

- [x] 基础 RAG 问答
- [x] 多轮对话支持
- [x] 流式响应（SSE）
- [x] 混合检索（Dense + Sparse + RRF 融合）
- [x] Reranker 重排序（bge-reranker-onnx + sigmoid 归一化）
- [x] 三层上下文管理架构（L1 永久层 + L2 会话层 + L3 短期窗口）
- [x] 不可变上下文锚定（Profile 跨会话持久化）
- [x] 增量状态检查点（Clinical Snapshot 增量更新）
- [x] 滑动窗口消息管理（后台异步 + per-thread 锁并发安全）
- [x] 语义上下文压缩（RAG 文档 doc_id 引用 + Redis 缓存）
- [x] 多级缓存（L0 答案缓存 + L2 语义缓存）
- [x] Docker 容器化
- [x] 规则优先路由 + 症状提取
- [x] 查询重写轻量化
- [x] 文档来源 SSE 元数据推送
- [x] MinerU PDF 解析
- [x] 图片问诊（聊天接口 `image_base64`：VLM 提取 + OCR 数值校准 + 追问闭环）
- [x] RAGAS 自动评估（四维指标 + 版本化 A/B 对比 + Bad Case 回归）
- [x] Bad Case 自动采集（用户差评按用户命名空间落库 + 跨用户聚合）
- [x] Bad Case AI 自动标注（判真伪 + 根因归类 + 期望回答草稿，低误杀原则）
- [x] Bad Case 失败大类归因（检索失败/知识缺失/生成失败）
- [x] Bad Case 管理后台（统计/筛选/人工审核）与审核通过自动并入黄金测试集
- [x] 并行检索架构（多子问题 ThreadPoolExecutor 并行检索）
- [x] 知识库零停机重建（影子集合 → 校验 → 原子切换 → 延迟清理）
- [x] 知识库增量更新双缓冲 + 版本化（0 窗口期、软删除、回滚）
- [x] 知识库管理 API（上传/删除/恢复/重建/回滚/审计日志/一致性校验）
- [x] 安全检查引擎（剂量上限/禁忌人群/相互作用/5 字段完整性/症状分诊）
- [x] 拒答机制（多数据加权置信度 + LLM 深度审查）
- [x] 查询/答案缓存版本化（kb_version + prompt_version 绑定，KB 更新自动失效）
- [ ] 全节点异步化
- [ ] 多语言支持

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
