# 系统优化更新日志

## v9.16 - 性能优化：检索提速 + 准确率提升

### 响应速率优化

**P1: Embedding缓存统一**（预计省 200~600ms）
- 语义缓存和 HybridRetriever 的 embedding LRU 缓存双向同步
- 避免同一 query 在两套缓存中各存一份、重复计算
- 优点：零成本提速，无精度损失
- 缺点：两套缓存的淘汰策略不同（LRU vs 无淘汰），极端情况下缓存一致性可能漂移

**P4: Reranker预热**（首次请求省 ~4s）
- 应用启动时触发 Reranker 首次推理，避免冷启动
- 优点：消除首请求冷启动延迟
- 缺点：启动时间增加 ~4s；如果 Reranker 不常用则浪费资源

### 准确率优化

**P2: Reranker阈值校准 + 动态K值**
- Reranker 阈值从 0.02 调整为 0.005（原阈值过高导致合理文档被过滤后降级兜底）
- symptom 类型检索 K=8（多跳推理需更多候选），knowledge 类型 K=5
- 优点：减少误过滤，多跳问题召回率提升
- 缺点：K 值增大后 Reranker 输入增多，单次推理延迟略增（~100ms）；阈值降低可能引入更多噪声文档

**P3: 口语化同义词扩充**（+35条映射）
- 覆盖消化/呼吸/神经/妇科/皮肤/全身/用药 7 大类口语化表述
- "拉肚子→腹泻"、"嗓子疼→咽痛"、"退烧药→解热镇痛药"等
- 优点：口语化查询的检索召回率显著提升
- 缺点：替换可能过度（如"痒"替换为"瘙痒"可能改变上下文语义）；需持续维护映射表

**P5: 生成时引用标注**
- RAG 生成 Prompt 要求在关键事实后标注来源文档名
- 优点：答案可溯源，用户可验证信息来源
- 缺点：LLM 可能标注不准确（来源名与实际不符）；增加输出长度

## v9.15 - 三层拒答机制：敢于拒答比强行回答更重要

### 核心改进：检索前-检索中-生成后的三层防御体系

**检索层**：时效过滤 + 权威加权 + 版本去重 + 置信度评分
- 过期文档直接剔除（`doc_expire_date < today` → modifier=0）
- 文档时效衰减：>3年0.6、2-3年0.75、1-2年0.85、<1年1.0
- 权威等级加权：国家指南1.0 > 协会共识0.95 > 教材0.90 > FAQ0.55
- 同源多版本自动去重，保留最新版
- 四维置信度 × 时效衰减 × 权威权重 → 分级路由

**生成层**：分级拒答策略
- `confidence ≥ 0.7` → 正常RAG生成
- `0.4 ≤ confidence < 0.7` → 部分回答 + 声明缺失项
- `confidence < 0.4` → 拒答 + 引导补充信息
- symptom/knowledge类型 + 知识库无覆盖 → 拒答（不再走direct_answer）

**拒答日志**：SQLite持久化（refusal_logs表 + v_refusal_daily视图）
- 高频拒答聚类 → 定向补充知识库
- 误拒答回收 → 校准阈值参数
- 与node_metrics通过request_id关联，可追溯完整链路

## v9.14 - 文档元数据自动提取：四源交叉校验

### 核心改进：入库时自动提取版本/日期/权威等级等元数据，支持后续文档冲突裁决

**问题**：
- 文档入库时缺少版本号、生效日期、权威等级等元数据
- 检索到多份文档时无法做版本冲突裁决（如新旧版指南矛盾）
- 无法区分文档的适用范围（人群/地区/医学体系）

**方案**：四源提取 + 交叉校验引擎

```
入库时自动执行：

源1: 文件名解析
    命名规范：{文档名}_v{版本号}_{日期}_{权威等级}.{ext}
    示例：发热诊断指南_v2_20250301_national.pdf
    → 提取 version / effective_date / authority_level
    → 最可靠（人工命名，有意识填写）

源2: PDF/DOCX内嵌属性
    PDF: fitz.metadata → creationDate / modDate / author / title
    DOCX: core_properties → created / modified / author / title
    → 部分可靠（可能为空或默认值）

源3: LLM正文提取（仅当源1/2不足时触发）
    取文档前500字 → 3B模型结构化提取
    → 兜底方案（可能遗漏但不会编造）

源4: 文件系统时间
    mtime / ctime → 低置信度回退
    → 最不可靠（可能是复制/下载时间）

交叉校验：
    全部一致 → confidence=high（自动确认）
    多数一致 → confidence=mid（取多数值，标记待人工确认）
    来源冲突 → confidence=mid（取优先级最高的来源，标记待确认）
    仅单一来源 → confidence=low
    全部缺失 → confidence=none

写入规则：
    confidence=high/mid → 写入 doc_version / doc_effective_date 等
    confidence=low → 写入 doc_{field}_pending（不参与检索过滤）
    needs_manual_review=True → 标记待管理员审核
```

### 新增文件
- `app/rag/metadata_extractor.py`：四源提取 + 交叉校验引擎

### 修改文件
- `app/rag/loader.py`：`add_metadata()` 新增 `extract_doc_meta` 参数，自动调用提取
- `app/api/routes.py`：上传接口响应中附加 `meta_report`（含置信度和待审核字段）

### 元数据字段体系
| 字段 | 用途 | 来源 |
|------|------|------|
| doc_version | 版本冲突裁决 | 文件名/PDF属性/LLM |
| doc_effective_date | 时效性校验 | 文件名/PDF属性/LLM |
| doc_authority_level | 权威优先级 | 文件名/PDF属性/LLM |
| doc_issuing_body | 发布机构 | PDF属性/LLM |
| doc_medical_system | 医学体系区分 | 文件名/LLM |
| doc_applicable_population | 适用人群 | 文件名/LLM |
| doc_expire_date | 自动计算（生效+3年） | 派生 |
| doc_meta_confidence | 整体置信度 | 交叉校验 |
| doc_needs_meta_review | 需人工审核 | 交叉校验 |

## v9.13 - 图片问诊方案C：VLM结构化提取 + OCR校准 + RAG生成

### 核心改进：图片问诊从"VLM直接回答"升级为"VLM提取→OCR校准→RAG生成"

**问题**：
- VLM对数字识别准确率低（如"115g/L"可能识别为"118g/L"）
- VLM纯生成回答无可溯源依据，幻觉风险高
- 图片问诊不走RAG，无法引用知识库文档佐证

**方案C流程**：
```
图片 + 问题
    ↓
Step 1: VLM结构化提取（强制输出JSON）
    → image_type / objective_description / extracted_data / possible_directions / confidence / needs_followup
    ↓
Step 2: OCR校准（仅数据类图片：报告/处方/药盒）
    → PaddleOCR精确提取数值 → 覆盖VLM猜测 → 标记ocr_verified/vlm_only
    ↓
Step 3: 不确定性处理
    → confidence=low 或 needs_followup=True → 追问用户（直接返回）
    ↓
Step 4: 构造RAG查询 → 路由到 knowledge_retrieval
    → VLM不再是"回答者"，而是"信息提取器"
    → 医学回答由RAG管线基于知识库文档生成
```

### 安全机制

1. **VLM输出标准化**：`VisionAnalysisOutput` 强制7个字段，Pydantic校验
2. **禁止诊断**：Prompt约束"只提取信息，不做诊断"
3. **OCR优先**：数据类图片数值以OCR为准，VLM仅语义理解
4. **不确定性追问**：`needs_followup=True` → 直接追问用户
5. **低置信度拦截**：`confidence=low` → 建议重拍或文字描述

### 改动文件

| 文件 | 改动 |
|------|------|
| `app/graph/nodes/models.py` | 新增 `VisionAnalysisOutput` Pydantic模型（7字段+2校验器） |
| `app/graph/nodes/prompts.py` | 新增 `VISION_STRUCTURED_EXTRACT_PROMPT` + `VISION_OCR_INJECTED_PROMPT` |
| `app/graph/nodes/nodes.py` | 重构 `vision_analysis_node`（4步流程+Command路由）；新增5个辅助函数；重构 `stream_vision_answer`（图片摘要+流式RAG） |
| `app/graph/graph.py` | `vision_analysis` 从固定边改为 Command 动态路由（→knowledge_retrieval 或 →safety_check） |

---

## v9.12 - 路由评估体系：意图识别可量化、可归因、可迭代

### 核心改进：评估基础设施 + 分层归因 + Bad Case 反哺闭环

**问题**：意图识别（route_node）的三层路由（规则→上下文→LLM）没有准确率度量，无法判断：
- 规则层关键词是否够全
- 上下文层是否有误判
- LLM层兜底的比例和准确率
- Bad Case 如何反哺测试集防止回归

**方案**：构建路由评估闭环

```
评估脚本 → 测试集(route_test_set.jsonl) → 分层归因报告
                                              ↓
                          Bad Case采集(👎+route_misclassification) → 人工审核 → 补入测试集
                                              ↓
                          BadCaseRegressionRunner.run_batch_route() → 回归验证
```

### 新增文件

| 文件 | 功能 |
|------|------|
| `tests/data/route_test_set.jsonl` | 路由评估测试集（85条：54条golden + 31条边界case） |
| `scripts/evaluate_router.py` | 路由评估脚本（规则/上下文/LLM三层评估+指标+对比基线） |

### 改动文件

| 文件 | 改动 |
|------|------|
| `app/graph/nodes/nodes.py` | route_node 增加 route_layer 记录 + _record_route_metrics 指标采集 |
| `app/evaluation/bad_case_runner.py` | 新增 run_single_route / run_batch_route 路由回归测试 |
| `app/core/metrics.py` | 新增 get_route_stats() 路由分层统计查询 |
| `tests/test_nodes.py` | 新增 TestDetectRouteFromContext 7个测试 + 规则层6个扩展测试 |

### 评估脚本用法

```bash
# 仅评估规则层（无需LLM，快速）
python scripts/evaluate_router.py --layer rule

# 评估规则+上下文层
python scripts/evaluate_router.py --layer context

# 全量评估（含LLM，需Ollama）
python scripts/evaluate_router.py --layer all

# 保存基线
python scripts/evaluate_router.py --layer rule --save-baseline data/evaluation/router_baseline.json

# 与基线对比
python scripts/evaluate_router.py --layer rule --compare data/evaluation/router_baseline.json
```

### 输出指标

- 整体准确率 + 宏平均/加权平均 F1
- 分类别 Precision / Recall / F1（symptom / knowledge / general）
- 分层命中率（rule / context / llm / miss）
- 分类别/分难度准确率
- 边界case专项准确率
- 错误用例明细（含boundary_reason归因）

---

## v9.11 - 双集合零停机：全库重建窗口期消除

### 核心改进：影子集合 + 别名指针 + 原子切换

**问题**：v9.10 的全量重建流程是"清空旧数据 → 写入新数据"，中间存在 80s+ 窗口期，
期间用户检索到空库 → "文档中未提及"。单文档更新已用双缓冲解决，但全库重建
的重建周期更长（1-5 分钟），必须用更彻底的隔离方案。

**方案**：双集合（Dual-Collection）+ 别名指针切换

```
旧方案（原地重建，有 80s+ 窗口期）：
  t0  clear_vector_store()     → 旧数据全部删除！
  t1  加载 → 切分 → 索引       → ~80s
  t2  写入向量库               → 期间检索全空

新方案（双集合，零停机）：
  t0  创建影子集合 medical_kb_v20260726_185100
  t1  加载 → 切分 → 写入影子集合 → 对线上完全不可见
  t2  校验（chunk数、召回率、模型一致性）
  t3  校验通过 → 原子切换指针 → 毫秒级生效
  t4  5 分钟后延迟清理旧集合
  ↑ 全程：用户要么看到完整旧版，要么看到完整新版
```

### 四阶段流程

| 阶段 | 操作 | 用户感知 |
|------|------|---------|
| **构建** | 影子集合写入，对线上不可见 | 始终走旧集合，完整结果 |
| **校验** | chunk数 + 抽样召回率 + 模型一致性 | 同上 |
| **切换** | 别名指针原子更新 + 刷新全局实例 | 毫秒级切换到新集合 |
| **清理** | 5 分钟后删除旧集合 | 无感知 |

### 新增组件

| 组件 | 说明 |
|------|------|
| `DualCollectionManager` | 影子集合创建/校验/切换/回滚/清理 |
| `kb_active.json` | 别名指针配置（active_collection + previous_collection） |
| `validate_shadow_collection()` | 四维校验（chunk数/召回率/模型一致/pending状态） |
| `switch_active_collection()` | 原子切换（temp文件→rename） |
| `schedule_cleanup_old_collection()` | 延迟5分钟清理旧集合 |
| `rollback_to_previous()` | 紧急回滚到上一活跃集合 |

### 新增 API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/admin/kb/rebuild` | POST | 双集合重建（替代原地重建） |
| `/api/admin/kb/rollback` | POST | 紧急回滚到上一集合 |
| `/api/admin/kb/collection-info` | GET | 查询当前/上一集合信息 |

### 改动文件

- `vector_store.py`：新增 `DualCollectionManager`（~340 行）
- `routes.py`：`kb_rebuild` 改用双集合流程 + 新增回滚/集合信息 API
- `hybrid_retriever.py`：新增 `reset_hybrid_retriever()` + `get_cached_hybrid_retriever()`
- `admin.html`：重建按钮改为"零停机" + 新增回滚按钮/集合信息按钮

---

## v9.10 - 双缓冲：消除知识库更新检索空窗期

### 核心改进：pending → active → deprecated 状态机

**问题**：v9.9 的增量更新流程是"先软删旧版本 → 再写新版本"，中间存在 ~1s 检索空窗期，
该文档在此期间对用户完全不可见。

**解决**：双缓冲——新版本写入时 `status=pending`（不可检索）→ 校验通过 →
批量激活 `status=active`（原子操作）→ 旧版本 `status=deprecated`（5 分钟后清理）。

```
旧方案（有检索空窗）：
  t0  soft_delete(旧)     → is_deleted=True → 该文档不可检索
  t1  写入新版本          → 新 chunk 可检索
  ↑ t0~t1 之间：该文档 = 不存在（空窗 ~1s）

新方案（双缓冲，无空窗）：
  t0  写入新版本(status=pending) → 对检索不可见，旧版本仍可检索 ✅
  t1  校验新版本 → 激活(status=active) → 原子操作
  t2  旧版本(status=deprecated) → 5 分钟后物理清理
  ↑ 全程无空窗：t0 前走旧版本，t1 后走新版本
```

### 状态机

```
  ┌─────────┐     校验通过      ┌─────────┐     5分钟后      ┌────────────┐
  │ pending │ ─────────────→  │  active │ ─────────────→  │ (物理删除) │
  │ 不可检索 │     激活         │  可检索  │     清理         │            │
  └─────────┘                  └─────────┘                  └────────────┘
       ↑                            │
       │ 校验失败                    │ 有新版本激活时
       │ (兜底：旧版本仍可用)        ↓
       │                       ┌────────────┐
       └───────────────────── │ deprecated │ ← 可被紧急回滚
                               │  不可检索   │
                               └────────────┘
```

### 检索层过滤

```python
# ChromaDB 查询强制过滤
where={"is_deleted": False, "status": "active"}

# 效果：
# pending chunk    → 不可检索（正在写入/校验中）
# active chunk     → 可检索（正常状态）
# deprecated chunk → 不可检索（旧版本，5 分钟后物理删除）
# is_deleted=True  → 不可检索（管理员主动删除，30 天后物理删除）
```

### 更新期间降级提示

前端发送消息前检测知识库更新状态，若正在更新则显示非阻断提示：
> "知识库正在同步最新指南，部分结果可能略有延迟"

提示 5 秒后自动消失，不阻断用户操作。

### 新增函数

| 函数 | 说明 |
|------|------|
| `activate_document_version()` | 校验并激活新版本：pending→active + 废弃旧版本 |
| `deprecate_old_versions()` | 旧版本 active→deprecated（非删除，可回滚） |
| `cleanup_deprecated_chunks()` | 清理超过 5 分钟的 deprecated chunk |

### 改动文件

- `kb_updater.py`：新增 3 个双缓冲函数 + `enrich_chunk_metadata` 增加 `status` 参数
- `hybrid_retriever.py`：ChromaDB 查询加 `status=active` 过滤
- `routes.py`：上传接口改用双缓冲流程
- `index.html`：发送消息前检测更新状态，非阻断提示

---

## v9.9 - 知识库更新架构优化（参考日志1最佳实践）

### 核心改进：6 大问题修复

| # | 日志1 指出的问题 | 修复 | 效果 |
|---|----------------|------|------|
| 1 | 只插入新向量，不删除旧向量 | 软删除旧版本 + version_id +1 | 修改5次不会留下5个版本 |
| 2 | Embedding 模型混用 | embedding_model/dimension 元数据 + 一致性校验 | 索引=查询模型不一致时告警 |
| 3 | Chunk 策略变了历史数据不重建 | chunk_strategy 版本化 + reconciliation 检测 | 策略变更触发全量重建 |
| 4 | 文档删除后仍被召回 | is_deleted 软删除 + 检索过滤 | 删除文档不再被检索命中 |
| 5 | 变更检测漏检 | 变更检测 API（磁盘 mtime vs 索引 updated_at） | 轮询兜底防漏检 |
| 6 | 更新无审计记录 | SQLite 审计日志（doc_id/change_type/result/elapsed） | 出问题可定位到具体环节 |

### 元数据体系（每个 chunk 必备）

```json
{
  "doc_id": "发热诊断与家庭护理指南.txt",
  "chunk_id": "发热诊断与家庭护理指南.txt_a3f7b2c1d4e5f6a7",
  "content_hash": "a3f7b2c1d4e5f6a7",
  "version_id": 3,
  "is_deleted": false,
  "embedding_model": "embedding-3",
  "embedding_dimension": 2048,
  "chunk_strategy": "v9.9_row_level_table_aware",
  "updated_at": "2026-07-24T15:30:00",
  "source_trace": "指南.txt | 药物对比 | 行5: 每日最大量"
}
```

### 新增接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/admin/kb/status` | GET | 知询知识库状态 + 一致性校验 + Embedding 信息 |
| `/api/admin/kb/upload` | POST | 上传文档（增量去重 + 版本管理 + 软删除旧版本 + 审计） |
| `/api/admin/kb/documents/{filename}` | DELETE | 软删除文档（is_deleted=True） |
| `/api/admin/kb/restore/{filename}` | POST | 恢复误删文档 |
| `/api/admin/kb/rebuild` | POST | 重建知识库（审计日志） |
| `/api/admin/kb/audit-log` | GET | 查询审计日志 |
| `/api/admin/kb/reconcile` | GET | 一致性校验（磁盘/向量库/Embedding/策略） |
| `/api/admin/kb/stale-detect` | GET | 变更检测（陈旧文档） |

### 增量更新流程（content_hash 去重）

```
上传文档 → 切分 → content_hash 计算
  → 对比已有 hash
    → 未变化：跳过 Embedding（0ms vs ~200ms/chunk）
    → 已变化：软删除旧版本 → 写入新版本 → version_id +1
  → 审计日志记录
```

### 一致性校验（reconciliation）

```
磁盘文件列表 ⊕ 向量库 source 列表
  → missing_in_index：磁盘有但索引无（新文档未入库）
  → missing_on_disk：索引有但磁盘无（删除未同步）
  → embedding_mismatch：Embedding 模型不一致（需重建）
  → chunk_strategy_mismatch：切分策略不一致（需重建）
  → soft_deleted_count：软删除记录数
```

### 检索时软删除过滤

- ChromaDB 查询自动加 `where={"is_deleted": False}`
- BM25 在重建时跳过 is_deleted=True 的文档
- 软删除后 30 天由 `physical_cleanup_stale_deletes()` 物理清理

**新增文件**：
- `app/rag/kb_updater.py`：知识库更新管理核心模块

**改动文件**：
- `routes.py`：6 个新接口 + 上传/删除集成 kb_updater
- `hybrid_retriever.py`：ChromaDB 查询加 is_deleted 过滤

---

## v9.8 - 知识库管理 API + 并发安全

### 新增1：知识库管理接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/admin/kb/status` | GET | 知询知识库状态（文档列表、向量数、kb_version、更新状态） |
| `/api/admin/kb/upload` | POST | 上传文档（multipart/form-data，支持多文件），增量入库 |
| `/api/admin/kb/documents/{filename}` | DELETE | 删除指定文档（ChromaDB chunks + 磁盘文件 + 缓存清除） |
| `/api/admin/kb/rebuild` | POST | 重建知识库（清空→重新加载→切分→索引→写入） |

**所有接口需管理员认证**：`X-Admin-API-Key` header 或本地访问（127.0.0.1）

**上传流程**：
```
POST /api/admin/kb/upload (multipart/form-data)
  → 保存文件到 docs/medical/
  → load_single_file() 按扩展名自动选择加载器
  → split_documents() 切分（表格感知→行级切片）
  → ParentChildManager.build_index() 构建父子索引
  → add_documents_to_store() 增量写入 ChromaDB
  → 清除缓存（Redis + 语义缓存）
```

**删除流程**：
```
DELETE /api/admin/kb/documents/xxx.txt
  → ChromaDB._collection.get(where={"source": "xxx.txt"})
  → ChromaDB._collection.delete(ids=...)
  → 磁盘删除文件 + BM25 缓存失效
  → 清除缓存
```

### 新增2：知识库更新并发安全（写锁 + 状态追踪）

**问题**：重建知识库期间（清空旧数据 → 写入新数据），用户检索请求可能命中半空向量库，返回不完整结果；并发上传/重建可能造成数据覆盖

**解决**：

| 机制 | 实现 | 保证 |
|------|------|------|
| **asyncio.Lock 写锁** | `_kb_update_lock` | 同一时间只有一个更新操作（上传/删除/重建） |
| **更新状态追踪** | `_kb_update_status` | 前端可查询进度，更新中拒绝新的更新请求（409 Conflict） |
| **非阻塞检索** | 检索不加锁 | 更新期间检索继续使用当前数据，新数据就绪后原子切换 |
| **run_in_executor** | CPU 密集操作在线程池执行 | 不阻塞 FastAPI 事件循环，其他请求正常处理 |
| **缓存清除** | 更新完成后统一清除 | 防止旧缓存返回过期答案 |

**并发场景处理**：

| 场景 | 行为 |
|------|------|
| 重建中 + 用户查询 | 查询走旧数据（ChromaDB persist），重建完成后下次查询走新数据 |
| 重建中 + 再次重建 | 返回 409 Conflict，提示"知识库正在更新中" |
| 上传中 + 用户查询 | 查询走当前数据，上传完成后下次查询包含新文档 |
| 上传中 + 删除 | 写锁排队，串行执行 |

**改动文件**：
- `routes.py`：新增 4 个知识库管理接口 + `_kb_update_lock` + `_kb_update_status`

---

## v9.7 - 表格数据知识库处理（行级切片+双格式）

### 设计原则（参考日志1：复杂表格入库最佳实践）

> 核心原则：单个切片被检索出来时，仍然能够解释自己在原表中的位置和含义。

**当前实现与日志1方案的差异对比**：

| 维度 | 日志1推荐 | v9.7 旧方案（整表保留） | v9.7 新方案（行级切片） |
|------|-----------|----------------------|---------------------|
| 切片粒度 | 按行/按业务分组 | 整表（≤1500字符）或按行组 | **每行独立 chunk + 概览 chunk** |
| 上下文完整性 | 每个切片携带表头+标题+单位+页码 | HTML 注释上下文提示 | **自然语言摘要 + 字段路径 + 表格标题** |
| 合并单元格 | fill-down 继承 | 不处理 | **fill-down 自动继承** |
| 可追溯性 | 文档名+表格标题+页码+行号 | 仅 source | **row_primary_key + row_index + table_title** |
| 检索精度 | 行级精确检索 | 整表返回（含无关行） | **行级精确检索，对比查询走概览 chunk** |

### 新增1：Excel/CSV/Markdown 文档加载器

**支持格式**：

| 格式 | 加载器 | 依赖 | 说明 |
|------|--------|------|------|
| `.xlsx` | `load_xlsx` | pandas + openpyxl | 每个 Sheet → 一个 Document（Markdown 表格） |
| `.xls` | `load_xlsx` | pandas + xlrd（降级） | 旧版 Excel 格式 |
| `.csv` | `load_csv` | pandas | 自动检测编码（utf-8/gbk/gb2312/latin-1） |
| `.md` | `load_md` | 无 | 保留原始 Markdown 格式（含表格） |

**转换流程**：Excel/CSV → DataFrame → Markdown 表格格式 → Document（含表格元数据）

**表格元数据**：
- `is_table: True` — 标记为表格文档
- `table_headers: List[str]` — 列名列表（检索增强用）
- `table_row_count / table_col_count` — 行列数
- `table_header_summary: str` — 表头摘要（如"表格列：药物, 剂量, 频次"）
- `sheet_name: str` — Excel Sheet 名

**LOADERS 字典扩展**：`.txt` `.pdf` `.docx` **`.md` `.xlsx` `.xls` `.csv`**

### 新增2：行级切片 + 双格式（核心改动）

**问题**：原有方案"整表保留"→ 查询"布洛芬的每日最大量"返回整个对比表（7 行），LLM 需自己从表格中提取，容易出错；且 Embedding/BM25 对表格格式文本检索效果差

**解决**：行级切片 + 双格式策略

| chunk 类型 | 数量 | 用途 | 示例查询 |
|-----------|------|------|---------|
| **概览 chunk** | 1 个/表 | 对比类查询 | "布洛芬和对乙酰氨基酚哪个好？" |
| **行级 chunk** | N 个/表 | 精确查询 | "布洛芬的每日最大量是多少？" |

**行级 chunk 示例**（药物对比表第 6 行）：
```
【表格上下文】
表格：对乙酰氨基酚 vs 布洛芬对比
字段：项目, 对乙酰氨基酚, 布洛芬
---
| 项目 | 对乙酰氨基酚 | 布洛芬 |
|------|------------|--------|
| 每日最大量 | 2000mg | 1200mg |
---
摘要：在对乙酰氨基酚 vs 布洛芬对比中，每日最大量：对乙酰氨基酚为2000mg，布洛芬为1200mg
```

**自然语言摘要 `_generate_row_summary()`**：
- 对比表（第一列 header ∈ {项目,症状,指标...}）：`每日最大量：对乙酰氨基酚为2000mg，布洛芬为1200mg`
- 实体表（其他）：`布洛芬：退热效果良好，止痛效果中重度，抗炎作用有`
- 摘要直接参与 Embedding/BM25 索引 → "布洛芬"+"每日最大量"精确命中

### 新增3：合并单元格继承（fill-down）

**问题**：Excel 中合并单元格（如"解热镇痛药"分类覆盖 3 行），转 Markdown 后只有第一行有分类名，其余行为空 → 行级切片丢失分类上下文

**解决**：`_dataframe_to_markdown_table(fill_down=True)` 逐列向前填充空值
```
修复前：                修复后：
| 分类     | 药物   |   | 分类     | 药物   |
|----------|--------|   |----------|--------|
| 解热镇痛 | 布洛芬 |   | 解热镇痛 | 布洛芬 |
|          | 阿司匹林|   | 解热镇痛 | 阿司匹林 |
|          | 对乙酰 |   | 解热镇痛 | 对乙酰 |
```

### 新增4：行级元数据（可追溯性）

每个行级 chunk 携带以下元数据，确保被检索出后可独立解释含义：

| 元数据字段 | 说明 | 示例 |
|-----------|------|------|
| `chunk_type` | `"table_row"` or `"table_overview"` | `table_row` |
| `table_title` | 最近的上层 Markdown 标题 | `"对乙酰氨基酚 vs 布洛芬对比"` |
| `table_headers` | 完整表头列名 | `["项目","对乙酰氨基酚","布洛芬"]` |
| `row_index` | 行号（0-based） | `5` |
| `row_primary_key` | 第一列值（行主键） | `"每日最大量"` |
| `row_summary` | 自然语言摘要 | `"在对乙酰氨基酚...布洛芬为1200mg"` |

**改动文件**：
- `loader.py`：新增 `load_xlsx`、`load_csv`、`load_md`、`_dataframe_to_markdown_table`（含 fill-down）
- `loader.py`：新增 `_detect_markdown_tables`、`_enrich_table_metadata`、`_extract_table_title`
- `loader.py`：新增 `_split_table_aware`（行级切片）、`_generate_row_chunks`、`_generate_row_summary`
- `loader.py`：新增 `_segment_by_table`（表格/非表格分段）
- `loader.py`：修改 `split_documents`、`_split_by_markdown_headers`（含表格章节走行级切片）
- `loader.py`：增强 `add_metadata`（溯源元数据：file_type/file_size/doc_hash/source_trace）
- `LOADERS` 字典：新增 `.md` `.xlsx` `.xls` `.csv`

### 新增6：扫描件 OCR 模块（PaddleOCR PP-Structure）

**流程**（参考日志1：先还原结构，再输出结构化内容）：
```
扫描件图片/PDF
  → pdf2image（PDF逐页转图片，DPI=300）
  → PaddleOCR PP-Structure
    → 版面分析（layout）：检测 text/table/title/figure 区域
    → 文本区域 → OCR → 纯文本
    → 表格区域 → 表格识别（table rec）→ HTML → _html_table_to_markdown → Markdown 表格
    → 标题区域 → OCR → Markdown 标题层级（bbox 高度启发式推断 H1/H2）
    → 页眉/页脚 → 忽略（不进入正文）
  → 按阅读顺序组装 Markdown
  → 标准 chunking pipeline（表格感知 → 行级切片）
```

**扫描件 PDF 自动检测**：
- MinerU 解析后总文本 < 50 字符 → 疑似扫描件 → 自动降级 `load_scanned_pdf()`
- 无需手动标注"这是扫描件"

**新增函数**：
- `_ocr_image_to_markdown()`：PP-Structure 版面分析 + OCR + 表格识别
- `_html_table_to_markdown()`：HTML 表格 → Markdown 表格（处理合并单元格）
- `load_scanned_image()`：图片扫描件加载器
- `load_scanned_pdf()`：PDF 扫描件加载器（逐页 OCR）

**依赖**：
- `paddleocr` + `paddlepaddle`（或 GPU 版 `paddlepaddle-gpu`）
- `pdf2image` + `poppler`（扫描件 PDF 逐页转图片）
- `Pillow`

**LOADERS 字典新增**：`.png` `.jpg` `.jpeg` `.tiff` `.tif` `.bmp` `.webp`

### 新增7：文档溯源元数据增强

**问题**：原有 `add_metadata` 仅添加 `source` 和 `file_path`，无法满足溯源需求——答案引用了某条数据，却无法追溯来自哪个文档的哪一行

**解决**：`add_metadata()` 增强为溯源元数据清单

| 元数据 | 说明 | 溯源用途 | 示例 |
|--------|------|---------|------|
| `source` | 文档文件名 | 定位文档 | `"发热诊断与家庭护理指南.txt"` |
| `file_path` | 完整路径 | 打开原文 | `"d:/Agent/.../发热诊断与家庭护理指南.txt"` |
| `file_type` | 文件类型 | 选择打开方式 | `"txt"` |
| `file_size` | 文件大小 | 完整性校验 | `15234` |
| `doc_hash` | 内容 MD5 前8位 | 防篡改校验 | `"a3f7b2c1"` |
| `page_number` | 页码（PDF/扫描件） | 定位页码 | `5` |
| `source_trace` | 溯源路径 | 一键追溯 | `"指南.txt \| 药物对比 \| 行5: 每日最大量"` |

**`source_trace` 格式**：`文档名 | 表格标题 | 行号: 行主键`
- 表格行级 chunk：`"发热诊断与家庭护理指南.txt | 对乙酰氨基酚 vs 布洛芬对比 | 行5: 每日最大量"`
- 表格概览 chunk：`"发热诊断与家庭护理指南.txt | 对乙酰氨基酚 vs 布洛芬对比 | 概览（前3行）"`
- 非表格 chunk：无 `source_trace`（通过 `source` + `page_number` 追溯）

---

## v9.6.2 - 无标点复合问题拆解增强 + 子问题并行检索 + 过滤修复

**场景**：用户输入"发烧怎么处理便秘怎么处理"（无问号、无连接词），当前检测逻辑（问号数 ≥ 2 或连接词）无法识别为复合问题，跳过拆解，导致便秘文档未召回

**修复**：在 `question_decompose_node` 中增加**症状实体检测**作为第二层判断：

1. **前置检测三层逻辑**：
   - 条件1（原有）：问号数 ≥ 2 或 1 个问号 + 连接词
   - 条件2（新增）：用 `get_symptom_matcher()` AC 自动机识别 ≥2 个不同症状实体
   - 两个条件都不满足时才跳过拆解

2. **规则降级拆解增强**（`_rule_based_decompose`）：
   - 优先按问号切分（原有逻辑）
   - 问号不足时按**症状实体边界**切分：根据症状在文本中的位置，从每个症状开始到下一个症状之前切出一个子问题
   - 例："发烧怎么处理便秘怎么处理" → ["发烧怎么处理", "便秘怎么处理"]

### 修复2：子问题检索串行执行导致首字延迟翻倍

**问题**：`knowledge_retrieval_node` 中多子问题用 `for` 循环串行调用 `retriever.invoke()`，2 个子问题串行约 4s

**修复**：改用 `ThreadPoolExecutor` 真正并行检索，检索耗时从 ~4s 降到 ~2s

### 修复3：adaptive_threshold 竞态导致 Reranker 被整体跳过

**问题**：`ThreadPoolExecutor` 并行检索时，`get_adaptive_threshold()` 阈值尚未注册，触发 `KeyError("未注册的阈值：RERANKER_THRESHOLD")`，整个 Reranker 被 `except` 跳过，返回 7 篇未精排的低质量文档 → 启发式过滤 10→2 → LLM 说"文档未提及"

**修复**（双重保险）：
1. `adaptive_threshold.py`：`get_adaptive_threshold()` 改为双重检查锁定（DCL），确保初始化+注册原子完成
2. `hybrid_retriever.py`：阈值获取失败时**用默认值**（RERANKER_THRESHOLD=0.02, HIGH_CONFIDENCE_THRESHOLD=0.08），而非让整个 Reranker 被跳过。这是防御性修复，即使 DCL 失败也能保证 Reranker 正常执行

### 修复4：邻域扩展导致 sub_question 与文档内容不匹配，启发式过滤误杀关键文档

**问题**：多子问题检索后，邻域扩展（sibling expansion）把同一文档的相邻章节拉入结果。但兄弟章节继承的是触发扩展的子问题的 `sub_question`，而非章节自身主题。例如：
- "发热怎么处理" 检索到 `常见疾病症状与家庭护理指南` 发热章节 → 邻域扩展拉入便秘章节 → 便秘章节的 `sub_question = "发热怎么处理"`
- 去重后，便秘章节的 `sub_question` 是 "发热怎么处理"
- `has_query_overlap("发热怎么处理", "## 便秘...")` → "发热" 不在便秘内容中 → 被过滤

**根因**：多子问题检索的文档已过各自 Reranker 精排，再用 `has_query_overlap` 做交叉过滤会导致兄弟章节被误杀。邻域扩展引入的跨主题章节虽然 `sub_question` 不匹配，但它们是原始检索结果的结构邻居，对回答完整性有价值

**修复**：多子问题检索直接跳过启发式过滤，保留全部文档。单一问题仍走原有的 Reranker + 重叠过滤逻辑

### 修复5：RAG Prompt 大小日志错误

**问题**：`len(prompt)` 返回消息条数（2条），不是字符数，日志显示"2字符"

**修复**：改为 `sum(len(msg.content) for msg in prompt)`

**改动文件**：
- `nodes.py`：`question_decompose_node`、`_rule_based_decompose`、`knowledge_retrieval_node`、`filter_relevant_docs`、`stream_answer_generation`
- `adaptive_threshold.py`：`get_adaptive_threshold()` 线程安全（DCL）
- `hybrid_retriever.py`：阈值获取失败时用默认值，防御性修复

---

## v9.6.1 - 问题拆解延迟修复 + 拆解条件修复

### 修复1：移除长度阈值（"发烧？便秘？"仅 8 字符但也应拆解）

**根因**：`len(question) <= 20` 跳过了 14 字符的复合问题"发烧怎么处理？便秘怎么处理？"
**修复**：移除长度阈值，仅用问号数 ≥ 2 或连接词作为复合检测条件

### 修复2：来源多样性过滤 max_per_source 2→3

**根因**：多子问题检索后，5 篇文档经 `max_per_source=2` 过滤只剩 2 篇（且同源），便秘文档被丢弃
**修复**：`max_per_source=3`，给多子问题检索更多空间

### 修复3：问题拆解节点延迟 4850ms → ~1200ms

**根因**（日志1.txt 分析）：
1. `invoke_structured` 尝试 2 次 × 3 种策略 = 6 次 LLM 调用，Ollama 本地模型不支持 Tool Calling/JSON Mode，前两层各报错，Layer 3 也解析失败 → 4 次 HTTP × ~1.2s = 4.85s
2. 子问题2 语义缓存误命中子问题1 的结果（83.86% 相似度命中），导致药物文档未检索
3. 文档去重过度：6 篇 → 3 篇 → 2 篇

**修复**：
- `max_attempts=2 → 1`，`force_strategy="text_only"`：跳过 Tool Calling 和 JSON Mode，直接用 Layer 3 纯文本解析
- 2 秒超时保护：超过 2s 直接降级规则拆解
- 子问题检索用 `original_query=sub_q`（子问题自身），避免缓存误命中
- 降级规则拆解按问号切分（无需 LLM，0ms）

---

## v9.6 - 长问题拆解 + 多子问题并行检索

### 新增：question_decompose_node 长问题拆解节点

**背景**：用户提出复合问题（如"感冒了怎么办？布洛芬和对乙酰氨基酚哪个好？"）时，单一检索无法同时覆盖多个主题，导致部分子问题答案缺失

**流程**：路由 → 查询重写 → **问题拆解** → 并行检索 → 评分 → 答案生成

**拆解策略**（三层降级）：

1. **前置检测**（不用 LLM）：
   - 问题 ≤ 20 字符 → 不拆解
   - 问号 < 2 个且无连接词 → 不拆解
   - 检测连接词：另外/还有/同时/而且/以及/并且/再问

2. **LLM 拆解**：`QUESTION_DECOMPOSE_PROMPT` + `invoke_structured` → `QuestionDecomposeOutput`
   - `need_decompose`: 是否需要拆解
   - `sub_questions`: 独立子问题列表（最多 4 个）
   - 每个子问题必须自包含（不依赖其他子问题的答案）

3. **规则降级**：LLM 失败时按问号切分

**并行检索**（`knowledge_retrieval_node` 增强）：

- `sub_questions` 有多个时，逐一检索每个子问题
- 每个子问题独立做同义词预处理
- 文档标注 `sub_question` + `sub_question_idx` 元数据
- 检索结果按 `(source, page_content[:100])` 去重
- 单一问题仍走原有逻辑（兼容）

**新增文件/模型**：
- `prompts.py`：`QUESTION_DECOMPOSE_PROMPT`
- `models.py`：`QuestionDecomposeOutput`（need_decompose + sub_questions + field_validator）
- `state.py`：`sub_questions: Optional[List[str]]`
- `streaming.py`：阶段 2.5 插入问题拆解

**示例**：
```
用户：感冒了怎么办？布洛芬和对乙酰氨基酚哪个好？
  ↓ question_decompose_node
子问题1：感冒了怎么办？
子问题2：布洛芬和对乙酰氨基酚哪个好？
  ↓ knowledge_retrieval_node（并行检索）
子问题1 → 5 篇文档（常见疾病症状与家庭护理、呼吸系统诊疗...）
子问题2 → 5 篇文档（常见药物使用指南、药物相互作用...）
  ↓ 去重
共 8 篇文档 → 答案生成
```

---

## v9.5 - 四层上下文压缩策略（L1→L3→L2→L4）

### 新增：context_manager.py 四层上下文压缩模块

**背景**：对话历史随轮次增长，messages 列表无限膨胀导致 Prompt 超出 4K token 限制，LLM 遗忘早期对话信息

**架构**：四层压缩 Pipeline，执行顺序 L1 → L3 → L2 → L4

| 层级 | 策略 | 触发条件 | 是否用 LLM | 效果 |
|------|------|----------|-----------|------|
| L1 | 中间输出清除 | 消息数 > 3 | 否 | 中间 AI 回答只保留首句摘要 |
| L3 | 大输出持久化 | 单条 > 30KB | 否 | 写入磁盘，占位符 `<persisted-output>` |
| L2 | 工具调用裁剪 | RAG 文档块/工具痕迹 | 否 | `[参考了 N 篇文档]` 占位符 |
| L4 | LLM 摘要压缩 | 总量 > 50000 字符 | 是 | 保留 5 类关键信息，原始存 transcript |

**L4 五类关键信息**（借鉴 MemGPT）：
1. `current_goal` — 当前目标（"用户在咨询感冒"）
2. `key_findings` — 关键发现和决策（"确诊为普通感冒"）
3. `files_referenced` — 参考过的文档来源列表
4. `remaining_work` — 尚未解决的问题（"待确认过敏史"）
5. `user_constraints` — 用户约束（"对青霉素过敏"）

**集成**：
- `get_conversation_history_text()` 新增 `enable_context_compression=True` 参数
- RAG 场景自动启用四层压缩
- L4 降级：LLM 摘要失败时自动切换规则提取（不依赖 LLM）

**新增文件**：
- `app/graph/nodes/context_manager.py` — 四层压缩核心模块
- `data/persisted_outputs/` — L3 持久化输出目录

**新增 Pydantic 模型**：
- `ContextSummaryOutput` — L4 摘要结构化输出（5 个字段 + field_validator）

---

## v9.4 - RAG 召回修复：查询预处理条件逻辑错误 + 同义词补全

### Bug1：`_preprocess_query` 从未执行（条件判断逻辑错误）

**问题**：自包含查询（如"流鼻血怎么处理？"、"头疼咋办"）的同义词预处理从未生效，26 条同义词映射形同虚设

**根因**：`query_rewrite_node` 跳过重写时设 `rewritten_query = question`（非空），但 `retrieve_node` 的预处理条件为 `if not rewritten_query`，非空字符串永远为 `True`，条件永远不满足

**影响范围**：所有 26 条同义词映射（"头疼"→"头痛"、"拉肚子"→"腹泻"、"退烧"→"退热"等）在自包含查询场景下全部失效

**修复**：
```python
# 修复前：not rewritten_query 永远 False
if not rewritten_query:
    preprocessed = _preprocess_query(original_query)

# 修复后：只有真正被 LLM 重写过的查询才跳过预处理
_was_actually_rewritten = rewritten_query and rewritten_query != original_query
if not _was_actually_rewritten:
    preprocessed = _preprocess_query(original_query)
```

### Bug2："流鼻血"→"鼻出血" 同义词缺失

**问题**：查询 `流鼻血怎么处理？` 召回了《儿童常见疾病护理指南》（鼻塞/流涕），而非《外伤急救与处理指南》（鼻出血止血）

**根因**：同义词字典缺失 `"流鼻血"→"鼻出血"` 映射 + Bug1 导致预处理从未执行

**修复**：

1. **同义词字典补全**（`nodes.py` `_SYNONYMS`）：
   - `"流鼻血"→"鼻出血"`, `"鼻子出血"→"鼻出血"`, `"鼻流血"→"鼻出血"`
   - `"拉血"→"便血"`, `"大便出血"→"便血"`, `"吐血"→"咯血"`, `"咳血"→"咯血"`, `"尿血"→"血尿"`

2. **AC 自动机路由关键词补全**（`keyword_matcher.py` `build_route_symptom_matcher`）：
   - 新增 `"流鼻血"`, `"鼻出血"`, `"鼻流血"` → `"symptom"` 路由分类

3. **黄金测试集**（`golden_test_set.jsonl`）：
   - 新增第 54 条：`{"query": "流鼻血怎么处理？", "source_doc": "外伤急救与处理指南.txt", ...}`

**效果**：`流鼻血怎么处理` → 预处理 `鼻出血怎么处理` → BM25 精确匹配外伤指南 → 正确召回

### Bug3：RAG 答案假设患者人群（"让儿童坐下"出现在成人查询中）

**问题**：查询 `流鼻血怎么处理？` 的回答中出现"让儿童坐下或站立"，但用户未提及儿童，外伤指南原文也无儿童指向

**根因**：RAG Prompt 约束了"不得编造药物/剂量/治疗方案"，但未禁止假设患者人群。LLM 训练数据中"流鼻血"高频关联儿童，导致自动脑补"让儿童"等未见于文档的内容

**修复**：`prompts.py` `RAG_ANSWER_PROMPT` 重写为两步接地生成（Grounded Generation）：
1. **System Prompt**：核心原则 + 反例清单（"文档没提到儿童就不能说让儿童"等）
2. **Human Prompt**：强制两步流程——先从文档提取事实（第一步），再仅基于事实组织回答（第二步）
3. 软约束改为结构化约束：LLM 必须先列出文档事实，再据此回答，从流程上切断脑补路径

### Bug5：L3 对话历史中的旧答案被 LLM 当作事实引用

**问题**：修改 Prompt 后仍出现"让儿童"，LLM 标注来源为"L3 对话历史"——之前的错误回答存在 PostgresStore 对话历史中，LLM 直接复制了旧答案

**根因**：`get_conversation_history_text()` 将 AI 完整回答注入 RAG Prompt，LLM 无法区分"历史助手回答"和"检索文档事实"，将旧答案当作可信来源引用

**修复**：
1. `get_conversation_history_text()` 新增 `compress_ai_answers` 参数：
   - `True`（RAG 场景）：AI 回答压缩为首句摘要 + `[已回复]`，防止复制
   - `False`（非 RAG 场景）：保留完整回答，支持追问补全
2. `build_rag_prompt()` 调用时启用 `compress_ai_answers=True`
3. L3 标签改为"【L3 对话历史（仅供理解上下文，不是事实来源）】"
4. Prompt 明确声明"【L3 对话历史】中的助手回答可能包含错误，绝对不能作为事实引用"

### Bug6：检索结果缺乏文档来源多样性

**问题**：查询"感冒了怎么办？"时，7 个文档含感冒内容但 top 3 全部来自同一文档《常见疾病症状与家庭护理指南》

**根因**：Reranker 无来源多样性约束——内容最全面的文档在 RRF + Reranker 中包揽全部 top-k 位置，其他文档的补充信息（如用药禁忌、呼吸系统诊疗）被完全排除

**修复**：
1. 新增 `_apply_source_diversity()` 过滤函数：同来源文档最多保留 2 个 chunk
2. 检索 `k=3→5`，Reranker 输入 `RERANKER_INPUT_CAP=8→10`，确保多样性后有足够候选
3. 预期效果："感冒"查询召回至少 3 个不同来源文档

### Bug4：Prompt 变更后语义缓存未失效

**问题**：修改 RAG Prompt 后，旧缓存仍返回修改前的答案（含"让儿童"），Prompt 约束无效

**根因**：缓存 key 仅绑定 `kb_version`（知识库版本），Prompt 变更不改变 `kb_version`，旧缓存永不过期

**修复**：
1. `semantic_cache.py`：缓存 key 加入 `prompt_version`（基于 `prompts.py` 文件内容的 MD5 前 8 位）
2. `redis_cache.py`：`_generate_key` 自动注入 `prompt_version`
3. Prompt 变更 → 文件 MD5 变化 → 缓存 key 变化 → 旧缓存自动失效

---

## v9.3 - 前端来源展示与反馈按钮修复

### Bug1：SSE 完成事件类型不匹配导致来源和反馈按钮丢失

**问题**：后端发送的完成事件为 JSON `{"type": "done", "request_id": "..."}` ，但前端检查的是字符串 `data === '[DONE]'`，两者永远不匹配
- 导致来源展示代码和 `addFeedbackButtons()` 调用从未执行
- 用户看不到文档来源，也看不到 👍/👎 反馈按钮

**修复**：
1. **完成事件处理**：`if (data === '[DONE]')` → `else if (parsed.type === 'done')`，正确匹配 JSON 完成事件
2. **request_id 传递**：从 done 事件中提取 `request_id`，传入 `addFeedbackButtons()`，反馈提交时关联请求
3. **来源渲染**：新增 `renderSources()` 函数，来源显示为带编号的蓝色标签 `[1] xxx.md [2] yyy.txt`
4. **来源样式**：新增 `.source-item` 样式（蓝底圆角标签）+ `.sources` 左边框加粗
5. **👍 按钮修复**：之前只切换 UI 状态不提交反馈 → 现在调用 `submitFeedbackAPI('up', ...)` 真正提交
6. **反馈数据修复**：`submitFeedback()` 之前缺失 `rating` 和 `request_id` 字段 → 现在完整发送 `rating`/`request_id`/`question`/`reason`/`note`/`answer_preview`
7. **同步请求**：同样适配 `renderSources()` 和 `addFeedbackButtons()` 新参数

**效果**：流式和同步模式下都能正确显示文档来源 + 反馈按钮，反馈数据完整关联 request_id

---

## v9.2 - 时间退化风险修复（5 项安全加固）

### 🔴 漏洞1：语义缓存毒化（最高优先级）✅ 已修复

**问题**：语义缓存 key 仅含 `md5(query)`，无知识库版本绑定。知识库更新后旧缓存仍返回过期答案 → 医疗安全风险
- 衰减速度：1~2 周
- 安全影响：⚠️ 用药剂量/禁忌症过期

**修复**：

1. **`vector_store.py` 新增 `get_kb_version()`**：基于 ChromaDB 所有 doc_id 排序哈希 + 文档数量生成 8 位版本指纹
   - 首次调用计算并缓存，后续 O(1) 读取
   - `add_documents()` / `delete_collection()` 后自动调用 `invalidate_kb_version()` 使指纹失效
   - 知识库更新 → kb_version 变化 → 旧缓存 key 不再匹配 → 自动失效

2. **`semantic_cache.py` 修复**：
   - `set()`：缓存 key 从 `md5(query)` 改为 `md5(query:kb_version)`，写入时记录 `kb_version`
   - `get()`：命中相似查询后校验 `cached_kb_version == current_kb_version`，不匹配则删除过期条目并 miss

3. **`redis_cache.py` 修复**：
   - `_generate_key()`：自动注入 `kb_version` 到 key 哈希输入，L0 检索缓存同样防毒化

**效果**：知识库每次更新后，L0（Redis 缓存）和 L2（语义缓存）自动失效，不再返回过期医疗答案

### 🔴 漏洞2：临床快照状态腐烂 ✅ 已修复

**问题**：`clinical_checkpoint` 由 LLM 增量更新，无字段级合并策略，`medication_history`/`red_flags` 被全量覆盖
- LLM 可能遗忘已有用药记录、篡改 chief_complaint、凭空编造药物
- 无快照历史版本、无回滚机制、无上限约束
- 衰减速度：2~4 周
- 安全影响：⚠️ 过敏史丢失 → 用药安全风险

**修复**：

1. **字段级合并策略**（`_apply_checkpoint_merge()`）：
   - `chief_complaint`：不变性约束（旧值优先，LLM 不可覆盖）
   - `medication_history`：追加去重（按 drug 名称匹配，旧记录为基础，仅补充空字段）
   - `red_flags`：追加去重（规范化字符串比较，忽略标点差异）
   - `confirmed_facts`：**只增不减**（过敏史/既往史不可被 LLM 删除，关键安全约束）
   - `ruled_out`：追加去重
   - `symptom_timeline`：LLM 可更新（允许修正），但需裁剪

2. **字段上限**（`_CHECKPOINT_FIELD_LIMITS`）：
   - medication_history ≤ 15, red_flags ≤ 10, symptom_timeline ≤ 20, confirmed_facts ≤ 20, ruled_out ≤ 15
   - 防止 LLM 编造过多条目导致 Prompt token 膨胀

3. **合并变更日志**：每次合并后记录字段条目数变化，便于审计

**效果**：过敏史/既往史不会因 LLM "遗忘" 而丢失，用药记录不会因全量覆盖而消失，主诉不会在追问中被篡改

### 🟡 漏洞3：PostgresStore Append-Only 无界膨胀 ✅ 已修复

**问题**：`symptom_events`/`medication_events`/`bad_cases`/`query_history` 四个命名空间只有 append，无 prune/compact
- `get_symptom_events()` 全量加载后截断，数据量增大后性能退化
- 衰减速度：1~2 个月
- 安全影响：性能退化 + 早期记录被遗忘

**修复**：

1. **Prune 机制**（`long_term_memory.py` 新增）：
   - `prune_namespace()`：按保留天数 + 条目上限双重清理，自动删除过期和超量记录
   - `prune_all_namespaces()`：批量清理用户全部命名空间
   - `prune_all_users()`：管理员接口，清理多用户数据
   - 各命名空间默认保留天数：symptom_events=90, medication_events=90, bad_cases=180, query_history=30
   - 各命名空间条目上限：symptom_events=500, medication_events=300, bad_cases=500, query_history=200

2. **查询优化**（提前截断）：
   - `get_symptom_events()`：最多读取 `limit * 3` 条后排序截断，避免全量加载
   - `get_query_history()` / `get_bad_cases()` / `get_medication_events()`：最多读取 `limit * 2` 条
   - 降低内存峰值，减少排序耗时

**效果**：数据量 3 个月后稳定在配额内，查询性能不再随时间退化

### 🟡 漏洞4：硬编码阈值随数据分布漂移失效 ✅ 已修复

**问题**：6 个关键阈值（0.08/0.02/0.01/0.05/0.92）在特定数据分布下调优，知识库扩张后静默失效
- `HIGH_CONFIDENCE_THRESHOLD=0.08`：稀疏空间"极度相似" → 密集空间"有点相关"
- 衰减速度：1~3 个月
- 安全影响：⚠️ 跳过 Reranker 导致幻觉

**修复**：

1. **新增 `app/core/adaptive_threshold.py`**：`AdaptiveThreshold` 自适应阈值管理器
   - 基于运行时百分位统计动态调整阈值
   - 冷启动：前 100 个样本使用默认值
   - 自动校准：每 1000 个样本重新计算百分位数
   - 持久化：校准值写入 SQLite（`data/adaptive_thresholds.db`），重启后恢复
   - 管理员接口：`force_recalibrate()` 手动触发校准，`get_stats()` 查看统计

2. **注册的三个自适应阈值**：

   | 阈值 | 默认值 | 策略 | 百分位 | 范围 |
   |------|--------|------|--------|------|
   | HIGH_CONFIDENCE_THRESHOLD | 0.08 | percentile | P5 | [0.01, 0.20] |
   | RERANKER_THRESHOLD | 0.02 | percentile | P5 | [0.005, 0.10] |
   | SEMANTIC_CACHE_THRESHOLD | 0.92 | percentile | P95 | [0.85, 0.99] |

3. **`hybrid_retriever.py` 修改**：
   - `HIGH_CONFIDENCE_THRESHOLD`：从硬编码 `0.08` → `at.get("HIGH_CONFIDENCE_THRESHOLD")`
   - `RERANKER_THRESHOLD`：从 `config.RERANKER_THRESHOLD` → `at.get("RERANKER_THRESHOLD")`
   - 每次检索后 `at.observe()` 记录观察值，用于后续校准

**效果**：知识库扩张后阈值自动跟随数据分布漂移，不再静默失效

### 🟡 漏洞5：AC 自动机规则与知识库脱耦

**问题**：药物关键词表/同义词字典/症状关键词均为人工维护，与知识库文档无同步机制
- 新增文档后规则不更新 → 覆盖率从 80% 降至 60%+ → 新药物查询绕过 RAG
- 衰减速度：渐进式
- 安全影响：降级风险

**修复计划**：知识库变更时自动扫描新实体，提醒更新规则

---

## v9.0 - RAG 流水线性能优化（TTFT 预计 -600~900ms）

### 优化1：Reranker 三阶段化（970ms → ~300ms）

**问题**：Reranker 入参 10~20 篇文档，max_length=512，CPU 推理 400~970ms

**修复**：
- RRF 融合后先轻量截断 top 8（`RERANKER_INPUT_CAP=8`），再进 Reranker 精排
- `max_length` 512 → 256（200 字中文 ≈ 256 tokens，覆盖 95%+ 关键信息）
- `MAX_RERANK_DOC_CHARS` 300 → 200（头 134 + 尾 66）
- `DEFAULT_K` 5 → 3（减少送入 LLM 的文档数，缩短 Prompt token）
- `rerank_top_k` 5 → 8（入参数，RRF 融合后截断数）

**预期**：Reranker 970ms → ~300ms，TTFT -600ms

### 优化2：Embedding LRU 缓存（重复查询 0ms vs API 200~400ms）

**问题**：每次查询调智谱 embedding-3 API，网络延迟 200~400ms

**修复**：
- 新增 `_EmbeddingLRUCache` 类（LRU，128 条上限，30 分钟 TTL）
- 相同查询复用 embedding 向量，命中时 0ms
- 跨请求复用（同一用户多次查询相同问题）
- 约占 1MB 内存（128 * 2048 * 4 bytes）

**预期**：重复查询 Embedding 400ms → 0ms

### 优化3：邻域扩展字符上限（2000 → 1500）

**问题**：邻域扩展后文档过长，Prompt token 数膨胀，TTFT 增加

**修复**：`MAX_SIBLING_CHARS` 2000 → 1500

**预期**：TTFT -200~300ms

### 优化4：知识库无覆盖早退（避免无效重试 1~2s）

**问题**：Reranker 最高分 < 0.01 时仍走自纠正循环，浪费 1~2s

**修复**：新增 `RERANK_NO_COVERAGE_THRESHOLD=0.01`，低于此值直接走 `direct_answer`

**预期**：知识库无覆盖场景 TTFT -1000~2000ms

### 优化5：查询预处理（纠错 + 同义词 + 语气词清理）

**问题**：口语化查询（"头疼咋办啊"）BM25 命中率低，Embedding 噪声大

**修复**：
- 同义词标准化（"头疼"→"头痛"、"拉肚子"→"腹泻"、"退烧"→"退热" 等 18 条）
- 语气词前缀清理（"我想问一下"→""）
- 语气词后缀清理（"啊呀呢吧"→""）
- 仅对未重写的原始查询生效，重写查询已是高质量查询

**预期**：口语化查询检索召回率 +10~15%

### 优化6：pyahocorasick C 扩展安装

**问题**：纯 Python AC 自动机实现，关键词库扩展后性能差距大

**修复**：`pip install pyahocorasick==2.3.1`，`keyword_matcher.py` 自动使用 C 扩展版

**预期**：规则层关键词匹配提速 5~10x（大规模关键词库时）

### 优化7：黄金测试集 + 评估模块适配

**新增**：
- `tests/data/golden_test_set.jsonl`：53 条人工精选黄金测试集（覆盖用药安全/症状分诊/急救/慢性病/剂量/知识问答）
- `scripts/generate_golden_test_set.py`：自动生成脚本（73 条模板）
- `evaluation.py`：新增 `query` 字段和 `key_facts` 字段支持

### 优化8：渐进式进度反馈（SSE progress events）

**问题**：用户发出问题后等 2~3s 才看到第一个 token，期间无任何反馈

**修复**：
- `_run_rag_pipeline` 改为 async generator，在各阶段 yield SSE progress 事件
- 4 个进度阶段：`analyzing`（正在分析症状）、`searching`（正在检索知识库）、`rewriting`（正在优化查询）、`generating`（正在生成建议）
- 自纠正重试时额外发送 `rewriting` + `searching` 进度

**前端 SSE 事件格式**：
```
data: {"type": "progress", "stage": "searching", "message": "正在检索医学知识库..."}
```

**预期**：用户感知等待时间 -50%

### 优化9：可观测性基础设施（SQLite Metrics）

**问题**：只有 timing_decorator 写日志，无结构化指标，无法做 P50/P95/P99 分析

**新增**：
- `app/core/metrics.py`：`MetricsCollector` 类，SQLite 存储（3 张表）
  - `node_metrics`：节点级耗时，支持 P50/P95/P99 分析
  - `token_usage`：LLM Token 用量，成本估算，按模型/节点/每日趋势
  - `feedback`：用户反馈闭环，满意度率，差评原因分布
- `timing_decorator` / `async_timing_decorator` 自动写入 node_metrics
- 自动清理 30 天旧数据
- 支持 `get_request_stats(hours=24)` 查询请求级总耗时排名
- 自动清理 7 天旧数据

**查询示例**：
```python
from app.core.metrics import get_metrics_collector
collector = get_metrics_collector()
stats = collector.get_node_stats(hours=24)
# → [{"node_name": "知识检索", "avg_ms": 800, "p50_ms": 750, "p95_ms": 1200, ...}]
```

### 优化10：外部 API 熔断器（Circuit Breaker）

**问题**：Embedding API 连续故障时仍反复调用，导致请求堆积和超时

**新增**：
- `app/core/circuit_breaker.py`：`CircuitBreaker` 类，CLOSED → OPEN → HALF_OPEN 三态
- 集成到 `hybrid_retriever.py` 的 Embedding API 调用处
- 连续 3 次失败 → OPEN 状态（快速失败，跳过 API 调用）
- 30 秒后 → HALF_OPEN（放行一次探测请求）
- 熔断时降级为 BM25-only 检索（保证系统可用性）
- 探测成功 → CLOSED（恢复正常）

### 优化11：用户反馈闭环

**问题**：旧 `/api/feedback` 只写 Bad Case，无 👍/👎 区分、无统计、无闭环

**修复**：
- 重写 `FeedbackRequest`：支持 `rating`（👍/👎）+ `request_id` 关联 + `question` + `reason`
- 差评自动创建 Bad Case（写入 PostgresStore `long_term_memory`）
- 反馈统计：满意度率、差评原因分布、每日趋势
- 黄金测试集候选：`get_feedback_candidates_for_golden_set()` 从差评中提取待转化条目
- SSE 完成事件附带 `request_id`，前端可关联反馈

**反馈闭环流程**：
```
用户 👎 → record_feedback() → 自动 append_bad_case()
→ 人工审核 → 补填 ground_truth → 加入黄金测试集
→ 每次系统迭代后重跑评估 → 验证修复
```

### 优化12：Token 用量监控

**问题**：无法追踪 LLM API 的 token 消耗和成本

**新增**：
- `app/core/token_tracker.py`：从 `AIMessage.response_metadata` 自动提取 token 用量
- 集成到 3 个 LLM 调用节点：答案生成、直接回答、查询重写
- SQLite 存储：`token_usage` 表（request_id, model, prompt_tokens, completion_tokens, estimated_cost）
- 成本估算：基于智谱官方定价（glm-4-flash ¥0.0001/千tokens, glm-4v-plus ¥0.01/千tokens 等）
- 按模型/节点/每日趋势聚合统计

**查询示例**：
```bash
# Token 用量统计
curl http://localhost:8000/api/metrics/tokens?hours=24
# → {"total_tokens": 150000, "total_cost": 0.15, "by_model": [...], "daily_trend": [...]}

# 反馈统计
curl http://localhost:8000/api/metrics/feedback?hours=24
# → {"satisfaction_rate": 0.85, "by_reason": [{"reason": "answer_inaccurate", "count": 5}], ...}
```

### 优化13：Prompt 模板化（ChatPromptTemplate）

**问题**：10 个 Prompt 全部用 f-string 拼接，无角色区分，无法 A/B 测试

**修复**：
- 新增 `app/graph/nodes/prompts.py`：10 个 ChatPromptTemplate 集中管理
- 所有 Prompt 改为 `ChatPromptTemplate.from_messages()`，区分 System/Human 角色
- `build_rag_prompt()` 和 `build_direct_answer_prompt()` 返回 `List[BaseMessage]` 而非 str
- `llm.invoke()` 同时支持 str 和 List[BaseMessage]，无缝兼容

**Prompt 清单**：

| Prompt | 角色 | 变量 |
|--------|------|------|
| RAG_ANSWER_PROMPT | system + human | context, question, frozen_profile, time_facts, checkpoint, history, followup |
| RAG_ANSWER_NO_CONTEXT_PROMPT | system + human | question, frozen_profile, history |
| DIRECT_ANSWER_PROMPT | system + human | question, frozen_profile, checkpoint, history |
| ROUTER_PROMPT | system + human | question |
| QUERY_REWRITE_PROMPT | system + human | history_summary, question |
| SAFETY_CHECK_PROMPT | system + human | answer, clinical_snapshot |
| PROFILE_EXTRACTION_PROMPT | system + human | question |
| CHECKPOINT_UPDATE_PROMPT | system + human | existing_snapshot, new_messages |
| CHECKPOINT_NEW_PROMPT | system + human | new_messages |
| HYDE_PROMPT | system + human | question |
| VISION_ANALYSIS_PROMPT | system + human(多模态) | question, image_url |

### 优化14：Pydantic 结构化输出校验补全

**问题**：7 个 Pydantic 模型中只有 3 个被实际使用，路由和查询重写无校验

**修复**：

1. **路由节点**：f-string + `parse_router_output()` 正则 → `ROUTER_PROMPT` + `invoke_json_once_with_fallback` + `RouterOutput`
   - `RouterOutput.question_type`：Literal["symptom", "knowledge", "general"]
   - `field_validator`：兼容中文/复数/变体输入（"症状"→"symptom"，"symptoms"→"symptom"）
   - 校验失败 → 兜底 "general"

2. **查询重写节点**：正则提取 `FINAL:` / `SEARCH:` → `QUERY_REWRITE_PROMPT` + `invoke_json_once_with_fallback` + `QueryRewriteOutput`
   - `QueryRewriteOutput` 扩展：`rewritten_query` → `final_question` + `search_keywords`
   - `field_validator`：自动去除 `FINAL:` / `SEARCH:` 残留前缀
   - 支持 `max_attempts=2` 重试

3. **models.py 增强**：
   - `RouterOutput`：新增 `normalize_question_type` field_validator
   - `QueryRewriteOutput`：扩展为双字段 + 两个 field_validator
   - `ProfileExtractionOutput`：新增 `coerce_age`（"30岁"→30）、`coerce_allergies`（"青霉素,头孢"→["青霉素","头孢"]）
   - 所有模型添加对应 Prompt 的文档说明

4. **helpers.py 增强**：
   - `invoke_json_once_with_fallback`：prompt 参数支持 `List[BaseMessage]`
   - 新增 `max_attempts` 参数（路由/重写用 2 次重试）
   - 重试逻辑：校验失败自动重试，日志记录每次尝试

**校验覆盖演进**：3/7 → 5/7（路由+重写+安全+档案+快照），剩余 2 个（症状解析=规则引擎、文档评分=启发式）暂不需要

### 优化15：结构化输出三层降级策略（Tool Calling → JSON Mode → 纯文本）

**问题**：`with_structured_output` 完全未被使用（死代码），`invoke_json_once_with_fallback` 只做后处理容错，缺少采样层约束

**根因**：
- `with_structured_output` 要求模型原生支持 function calling，旧版 Ollama 不支持就报错
- 当前方案完全依赖后处理（extract_json_block + json_repair + Pydantic），是"打补丁"思维
- Tool Calling 可从采样层约束 LLM 输出格式，比后处理更可靠

**新增**：
- `app/graph/nodes/structured_output.py`：`invoke_structured()` 统一入口
- 三层降级策略：
  - **Layer 1: Tool Calling**（最可靠）
    - `convert_to_openai_tool(schema)` 将 Pydantic 模型转为 OpenAI tool 定义
    - `llm.bind_tools([tool], tool_choice={"type":"function","function":{"name":schema_name}})`
    - LLM 返回 `tool_calls[0].args` → Pydantic `model_validate`
    - 支持：glm-4-flash、glm-4-plus、glm-4、Ollama qwen2.5 (v0.3+)
  - **Layer 2: JSON Mode**（中等可靠）
    - `response_format={"type":"json_object"}` 保证合法 JSON
    - `extract_json_block` + Pydantic 校验
    - 需要 prompt 含 "JSON" 字样
  - **Layer 3: 纯文本 + 本地解析**（兜底）
    - 无格式约束，完全依赖后处理
    - `extract_json_block` → `json.loads` → `json_repair` → `ast.literal_eval`
    - `_coerce_list_fields` + `field_validator` 容错

**替换**：
- 5 个节点从 `invoke_json_once_with_fallback` 切换到 `invoke_structured`
- `invoke_structured_with_fallback` 保留为旧接口（向后兼容）
- `invoke_json_once_with_fallback` 保留为 Layer 3 的底层实现

**各模型实际降级路径**：

| 模型 | Layer 1 | Layer 2 | Layer 3 |
|------|---------|---------|---------|
| glm-4-flash | ✅ Tool Calling | ✅ JSON Mode | 兜底 |
| glm-4-plus | ✅ Tool Calling | ✅ JSON Mode | 兜底 |
| Ollama qwen2.5 (v0.3+) | ✅ Tool Calling | ✅ JSON Mode | 兜底 |
| Ollama qwen2.5 (旧版) | ❌ 不支持 | ✅ JSON Mode | 兜底 |

### 累计性能演进

| 版本 | TTFT (自包含) | TTFT (追问) | 核心优化 |
|------|-------------|------------|---------|
| v4.3 | ~4000ms | ~6000ms | 症状短路+缓存+Prompt精简 |
| v5.5 | ~2700ms | ~4000ms | 自包含跳过重写 |
| v8.4 | ~2700ms | ~3500ms | 症状移除LLM+HyDE短路 |
| v8.5 | ~2700ms | ~3500ms | HyDE默认关闭 |
| **v9.0** | **~2000ms** | **~2800ms** | Reranker三阶段+Embedding缓存+邻域缩减 |

---

## v8.5 - 指代词误判修复 + HyDE 移除 + Faithfulness 提升

### Bug1：_ANAPHORA_PATTERNS 多义字导致自包含查询强制重写

"流鼻血了怎么办？"被判定为"含指代词/省略结构"→ 强制调用 LLM 重写（3008ms）→ 记录 Bad case。

修复：移除 `"呢"`、`"再"`、`"也"` 多义字；补充实体词表

### Bug2：自包含查询跳过重写但没跳过 HyDE（5180ms 白跑）

修复：自包含查询统一跳过重写 + HyDE

### 决策：HyDE 基于实测数据默认关闭

A/B 测试结果：Recall -13.3%，耗时 +1574ms，4 条负向仅 2 条正向。

处置：`ENABLE_HYDE=False` 默认关闭

### Bug3：规则引擎中文分词失效（评估分数全为 0）

`_tokenize_chinese` 用空格分词，中文无空格 → 整段文本变成 1 个 token → 不重叠 → 分数为 0

修复：改用 jieba 精确模式分词 + 停用词过滤。修复后分数：faithfulness 0.36, relevance 0.81, context_relevance 0.22, context_precision 0.48

### Faithfulness 提升：强化 RAG Prompt 忠实度约束

| 维度 | 旧 Prompt | 新 Prompt |
|------|-----------|-----------|
| 约束 | "基于文档回答" | "严格基于【文档】内容回答，不得编造文档中未提及的药物/剂量/治疗方案" |
| 无信息时 | "无相关信息则说明" | "明确告知'根据现有资料无法回答'，不要用自身知识补充" |
| 引用约束 | 无 | "引用药物/剂量时，必须与文档原文一致" |

### RAGAS 评估模块

| 问题 | 回答 |
|------|------|
| 需要魔法？ | 不需要——评估 LLM 用智谱 API（`get_llm()`），不用 OpenAI |
| 免费？ | RAGAS 框架免费（MIT），LLM API 调用少量成本（~¥0.04/10条） |
| 评估模式 | RAGAS（`pip install ragas`）/ 规则引擎（默认，jieba 分词） |

### Bug2：自包含查询跳过重写但没跳过 HyDE（5180ms 白跑）

"便秘怎么办？"：重写正确跳过，但 HyDE 未跳过（HyDE 跳过条件与重写条件独立）

修复：自包含查询统一跳过重写 + HyDE

### 决策：HyDE 基于实测数据默认关闭

**A/B 测试结果（10 条典型医疗查询）**：

| 指标 | 无 HyDE | 有 HyDE | 差异 |
|------|---------|---------|------|
| 平均 Recall | 56.7% | 43.3% | **-13.3%** |
| 平均耗时 | 1473ms | 3048ms | +1574ms |
| Recall 正向 | — | 2 条 | — |
| Recall **负向** | — | **4 条** | — |
| 文档重叠度 | 10% | — | 几乎完全不同 |

**结论**：HyDE 在当前架构下为负收益组件。原因：
1. 规则引擎 + 症状解析已做查询标准化，填平了"查询-文档语义鸿沟"
2. 现代中文 Embedding（bge/m3e）语义理解已足够强
3. 混合检索（BM25 + Dense）精确匹配不依赖语义桥接
4. Ollama 1.5b 本地模型生成质量有限，假想答案含噪反而污染检索

**处置**：默认关闭（`ENABLE_HYDE=False`），保留开关供未来长尾模糊查询按需启用---|------|
| `app/graph/nodes/nodes.py` | `_ANAPHORA_PATTERNS` 移除多义字；`_DOMAIN_ENTITY_KEYWORDS` 新增 7 个症状关键词；HyDE 跳过逻辑改为自包含判断 |

---

## v8.4 - 症状解析架构瘦身 + TTFT 优化（6.3s → 2.7s）

### 核心设计决策：移除症状解析 LLM 兜底

**问题**：症状解析和查询改写做的是同一件事（理解查询医学语义），但分别调用 LLM，导致：
- 症状解析 LLM 兜底 2.8s，结果常校验失败（如 severity="轻至中度"），白跑
- 查询改写对自包含查询跳过，但 HyDE 仍执行 734ms
- 两次 LLM 调用做重复劳动，合计 3.5s

**决策**：症状解析只保留规则引擎（<5ms），移除 LLM 兜底。规则未命中时降级为原始查询检索，下游完全能处理。

**理由**：
1. symptoms=None 时：`_build_followup_hints()` 返回空，RAG prompt 不追加追问，LLM 照常生成答案
2. 时间锚定已独立于症状关键词（新增 `_extract_time_grounding()`），规则未命中时 onset_ts 仍能计算
3. 查询改写节点本身就是 LLM 语义理解的入口，不需要症状解析重复做
4. 词表是性能优化手段，不是功能完整性保障——Top 200 高频症状覆盖 80% 场景即可

### 改动1：症状解析移除 LLM 兜底（2809ms → <5ms）

```
旧架构：
  规则命中(<5ms) → 返回 ✅
  规则未命中 → LLM提取(2809ms) → 校验失败 → None ❌

新架构：
  规则命中(<5ms) → 返回 ✅
  规则未命中 → 时间锚定(<5ms) + symptoms=None → 降级为原始查询检索 ✅
```

### 改动2：时间锚定独立于症状解析

`_extract_time_grounding(question)` 从 `_extract_symptoms_by_rules` 中提取，
不再依赖症状关键词命中。即使规则未命中"流鼻血"，"3天前开始流鼻血"的 `onset_ts` 仍能正确计算。

### 改动3：症状关键词覆盖增强

新增：流鼻血、鼻出血、鼻流血、便血、咯血、尿血、血尿

### 改动4：HyDE 短路条件补全（734ms → 0ms）

`_hyde_symptom_words` 新增"流血"、"血"，修复"流鼻血了怎么办？"未触发 HyDE 跳过。

### TTFT 改善

| 节点 | 修复前 | 修复后 | 节省 |
|------|--------|--------|------|
| 症状解析 | 2809ms（LLM白跑） | <5ms（规则/降级） | 2804ms |
| HyDE | 734ms（未跳过） | 0ms（跳过） | 734ms |
| 知识检索 | 1559ms | 1559ms | — |
| LLM 首token | ~1042ms | ~1042ms | — |
| **TTFT** | **6269ms** | **~2700ms** | **3538ms（-56%）** |

### 修改文件

| 文件 | 改动 |
|------|------|
| `app/graph/nodes/nodes.py` | `symptom_analysis_node` 移除 LLM 兜底；新增 `_extract_time_grounding()`；`_hyde_symptom_words` 补充"流血""血" |
| `app/core/keyword_matcher.py` | 症状映射新增 7 个关键词 |

### 后续优化方向

| 方向 | 预期收益 | 复杂度 |
|------|----------|--------|
| 本地 Embedding 模型 | 730ms → <50ms | 中（需 GPU 内存） |
| 症状解析与检索并行 | 串行 4.3s → 并行 2.8s | 高（需重构节点依赖） |
| Reranker max_length=256 | 777ms → ~400ms | 低 |

---

## v8.3 - 邻域扩展（Sibling Expansion）：跨章节信息补全 + 幻觉消除

### 背景

用户问"头痛怎么办？"，答案天然分布在"危险信号（排除禁忌）"+"药物选择（治疗）"+"非药物治疗（辅助）"等多个 Parent 中。但 Reranker 只返回了 1 个 Parent（"头痛危险信号"），而"头痛的药物选择"（含布洛芬等药物信息）不在检索候选中（Dense 排名 #50+）。

LLM 拿到不含药物的文档，被迫靠自身知识编造布洛芬/对乙酰氨基酚 → 幻觉检测误报。

根因：Embedding 对"怎么办"和"药物选择"的语义映射不够，Dense 召回不了跨章节的相关内容。这不是调参能解决的——调大 top_k 从 6 到 50 不现实。

### 解决方案：Parent 邻域扩展

利用 Markdown 文档结构，检索命中后自动拉取同一文档中的相邻章节：

```
Reranker 返回：Parent section=5 "头痛危险信号"
    ↓ 邻域扩展 window=1
自动拉取：Parent section=4 "头痛的药物选择"（含布洛芬）  ← 补足！
自动拉取：Parent section=6 "头晕"
    ↓ 合并去重
注入 LLM：3 个 Parent（含药物选择）→ 无幻觉
```

### 技术栈对比

| 维度 | 旧方案（v8.2） | 新方案（v8.3） |
|------|---------------|---------------|
| 检索粒度 | 仅 Reranker 返回的 Parent | Parent + 相邻兄弟章节 |
| 跨章节信息 | 缺失 → LLM 编造 → 幻觉 | 邻域扩展自动补全 |
| 新增元数据 | 无 | `doc_id`（所属文档）+ `section_index`（章节序号） |
| 字符上限 | 无 | `MAX_SIBLING_CHARS=2000`（防撑爆 LLM 上下文） |
| 额外延迟 | — | <1ms（内存查找） |
| 幻觉检测 | 误报（文档不含布洛芬） | 正确（扩展后含布洛芬） |

### 实测效果

```
查询"头痛怎么办？"
  Reranker 返回：1 个 Parent（section=5，头痛危险信号）
  邻域扩展：1 → 3 个（+2 个兄弟章节），总字符数=432

  [1] 🔍 原始  | section=5 | "头痛危险信号（红旗征）"          ← Reranker 召回
  [2] 📱 扩展 💊 | section=4 | "头痛的药物选择"（含布洛芬）   ← 邻域扩展
  [3] 📱 扩展  | section=6 | "头晕"                           ← 邻域扩展
```

### 配置项

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `SIBLING_WINDOW` | 1 | 邻域窗口大小，1=前后各取1个章节 |
| `MAX_SIBLING_CHARS` | 2000 | 扩展后最大总字符数 |

### 修改文件

| 文件 | 改动 |
|------|------|
| `app/rag/parent_child_store.py` | 新增 `expand_with_siblings()` 方法；`build_index()` 写入 `doc_id`+`section_index` 元数据；持久化格式升级（含章节索引） |
| `app/rag/hybrid_retriever.py` | parent 映射后调用 `expand_with_siblings()` |
| `app/core/config.py` | 新增 `SIBLING_WINDOW`、`MAX_SIBLING_CHARS` 配置项 |
| `scripts/rebuild_vector_store.py` | 重建前删除旧 parent_store.pkl，避免旧数据叠加 |

### 兼容性

- 旧版 `parent_store.pkl`（无 `doc_id`/`section_index`）自动降级：`_rebuild_index_from_store()` 从 Parent 元数据中重建索引
- 无章节索引时跳过邻域扩展，不影响现有功能

---

## v8.2 - 严重 Bug 修复：Dense 检索分数永远为 0.0 → Reranker 被误跳过 → 召回错误文档 + 幻觉检测误报

### 背景

用户查询"头痛怎么办？"，系统返回了**皮肤疾病诊疗指南**的内容，而非**神经系统症状鉴别指南**（包含头痛药物选择）。LLM 拿到无关文档后自由编造了布洛芬/曲马多等药物名，触发幻觉检测。但幻觉检测也误报了——文档中实际包含布洛芬等药物。

日志追踪：
```
top1_dense_dist=0.0000        ← 看似"完美匹配"，实际是 Bug
Dense Top-1 置信度极高（distance=0.0000 < 0.08），跳过重排  ← Reranker 被误跳过
文档启发式过滤结果：3 -> 1 相关   ← 神经系统文档也被误过滤
疑似幻觉检测：答案提到 {'布洛芬', '曲马多', '对乙酰氨基酚'}，但检索文档中未出现  ← 误报
```

### 根因 1：`similarity_search_by_vector_with_score` 方法不存在

| 维度 | 旧代码 | 实际 |
|------|--------|------|
| 调用方法 | `Chroma.similarity_search_by_vector_with_score()` | **此方法不存在！** |
| `hasattr` 检查 | `if hasattr(...)` → `False` → 跳过 | 永远跳过带分数的分支 |
| 回退路径 | `similarity_search_by_vector()` → 无分数 | `top1_score` 保持默认值 `0.0` |
| High-Confidence Bypass | `0 <= 0.0 < 0.08` = `True` → 跳过 Reranker | **每次请求都跳过 Reranker！** |

**langchain_chroma.Chroma 的正确方法名**：
- `similarity_search_with_score` → 文本查询 + ChromaDB distance ✅
- `similarity_search_by_vector_with_relevance_scores` → 向量查询 + 归一化分数 ✅
- ~~`similarity_search_by_vector_with_score`~~ → **不存在** ❌

### 根因 2：幻觉检测读取过滤后的文档

| 维度 | 旧实现 | 新实现 |
|------|--------|--------|
| 检测依据 | `state["retrieved_docs"]`（过滤后） | `state["all_retrieved_docs"]`（过滤前） |
| 问题 | grade_documents_node 过滤后只剩 1 篇不相关文档 | 全量文档包含所有检索结果 |
| 后果 | 布洛芬在神经系统文档中，但该文档已被过滤 → 误报幻觉 | 正确识别布洛芬在检索文档中存在 |

### 根因 3：`has_query_overlap` 口语虚词误判

| 维度 | 旧实现 | 新实现 |
|------|--------|--------|
| 查询 "头痛怎么办？" | tokens = ["头痛", "怎么办"] | tokens = ["头痛"]（过滤"怎么办"虚词） |
| 神经系统文档 | "头痛"匹配1个，"怎么办"不匹配 → match=1 → `1 >= 1.5` = **False** | "头痛"匹配1个 → `1 >= 1` = **True** ✅ |
| 后果 | 含"头痛"的正确文档被误过滤掉 | 正确保留 |

### 修复 1：三层回退策略获取真实 distance

| 优先级 | 策略 | 方法 | 返回值 |
|--------|------|------|--------|
| 1 | 底层 collection 直接查询 | `col.query(query_embeddings=..., include=["distances"])` | ChromaDB cosine distance ✅ |
| 2 | LangChain 向量查询 | `similarity_search_by_vector_with_relevance_scores` | 归一化分数 → 转换为 distance |
| 3 | 无分数兜底 | `similarity_search_by_vector` | 默认值 1.0（不触发 Bypass） |

### 修复 2：`top1_score` 默认值 0.0 → 1.0

| 维度 | 旧实现 | 新实现 |
|------|--------|--------|
| 默认值 | `0.0` | `1.0` |
| 异常时返回 | `([], 0.0)` → 触发 Bypass | `([], 1.0)` → 不触发 Bypass |

### 修复 3：幻觉检测使用过滤前文档 + 口语虚词过滤

- 新增 `all_retrieved_docs` state 字段，保存知识检索节点的完整结果
- 幻觉检测优先读 `all_retrieved_docs`
- `has_query_overlap` 增加 `ORAL_FILLERS` 过滤"怎么办/怎么样/好不好"等口语虚词
- 过滤后只需 1 个实质关键词命中即判定相关

### 修复后 Dense 检索实际排名（查询"头痛怎么办？"）

| 排名 | distance | 文档 | Reranker 执行 |
|------|----------|------|--------------|
| 1 | 0.4212 | 皮肤疾病诊疗指南 | ✅ 正常执行 |
| 5 | 0.4646 | 神经系统症状鉴别指南 | ✅ Reranker 可将其提升至 Top-3 |

### 修改文件

| 文件 | 改动 |
|------|------|
| `app/rag/hybrid_retriever.py` | `_dense_search()` 重写：三层回退策略获取真实 distance；默认值 0.0→1.0 |
| `app/graph/state.py` | 新增 `all_retrieved_docs` 字段 |
| `app/graph/nodes/nodes.py` | 知识检索节点保存 `all_retrieved_docs`；`has_query_overlap` 口语虚词过滤 |
| `app/graph/streaming.py` | 幻觉检测优先读 `all_retrieved_docs` |

### 验证方式

```bash
D:\Agent\software\envs\my_medical_env\python.exe scripts\diagnose_dense.py
```

---

## v8.1 - 紧急修复：Embedding 超时失效 + High-Confidence Bypass 误排除完美匹配

### 背景

日志分析发现单次查询耗时 **62 秒**，其中 Embedding API 调用占 60 秒：
```
query_embedding=60177.97ms    ← 60秒！Embedding API 一个调用
top1_dense_dist=0.0000        ← 完美匹配，但没触发 High-Confidence Bypass（注：此0.0实际是v8.2修复的Bug）
rerank=2007.90ms              ← 本可跳过，浪费2秒
总耗时=62337.51ms
```

### 修复 1：Embedding API 超时失效（60s → 10s 强制超时）

| 维度 | 旧实现 | 新实现 |
|------|--------|--------|
| 超时参数 | `request_timeout=10` | `httpx.Timeout(connect=5, read=10, write=10, pool=5)` |
| 实际生效 | ❌ 新版 openai>=1.0 已弃用 `request_timeout`，回退到默认 600s | ✅ 显式 `httpx.Timeout` 强制各阶段超时 |
| 60s 卡顿 | API 响应慢时阻塞 60 秒无超时 | 最长 10s 超时 + 1 次重试 = 最多 20s |
| 兼容性 | 无 | 旧版 langchain-openai 不支持 `timeout` 参数时自动回退 `request_timeout=15` |

**改动**：[embeddings.py](file:///d:/Agent/medical_assistant_agent/app/core/embeddings.py)

### 修复 2：High-Confidence Bypass 误排除完美匹配（distance=0.0）

| 维度 | 旧实现 | 新实现 |
|------|--------|--------|
| 条件 | `top1_dense_score > 0 and top1_dense_score < 0.08` | `0 <= top1_dense_score < 0.08` |
| distance=0.0 | ❌ `0.0 > 0` = False，不触发跳过 | ✅ `0 <= 0.0` = True，正确跳过 |
| 后果 | 完美匹配仍跑 Reranker（浪费 2s） | 完美匹配直接跳过 Reranker |

**根因**：ChromaDB cosine distance 中 `0.0` 是完美匹配（两向量完全一致），但旧代码 `> 0` 把它排除了。正确逻辑：`distance ∈ [0, 0.08)` 都是高置信度，应跳过 Reranker。

**改动**：[hybrid_retriever.py](file:///d:/Agent/medical_assistant_agent/app/rag/hybrid_retriever.py#L276)

### 修复后预期

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| Embedding API 超时 | 60s+ (无超时) | ≤10s (强制超时) |
| 完美匹配时 Reranker | 仍执行 (2s) | 跳过 (0ms) |
| 理想 TTFT | 62s | ≤3s (API正常时 ~500ms embedding + 跳过 reranker) |

---

## v8.0 - Skills 体系扩展 + Bad Case 回归测试闭环 + RAGAS 评估重写

### 背景

项目已具备基础的医疗安全审查 Skill、Bad Case 自动采集和 RAGAS 评估模块，但三个方向均存在明显短板：
- **Skills**：仅有 `medical_safety_review` 1 个 Skill，用药指导和症状分诊等高频场景无覆盖
- **Bad Case**：自动采集（4 触发点）+ 导出脚本 + 人工审核 API 已有，但缺回归测试运行器，无法验证修复效果
- **RAGAS 评估**：基础代码存在但硬编码 `gpt-4o`、兼容层冗余、无增量评估、无版本对比、未接入 bad case 数据

### 改动 1：新增 2 个结构化 Skill（Anthropic 范式）

#### 1.1 用药指导 Skill（`medication_guide`）

| 维度 | 说明 |
|------|------|
| Trigger | 用户问题涉及药物用法用量、相互作用、禁忌人群时触发 |
| Workflow | 药物实体识别 → 禁忌人群交叉检查 → 相互作用初筛 → 用量安全验证 → 规范性校验 |
| 输出 | `{status: pass/revise, revised_answer, risk_tags}` |

规则引擎覆盖：
- **药物实体识别**：AC 自动机匹配 + 正则兜底（剂型后缀"XX片""XX胶囊"）
- **禁忌人群交叉检查**：6 类人群（孕妇/儿童/老年/肝肾功能不全/消化道溃疡）与 4 种常见药物安全规则交叉
- **药物相互作用**：布洛芬↔阿司匹林/华法林等已知高风险组合
- **用量安全**：检测回答中剂量是否超过每日上限（如布洛芬 1200mg）
- **5 字段完整性**：适应症/用法用量/注意事项/禁忌/如症状持续请就医

#### 1.2 症状分诊 Skill（`symptom_triage`）

| 维度 | 说明 |
|------|------|
| Trigger | 路由为 `symptom_analysis` 时触发 |
| Workflow | 紧急度分级 → 危险症状组合检测 → 持续时间评估 → 分诊建议生成 |
| 输出 | `{status: pass/revise, triage_result, advice_text, risk_tags}` |

分诊等级：
- 🔴 紧急（立即就医/120）：胸痛+呼吸困难、意识不清、大出血等 12 类
- 🟡 建议就诊（48h 内）：症状 >72h 无缓解、反复发作、用药无改善
- 🟢 居家观察：轻微症状、首次出现、无危险信号

危险症状组合（5 种）：
- 头痛+发热+颈僵 → 脑膜炎高风险
- 胸痛+呼吸困难+冷汗 → 心梗高风险
- 头痛+呕吐+视力模糊 → 颅内压增高
- 发热+皮疹+呼吸困难 → 严重过敏反应
- 腹痛+发热+呕吐 → 急腹症

### 改动 2：Bad Case 回归测试闭环

| 维度 | 旧方案 | 新方案 |
|------|--------|--------|
| 采集 | ✅ 4 触发点自动采集 | ✅ 不变 |
| 导出 | ✅ JSONL 导出脚本 | ✅ 不变 |
| 人工审核 | ✅ API + PostgresStore | ✅ 不变 |
| **回归测试** | ❌ 无 | ✅ `BadCaseRegressionRunner` |
| **统计报告** | ❌ 无 | ✅ 按类型/通过率/失败分布 |
| **CLI 运行** | ❌ 无 | ✅ `scripts/run_bad_case_regression.py` |

回归测试流程：
```
加载 JSONL 测试集 → 构建 state → 调用 query_rewrite_node → 
模糊匹配对比(三级) → 生成统计报告 → 保存
```

模糊匹配三级策略（任一通过即算 PASS）：
1. 归一化精确匹配（去标点/空格后相等）
2. 子串包含（期望是实际的子串）
3. 核心实体覆盖（期望中的药物/症状词全部在实际中出现）

### 改动 3：RAGAS 评估模块重写

| 维度 | 旧方案 | 新方案 |
|------|--------|--------|
| 评估 LLM | 硬编码 `gpt-4o` | 项目配置 `get_llm()` |
| RAGAS 兼容 | 逐指标 try/except 冗长链 | 顶层一次性导入，仅支持 `>=0.1` |
| 降级策略 | 无（报错退出） | 规则引擎简易评估（关键词覆盖率） |
| 增量评估 | ❌ 每次全量重跑 | ✅ 按 question 去重跳过已评估 |
| 版本对比 | ❌ 无 | ✅ A/B 对比 + delta + 改进方向 |
| 测试集 | 内嵌 5 条 | JSONL 文件 10 条（4 类场景） |
| Bad Case 接入 | ❌ 不支持 | ✅ 自动识别 bad case 格式 |

四维评估指标：
- **Faithfulness（忠实度）**：答案是否基于检索上下文
- **Answer Relevance（答案相关性）**：答案是否切题
- **Context Precision（上下文精确度）**：检索结果中相关文档的比例
- **Context Relevance（上下文相关性）**：检索结果与问题的相关程度

规则引擎降级指标（ragas 未安装时）：
- `rule_based_faithfulness`：答案关键词在上下文中的覆盖率
- `rule_based_relevance`：问题关键词在答案中的覆盖率
- `rule_based_context_relevance`：问题关键词在上下文中的覆盖率
- `rule_based_context_precision`：上下文中与问题相关文档的比例

### 新增文件

| 文件 | 说明 |
|------|------|
| `app/skills/medication_guide.md` | 用药指导 Skill 定义（Anthropic 范式） |
| `app/skills/medication_guide_engine.py` | 用药指导规则引擎 |
| `app/skills/symptom_triage.md` | 症状分诊 Skill 定义（Anthropic 范式） |
| `app/skills/symptom_triage_engine.py` | 症状分诊规则引擎 |
| `app/evaluation/__init__.py` | 评估模块入口 |
| `app/evaluation/bad_case_runner.py` | Bad Case 回归测试运行器 |
| `scripts/run_bad_case_regression.py` | Bad Case 回归测试 CLI |
| `tests/data/rag_eval_test_set.jsonl` | RAGAS 评估测试集（10 条/4 类场景） |

### 修改文件

| 文件 | 改动 |
|------|------|
| `app/skills/__init__.py` | 新增用药指导 + 症状分诊导出 |
| `app/rag/evaluation.py` | 重写：项目 LLM + 增量评估 + 版本对比 + bad case 接入 |
| `scripts/evaluate_rag.py` | 重写：新参数 + 版本对比 CLI |

### 使用方式

```bash
# Bad Case 回归测试
python scripts/run_bad_case_regression.py
python scripts/run_bad_case_regression.py --test-set tests/data/custom.jsonl --verbose

# RAGAS 评估
python scripts/evaluate_rag.py
python scripts/evaluate_rag.py --test-set tests/data/bad_cases_export.jsonl
python scripts/evaluate_rag.py --compare data/evaluation/eval_v1.json data/evaluation/eval_v2.json
```

---

## v7.0 - 系统性代码质量审计：P0/P1 缺陷修复 + AC 自动机引擎

### 背景

对项目所有模块进行了系统性审查，识别出 30 个"朴素实现可用更优算法替代"的问题（类似"300 字符截断"模式）。按优先级从 P0 开始修复，本次修复 4 项。

### 修复 1：warnings 字段覆盖→累积（P0 Bug）

| 维度 | 旧实现 | 新实现 |
|------|--------|--------|
| 声明 | `warnings: List[str]  # 覆盖警告信息` | `warnings: Annotated[List[str], add]` |
| 行为 | 后序节点的 warnings 覆盖前序节点 | 多个节点的 warnings 自动合并累积 |
| Bug | `knowledge_retrieval_node` 的检索警告被 `safety_check_node` 覆盖丢失 | 所有节点的 warnings 都保留 |

**改动**：[state.py](file:///d:/Agent/medical_assistant_agent/app/graph/state.py#L65) 1 行

### 修复 2：onset_dates 合并语义错误→取最早 ts（P0 Bug）

| 维度 | 旧实现 | 新实现 |
|------|--------|--------|
| 合并方式 | `{**a, **b}` 简单覆盖 | `_merge_onset_dates(a, b)` 深度合并 |
| 同一症状 | 后值覆盖前值（L2 覆盖 L1） | 取 `ts` 更小（更早首发）的记录 |
| Bug | L1 记录"头痛首次出现在3天前"，L2 记录"头痛出现在1天前"→ 1天前覆盖3天前 | 保留3天前（更早的首发时间） |

**改动**：[nodes.py](file:///d:/Agent/medical_assistant_agent/app/graph/nodes/nodes.py#L2074-L2104) 新增 `_merge_onset_dates()` 函数，替换 2 处简单字典合并

### 修复 3：关键词匹配→AC 自动机（P1 性能+精度）

| 维度 | 旧实现 | 新实现 |
|------|--------|--------|
| 算法 | `any(keyword in text for keyword in keywords)` 线性扫描 | Aho-Corasick 自动机 O(m) 一次扫描 |
| 复杂度 | O(n×m)（n=关键词数，m=文本长度） | O(m)（与关键词数无关） |
| 误匹配 | "心疼"命中"疼"→误判为症状 | AC 自动机 + 边界检测消除子串误匹配 |
| 关键词维护 | 5 个文件各自维护一份列表 | 集中式 `keyword_matcher.py`，单一真相源 |
| 降级策略 | 无 | pyahocorasick 未安装时自动降级为线性扫描 |

**受影响的模块**：

| 模块 | 旧方式 | 新方式 |
|------|--------|--------|
| `detect_rule_based_route` | 2 个关键词列表 × 线性扫描 | `get_route_symptom_matcher()` / `get_route_knowledge_matcher()` |
| `_extract_symptoms_by_rules` | 70+ 条 `symptom_map` × 线性遍历 | `get_symptom_matcher().get_matched_keywords()` |
| `_build_rewrite_context` | 2 个内联关键词集合 × 线性扫描 | `get_drug_matcher()` / `get_symptom_matcher()` |
| `_check_emergency_risks` | 双向子串 `emerg in sym or sym in emerg` | `get_emergency_matcher().contains_any()` |

**新增文件**：[keyword_matcher.py](file:///d:/Agent/medical_assistant_agent/app/core/keyword_matcher.py) — AC 自动机引擎 + 5 个集中式匹配器构建器

**新增依赖**：`pyahocorasick`（纯 C 实现，<1MB，自动降级）

### 修复 4：语义缓存伪 LRU→真 LRU（P1 正确性）

| 维度 | 旧实现 | 新实现 |
|------|--------|--------|
| 数据结构 | Redis Set（无序） | Redis Sorted Set（score=访问时间戳） |
| 淘汰策略 | `list(all_keys)[:20%]` → 本质是随机淘汰 | `ZRANGE` 按 score 升序 → 淘汰最久未访问的 |
| 访问刷新 | 无（命中不更新顺序） | 命中时 `ZADD` 更新 score → 最近访问的排在后面 |
| 与注释一致性 | 注释说"LRU"但实际是随机淘汰 ❌ | 真正的 LRU 行为 ✅ |

**改动**：[semantic_cache.py](file:///d:/Agent/medical_assistant_agent/app/cache/semantic_cache.py)
- 新增 `_keys_zset` Sorted Set 索引
- `set()` 写入时 `ZADD {key: timestamp}`
- `get()` 命中时 `ZADD` 刷新时间戳（读时刷新）
- 淘汰时 `ZRANGE` 按 score 升序取最早的 20%
- `clear()` 同时清理 ZSET

### 修改文件清单

| 文件 | 改动类型 |
|------|---------|
| `app/graph/state.py` | Bug 修复：warnings 覆盖→累积 |
| `app/graph/nodes/nodes.py` | Bug 修复：onset_dates 深度合并；AC 自动机集成（4 处） |
| `app/core/keyword_matcher.py` | 新增：AC 自动机引擎 + 5 个集中式匹配器 |
| `app/skills/safety_review_engine.py` | 优化：紧急症状检测使用 AC 自动机 |
| `app/cache/semantic_cache.py` | 修复：伪 LRU→真 LRU（Sorted Set + 读时刷新） |

---

## v6.2 - 父子索引（Parent-Child Index）：检索精度 + 上下文完整性双提升

### 背景

用户提问"布洛芬的用法用量"，文档中包含完整答案但系统未能正确回答。根因：`build_rag_prompt` 将文档盲目截断到 300 字符，导致上下文丢失。提高截断限制（如 500）只是推迟问题——如果信息在第 800 字符处仍然会丢失。

### 核心改动：Parent-Child Index 架构

```
旧架构（盲目截断）：
  文档 400 字符 → 截断到 300 字符 → 送入 LLM（可能丢失关键信息）

新架构（父子索引）：
  Parent（完整章节 ~400 字符）→ 切分为 Child（~150 字符）
  Child 写入向量库 + BM25（精准匹配）
  Child 经 Reranker 重排（序列更短 = 推理更快）
  → 取回完整 Parent → 送入 LLM（无需截断）
```

### 新增文件

| 文件 | 说明 |
|------|------|
| `app/rag/parent_child_store.py` | ParentChildManager：父子索引管理器（InMemoryStore + 磁盘持久化） |

### 修改文件

| 文件 | 改动 |
|------|------|
| `app/rag/hybrid_retriever.py` | Reranker 后增加 `parent_manager.get_parents()` 映射；BM25 缓存兼容检测（无 parent_id 视为旧缓存重建） |
| `app/graph/nodes/nodes.py` | `build_rag_prompt` 移除 300 字符截断，仅对 >2000 字符做安全兜底 |
| `app/rag/__init__.py` | 导出 ParentChildManager |
| `scripts/rebuild_vector_store.py` | 重建脚本支持父子索引：Parent 切分 → Child 入库 → Parent 持久化 |

### 技术栈对比

| 维度 | 旧方案 | 新方案 |
|------|--------|--------|
| 索引粒度 | 单层（大 chunk ~400 字符） | 双层（Parent ~400 + Child ~150） |
| 检索精度 | 大 chunk 语义模糊 | Child 小块精准匹配 |
| 上下文完整性 | 截断到 300 字符 | 完整 Parent 注入 |
| Reranker 性能 | 5 篇 × 300 字符 = ~969ms | 5 篇 × ~80 字符 = 预期 ~400ms |
| 信息丢失风险 | 高（第 301+ 字符丢失） | 无（Parent 完整保留） |

### 使用方式

```bash
# 重建向量库（自动构建父子索引）
python scripts/rebuild_vector_store.py
```

### 兼容性

- 未重建索引时，系统自动降级为旧模式（child 文档无 parent_id → 跳过 parent 映射）
- BM25 缓存自动检测旧版数据并重建

---

## v6.1 - 性能修复：Dense 检索 759ms → ~100ms（三个根因修复）

### 背景

日志分析发现 Dense 检索耗时 759ms（244 篇文档的向量库，正常应 <100ms）。排查发现三个叠加问题导致 Chroma 被重复实例化、Embedding API 被重复调用。

### 三个根因与修复

| 根因 | 文件 | 问题 | 修复 |
|------|------|------|------|
| Chroma 重复实例化 | `vector_store.py` | `create_vector_store()` 每次调用都新建 Chroma 实例，即使已存在 | 增加 `if self.vector_store is not None and not force_rebuild: return self.vector_store` |
| lru_cache 参数不匹配 | `routes.py` | 启动预热 `get_hybrid_retriever()`（k=5, rerank_top_k=10）与搜索节点 `get_hybrid_retriever(k=3, rerank_top_k=5)` 参数不同 → 缓存未命中 → 新建 HybridRetriever → 触发 Chroma 重新加载 | 启动预热改为 `get_hybrid_retriever(k=3, alpha=0.5, use_reranker=True, rerank_top_k=5)` |
| Embedding 未预计算 | `hybrid_retriever.py` | `elif query_embedding is None` 在 L2 缓存开启时永远不会执行（`if` 条件已为 True）→ `query_embedding` 为 None → Chroma 内部调 Embedding API（~200-300ms） | `elif` 改为 `if`，L2 缓存为空时仍计算 embedding 供 Dense 复用 |

### 修复前后对比

| 指标 | 修复前 | 修复后（预期） |
|------|--------|--------------|
| Dense 检索耗时 | 759ms | ~100ms |
| Chroma 实例化 | 每次请求重新加载 | 启动时加载一次，后续复用 |
| Embedding API 调用 | Chroma 内部调用（不可控） | 预计算后传入 Chroma（可复用） |
| BM25 缓存加载 | 每次请求从磁盘重新加载 | lru_cache 命中后跳过 |

### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `app/rag/vector_store.py` | `create_vector_store()` 增加实例缓存判断 |
| `app/api/routes.py` | 启动预热参数与搜索节点一致（k=3, rerank_top_k=5） |
| `app/rag/hybrid_retriever.py` | `elif query_embedding is None` → `if query_embedding is None` |

---

## v6.0 - Skill 增强：医疗合规与安全审查（结构化 Prompt 范式）

### 背景

项目原有的 `safety_check_node` 仅调用 LLM 做简单风险评估，追加 warnings，不修改回答内容。在医疗场景中，大模型即使拿到正确文档，仍可能在生成阶段出现超适应症建议、诊断性断言或遗漏紧急就医指引。本次升级基于 Anthropic Skill 范式（结构化 Prompt），将安全审查从"附加警告"升级为"三态决策阀门"。

### Skill 定义

新增 [medical_safety_review.md](file:///d:/Agent/medical_assistant_agent/app/skills/medical_safety_review.md)，按 Anthropic Skill 范式定义五大模块：
- 🎯 Trigger：答案生成后、缓存写入前自动触发
- ⚙️ Workflow：5 步审查流程（诊断断言→用药安全→紧急风险→免责声明→决策输出）
- 📤 Output：{status: pass|revise|block, revised_answer, risk_tags}
- 🛡️ Guardrails：禁止生成新医学建议、800ms 超时兜底、规则引擎优先

### 技术栈对比

| 维度 | 旧方案（v5.x） | 新方案（v6.0） |
|------|---------------|---------------|
| 审查架构 | 仅 LLM 单步审查 | 规则引擎（0ms）+ LLM 深度审查（高风险时触发） |
| 审查决策 | 仅追加 warnings | 三态决策：pass（透传）/ revise（修订）/ block（拦截） |
| 诊断性断言 | 未检测 | 10 类正则模式检测 + 自动替换为风险提示句式 |
| 紧急风险 | 未关联快照 | 交叉检查 clinical_checkpoint 中的危急重症信号 |
| 免责声明 | 固定追加 | 检测缺失后自动注入 |
| 流式集成 | 未集成 | 流式结束后执行审查，修订时发送 safety_revision SSE 事件 |
| 缓存保护 | 无 | block 的回答不写入 Redis，防止污染缓存 |

### 审查流程

```
答案生成 → [规则引擎 0ms] → 无风险 → pass → 缓存写入
                         ↓ 有风险
                    [LLM 深度审查] → revise → 修订后缓存 + 发送修正 SSE
                                  → block → 不缓存 + 返回安全拒答模板
```

### 新增文件

| 文件 | 说明 |
|------|------|
| `app/skills/__init__.py` | Skills 模块入口 |
| `app/skills/medical_safety_review.md` | 结构化 Skill 定义（Anthropic 范式） |
| `app/skills/safety_review_engine.py` | 规则引擎：诊断断言检测 + 紧急风险拦截 + 免责声明注入 |

### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `app/graph/nodes/nodes.py` | `safety_check_node` 重写：规则引擎 → LLM 深度审查 → 三态决策 |
| `app/graph/streaming.py` | 新增 `_run_safety_review()` 方法；所有答案路径（direct/vision/RAG/cached_docs）流式结束后执行安全审查，block 不缓存 |

### 设计要点

1. **规则引擎优先**：0ms 正则检测诊断性断言、紧急症状、免责声明，仅高风险时才触发 LLM
2. **三态决策**：pass（透传）/ revise（自动替换诊断断言 + 注入紧急提示 + 补全免责声明）/ block（返回安全引导模板）
3. **流式安全修正**：流式输出已发送给用户后，如审查发现风险，发送 `safety_revision` SSE 事件推送修正
4. **缓存保护**：block 的回答不写入 Redis，避免错误回答被缓存后持续返回
5. **临床快照关联**：审查时读取 `clinical_checkpoint`，检测用户症状快照中的危急重症信号是否在回答中被遗漏

---

## v5.7 - 修复：LangGraph BaseStore.search() namespace_prefix 位置参数兼容

### 问题

日志持续输出警告：
```
从L1加载症状首发时间失败（不影响主流程）：
BaseStore.search() got some positional-only arguments passed as keyword arguments: 'namespace_prefix'
```

LangGraph 新版中 `BaseStore.search()` 的 `namespace_prefix` 参数改为**位置参数**（positional-only），
不能再以关键字形式传递。`long_term_memory.py` 中 5 处 `store.search(namespace_prefix=xxx)` 全部报此警告。

虽说不影响主流程（try/except 兜底），但症状历史、用药记录、查询记录等 L1 数据实际未被加载，
影响症状快照继承和上下文补全的准确性。

### 修复

5 处调用全部改为位置参数：
```python
# 修复前
items = self.store.search(namespace_prefix=("symptom_history", user_id))

# 修复后
items = self.store.search(("symptom_history", user_id))
```

涉及方法：`get_symptom_history`、`get_query_history`、`get_symptom_events`、`get_bad_cases`、`get_medication_events`

### 修改文件

- `app/memory/long_term_memory.py` — 5 处 `store.search()` 调用 `namespace_prefix=` → 位置参数

---

## v5.6 - 查询重写切换本地模型 + 前端新会话按钮

### 1. 查询重写 LLM 从远端 API 切换到本地模型

`query_rewrite_node` 的重写和 HyDE 生成原来使用 `get_rewrite_llm()`（调用智譜 API，~4.5s），
现改为 `get_local_llm()`，逻辑如下：

```
LOCAL_MODEL_ENABLED=true  → Ollama qwen2.5:1.5b（本地 GPU 推理，<1s）
LOCAL_MODEL_ENABLED=false → 降级回 API（glm-4-flash）
```

- 本地模型不再有网络延迟，重写耗时预计从 4.5s 降至 <1s
- 通过 `.env` 的 `LOCAL_MODEL_ENABLED` 控制切换，无需改代码
- 当 v5.5 的短路逻辑生效（自包含查询跳过重写）时，此切换不影响首轮查询

### 2. 前端"新会话"按钮

**问题**：`thread_id` 留空时自动从 `user_id` 派生（`thread_{user_id}`），
同一用户多次请求会共享 checkpointer 历史，导致自包含查询被误判为追问。

**修复**：在配置栏新增"新会话"按钮，点击后：
- 生成唯一 `thread_id`（`thread_` + UUID 前 8 位）
- 填入 thread_id 输入框
- 清空对话界面，显示"新会话已创建"提示

与"清空对话"的区别：
- 清空对话：仅清 UI，thread_id 不变 → checkpointer 历史继续累积
- 新会话：生成新 thread_id → checkpointer 从零开始

### 修改文件

- `app/graph/nodes/nodes.py` — `query_rewrite_node` 和 HyDE 的 LLM 调用从 `get_rewrite_llm` 改为 `get_local_llm`
- `app/static/index.html` — 新增"新会话"按钮 + `newSession()` 函数

---

## v5.5 - TTFT 优化：自包含查询跳过远端 LLM 重写

### 问题

v5.4 Reranker 修复后 RAG 管道走通，但 TTFT = 7269ms，仍超 5s 目标。
耗时分解：查询重写 4536ms (62%) + 知识检索 1926ms (26%) + 其他 807ms。

问题出在 `query_rewrite_node`：只要 checkpointer 中有历史消息（即使是旧会话残留），
就对**所有**问题调用远端 LLM 重写，包括完全自包含的首轮问题"头痛怎么办？"。

更严重的是，LLM 重写时把历史中的"发烧、持续3天"编入了当前问题，
产生了**幻觉症状**："头痛伴有发烧，持续3天"——实际问题根本没提发烧。

```
代码缺陷：
  if not messages:     ← 跳过重写
  else:                ← 不管问题是否自包含，一律调 LLM！
      llm.invoke()     ← 4536ms，还可能编造症状
```

`_anaphora_detected` 在上方已算出为 `False`（"头痛怎么办？"无指代词），但未被使用。

### 修复

新增 `elif not _anaphora_detected` 分支：有历史但查询自包含 → 跳过重写。

```
修复前：not messages → skip | else → LLM rewrite（4536ms）
修复后：not messages → skip | not anaphora → skip | else → LLM rewrite
```

### 收益

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 查询重写 | 4536ms | **0ms** |
| 重写引入幻觉症状 | ✅ 可能发生 | ❌ 杜绝 |
| TTFT | 7269ms | **~2700ms** |
| 智譜 API 调用次数 | 每请求 1 次 | 仅追问时 1 次（减少 ~70%） |

### 修改文件

- `app/graph/nodes/nodes.py` — `query_rewrite_node` 添加 `elif not _anaphora_detected` 短路分支

---

## v5.4 - 修复：Reranker 双重截断导致文档评分失效

### 问题

v5.3 修复了 ChromaDB 距离度量（L2→cosine），`top1_dense_dist` 从 0.9333 降到 0.4666。
但 Reranker 分数仍为 0.0031（远低于 0.02 阈值），文档评分节点依然判定"无相关文档"跳过 RAG。

根因是 **Reranker 对文档做了双重截断**，大部分内容在到达模型前就被丢弃了：

```
文档 500 字符（含头痛诊断、治疗方案、药物推荐）
  → truncate_for_rerank: 取前 400 字符（丢弃后 100 字符）
  → tokenizer max_length=128: 只取前 ~80 字符（丢弃剩余 320 字符）
  → Reranker 只看到文档的前 80 字符 = 16% 的内容
  → 治疗方案/药物推荐在截断部分 → 评分 0.0031
```

两层截断叠加后，Reranker 实际只看到文档的 **前 ~80 个中文字符**（约 16%），
如果头痛的治疗建议在后半段，模型根本看不到。

### 为什么 max_length=128 对中文太短

| 维度 | 英文 | 中文（BERT tokenizer） |
|------|------|----------------------|
| 1 个 token 覆盖 | 0.75 个单词 | 0.5-0.7 个汉字 |
| 128 tokens 覆盖 | ~96 个单词 | **~65-90 个汉字** |
| 400 字文档覆盖率 | ~100% | **~20%** |

中文在 BERT tokenizer 下每个汉字常被拆为 1-3 个 subword token，
128 token 窗口对英文够用但对中文严重不足。

### 修复

**1. tokenizer max_length：128 → 512**

BGE-reranker 模型上限为 512 tokens，之前仅用了 25%。
512 tokens 可覆盖约 300 个中文字符 + query + 特殊 token。

**2. 截断策略：纯取头 → 头尾各取**

| 维度 | 修复前 | 修复后 |
|------|--------|--------|
| 策略 | `text[:400]` | `text[:200] + text[-100:]` |
| 保留开头（诊断/主题） | ✅ | ✅ |
| 保留结尾（治疗/药物） | ❌ 被丢弃 | ✅ 保留最后 100 字 |
| 文档有效覆盖率 | ~16%（80/500） | ~60%（300/500） |

**3. MAX_RERANK_DOC_CHARS：400 → 300**

300 字中文 ≈ 450-500 tokens，加上 query + 特殊 token 可稳定装入 512 窗口，
避免 BERT tokenizer 的二次截断。

### 修改文件

- `app/rag/reranker.py` — `max_length` 128→512；`truncate_for_rerank` 改为头尾各取；`MAX_RERANK_DOC_CHARS` 400→300

---

## v5.3 - 修复：ChromaDB L2 距离导致检索相似度异常

### 问题

用户问"头疼怎么缓解？"，知识库中有头痛处理文档，但 Dense 检索 `top1_dense_dist=0.9333`，
Reranker 最高分仅 0.0031，最终判定为"无相关文档"，返回澄清追问。

根因是 **ChromaDB 默认使用 L2 距离**（欧几里得距离），而代码注释和阈值全部按余弦距离设计。
ChromaDB `space="l2"` 时，`similarity_search_by_vector_with_score` 返回的是 L2² 距离而非余弦距离。

### 为什么 RAG 场景必须用余弦相似度而非 L2 距离

Embedding 模型的本质是将文本映射为高维空间中的向量，其中**方向编码语义，长度编码强度**。

**L2 距离（欧几里得距离）——量尺子**

计算两个向量在多维空间中的直线距离：`√(Σ(ai - bi)²)`

```
向量A（查询"头疼怎么缓解"）: 方向↗, 长度=1.0
向量B（文档"头痛的治疗方法包括..."）: 方向↗, 长度=3.2

L2 距离 = 很大  ← 虽然方向一致，但长度差拉大了距离
```

问题：文本较长的文档 chunk 的向量模长天然更大，L2 会把"长度差异"误判为"语义差异"。

**余弦相似度——量角器**

计算两个向量的夹角：`cos(θ) = (A·B) / (|A|·|B|)`

```
向量A（查询"头疼怎么缓解"）: 方向↗, 长度任意
向量B（文档"头痛的治疗方法包括..."）: 方向↗, 长度任意

余弦相似度 ≈ 1.0  ← 只看方向，与长度无关
```

余弦相似度**归一化了向量长度**，只比较方向。这意味着：
- 短查询"头疼"和长文档"头痛的病理机制与临床治疗指南..."可以正确匹配
- 高频词导致的向量模长膨胀不会影响相似度
- 语义相同但字数差异巨大的文本不会被误判

**为什么现代 Embedding 模型都为余弦设计**

| 模型 | 训练目标 | 输出 |
|------|----------|------|
| OpenAI text-embedding-3 | 余弦相似度对比学习 | 自动归一化 |
| 智谱 embedding-3 | 余弦相似度对比学习 | 自动归一化 |
| BGE / M3E 系列 | InfoNCE + 余弦损失 | 自动归一化 |

所有主流 Embedding 模型的训练目标都是**最小化正样本对的余弦距离、最大化负样本对的余弦距离**。归一化后的向量落在单位超球面上，此时 L2² = 2×(1−cos_sim)，两种度量等价——但前提是**索引和查询使用同一种距离**。ChromaDB 默认 L2，而模型为余弦优化，这正是错配的根源。

**一句话总结：语义存在于方向中，不在长度中。余弦相似度度量的是"两个文本在说什么"，L2 度量的是"两个向量有多长"。RAG 需要前者。**

### 三个叠加问题

**1. ChromaDB 距离度量错误（主因）**

| 维度 | 修复前 | 修复后 |
|------|--------|--------|
| ChromaDB 空间 | `l2`（默认） | `cosine` |
| 距离 0.9333 的含义 | L2² 距离 ≈ 余弦相似度 0.53 | cosine 距离 ≈ 余弦相似度 0.88 |
| HIGH_CONFIDENCE_THRESHOLD | 0.08（注释写余弦但实际对 L2 无效） | 0.08（与 cosine 距离正确对应） |

- `vector_store.py`：`Chroma.from_documents()` 时显式传入 `collection_metadata={"hnsw:space": "cosine"}`
- 需重建向量库：`python scripts/rebuild_vector_store.py`

**2. 文档分块参数不匹配**

- `loader.py` `split_documents()` 默认 `chunk_size=1000`，但 config 配置为 500
- 1000 字符的 chunk 会稀释短查询的语义信号
- 修复：默认参数改为 `chunk_size=500, chunk_overlap=50`，对齐 config

**3. 高置信度绕过逻辑失效**

- `hybrid_retriever.py` 的 `HIGH_CONFIDENCE_THRESHOLD = 0.08` 注释写"cosine distance"
- L2 距离下此阈值永远无法触发，导致所有查询都走 Reranker（额外 627ms）
- 切换 cosine 后阈值自然生效

### 修改文件

- `app/rag/vector_store.py` — `Chroma.from_documents()` 添加 `collection_metadata={"hnsw:space": "cosine"}`
- `app/rag/loader.py` — `split_documents()` 默认参数对齐 config（500/50）

### 部署注意

**必须重建向量库**才能使 cosine 距离生效：
```bash
python scripts/rebuild_vector_store.py
```
旧 L2 空间的 ChromaDB 集合不会自动迁移。

---

## v5.2 - 前端用户反馈通道 + 正例测试集

### 问题

Bad case 采集完全依赖后端自动检测，缺少真实用户反馈信号。
测试集 20 条全为负例（不自包含的追问），`_has_anaphora_pattern` 的误杀率未被测量。

### 修复

**1. 前端反馈按钮（👍/👎）**

- 每条 AI 回答后显示 👍/👎 按钮
- 点击 👎 弹出反馈面板，选择原因：
  - 答案不准确 / 没回答我的问题 / 缺少关键信息 / 内容不安全 / 其他
- 支持补充说明文字
- 反馈以 `user_negative_feedback` 类型写入 bad_cases
- 后端新增 `POST /api/feedback` 端点

**2. 正例测试集（18 条自包含查询）**

- 新增 `bc_pos_001` ~ `bc_pos_018`，均为自包含的医疗查询
  （如"布洛芬的副作用是什么？""头痛怎么缓解？"）
- 测试集从 20 条扩至 38 条：负例 20 + 正例 18

**3. 修复 `_has_anaphora_pattern` 误杀**

- 第二层（<15字短查询）增加领域实体检测：
  `len(text) < 15 AND 缺少领域实体 → 不自包含`
- 修复前：`"头痛怎么缓解？"`（7字）被误判为不自包含
- 修复后：含实体的短查询正确识别为自包含
- 准确率：100%（漏检 0/20，误杀 0/18）

### 修改文件

- `app/static/index.html` — 反馈按钮 UI + Modal + JS 逻辑
- `app/api/routes.py` — 新增 `FeedbackRequest` 模型 + `POST /api/feedback` 端点
- `app/graph/nodes/nodes.py` — `_has_anaphora_pattern` 短查询实体检测修复
- `tests/data/self_containment_test_set.jsonl` — 新增 18 条正例

---

## v5.1 - Bad Case 采集面扩展（3 个新采集点）

### 问题

Bad case 只覆盖了查询重写环节（3 个采集点），以下关键失败模式完全未被采集：
- LLM 在答案中编造检索文档不存在的药物（幻觉）
- 检索返回零文档（索引/查询词匹配问题）
- 含症状词的问题被错误路由到 direct_answer

### 新增采集点

| 采集点 | 触发条件 | case_type | 位置 |
|--------|----------|-----------|------|
| 幻觉检测 | 答案含药物名但检索文档中未出现 | `hallucination_suspected` | `streaming.py` RAG 答案生成后 |
| 检索失败 | RAG 管道返回零文档 | `retrieval_miss` | `streaming.py` 零文档分支 |
| 路由异常 | 问题含症状词但路由到 direct_answer | `route_misclassification` | `streaming.py` direct_answer 分支 |

### 配套更新

- `long_term_memory.py`：`append_bad_case` docstring 补充 5 个新 case_type
- `scripts/export_bad_cases.py`：`--case-type` 选项新增所有类型

### 修改文件

- `app/graph/streaming.py` — 新增 `_check_hallucination`、`_record_retrieval_miss`、`_check_route_misclassification` 三个方法；在 `run()` 的关键路径插入调用
- `app/memory/long_term_memory.py` — 更新 case_type 文档
- `scripts/export_bad_cases.py` — 扩展 `--case-type` 选项

---

## v5.0 - 自包含性检测 + Bad Case 采集 + 低分澄清

### 问题

"语法完整性陷阱"：查询"还有其他什么可以吃的吗？"语法完美但语义残缺，
缺少核心实体（头痛/缓解药物），传统基于"查询质量/长度/语法"的静态规则完全失效。
低分检索结果直接进入 LLM 自由生成，产生幻觉回答。

### 修复

**1. 自包含性前置检测（方案A P0）**

| 维度 | 修复前 | 修复后 |
|------|--------|--------|
| 指代词检测 | ❌ 无 | ✅ 15个指代词黑名单（其他/还有/这个/那个/它/呢/...） |
| 极短查询 | ❌ 无 | ✅ <15字 + 有历史 → 强制重写 |
| 疑问词+缺实体 | ❌ 无 | ✅ 以"怎么/如何/什么/哪些"开头但缺少领域实体 → 强制重写 |

三层检测逻辑：`_has_anaphora_pattern(query)` → 误杀代价远小于漏改导致的幻觉

**2. Bad Case 自动采集**

| 采集点 | 触发条件 | case_type |
|--------|----------|-----------|
| 重写后 | 指代词检测命中但重写结果与原问题一致 | `rewrite_same_as_original` |
| 重写后 | 指代词检测命中但重写后仍缺领域实体 | `rewrite_missed_anaphora` |
| 低分时 | 检索低分但未触发澄清 | `low_score_no_clarify` |

存储：PostgresStore `("bad_cases", user_id)` 命名空间，支持人工审核补填 `expected_rewrite` 和 `is_self_contained`

**3. 低分澄清机制（消除幻觉出口）**

| 场景 | 修复前 | 修复后 |
|------|--------|--------|
| 无检索文档 | 降级为 LLM 自由生成（幻觉风险） | 返回结构化澄清追问 |
| 低分检索 | 直接生成兜底答案 | 记录 bad case + 澄清追问 |

**4. 测试集和工具**

- 种子测试集：20 条手工标注 bad case（`tests/data/self_containment_test_set.jsonl`）
- 导出脚本：`scripts/export_bad_cases.py`（PostgresStore → JSONL）
- 回归测试：`tests/test_self_containment.py`（验证 `_has_anaphora_pattern` 准确率）

### 修改文件

- `app/graph/nodes/nodes.py` — 新增 `_has_anaphora_pattern`、`_record_bad_case_if_needed`、`_ANAPHORA_PATTERNS`、`_QUESTION_STARTS`、`_DOMAIN_ENTITY_KEYWORDS`；`query_rewrite_node` 增加前置检测和 bad case 采集
- `app/graph/streaming.py` — 新增 `_build_clarification_answer`、`_record_low_score_bad_case`；无检索文档时返回澄清追问
- `app/memory/long_term_memory.py` — 新增 `append_bad_case`、`get_bad_cases`、`update_bad_case_review`
- `tests/data/self_containment_test_set.jsonl` — 新增种子测试集（20条）
- `scripts/export_bad_cases.py` — 新增导出脚本
- `tests/test_self_containment.py` — 新增回归测试脚本

---

## v4.4 - 修复：`_build_rewrite_context` 截断导致药物名丢失

### 问题

`_build_rewrite_context` 对 AI 回复做头尾截断（保留前 2/3 + 后 1/3）时，
LLM 推荐的具体药品名称可能出现在回复的**中间部分**（如药理说明段落），截断后丢失。
后续用户追问"还有什么药可以吃？"时，重写提示词中看不到第一次推荐的药物名，
只能依赖用户问题中残留的关键词，造成上下文断层。

### 修复

**截断前全文扫描提取医疗实体**：
1. AI 回复**截断前**，先扫描全文匹配药物关键词（与 `_DRUG_KEYWORDS` 对齐，~45个）和症状关键词（~30个）
2. 匹配到的实体以 `[提及：布洛芬、对乙酰氨基酚]` 格式前置到截断文本前
3. 实体上限 12 个（按长度排），避免提示词膨胀
4. 即使药物名在回复中间第 400 个字符处，截断后也能通过前置标签找回

### 关键逻辑

```
AI 回复全文（可能 800+ 字）
  ↓ 先扫描全文 → found_entities = {布洛芬, 头痛, 剂量}
  ↓ 再截断头尾（head...tail，丢失中间药物名）
  ↓ 前置实体标签 → "[提及：布洛芬、头痛、剂量] 头部内容...尾部内容"
```

### 修改文件

- `app/graph/nodes/nodes.py` — `_build_rewrite_context` 重构

---

## v4.3 - TTFT 优化：首 token 目标 <5s

### 问题

第二次提问"还有其他什么可以吃吗？"TTFT = 9341ms，远超 5s 目标。
耗时分解：症状解析 (2871ms, 31%) + 查询重写 (3577ms, 38%) + 检索 (1675ms,18%) + L2缓存 (397ms, 4%) + 答案LLM (817ms, 9%)

### 优化

**1. 症状解析追问短路**（2871ms → 0ms）
- `symptom_analysis_node`：有对话历史且问题不含症状词时，跳过本地模型调用
- 理由：追问"还有其他什么可以吃吗？"不含任何症状词，LLM 推理 2.8s 只返回 `[]`
- 症状由节点末尾的快照继承逻辑补充

**2. 路由优先，按类型缓存**（397ms → 0ms for symptom）
- `streaming.py` `run()`：先跑路由（规则+上下文 0ms），再按类型决定缓存深度
- `symptom` / `general` → 仅 L0 答案缓存（无 embedding API）
- `knowledge` → L0 + L2 语义缓存（知识查询常重复）
- 新增 `_check_l0_cache()` 方法

**3. 重写提示词精简**（3577ms → ~1500ms）
- `query_rewrite_node`：Prompt 从 ~1500 字缩减到 ~300 字
- 移除冗长规则说明和重复示例，保留核心输出格式
- 减少 token 数 → 降低 LLM 首 token 延迟

### 预期收益

| 阶段 | 优化前 | 优化后 |
|------|--------|--------|
| 症状解析 | 2871ms | 0ms (短路) |
| L2 语义缓存 | 397ms | 0ms (跳过) |
| 查询重写 | 3577ms | ~1500ms (短 prompt) |
| 知识检索 | 1675ms | 1675ms (不变) |
| 答案 LLM | 817ms | 817ms (不变) |
| **TTFT** | **9341ms** | **~4000ms** |

### 修改文件

- `app/graph/nodes/nodes.py` — `symptom_analysis_node` 追问短路；`query_rewrite_node` prompt 精简
- `app/graph/streaming.py` — `run()` 路由优先 → 按类型缓存；新增 `_check_l0_cache()`

---

## v4.2 - 重构：查询强制重写 + 问题拆解

### 问题

上一版修复了上下文注入缺失，但查询重写仍存在"是否要重写"的判断门。
对"还有其他什么可以吃吗？"这类追问，LLM 偶尔返回 `need_rewrite=False`，
导致问句未补全，后续检索和答案生成都缺少上下文。

### 方案

参考业界 2026 年多轮 RAG 最佳实践（Constrained Rewrite + Query Decomposition）：

1. **废除判断门**：有对话历史 → 强制重写，不再问"是否需要"
2. **一次调用产出两份结果**：
   - `FINAL`：完整的自包含问句 → 用于答案生成 + HyDE
   - `SEARCH`：检索关键词 → 用于 BM25 稀疏检索
3. **对话历史完整保留**：AI 回复不再粗暴截断 150 字，医疗关键词消息保留 500 字

流程示例：
```
追问："还有其他什么可以吃吗？"
  → FINAL:  "缓解头痛，除了布洛芬，还有什么药物可以服用？"
  → SEARCH: "头痛 缓解 药物"
  → Dense: HyDE(FINAL)  → 向量检索
  → Sparse: BM25(SEARCH) → 关键词检索
  → 生成: build_rag_prompt(FINAL, docs, history)
```

### 修改文件

- `app/graph/state.py` — 新增 `final_question` 字段
- `app/graph/nodes/nodes.py`：
  - `query_rewrite_node`：重写提示词重构为 FINAL/SEARCH 双输出格式；
    移除 yes/no 判断门，有历史就强制重写+拆解；
    HyDE 改用 FINAL（完整上下文）生成假想答案
  - `_build_rewrite_context`：取最近 2 轮对话，
    AI 回复根据医疗关键词智能截断（500/250 字），保留头尾关键信息
  - `answer_generation_node` / `stream_answer_generation` / `stream_direct_answer`：
    答案生成统一使用 `final_question`（无重写时回退到原问题）

### 收益

- 追问不再被误判为"无需重写"，上下文可靠传递到检索和生成
- 检索用关键词 + 生成用完整问句，各司其职
- HyDE 用完整问句生成假想答案，语义召回更精准

---

## v4.1 - 修复：短期记忆丢失——追问上下文链路断裂

### 问题

用户追问"还有其他什么可以吃吗？"时，系统完全丢失了上文"头痛→布洛芬"的上下文，
推荐了无关内容。日志分析发现三层逐级断裂：

1. **查询重写误判**：LLM 提示词太宽松，对明确追问返回 `need_rewrite=False`
2. **症状提取为空**：追问本身不含症状词，提取结果 `[]`
3. **直接回答无对话历史**：RAG 降级到 `direct_answer` 后，
   `build_direct_answer_prompt` 未注入 L3 对话历史，LLM 只看到孤立的追问

### 修改内容

- `app/graph/nodes/nodes.py` — 三处修复：
  - `query_rewrite_node`：重写提示词重构，从"是否需重写"改为"必须补全上下文"，
    新增 3 个正反示例（追问药物/剂量/重复），降低误判率
  - `symptom_analysis_node`：追问症状继承——当前问题无显式症状时，
    从 `clinical_checkpoint` 补充历史症状/部位/发作时间
  - `build_direct_answer_prompt`：新增 L3 对话历史注入，
    追加指令"追问必须结合对话历史中的症状和药物回答"
  - `stream_vision_answer`：新增 L1+L2+L3 三层上下文注入，
    追加指令"结合对话历史中的症状/用药信息解读图片"

---

## v3.9 - 紧急修复：路由结果被丢弃导致 RAG 全部跳过

### 问题

`streaming.py` 的 `run()` 方法中，路由和缓存并行执行后，
`asyncio.gather(_check(), self._run_route_sync())` 的返回值未被接收。
`_run_route_sync()` 返回的 `Command`（含 `goto=symptom_analysis` 等路由目标）
被丢弃，导致 `route_command` 始终为 `None`。

下游判断逻辑 `route_command or "direct_answer"` 永远命中默认值，
**所有无缓存请求都走 `direct_answer`，RAG 管道被完全绕过。**

### 修复

- `app/graph/streaming.py` — `_, route_command = await asyncio.gather(...)` 接收路由结果

---

## v3.7 - 运维优化：配置热更新接口

### 优化背景

缓存 TTL、速率限制、模型参数等配置修改后需要重启服务才能生效，
开发调试和运维应急时不够灵活。

### 修改内容

**修改文件**：
- `app/core/config.py` — 新增 `reload_config()` 函数：
  - 重新读取 `.env` 文件创建新 Settings 实例
  - 对比新旧值，返回变更字段列表
  - 异常时回退到旧配置，保证服务不中断
- `app/api/routes.py` — 新增 `POST /api/admin/reload-config` 端点：
  - 需要 `X-Admin-API-Key` 认证（复用已有 `_verify_admin_key`）
  - 返回变更字段列表和重载状态

**使用方式**：
```bash
curl -X POST http://localhost:8000/api/admin/reload-config \
  -H "X-Admin-API-Key: your-admin-key"
```

**响应示例**：
```json
{"reloaded": true, "changed_fields": ["CACHE_TTL_SECONDS", "RATE_LIMIT_PER_MINUTE"],
 "message": "配置已重新加载，2 个字段发生变化"}
```

**可热更新的配置项**：所有 `Settings` 字段均支持热更新，包括 `CACHE_TTL_SECONDS`、
`RATE_LIMIT_PER_MINUTE`、`MODEL_TEMPERATURE`、`ENABLE_SAFETY_CHECK` 等。

---

## v3.8 - 运维优化：Dockerfile 路径环境变量化

### 优化背景

Dockerfile 中 `/app/models` 路径硬编码，docker-compose.yml 中
`RERANKER_MODEL_PATH` 也写死为容器内绝对路径，本地开发时需手动覆盖。

### 修改内容

**修改文件**：
- `Dockerfile` — `RUN mkdir -p /app/models` 改为 `ARG MODEL_DIR=/app/models` + `RUN mkdir -p ${MODEL_DIR}`，支持构建时通过 `--build-arg MODEL_DIR=/custom/path` 覆盖
- `docker-compose.yml` — `RERANKER_MODEL_PATH` 从硬编码改为 `${RERANKER_MODEL_PATH:-/app/models/bge-reranker-onnx}`，支持 `.env` 文件或环境变量覆盖
- `app/core/config.py` — `RERANKER_MODEL_PATH` 默认值从 `/app/models/bge-reranker-onnx` 改为 `PROJECT_ROOT / "bge-reranker-onnx"`，本地开发无需额外配置

**使用方式**：
```bash
# .env 文件中覆盖
RERANKER_MODEL_PATH=/home/user/models/bge-reranker-onnx

# 或 docker-compose 构建时
docker compose build --build-arg MODEL_DIR=/opt/models
```

---

## v3.6 - 质量保障：核心节点单元测试

### 优化背景

项目缺少单元测试。LangGraph 节点的纯函数特性非常适合单元测试，
但没有覆盖时，重构和回归都缺乏安全网。

### 修改内容

**新增文件**：
- `tests/__init__.py`
- `tests/conftest.py` — pytest 配置和共享 fixtures（mock_llm, base_state 等）
- `tests/test_helpers.py` — `extract_json_block` 5 层回退、`_coerce_list_fields` 列表规范化、
  药物关键词常量测试（共 14 个用例）
- `tests/test_nodes.py` — 路由规则、标签规范化、症状规则提取、查询相似性、
  文档评分振荡检测测试（共 19 个用例）
- `pytest.ini` — pytest 配置

**修改文件**：
- `requirements.txt` — 添加 `pytest~=8.0`

**测试覆盖**：
- `extract_json_block`: 直接 JSON / Markdown 代码块 / 嵌套 / 花括号提取 / 空输入
- `_coerce_list_fields`: 字符串转列表 / 已是列表 / None / 嵌套展平 / 中文逗号
- `detect_rule_based_route`: 症状/知识/问候/未知/优先级 路由
- `normalize_router_label`: 合法标签 / 中文标签 / 兜底默认值
- `_extract_symptoms_by_rules`: 单症状 / 多症状 / 严重程度 / 部位 / 持续时间 / 去重 / 疼痛模式兜底
- `grade_documents_node`: 振荡检测无改善时跳过重试

**运行方式**：
```bash
cd D:/Agent/medical_assistant_agent
pytest tests/ -v
```

---

## v3.5 - 可靠性优化：自纠正循环振荡检测

### 优化背景

`grade_documents_node` 在检索结果不相关时触发自纠正（重写→检索→评分），上限 2 次。
但如果 Reranker 分数刚好在阈值附近反复横跳，重试不会改善结果，反而浪费 3-5s。

### 修改内容

**修改文件**：
- `app/graph/nodes/nodes.py` — `grade_documents_node()` 增加振荡检测：
  - 重试前记录 `_prev_max_score` 和 `_prev_relevant_count` 到状态
  - 重试后检测：score_delta < 0.05 且 doc_delta < 1 → 无改善，跳过二次重试
  - 无检索文档且前次有重试历史时同样检测
- `app/graph/streaming.py` — `_run_rag_pipeline()` 重试时递增 `retrieval_attempts`

**收益**：
- 避免无效重试，节省 3-5s 的无关等待
- 日志明确记录每次重试前后的分数变化

---

## v3.4 - 可靠性优化：L1 写入失败本地缓冲

### 优化背景

快照更新中的 L1 写入（症状事件/用药记录同步到 PostgresStore）失败时只打 warning 日志，
不做任何补偿。如果 PostgresStore 暂时不可用，症状事件会永久丢失。

### 修改内容

**新增文件**：
- `app/memory/fallback_buffer.py` — 本地 SQLite 缓冲队列：
  - `enqueue_symptom_event()` / `enqueue_medication_event()` — L1 写入失败时入队
  - `flush()` — 服务恢复时重新写入 L1，超过 10 次重试自动丢弃
  - `start_background_flush()` — 启动时立即 flush + 每 5 分钟定期 flush
  - 过期清理：超过 7 天的事件自动删除

**修改文件**：
- `app/graph/nodes/nodes.py` — `update_clinical_snapshot_node()` 的 except 块增加缓冲写入
- `app/api/routes.py` — lifespan 启动/关闭时调用 `start_background_flush()` / `stop_background_flush()`

**收益**：
- L1 不可用时症状事件不再丢失，恢复后自动补写
- 双重保险：缓冲写入失败时才丢失事件（概率极低）

---

## v3.3 - 性能优化：语义缓存 SCAN 替换为 Set + MGET

### 优化背景

语义缓存的 `_find_similar_query` 每次都用 SCAN 遍历所有 `semantic_cache:*` 键，
然后对每个键单独执行 GET，N 个条目需要 N+1 次 Redis 往返。随着缓存增长到数千条，
这会成为显著的性能瓶颈。

### 修改内容

**修改文件**：
- `app/cache/semantic_cache.py`：
  - 新增 Redis Set (`semantic_cache:keys`) 追踪所有缓存键，`set()` 时 SADD，`clear()` 时 SMEMBERS + DEL
  - `_find_similar_query` 用 SMEMBERS 替代 SCAN + N×GET 改为单次 MGET，从 N+1 次往返降为仅 2 次
  - `set()` 方法增加 LRU 淘汰：超过 `max_keys`（默认 5000）时删除最早 20% 的条目
  - 空集合检查改用 SCARD（O(1)）
- `app/graph/streaming.py`：L2 缓存为空检查改用 `scard()` 替代 SCAN

**收益**：
- 缓存查找从 O(n) 次 Redis 往返降为 2 次（SMEMBERS + MGET）
- 缓存写入自动淘汰，防止无限增长
- 1000 条缓存时查找耗时从 ~50ms 降到 ~5ms

---

## v3.2 - 架构优化：拆分 nodes.py 为子模块包

### 优化背景

`nodes.py` 包含 2600+ 行代码、17 个节点函数 + 10+ 个辅助函数 + 7 个 Pydantic 模型，
是项目中最庞大的单文件。修改任何节点都需在巨型文件中定位。

### 修改内容

**新增文件**：
- `app/graph/nodes/__init__.py` — 包入口，重导出所有公开接口，保持向后兼容
- `app/graph/nodes/helpers.py` (221 行) — 工具函数：药物关键词常量、计时装饰器、
  `extract_json_block`（5 层 JSON 回退解析）、`invoke_structured_with_fallback` 等
- `app/graph/nodes/models.py` (61 行) — 7 个 Pydantic 结构化输出模型

**移动文件**：
- `app/graph/nodes.py` → `app/graph/nodes/nodes.py` — 原文件移入包内，删除已迁移的
  常量/装饰器/模型/工具函数定义，改为从子模块相对导入

**收益**：
- nodes.py 从 2619 行缩减到 2382 行，移除了 ~240 行已提取的代码
- helpers.py 和 models.py 可独立导入和测试，无需加载整个节点模块
- 外部代码通过 `from app.graph.nodes import router_node` 继续工作，零破坏性变更

---

## v3.1 - 架构优化：流式编排模块化

### 优化背景

routes.py 中的 `event_generator()` 闭包包含了 400+ 行的节点编排逻辑，
与 graph.py 中的边定义形成双维护。每次修改 Graph 节点都需手动同步两处代码。

### 修改内容

**新增文件**：
- `app/graph/streaming.py` — `StreamingOrchestrator` 类，封装完整的流式编排逻辑：
  - 缓存检查（L0 答案缓存 + L2 语义缓存）
  - 并行路由 + 缓存检查
  - RAG 流水线编排（症状→重写→检索→评分→自纠正）
  - 对话历史保存 + 后台快照更新
  - SSE 事件发射

**修改文件**：
- `app/api/routes.py` — stream 端点从 400+ 行削减到 35 行，仅负责参数提取和 SSE 响应包装。移除了不再需要的节点级导入和 L0 缓存函数

**收益**：
- routes.py 代码量减少 ~40%（845 → 500 行）
- 消除 routes.py 和 graph.py 的双维护问题——编排逻辑现在是 graph 定义的唯一消费者
- `validate_streaming_sync()` 仍作为安全网在启动时自动检测一致性

---

## v3.0 - 参考蚂蚁阿福方案的功能增强

### 增强背景

参考蚂蚁阿福（支付宝医疗AI）的技术方案，从功能维度增强医疗助手：

| 能力 | 蚂蚁阿福 | 优化前本项目 | 优化后本项目 |
|------|---------|------------|------------|
| 主动追问 | 多轮追问补全信息 | 无 | 症状模糊时追加追问引导 |
| 图片问诊 | OCR+VLM混合 | 不支持 | VLM直接理解图片 |
| 循证标注 | 证据等级A/B/C | 无 | RAG回答标注来源和证据等级 |
| 安全拒答 | 超范围问题拒答 | 无 | LLM路由新增refuse类型 |

---

### 增强项 1：主动追问机制

**参考**：蚂蚁阿福的"模拟真人医生问诊逻辑"，当用户描述模糊时主动追问补全关键信息。

**方案**：新增 `_build_followup_hints()` 函数，根据症状提取结果检测缺失字段（部位、持续时间、严重程度），在 RAG prompt 末尾追加追问引导。

**修改文件**：
- `app/graph/nodes.py` - 新增 `_build_followup_hints()` 函数
- `app/graph/nodes.py` - `build_rag_prompt()` 新增 `symptoms` 参数和追问逻辑

**效果示例**：
- 用户："我肚子不舒服"
- 回答末尾追加："💡 为了更准确地帮助您，您可以补充以下信息：具体部位（如：头部、腹部、四肢等）、持续时间（如：3天、1周等）、严重程度（如：轻微、中等、剧烈）。"

---

### 增强项 2：多模态图片问诊

**参考**：蚂蚁阿福的图片问诊架构——报告类走OCR+结构化，皮肤类走VLM。

**方案**：采用方案A（直接VLM），使用智谱 `glm-4v-plus` 多模态模型直接理解图片内容。理由：
1. 医疗图片信息密度高（箭头↑↓、灰度、颜色分布），OCR无法捕捉
2. 实现简单，改动最小
3. 首token延迟反而可能更快（省去RAG检索+Reranker的7-9秒）

**修改文件**：
- `app/core/config.py` - 新增 `VISION_MODEL_NAME: str = "glm-4v-plus"`
- `app/core/llm.py` - 新增 `get_vision_llm()` 函数
- `app/graph/state.py` - `MedicalAssistantState` 新增 `image_base64` 字段
- `app/graph/nodes.py` - 新增 `vision_analysis_node()` 和 `stream_vision_answer()`
- `app/graph/nodes.py` - `router_node()` 添加图片检测优先路由
- `app/graph/graph.py` - 注册 `vision_analysis` 节点和边
- `app/api/routes.py` - `ChatRequest` 新增 `image_base64` 字段
- `app/api/routes.py` - `event_generator()` 添加 vision 分支

**使用方式**：
```json
POST /api/chat/stream
{
    "question": "请帮我解读这份血常规报告",
    "image_base64": "/9j/4AAQSkZJRg..."
}
```

**预期首token延迟**：4-9秒（vs 文字RAG的12-14秒）

---

### 增强项 3：循证医学标注

**参考**：蚂蚁阿福的回答标注证据等级（A级=随机对照试验，B级=学会共识，C级=临床经验）。

**方案**：在 `build_rag_prompt` 的回答要求中添加循证标注指令：
- `[来源：文档N]` — 标注建议来源
- `[证据等级：A/B/C]` — 标注证据可信度

**修改文件**：
- `app/graph/nodes.py` - `build_rag_prompt()` 添加循证标注要求

**效果示例**：
> 建议多饮水、注意休息 [来源：文档1] [证据等级：B]

---

### 增强项 4：安全拒答机制

**参考**：蚂蚁阿福的安全边界——超出AI能力范围的问题引导至真人医生，非医疗问题礼貌拒绝。

**方案**：
- LLM路由新增 `refuse` 类型，识别非医疗相关问题
- 路由命中 `refuse` 时返回固定拒答话术
- RAG prompt 安全提醒升级为带⚠️的醒目格式

**修改文件**：
- `app/graph/nodes.py` - `router_node()` LLM路由新增 `refuse` 类型
- `app/graph/nodes.py` - `router_node()` 添加拒答分支
- `app/graph/nodes.py` - `build_rag_prompt()` 安全提醒升级

**效果**：
- 用户："帮我写个Python爬虫" → "抱歉，我是医疗健康助手，只能回答与健康相关的问题。"
- 用户："感冒了怎么办" → 正常回答 + "⚠️ 以上建议仅供参考，如有疑问请及时就医"

---

### 增强项 5：症状解析快速模型

**方案**：规则未命中时调用 `glm-4-flash`（智谱最快模型）替代主模型 `glm-4.5-air`。

**修改文件**：
- `app/core/config.py` - 新增 `SYMPTOM_MODEL_NAME: str = "glm-4-flash"`
- `app/core/llm.py` - 新增 `get_symptom_llm()` 函数
- `app/graph/nodes.py` - `symptom_analysis_node()` 规则未命中时调用 `get_symptom_llm()`

**效果**：规则未命中时从 ~13秒 降至 **~2-3秒**。

---

### 修改文件汇总

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `app/core/config.py` | 新增 | `VISION_MODEL_NAME`、`SYMPTOM_MODEL_NAME` |
| `app/core/llm.py` | 新增 | `get_vision_llm()`、`get_symptom_llm()` |
| `app/graph/state.py` | 新增 | `image_base64` 字段 |
| `app/graph/nodes.py` | 新增+修改 | `_build_followup_hints`、`vision_analysis_node`、`stream_vision_answer`、`build_rag_prompt`（追问+循证）、`router_node`（vision+refuse）、`symptom_analysis_node`（快速模型） |
| `app/graph/graph.py` | 新增 | `vision_analysis` 节点和边 |
| `app/api/routes.py` | 新增+修改 | `ChatRequest.image_base64`、`event_generator` vision分支 |

---

### 后续优化方向

1. **OCR结构化方案**：体检报告场景，OCR提取指标 → 知识图谱校验 → LLM解读（参考阿福的报告解读架构）
2. **药品知识图谱**：药盒识别 → 国药准字匹配 → 禁忌/相互作用检查
3. **多轮追问**：当前为单次追问，后续可改为多轮对话式追问
4. **并行检索**：Dense + Sparse 检索改为并行执行
5. **全节点异步化**：所有节点改为 async，支持并发请求

---

## v4.0 - RAG Pipeline 深度优化 + 三层记忆架构

### 更新背景

针对首 token 响应时间（TTFT）过长（9s+）、3B 小模型结构化输出不稳定、跨会话记忆丢失三大核心问题，进行 RAG Pipeline 深度优化和三层记忆架构重构。

### 核心指标变化

| 指标 | 优化前 | 优化后 | 降幅 |
|------|--------|--------|------|
| TTFT（明确查询） | ~9s | ~3-4s | **56%** |
| TTFT（追问查询） | ~9s | ~4-5s | **44%** |
| 3B模型 JSON 合法率 | ~60% | ~95% | **+35%** |
| 跨会话症状记忆 | ❌ 丢失 | ✅ L1持久化 | 新增 |
| Reranker 跳过率 | 0% | ~40%（高置信度查询） | 新增 |

---

### 更新项 1：3B 小模型结构化输出（方案1+3+4组合）

**问题**：Qwen2.5:3b 输出 JSON 格式不稳定，`"symptoms": "膝盖摔伤"`（字符串而非列表）、缺少引号、多余逗号等。

**原方案**：`get_local_llm()` + `with_structured_output()`（3B 模型不支持 function calling）

**新方案**：三层防线

| 层级 | 方案 | 技术 | 兜底场景 |
|------|------|------|----------|
| L1 | JSON Mode | `response_format={"type": "json_object"}` | 采样层提高 JSON 字符权重 |
| L2 | 鲁棒解析 | `extract_json_block`（json.loads → json_repair → ast.literal_eval） | 修复单引号、多余逗号、缺少引号 |
| L3 | 分隔符降级 | `parse_symptom_text`（`症状：xxx\n部位：xxx`） | JSON 完全无法解析时 |

**修改文件**：
- `app/core/llm.py` — 新增 `get_local_llm_json()`（JSON Mode）
- `app/graph/nodes.py` — `extract_json_block` 三层解析 + `_coerce_list_fields` 自动修复 + `parse_symptom_text` 分隔符降级
- `app/graph/nodes.py` — 症状解析/快照更新/档案提取/安全检查节点均改用 `get_local_llm_json()` + `invoke_json_once_with_fallback()`

**依赖新增**：`json_repair~=0.60.1`

---

### 更新项 2：查询重写重构（硬编码规则 → LLM 自主判断）

**问题**：`_should_rewrite_query` 用8层硬编码规则（字数/特征词/药物名）判断是否重写，维护成本无底洞，反例频出。

**原方案**：

```python
# ❌ 硬编码规则判断
if "呢" in question and has_history:
    should_rewrite = True  # "头痛怎么办"含"怎么办"→误判
if len(question) < 8 and has_history:
    should_rewrite = True  # 字数陷阱
```

**新方案**：LLM 一次调用完成判断+重写，输出分隔符格式

```
REWRITE: 是/否
QUERY: 重写后的查询或原查询
```

**核心逻辑**（仅2条，零维护）：
- 无对话历史 → 跳过重写（0ms）
- 有对话历史 → LLM 判断是否需要重写

**修改文件**：
- `app/graph/nodes.py` — 删除 `_should_rewrite_query`（~80行）、`_detect_current_question_route`（~25行）
- `app/graph/nodes.py` — 新增 `_build_rewrite_context`、`_rewrite_guard_check`
- `app/graph/nodes.py` — `query_rewrite_node` 重写为 LLM 判断模式

**删除的硬编码规则**：

| 删除的函数 | 行数 | 问题 |
|-----------|------|------|
| `_should_rewrite_query` | ~80行 | 8层规则，字数/特征词陷阱 |
| `_detect_current_question_route` | ~25行 | 药物名+意图词硬编码 |

---

### 更新项 3：模型切换（本地3B → glm-4-flash API）

**问题**：RTX 3050 Laptop 4GB VRAM，3B模型 GPU/CPU 混合推理，查询重写耗时 2.7-3.4s。

**原方案**：

| 节点 | 模型 | 耗时 |
|------|------|------|
| 查询重写 | Qwen2.5:3b（本地） | ~2.7-3.4s |
| HyDE | Qwen2.5:3b（本地） | ~2-3s |
| 症状解析 | Qwen2.5:3b（本地） | ~3s |
| 快照/档案 | Qwen2.5:3b（本地） | ~2-3s |

**新方案**：

| 节点 | 模型 | 耗时 |
|------|------|------|
| 查询重写 | **glm-4-flash（API）** | ~0.5-0.8s |
| HyDE | **glm-4-flash（API）** | ~0.5-0.8s |
| 症状解析 | Qwen2.5:1.5b（本地）+ 规则优先 | 0ms（规则命中）/ ~1s（LLM） |
| 快照/档案 | Qwen2.5:1.5b（本地）+ JSON Mode | ~1-1.5s |

**修改文件**：
- `app/core/config.py` — `LOCAL_MODEL_NAME: "qwen2.5:1.5b"`
- `app/graph/nodes.py` — `query_rewrite_node` / HyDE 改用 `get_rewrite_llm()`（glm-4-flash）

**本地模型对比**：

| | qwen2.5:3b | qwen2.5:1.5b |
|---|---|---|
| 模型大小 | 1.9 GB | 986 MB |
| VRAM 需求 | ~2.5-3 GB（4GB卡装不下） | ~1.2 GB（4GB卡纯GPU） |
| 推理速度 | 15-20 tokens/s（混合模式） | 50+ tokens/s（纯GPU） |

---

### 更新项 4：Reranker High-Confidence Bypass

**问题**：向量检索已找到完美匹配时，Reranker 仍耗时 1-2s 做无意义排序。

**原方案**：所有查询都经过 Reranker

**新方案**：Dense Top-1 cosine distance < 0.08（similarity > 0.92）时跳过 Reranker

```python
# 跳过逻辑
top1_dense_dist < 0.08  →  跳过重排（高置信度）
top1_dense_dist ≥ 0.08  →  执行重排
```

**修改文件**：
- `app/rag/hybrid_retriever.py` — `_dense_search` 返回 `(docs, top1_score)`
- `app/graph/nodes.py` — `_should_skip_reranker` 新增 `top1_dense_score` 参数

**回退的错误逻辑**：~~`candidate_count <= k * 2` → 跳过重排~~（数量少不等于质量高，忽略 Lost in the Middle 和噪声过滤）

---

### 更新项 5：时间锚定（相对时间 → 绝对时间戳）

**问题**：用户说"我现在头痛"，系统只记录"今天"，后续追问"头痛几天了"时无法精确计算。LLM 不知道"现在"是什么时候，多轮对话中容易丢失上下文。

**核心铁律**：绝不让 LLM 做时间运算，代码层完成所有时间转换和计算。

**原方案**：

```python
# ❌ 硬编码相对时间映射，无法覆盖所有场景
relative_time_map = {"现在": 0, "昨天": 1, "前天": 2, ...}
onset_date = "2026-06-22"  # 只有日期，没有时间戳
```

**新方案**：三层时间解析流水线（1-5ms）

| 层级 | 工具 | 场景 | 示例 |
|------|------|------|------|
| L1 | `dateparser` | 标准相对时间表达 | "前天"→2026-06-22 |
| L2 | 中文数字正则 | dateparser 不覆盖的中文 | "持续三天了"→3天 |
| L3 | 默认锚定 | 未提及任何时间 | "我现在头痛"→当前时刻 |

**存储结构升级**：

```python
# ❌ 之前：纯文本，无法计算
{"symptom_onset_dates": {"头痛": "2026-06-22"}}

# ✅ 现在：ISO + Unix时间戳 + 精度标记
{"symptom_onset_dates": {
  "头痛": {
    "iso": "2026-06-22T10:30:00",
    "ts": 1784567800,
    "precision": "exact"  # exact/approximate/vague/default
  }
}}
```

**Prompt 注入时间事实**：

```
【时间事实（系统计算，无需推算）】
- 头痛：首发于 2026-06-22T10:30:00，距今 2天3小时
```

**修改文件**：
- `app/graph/nodes.py` — `_extract_symptoms_by_rules` 重写时间解析逻辑
- `app/graph/nodes.py` — 新增 `_calculate_duration_from_checkpoint`（Unix 时间戳精确计算）
- `app/graph/nodes.py` — `build_rag_prompt` 注入【时间事实】段落
- `app/graph/nodes.py` — `ClinicalCheckpointOutput` 新增 `symptom_onset_dates` 字段（结构升级为 Dict[str, Dict]）
- `requirements.txt` — 新增 `dateparser~=1.4.1`

---

### 更新项 6：三层记忆协同架构

**问题**：症状首发时间只存在 L2 快照（绑定 thread_id），用户开新会话后完全失忆。

**原方案**：

| 信息类型 | L1 Profile | L2 Snapshot |
|---------|-----------|-------------|
| 姓名/年龄/过敏史 | ✅ | ✅ |
| 症状首发时间 | ❌ | ✅（跨会话丢失） |
| 用药记录 | ❌ | ✅（跨会话丢失） |

**新方案**：L1 新增 Append-Only 事件流，L2 作为活跃上下文缓存

```
┌─────────────────────────────────────────────────────┐
│  L3 短期窗口 (Messages, 6条)                         │
│  • 最近3轮对话原文                                    │
├─────────────────────────────────────────────────────┤
│  L2 活跃上下文 (Clinical Snapshot)                    │
│  • symptom_onset_dates ← L1填充 ← 当前轮症状解析      │
│  • medication_history                                │
├─────────────────────────────────────────────────────┤
│  L1 长期记忆 (PostgresStore, Append-Only)             │
│  • symptom_events:  {iso, ts, precision}             │
│  • medication_events: {drug, dosage, effect}          │
│  • user_profile: {name, age, allergies}              │
└─────────────────────────────────────────────────────┘
```

**数据流转**：

| 场景 | 数据流 | 结果 |
|------|--------|------|
| 新会话 | L1.get_all_symptom_onsets → L2.symptom_onset_dates | ✅ 跨会话记忆 |
| 当前会话 | 规则提取 → L2 → 快照更新时同步L1 | ✅ 双写保障 |
| 快照更新 | L2合并 → L1.append（保留最早记录） | ✅ 不覆盖更早记录 |
| 用药记录 | L2.medication_history → L1.append_medication_event | ✅ 跨会话可查 |

**修改文件**：
- `app/memory/long_term_memory.py` — 新增6个方法：
  - `append_symptom_event` / `get_symptom_events` / `get_latest_symptom_onset` / `get_all_symptom_onsets`
  - `append_medication_event` / `get_medication_events`
- `app/graph/nodes.py` — `memory_load_node` 新增 L1→L2 症状首发时间合并
- `app/graph/nodes.py` — `update_clinical_snapshot_node` 新增 L2→L1 异步同步（症状+用药）

**L1 新增命名空间**：

| 命名空间 | 用途 | 数据格式 |
|---------|------|---------|
| `symptom_events/{user_id}` | 症状报告事件流 | `{event_type, symptom, onset_iso, onset_ts, precision, source_query, created_at}` |
| `medication_events/{user_id}` | 用药记录事件流 | `{event_type, drug, dosage, effect, source_query, created_at}` |

---

### 更新项 7：L0 缓存日志可见性

**问题**：`has_profile=True` 时 L0 缓存完全跳过，日志中无任何 L0 相关记录。

**原方案**：有用户档案时静默跳过 L0

**新方案**：添加3条日志
- `L0答案缓存命中` / `L0答案缓存未命中` / `L0答案缓存跳过（用户有档案）`

**修改文件**：
- `app/api/routes.py` — L0 缓存检查处添加日志

---

### 修改文件汇总

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `app/core/llm.py` | 新增 | `get_local_llm_json()`（JSON Mode） |
| `app/core/config.py` | 修改 | `LOCAL_MODEL_NAME: "qwen2.5:1.5b"` |
| `app/graph/nodes.py` | 重构 | 查询重写（LLM判断）、时间锚定、三层记忆、Reranker Bypass、结构化输出 |
| `app/graph/state.py` | 修改 | `hyde_answer` 字段 |
| `app/memory/long_term_memory.py` | 新增 | 6个事件流读写方法 |
| `app/rag/hybrid_retriever.py` | 修改 | `_dense_search` 返回 top1 score |
| `app/api/routes.py` | 修改 | L0 缓存日志 |
| `requirements.txt` | 新增 | `dateparser~=1.4.1`、`json_repair~=0.60.1` |

---

### 依赖变化

| 依赖 | 版本 | 用途 |
|------|------|------|
| `dateparser` | ~1.4.1 | 相对时间→绝对时间解析（200+语言） |
| `json_repair` | ~0.60.1 | 3B模型输出 JSON 修复（单引号、多余逗号等） |

---

### 后续优化方向

1. **痊愈/恢复事件**：用户说"我头不痛了"时记录恢复时间，形成完整的症状生命周期
2. **时间范围过滤检索**：RAG 检索时支持按时间范围过滤文档
3. **L1 事件过期清理**：超过6个月的症状事件自动归档/清理
4. **异步 L1 写入**：快照更新时 L1 写入改为 `asyncio.create_task` 真正异步
5. **用户修改时间**："不是前天，是大前天" → 更新 L1 中已有的事件
