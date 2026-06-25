# 系统优化更新日志

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
