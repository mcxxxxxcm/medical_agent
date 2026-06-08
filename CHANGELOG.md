# 系统优化更新日志

## v2.0 - 三种请求场景性能优化

### 优化背景

针对三种典型用户请求场景进行系统级性能优化：

| 场景 | 请求示例 | 优化前延迟 | 核心瓶颈 |
|------|----------|-----------|----------|
| 简单问候 | "你好" | ~8s | 不必要的 LLM 调用 |
| 自我介绍 | "你好，我是王艺涵" | ~18s | 档案提取阻塞首 token |
| 症状咨询 | "我是王艺涵，我有芒果过敏" | ~49s | 症状解析 LLM + Reranker + 查询重写 |

---

### 优化项 1：简单问候直接返回

**问题**：用户发送"你好"等简单问候时，仍需调用 LLM 生成回复，浪费 ~8 秒和 Token。

**方案**：添加问候映射表，精确匹配后直接返回预设回复，跳过所有 LLM 调用。

**修改文件**：
- `app/graph/nodes.py` - 新增 `_is_simple_greeting()` 函数
- `app/graph/nodes.py` - `direct_answer_node()` 添加问候快速返回
- `app/graph/nodes.py` - `stream_direct_answer()` 添加问候快速返回

**效果**：简单问候响应从 ~8s 降至 **<100ms**。

---

### 优化项 2：档案提取移至回答之后

**问题**：`profile_extraction_node` 在回答生成之前执行，耗时 ~10.5s，阻塞首 token 输出。

**方案**：将档案提取从回答前移至 `[DONE]` 之后执行，用户无需等待即可看到回答。

**修改文件**：
- `app/api/routes.py` - 将 `profile_extraction_node` 从第 287 行移至 `[DONE]` 之后

**效果**：自我介绍类请求首 token 延迟减少 ~10s。

---

### 优化项 3：症状解析规则优先 + LLM 降级

**问题**：`symptom_analysis_node` 每次都调用 LLM 解析症状，耗时 ~12.7s。

**方案**：添加 `_extract_symptoms_by_rules()` 规则提取函数，关键词匹配命中时跳过 LLM；LLM 失败时降级到规则提取。

**修改文件**：
- `app/graph/nodes.py` - 新增 `_extract_symptoms_by_rules()` 函数
- `app/graph/nodes.py` - `symptom_analysis_node()` 先规则后 LLM

**效果**：明确症状描述（如"我有芒果过敏"）解析从 ~12.7s 降至 **<10ms**。

---

### 优化项 4：Redis 连接超时 + 故障快速跳过

**问题**：Redis 连接失败时耗时 ~9s（无超时设置），严重拖慢响应。

**方案**：添加 `socket_timeout=2` 和 `socket_connect_timeout=2`；`get` 操作捕获 `ConnectionError`/`TimeoutError` 后临时降级。

**修改文件**：
- `app/cache/redis_cache.py` - `_connect()` 添加超时参数
- `app/cache/redis_cache.py` - `get()` 添加连接异常捕获

**效果**：Redis 故障时从 ~9s 阻塞降至 **<2s** 快速降级。

---

### 优化项 5：Reranker 条件跳过优化

**问题**：Reranker 推理耗时 5-7s，对简单查询和少量候选文档是不必要的开销；阈值 0.3 导致所有文档被过滤。

**方案**：
- 优化 `_should_skip_reranker()` 逻辑：候选数 <= k 直接跳过、简单问候跳过
- Reranker 阈值从 0.3 降至 0.0（ONNX 分数范围与原生模型不同）

**修改文件**：
- `app/rag/hybrid_retriever.py` - 优化 `_should_skip_reranker()` 逻辑
- `app/core/config.py` - `RERANKER_THRESHOLD` 默认值改为 0.0

**效果**：简单查询跳过 Reranker 节省 5-7s；文档不再被错误过滤。

---

### 优化项 6：查询重写使用轻量模型

**问题**：`query_rewrite_node` 使用主模型重写查询，延迟高；存在重复的死代码实现。

**方案**：
- 新增 `get_rewrite_llm()` 函数，支持配置独立轻量模型（`REWRITE_MODEL_NAME`）
- 合并清理 `query_rewrite_node` 中的死代码
- 添加医疗关键词跳过逻辑（明确提问无需重写）

**修改文件**：
- `app/core/llm.py` - 新增 `get_rewrite_llm()` 函数
- `app/core/config.py` - 新增 `REWRITE_MODEL_NAME` 配置项
- `app/graph/nodes.py` - `query_rewrite_node()` 使用 `get_rewrite_llm()` + 关键词跳过
- `app/graph/nodes.py` - 清理重复的 `query_rewrite_node` 死代码

**效果**：查询重写可使用更轻量的模型降低延迟；明确医疗问题直接跳过重写。

---

### 优化项 7：路由规则优先级修正

**问题**："你好我是王艺涵发烧了"被路由到 general，因为 general 问候匹配优先于 symptom。

**方案**：路由优先级改为 symptom > knowledge > general；general 改为精确匹配避免误判。

**修改文件**：
- `app/graph/nodes.py` - `detect_rule_based_route()` 优先级调整

**效果**：包含症状关键词的复合请求不再被误判为 general。

---

### 修改文件汇总

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `app/graph/nodes.py` | 新增+修改 | `_is_simple_greeting`、`_extract_symptoms_by_rules`、`direct_answer_node`、`stream_direct_answer`、`symptom_analysis_node`、`query_rewrite_node`、`detect_rule_based_route` |
| `app/api/routes.py` | 修改 | 档案提取移至回答之后 |
| `app/cache/redis_cache.py` | 修改 | 连接超时 + 故障快速降级 |
| `app/rag/hybrid_retriever.py` | 修改 | Reranker 跳过逻辑优化 + import re |
| `app/core/llm.py` | 新增 | `get_rewrite_llm()` 函数 |
| `app/core/config.py` | 修改+新增 | `RERANKER_THRESHOLD` 改为 0.0、新增 `REWRITE_MODEL_NAME` |
| `README.md` | 更新 | 全面更新，新增性能优化章节 |

---

### 未修改项（按用户指示）

- **structured output 相关代码**：`with_structured_output`、`invoke_structured_with_fallback` 等未修改，因为当前 LLM 不支持原生结构化输出，后续切换支持的模型即可。

---

### 后续优化方向

1. **并行检索**：Dense + Sparse 检索改为并行执行
2. **全节点异步化**：所有节点改为 async，支持并发请求
3. **档案提取异步化**：使用后台任务（如 `asyncio.create_task`）执行档案提取
4. **原生结构化输出**：切换支持原生结构化输出的模型后，移除 JSON 解析 fallback
