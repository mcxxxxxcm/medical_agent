# 系统优化更新日志

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
