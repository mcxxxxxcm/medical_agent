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
