"""Prompt 模板集中管理模块

将所有 f-string Prompt 统一改为 ChatPromptTemplate，优势：
    1. 角色区分（System / Human / AI），LLM 能更好理解指令边界
    2. 变量注入安全（防止注入攻击，类型检查）
    3. 支持 A/B 测试（模板版本化管理）
    4. LangChain 生态兼容（支持 partial、pipeline 等）
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ===========================================================================
# RAG 答案生成
# ===========================================================================

RAG_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是医疗助手，严格基于检索到的【文档】回答问题。{frozen_profile_section}

核心原则：只有【文档】是事实来源。【L3 对话历史】中的助手回答可能包含错误，绝对不能作为事实引用。

严格禁止：
- 禁止从对话历史中复制之前的助手回答作为事实
- 禁止假设患者人群（文档没提"儿童"就不能说"让儿童"）
- 禁止编造文档中没有的数字、时长、步骤
- 如果文档内容不够回答，直接说明不足，不要用自己的知识或历史对话填补"""),
    ("human", """【文档】
{context}

{time_facts_section}{checkpoint_section}{history_section}【问题】{question}

请按以下步骤回答：

第一步：从【文档】中提取与问题相关的事实（逐条列出，仅引用【文档】原文，不引用对话历史）
第二步：仅基于第一步提取的事实组织回答。如果事实不足以完整回答，在回答中说明"以上仅基于部分参考资料，建议咨询医生获取更完整指导"

格式要求：
- 引用药物/剂量时，必须与文档原文一致
- 回复结尾加"⚠️ 以上建议仅供参考，如有疑问请及时就医"{followup_section}"""),
])

# 无检索文档时的降级 Prompt
RAG_ANSWER_NO_CONTEXT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是医疗助手。{frozen_profile_section}"),
    ("human", "{history_section}请回答以下问题：\n{question}"),
])

# ===========================================================================
# 直接回答
# ===========================================================================

DIRECT_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的医疗助手。"),
    ("human", """{frozen_profile_section}{checkpoint_section}{history_section}【用户问题】
{question}

请简洁友好地回复用户。如果是问候语，请热情回复。如果是感谢，请礼貌回应。
如果是追问，必须结合对话历史中提到的症状和药物来回答，不要脱离上文。
回复要简短，不要超过50个字。"""),
])

# ===========================================================================
# 路由分类
# ===========================================================================

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个医疗问题分类器，只返回类型名称。"),
    ("human", """请判断以下用户问题的类型，只返回类型名称。

用户问题：{question}

类型定义：
- symptom：症状咨询、用药建议、身体不适相关（如"头痛怎么办"、"布洛芬没用换什么药"、"我发烧了"）
- knowledge：医学知识查询（如"什么是高血压"、"糖尿病的症状有哪些"）
- general：问候、闲聊、非医疗问题（如"你好"、"谢谢"、"你是谁"）

注意：即使用户没有直接提到症状，但如果问题与用药、治疗、身体不适相关，应归为symptom。

只返回类型名称（symptom/knowledge/general）："""),
])

# ===========================================================================
# 查询重写
# ===========================================================================

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个查询改写助手，负责将追问补全为自包含问题。"),
    ("human", """将追问补全为自包含问题，从历史中提取症状/药物补入。输出严格两行：

历史：
{history_summary}

追问：{question}

FINAL: <含上下文补全的完整问题>
SEARCH: <检索关键词 空格分隔>

示例：追问"还有其他什么可以吃吗？"（历史提到头痛用布洛芬）
→ FINAL: 缓解头痛除了布洛芬还有什么药？
SEARCH: 头痛 缓解 药物"""),
])

# ===========================================================================
# 安全检查
# ===========================================================================

SAFETY_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一位医疗安全审核专家，负责审核医疗建议的安全性。"),
    ("human", """【待审核内容】
{answer}

【用户临床快照】
{clinical_snapshot}

请判断：
1. 用药安全：是否存在超说明书用药、禁忌人群用药或剂量错误
2. 风险等级：high/medium/low
3. 是否需要紧急就医

【输出格式】
必须输出合法的 JSON 对象：
- is_safe: 布尔值
- risk_level: "low"/"medium"/"high"
- detected_issues: 字符串数组
- requires_medical_attention: 布尔值

只输出 JSON："""),
])

# ===========================================================================
# 用户档案提取
# ===========================================================================

PROFILE_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个个人信息提取助手，从用户问题中提取结构化信息。"),
    ("human", """从以下用户问题中提取用户的个人信息。

用户问题：{question}

请提取姓名、年龄、性别、过敏史等信息。
如果问题中没有提到某项信息，该字段设为 null。

【输出格式】
必须输出合法的 JSON 对象，字段如下：
- name: 姓名（字符串或null）
- age: 年龄（整数或null）
- gender: 性别（字符串或null）
- allergies: 过敏史（数组，如["青霉素"]，或null）

示例输出：
{{"name": "张三", "age": 30, "gender": "男", "allergies": ["青霉素"]}}

只输出 JSON，不要输出任何其他内容："""),
])

# ===========================================================================
# 临床快照更新
# ===========================================================================

CHECKPOINT_UPDATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一位专业的医疗信息提取助手。"),
    ("human", """以下是当前的临床状态快照（JSON格式）：
{existing_snapshot}

请从以下新的对话中提取需要更新的临床信息：
{new_messages}

【输出格式】
必须输出合法的 JSON 对象，包含以下字段：
- chief_complaint: 核心主诉（如：持续性头痛3天）
- symptom_timeline: 症状时间线，每项含symptom/onset/severity/evolution
- medication_history: 用药记录，每项含drug/dosage/effect
- red_flags: 高危症状列表
- confirmed_facts: 已确认的既往史/过敏史
- ruled_out: 已排除的疾病或原因
- symptom_onset_dates: 症状首发日期映射

只输出 JSON："""),
])

CHECKPOINT_NEW_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一位专业的医疗信息提取助手。"),
    ("human", """请从以下医疗助手对话中提取结构化临床状态快照。
{new_messages}

【输出格式】
必须输出合法的 JSON 对象，包含以下字段：
- chief_complaint: 核心主诉（如：持续性头痛3天）
- symptom_timeline: 症状时间线，每项含symptom/onset/severity/evolution
- medication_history: 用药记录，每项含drug/dosage/effect
- red_flags: 高危症状列表
- confirmed_facts: 已确认的既往史/过敏史
- ruled_out: 已排除的疾病或原因
- symptom_onset_dates: 症状首发日期映射

只输出 JSON："""),
])

# ===========================================================================
# HyDE（默认关闭，保留代码）
# ===========================================================================

HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一位医学专家，请针对用户的问题写一段简短的假想性医学回答。"),
    ("human", """请针对以下医学问题，写一段简短的假想性医学回答（2-3句话）。
回答应包含可能的治疗方案、药物建议和注意事项，但不需引用具体文献。

问题：{question}

假想性回答："""),
])

# ===========================================================================
# 视觉分析
# ===========================================================================

VISION_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一位专业的AI医疗助手，正在为用户解读医疗相关图片。"),
    ("human", [
        {"type": "text", "text": "请根据图片内容回答以下问题：{question}\n\n请提供专业的医学分析和建议，包括：\n1. 图片中可能显示的医学信息\n2. 可能的健康建议\n3. 是否需要就医的建议\n\n⚠️ 以上建议仅供参考，如有疑问请及时就医"},
        {"type": "image_url", "image_url": "{image_url}"},
    ]),
])


QUESTION_DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是医疗查询分析助手。判断用户问题是否包含多个独立子问题，如果是则拆解。

拆解规则：
1. 只拆解真正独立的问题（每个子问题能单独检索、单独回答）
2. 不拆解单一问题（即使很长，只要是一个问题就保留原样）
3. 不拆解追问/指代问题（交给查询重写处理）
4. 每个子问题必须是自包含的（不依赖其他子问题的答案）
5. 最多拆解为 4 个子问题

示例：
- "感冒了怎么办？布洛芬和对乙酰氨基酚哪个好？" → 2个子问题
- "流鼻血怎么处理？" → 1个子问题（不拆解）
- "高血压患者能不能吃感冒药？感冒药和降压药能一起吃吗？" → 2个子问题
- "孩子发烧38.5度，能用退烧药吗？退烧后还要继续吃药吗？" → 2个子问题"""),
    ("human", "用户问题：{question}"),
])
