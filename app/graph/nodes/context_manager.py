"""上下文管理模块（四层压缩策略）

执行顺序：L1 → L3 → L2 → L4

L1：中间输出清除 —— 移除非首尾的 AI 中间推理内容，只保留关键输出
L3：大输出持久化 —— 超过阈值的内容写入磁盘，上下文中用 <persisted-output> 占位
L2：工具调用裁剪 —— 工具输出用占位符替代（如"进行了知识库检索"）
L4：LLM 摘要压缩 —— 总长度超阈值时，LLM 生成摘要保留 5 类关键信息

设计参考：Claude Code 的 MicroCompact 策略 + MemGPT 的分层记忆
"""
import json
import os
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from app.core.app_logging import get_logger

logger = get_logger(__name__)

# ===== 配置常量 =====
L3_SIZE_THRESHOLD = 30 * 1024        # L3: 30KB 触发持久化
L4_TOTAL_THRESHOLD = 50000           # L4: 总字符数 50000 触发 LLM 摘要
L4_SUMMARY_MAX_CHARS = 2000          # L4: 摘要最大字符数
PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                           "data", "persisted_outputs")


# ============================================================
# v9.17: 动态压缩阈值（P1）
# ============================================================

def estimate_message_tokens(messages: List[BaseMessage]) -> int:
    """估算消息列表的token占用

    简易估算：中文字符×1.5 + 英文单词×1.3
    不需要精确，只需在70%阈值附近做出判断
    """
    total_chars = sum(len(str(m.content)) for m in messages)
    # 粗估：混合中英文平均1字符≈1.5token
    return int(total_chars * 1.5)


def should_compress_by_token_ratio(
    messages: List[BaseMessage],
    context_window: int = 8192,
    compression_ratio: float = 0.7,
) -> bool:
    """判断是否需要基于token占用率触发压缩

    Args:
        messages: 当前消息列表
        context_window: LLM上下文窗口大小（token数）
        compression_ratio: 触发压缩的占用比例（0.7=70%）

    Returns:
        True = 应该压缩
    """
    if not messages:
        return False
    estimated_tokens = estimate_message_tokens(messages)
    threshold = int(context_window * compression_ratio)
    should = estimated_tokens > threshold
    if should:
        logger.info(
            f"上下文占用率触发压缩：估算{estimated_tokens} tokens > {threshold} "
            f"({compression_ratio*100:.0f}% × {context_window})"
        )
    return should


def get_adaptive_max_rounds(
    messages: List[BaseMessage],
    context_window: int = 8192,
    compression_ratio: float = 0.7,
    min_rounds: int = 1,
    max_rounds: int = 10,
) -> int:
    """自适应计算最多保留几轮对话

    从最近的对话开始，逐步向前扩展，直到累计token接近70%阈值。
    这样短对话可以保留更多轮，长对话自动缩减。

    Args:
        messages: 当前消息列表
        context_window: LLM上下文窗口大小
        compression_ratio: 压缩触发比例
        min_rounds: 最少保留轮数
        max_rounds: 最多保留轮数

    Returns:
        应保留的对话轮数
    """
    if not messages:
        return min_rounds

    threshold = int(context_window * compression_ratio)

    # 从最后一条消息向前扫描
    accumulated_tokens = 0
    rounds_found = 0
    human_count = 0

    for msg in reversed(messages):
        msg_tokens = int(len(str(msg.content)) * 1.5)
        accumulated_tokens += msg_tokens

        if isinstance(msg, HumanMessage):
            human_count += 1
            if human_count >= 2:  # 一轮 = 1条Human + 1条AI
                rounds_found = human_count // 1  # 简化：每2条Human=1轮
                if accumulated_tokens > threshold:
                    break

    # 限制在 min_rounds ~ max_rounds 之间
    result = max(min_rounds, min(rounds_found, max_rounds))

    # 如果扫描完了所有消息都没超阈值，返回全部轮数
    if accumulated_tokens <= threshold:
        total_human = sum(1 for m in messages if isinstance(m, HumanMessage))
        result = max(min_rounds, min(total_human, max_rounds))

    return result


# ============================================================
# v9.17: 原文存档（P2）
# ============================================================

def archive_messages_to_disk(
    messages: List[BaseMessage],
    user_id: str = "",
    thread_id: str = "",
) -> str:
    """将消息原文存档到磁盘（压缩时不丢弃，可回溯）

    存储路径：data/persisted_outputs/archives/{archive_id}.json

    Returns:
        archive_id（用于后续回溯）
    """
    os.makedirs(os.path.join(PERSIST_DIR, "archives"), exist_ok=True)

    archive_id = f"archive_{uuid.uuid4().hex[:8]}_{int(time.time())}"
    archive_path = os.path.join(PERSIST_DIR, "archives", f"{archive_id}.json")

    archive_data = {
        "id": archive_id,
        "type": "conversation_archive",
        "user_id": user_id,
        "thread_id": thread_id,
        "message_count": len(messages),
        "messages": [
            {
                "role": m.__class__.__name__,
                "content": str(m.content)[:10000],  # 单条消息上限10K字符
            }
            for m in messages
        ],
        "total_chars": sum(len(str(m.content)) for m in messages),
        "created_at": time.time(),
        "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    try:
        with open(archive_path, 'w', encoding='utf-8') as f:
            json.dump(archive_data, f, ensure_ascii=False, indent=2)
        logger.info(
            f"原文存档完成：{len(messages)} 条消息 → {archive_id} "
            f"({archive_data['total_chars']} 字符)"
        )
        return archive_id
    except Exception as e:
        logger.warning(f"原文存档失败：{e}")
        return ""


def load_archive(archive_id: str) -> Optional[Dict[str, Any]]:
    """从磁盘加载存档的原文

    Args:
        archive_id: 存档ID

    Returns:
        存档数据（含messages列表），不存在时返回None
    """
    archive_path = os.path.join(PERSIST_DIR, "archives", f"{archive_id}.json")
    if not os.path.exists(archive_path):
        logger.warning(f"存档不存在：{archive_id}")
        return None
    try:
        with open(archive_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"加载存档失败：{e}")
        return None


# ============================================================
# L1：中间输出清除
# ============================================================

def apply_l1_intermediate_cleanup(messages: List[BaseMessage]) -> List[BaseMessage]:
    """L1：清除中间输出，只保留首尾重要输出

    规则（不用 LLM）：
    1. 第一条 HumanMessage 和最后一条 AIMessage 完整保留
    2. 中间的 AIMessage：只保留首句（≤80 字），其余用 [中间推理已省略] 替代
    3. SystemMessage（临床快照等）：保留（信息密度高）
    4. HumanMessage：完整保留（用户问题不可裁剪）

    这样做的好处：
    - LLM 能看到用户最初的问题（上下文锚点）
    - LLM 能看到自己最后的回答（连贯性）
    - 中间推理过程不影响最终答案质量
    """
    if len(messages) <= 3:
        return messages

    # 找出首尾关键消息的索引
    first_human_idx = None
    last_ai_idx = None
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage) and first_human_idx is None:
            first_human_idx = i
        if isinstance(msg, AIMessage):
            last_ai_idx = i

    result = []
    for i, msg in enumerate(messages):
        if isinstance(msg, SystemMessage):
            # 系统消息（临床快照等）信息密度高，保留
            result.append(msg)
        elif isinstance(msg, HumanMessage):
            # 用户消息完整保留
            result.append(msg)
        elif isinstance(msg, AIMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            # 首条 Human 之前的 AI（边界情况）和最后一条 AI 完整保留
            if i == last_ai_idx:
                result.append(msg)
            else:
                # 中间 AI 消息：只保留首句摘要
                first_sentence = content.split('。')[0].split('\n')[0]
                if len(first_sentence) > 80:
                    first_sentence = first_sentence[:80] + "..."
                compressed = AIMessage(content=f"{first_sentence}...[中间推理已省略]")
                # 保留原始 metadata
                if hasattr(msg, 'id') and msg.id:
                    compressed.id = msg.id
                result.append(compressed)

    original_chars = sum(len(str(m.content)) for m in messages)
    compressed_chars = sum(len(str(m.content)) for m in result)
    if original_chars > compressed_chars:
        logger.info(f"L1 中间输出清除：{len(messages)} 条消息，{original_chars} → {compressed_chars} 字符"
                     f"（节省 {(original_chars - compressed_chars) / max(original_chars, 1) * 100:.0f}%）")

    return result


# ============================================================
# L3：大输出持久化
# ============================================================

def apply_l3_persist_large_outputs(messages: List[BaseMessage],
                                    threshold: int = L3_SIZE_THRESHOLD) -> List[BaseMessage]:
    """L3：大输出写入磁盘，上下文中用 <persisted-output> 占位

    规则（不用 LLM）：
    - 任何消息内容超过 threshold（默认 30KB）时：
      1. 将完整内容写入磁盘文件（data/persisted_outputs/{id}.json）
      2. 上下文中替换为 <persisted-output id="xxx" preview="前 200 字...">

    适用场景：
    - 长文档检索结果（5 个父文档可能超过 30KB）
    - 大量历史对话积累
    """
    os.makedirs(PERSIST_DIR, exist_ok=True)

    result = []
    persisted_count = 0

    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        content_size = len(content.encode('utf-8'))

        if content_size > threshold:
            # 生成持久化 ID
            output_id = f"persist_{uuid.uuid4().hex[:8]}_{int(time.time())}"
            file_path = os.path.join(PERSIST_DIR, f"{output_id}.json")

            # 写入磁盘
            persist_data = {
                "id": output_id,
                "type": msg.__class__.__name__,
                "content": content,
                "size_bytes": content_size,
                "created_at": time.time(),
            }
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(persist_data, f, ensure_ascii=False, indent=2)

                # 替换为占位符 + 预览
                preview = content[:200] + "..." if len(content) > 200 else content
                placeholder_content = f'<persisted-output id="{output_id}" preview="{preview}">'

                if isinstance(msg, AIMessage):
                    new_msg = AIMessage(content=placeholder_content)
                elif isinstance(msg, HumanMessage):
                    new_msg = HumanMessage(content=placeholder_content)
                else:
                    new_msg = SystemMessage(content=placeholder_content)

                if hasattr(msg, 'id') and msg.id:
                    new_msg.id = msg.id
                result.append(new_msg)
                persisted_count += 1

                logger.info(f"L3 持久化：{content_size / 1024:.1f}KB → <persisted-output id={output_id}>")
            except Exception as e:
                logger.warning(f"L3 持久化写入失败，保留原文：{e}")
                result.append(msg)
        else:
            result.append(msg)

    if persisted_count > 0:
        logger.info(f"L3 大输出持久化完成：{persisted_count} 条消息已转存磁盘")

    return result


def load_persisted_output(output_id: str) -> Optional[Dict[str, Any]]:
    """从磁盘加载持久化输出"""
    file_path = os.path.join(PERSIST_DIR, f"{output_id}.json")
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"加载持久化输出失败：{e}")
        return None


# ============================================================
# L2：工具调用输出裁剪
# ============================================================

# 工具调用输出裁剪映射表
_TOOL_OUTPUT_TEMPLATES = {
    # 检索类工具
    "知识检索": "进行了知识库检索，找到 {count} 篇相关文档",
    "检索": "进行了知识库检索",
    # 评分类工具
    "文档评分": "对检索文档进行了相关性评分",
    "评分": "完成了文档相关性评估",
    # 分析类工具
    "症状解析": "完成了症状分析",
    "路由": "完成了查询路由判断",
    "查询重写": "优化了查询表述",
    # 安全检查
    "安全检查": "完成了安全审查",
    # 档案提取
    "档案提取": "更新了用户健康档案",
    # 快照更新
    "快照更新": "更新了临床状态快照",
}


def apply_l2_tool_output_trim(messages: List[BaseMessage]) -> List[BaseMessage]:
    """L2：工具调用输出裁剪，占位符替代

    规则（不用 LLM）：
    - AI 消息中如果包含工具调用痕迹（如 [文档N 来源：xxx]、检索结果等），
      用简短占位符替代详细输出
    - 保留关键数值（如文档数量），去除冗余细节
    - 用户消息不裁剪

    裁剪模式：
    1. RAG 文档块 → [参考了 N 篇文档]
    2. 已有 doc_id 的引用 → 保留（已经是精简格式）
    3. 工具调用痕迹 → 查表替换为占位符
    """
    import re

    result = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            result.append(msg)
            continue

        if isinstance(msg, SystemMessage):
            result.append(msg)
            continue

        if isinstance(msg, AIMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)

            # 模式1：RAG 文档块裁剪 [文档N 来源：xxx]\n... → [参考了 N 篇文档]
            doc_pattern = r'\[文档(\d+)\s+来源[：:](.*?)(?:\s+doc_id:(doc_\w+))?\]\n(.*?)(?=\[文档\d|\Z)'
            doc_matches = list(re.finditer(doc_pattern, content, flags=re.DOTALL))

            if doc_matches:
                doc_count = len(doc_matches)
                # 如果消息主体就是文档内容，直接替换
                non_doc_content = re.sub(doc_pattern, '', content, flags=re.DOTALL).strip()
                if not non_doc_content:
                    # 消息全都是文档 → 替换为占位符
                    content = f"[参考了 {doc_count} 篇文档]"
                else:
                    # 混合内容 → 保留非文档部分，文档部分替换
                    content = re.sub(doc_pattern,
                                     lambda m: f"[参考文档: {m.group(2).strip()}]",
                                     content, flags=re.DOTALL)

            # 模式2：工具调用痕迹裁剪
            for tool_name, template in _TOOL_OUTPUT_TEMPLATES.items():
                if tool_name in content:
                    # 检索结果数量提取
                    count_match = re.search(r'(\d+)\s*(?:篇|个|条)', content)
                    count = count_match.group(1) if count_match else "?"
                    replacement = template.format(count=count)
                    # 只替换第一次出现的长文本块
                    if len(content) > 200 and tool_name in content[:50]:
                        content = replacement + content[len(content) // 2:]

            # 如果裁剪后内容仍然很长（>2000 字符），截断中间
            if len(content) > 2000:
                head = content[:800]
                tail = content[-400:]
                content = f"{head}\n...[输出已裁剪]...\n{tail}"

            result.append(AIMessage(content=content))

    return result


# ============================================================
# L4：LLM 摘要压缩
# ============================================================

# 摘要 Prompt
_SUMMARY_PROMPT = """你是一个上下文压缩助手。请将以下对话历史压缩为结构化摘要，保留 5 类关键信息：

1. current_goal：当前目标（用户在咨询什么健康问题）
2. key_findings：关键发现（已确认的症状、诊断结论、重要医学判断）和决策（选择了什么方案）
3. files_referenced：参考过的文档来源列表
4. remaining_work：尚未解决的问题（如：待确认的过敏史、未回答的追问）
5. user_constraints：用户约束（如：过敏药物、年龄、孕哺状态、明确拒绝的治疗方案）

输出格式（严格 JSON）：
```json
{
  "current_goal": "...",
  "key_findings": ["...", "..."],
  "files_referenced": ["...", "..."],
  "remaining_work": ["...", "..."],
  "user_constraints": ["...", "..."]
}
```

对话历史：
{history}
"""


def apply_l4_llm_summary(messages: List[BaseMessage],
                          total_threshold: int = L4_TOTAL_THRESHOLD,
                          llm=None) -> List[BaseMessage]:
    """L4：总长度超阈值时，LLM 生成摘要保留 5 类关键信息

    规则：
    - 计算 messages 总字符数
    - 如果超过 total_threshold（默认 50000），触发 LLM 摘要
    - 摘要替换中间所有消息，保留第一条 HumanMessage + 最后一条 AIMessage
    - 原始对话存入磁盘（transcript），摘要中附带 transcript 路径

    如果 llm 为 None，则跳过 L4（无法生成摘要）
    """
    total_chars = sum(len(str(m.content)) for m in messages)

    if total_chars <= total_threshold:
        logger.debug(f"L4 未触发：总字符数 {total_chars} ≤ {total_threshold}")
        return messages

    if llm is None:
        logger.warning("L4 需要压缩但无 LLM 实例，跳过摘要生成")
        return messages

    logger.info(f"L4 触发：总字符数 {total_chars} > {total_threshold}，开始 LLM 摘要")

    # 原始对话存入磁盘（transcript）
    os.makedirs(PERSIST_DIR, exist_ok=True)
    transcript_id = f"transcript_{uuid.uuid4().hex[:8]}_{int(time.time())}"
    transcript_path = os.path.join(PERSIST_DIR, f"{transcript_id}.json")

    transcript_data = {
        "id": transcript_id,
        "type": "conversation_transcript",
        "messages": [
            {"role": m.__class__.__name__, "content": str(m.content)[:5000]}
            for m in messages
        ],
        "total_chars": total_chars,
        "created_at": time.time(),
    }
    try:
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"transcript 写入失败：{e}")

    # 准备历史文本给 LLM
    history_text = ""
    for msg in messages:
        role = "用户" if isinstance(msg, HumanMessage) else "助手" if isinstance(msg, AIMessage) else "系统"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        # 限制单条消息长度，避免 LLM 输入过长
        if len(content) > 2000:
            content = content[:1000] + "...[省略]..." + content[-500:]
        history_text += f"{role}：{content}\n\n"

    # LLM 生成摘要
    try:
        from app.graph.nodes.structured_output import invoke_structured
        from app.graph.nodes.models import ContextSummaryOutput

        prompt = _SUMMARY_PROMPT.format(history=history_text[:20000])  # 限制输入长度
        summary_result = invoke_structured(llm, prompt, ContextSummaryOutput, max_attempts=1)

        if summary_result:
            summary_text = (
                f"【对话摘要】\n"
                f"当前目标：{summary_result.current_goal}\n"
                f"关键发现：{', '.join(summary_result.key_findings)}\n"
                f"参考文档：{', '.join(summary_result.files_referenced)}\n"
                f"待解决问题：{', '.join(summary_result.remaining_work)}\n"
                f"用户约束：{', '.join(summary_result.user_constraints)}\n"
                f"[完整对话记录: {transcript_id}]"
            )
        else:
            raise ValueError("invoke_structured 返回 None")

    except Exception as e:
        logger.warning(f"L4 LLM 摘要生成失败，使用规则摘要：{e}")

        # 降级：规则提取摘要（不用 LLM）
        summary_text = _rule_based_summary(messages, transcript_id)

    # 构建压缩后的消息列表
    result = []

    # 保留第一条 HumanMessage
    first_human = None
    for msg in messages:
        if isinstance(msg, HumanMessage):
            first_human = msg
            break

    # 添加摘要作为 SystemMessage
    summary_msg = SystemMessage(content=summary_text)
    summary_msg.name = "context_summary"

    # 保留最后一条 AIMessage
    last_ai = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            last_ai = msg
            break

    if first_human:
        result.append(first_human)
    result.append(summary_msg)
    if last_ai:
        result.append(last_ai)

    compressed_chars = sum(len(str(m.content)) for m in result)
    logger.info(
        f"L4 摘要压缩完成：{len(messages)} → {len(result)} 条消息，"
        f"{total_chars} → {compressed_chars} 字符"
        f"（节省 {(total_chars - compressed_chars) / max(total_chars, 1) * 100:.0f}%）"
    )

    return result


def _rule_based_summary(messages: List[BaseMessage], transcript_id: str) -> str:
    """L4 降级：规则提取摘要（不用 LLM）

    当 LLM 摘要失败时使用，从消息中提取 5 类关键信息
    """
    current_goal = ""
    key_findings = []
    files_referenced = set()
    remaining_work = []
    user_constraints = []

    import re

    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)

        if isinstance(msg, HumanMessage):
            # 第一条 HumanMessage 作为 current_goal
            if not current_goal:
                current_goal = content[:100]
            # 提取用户约束（含"不要""不能""过敏"等）
            constraint_patterns = r'(不要|不能|过敏|忌|禁|孕|哺乳|儿童|老人)'
            constraints = re.findall(constraint_patterns, content)
            if constraints:
                user_constraints.append(content[:80])

        elif isinstance(msg, AIMessage):
            # 提取文档引用
            doc_refs = re.findall(r'来源[：:]\s*(\S+)', content)
            files_referenced.update(doc_refs)
            # 提取关键发现（含医学判断的句子）
            finding_patterns = r'((?:诊断|确诊|建议|推荐|应当|需要|避免|禁忌)[^。\n]{5,50})'
            findings = re.findall(finding_patterns, content)
            key_findings.extend(findings[:3])  # 每条 AI 消息最多 3 个发现

    # 限制各字段长度
    key_findings = key_findings[:5]
    files_referenced = list(files_referenced)[:5]

    return (
        f"【对话摘要（规则提取）】\n"
        f"当前目标：{current_goal}\n"
        f"关键发现：{', '.join(key_findings) if key_findings else '无'}\n"
        f"参考文档：{', '.join(files_referenced) if files_referenced else '无'}\n"
        f"待解决问题：{', '.join(remaining_work) if remaining_work else '无'}\n"
        f"用户约束：{', '.join(user_constraints) if user_constraints else '无'}\n"
        f"[完整对话记录: {transcript_id}]"
    )


# ============================================================
# 统一入口：四层压缩 Pipeline
# ============================================================

def compress_context(messages: List[BaseMessage],
                     llm=None,
                     l3_threshold: int = L3_SIZE_THRESHOLD,
                     l4_threshold: int = L4_TOTAL_THRESHOLD) -> List[BaseMessage]:
    """四层上下文压缩 Pipeline（L1 → L3 → L2 → L4）

    执行顺序说明：
    1. L1（中间输出清除）：先清理最大的噪声源——中间推理过程
    2. L3（大输出持久化）：对清理后仍超大的内容写入磁盘
    3. L2（工具调用裁剪）：对持久化后的内容进一步精简工具输出
    4. L4（LLM 摘要）：最后检查总量，仍超阈值则 LLM 压缩

    Args:
        messages: 原始消息列表
        llm: LLM 实例（L4 需要，为 None 则跳过 L4）
        l3_threshold: L3 持久化阈值（字节）
        l4_threshold: L4 总量阈值（字符数）

    Returns:
        压缩后的消息列表
    """
    if not messages:
        return messages

    original_chars = sum(len(str(m.content)) for m in messages)

    # L1：中间输出清除
    result = apply_l1_intermediate_cleanup(messages)

    # L3：大输出持久化
    result = apply_l3_persist_large_outputs(result, threshold=l3_threshold)

    # L2：工具调用输出裁剪
    result = apply_l2_tool_output_trim(result)

    # L4：LLM 摘要
    result = apply_l4_llm_summary(result, total_threshold=l4_threshold, llm=llm)

    final_chars = sum(len(str(m.content)) for m in result)
    if original_chars > final_chars:
        logger.info(
            f"上下文压缩完成：{len(messages)} → {len(result)} 条消息，"
            f"{original_chars} → {final_chars} 字符"
            f"（节省 {(original_chars - final_chars) / max(original_chars, 1) * 100:.0f}%）"
        )

    return result
