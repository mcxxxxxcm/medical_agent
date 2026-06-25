"""流式回答编排器

将流式接口的节点编排逻辑从 routes.py 中独立出来，
作为 Graph 定义的唯一流式消费者，消除双维护问题。

设计原则：
    1. 编排器是 routes.py 和 graph.py 之间的桥梁
    2. 节点调用顺序与 graph.py 的边定义保持一致
    3. 缓存策略（L0/L2）在编排器内部处理，不污染 Graph 定义
    4. 启动时 validate_streaming_sync() 自动检测节点不一致
"""

import asyncio
import json
import time

from langchain_core.messages import AIMessage, HumanMessage

from app.cache.redis_cache import get_cache
from app.cache.semantic_cache import get_semantic_cache
from app.core.app_logging import get_logger
from app.core.config import get_config
from app.graph.graph import get_graph
from app.graph.nodes import (
    grade_documents_node,
    knowledge_retrieval_node,
    memory_load_node,
    profile_extraction_node,
    query_rewrite_node,
    router_node,
    safety_check_node,
    should_update_snapshot,
    stream_answer_generation,
    stream_direct_answer,
    stream_vision_answer,
    symptom_analysis_node,
    update_clinical_snapshot_node,
)

logger = get_logger(__name__)

# L0 答案缓存 TTL（秒）
_ANSWER_CACHE_TTL = 1800  # 30 分钟


def _save_answer_cache(question: str, answer: str):
    """将完整答案写入 L0 答案缓存（仅无用户档案时调用）"""
    try:
        cache = get_cache()
        if cache._available:
            answer_cache_key = f"{cache.prefix}answer:{question}"
            cache._redis.setex(
                answer_cache_key,
                _ANSWER_CACHE_TTL,
                json.dumps(answer, ensure_ascii=False),
            )
            logger.info(f"L0 答案缓存已写入：{question[:30]}...")
    except Exception as e:
        logger.warning(f"L0 答案缓存写入失败：{e}")


class StreamingOrchestrator:
    """流式回答编排器

    负责：
        - 节点调用编排（与 graph.py 边定义一致）
        - 缓存检查（L0 答案缓存 + L2 语义缓存）
        - 对话历史保存
        - 后台快照更新
    """

    def __init__(
        self,
        question: str,
        user_id: str | None,
        thread_id: str,
        image_base64: str | None,
        request_id: str,
        request_start_time: float,
        snapshot_lock: asyncio.Lock,
    ):
        self.question = question
        self.user_id = user_id
        self.thread_id = thread_id
        self.image_base64 = image_base64
        self.request_id = request_id
        self.request_start_time = request_start_time
        self._snapshot_lock = snapshot_lock

        self._first_token_sent = False
        self._full_answer = ""
        self._state: dict = {
            "question": question,
            "user_id": user_id,
            "image_base64": image_base64,
            "thread_id": thread_id,
            "request_id": request_id,
            "warnings": [],
            "sources": [],
            "retrieval_attempts": 0,
        }

    async def _emit(self, payload) -> str:
        """生成 SSE data 行，记录首 token 延迟"""
        if not self._first_token_sent:
            latency_ms = (time.time() - self.request_start_time) * 1000
            logger.info(
                f"首个 token 已发送：request_id={self.request_id}, "
                f"thread_id={self.thread_id}, latency_ms={latency_ms:.2f}"
            )
            self._first_token_sent = True
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    async def _run_safety_review(self) -> tuple:
        """流式输出完成后的安全审查

        对已流式输出的答案执行规则引擎 + LLM 审查。
        如发现风险，返回修正 SSE 事件供调用方 yield。

        Returns:
            (can_cache: bool, revision_sse: str | None)
        """
        if not self._full_answer:
            return True, None

        self._state["final_answer"] = self._full_answer
        review_result = safety_check_node(self._state)

        if "final_answer" in review_result:
            revised = review_result["final_answer"]
            if revised != self._full_answer:
                revision_payload = {
                    "type": "safety_revision",
                    "original": self._full_answer,
                    "revised": revised,
                    "warnings": review_result.get("warnings", []),
                }
                revision_sse = f"data: {json.dumps(revision_payload, ensure_ascii=False)}\n\n"
                logger.info(
                    f"安全审查发现风险，发送修正：request_id={self.request_id}, "
                    f"warnings={review_result.get('warnings', [])}"
                )
                self._full_answer = revised
                return True, revision_sse  # revise 可以缓存修订版

        if any("拦截" in w for w in review_result.get("warnings", [])):
            logger.warning(f"安全审查拦截，不缓存：request_id={self.request_id}")
            return False, None

        return True, None

    async def _load_initial_state(self):
        """加载用户档案和对话历史"""
        memory_state = memory_load_node(self._state)
        if memory_state:
            self._state.update(memory_state)

        try:
            graph = await get_graph()
            checkpoint_config = {"configurable": {"thread_id": self.thread_id}}
            state_snapshot = await graph.aget_state(checkpoint_config)
            if state_snapshot and state_snapshot.values:
                if "messages" in state_snapshot.values:
                    self._state["messages"] = state_snapshot.values["messages"]
                    logger.info(
                        f"从 checkpointer 加载对话历史："
                        f"{len(state_snapshot.values['messages'])} 条消息"
                    )
                if "clinical_checkpoint" in state_snapshot.values:
                    self._state["clinical_checkpoint"] = state_snapshot.values["clinical_checkpoint"]
        except Exception as e:
            logger.warning(f"加载对话历史失败（不影响当前请求）：{e}")

    async def _check_l0_cache(self) -> str | None:
        """仅检查 L0 答案缓存（无 embedding API 调用，<1ms）"""
        try:
            cache = get_cache()
            if not cache._available:
                return None
            answer_cache_key = f"answer:{self.question}"
            cached_raw = cache._redis.get(f"{cache.prefix}{answer_cache_key}")
            if cached_raw:
                answer = json.loads(cached_raw)
                logger.info(f"L0 答案缓存命中：{self.question[:30]}...")
                return answer
            logger.info(f"L0 答案缓存未命中：{self.question[:30]}...")
        except Exception:
            pass
        return None

    async def _check_cache(self, has_profile: bool) -> tuple:
        """并行缓存检查（L0 答案 + L2 语义）"""
        cached_docs = None
        cached_answer = None

        try:
            cache = get_cache()

            # L0 答案缓存（仅无用户档案时）
            if not has_profile:
                answer_cache_key = f"answer:{self.question}"
                try:
                    if cache._available:
                        cached_raw = cache._redis.get(f"{cache.prefix}{answer_cache_key}")
                        if cached_raw:
                            cached_answer = json.loads(cached_raw)
                            logger.info(f"L0 答案缓存命中：{self.question[:30]}...")
                            return cached_docs, cached_answer
                        logger.info(f"L0 答案缓存未命中：{self.question[:30]}...")
                except Exception:
                    pass
            else:
                logger.info("L0 答案缓存跳过（用户有档案，答案个性化不可缓存）")

            # L2 语义相似匹配（文档级）
            config = get_config()
            if getattr(config, "ENABLE_SEMANTIC_CACHE", False) and cache._available:
                semantic_cache = get_semantic_cache()
                try:
                    # 用 SCARD 替代 SCAN，O(1) 判断集合是否为空
                    key_count = cache._redis.scard(semantic_cache._keys_set) if cache._available else 0
                    if not key_count:
                        logger.info("L2 语义缓存为空，跳过 Embedding 计算")
                    else:
                        query_embedding = semantic_cache.get_embedding(self.question)
                        l2_result = semantic_cache.get(
                            self.question, query_embedding=query_embedding
                        )
                        if l2_result:
                            cached_docs, l2_meta = l2_result
                            logger.info(
                                f"早期缓存 L2 命中(相似度 {l2_meta.get('similarity', 0):.1%})，"
                                f"跳过路由+重写：{self.question[:30]}..."
                            )
                except Exception as l2_err:
                    logger.warning(f"L2 缓存检查异常，跳过：{l2_err}")
        except Exception as cache_err:
            logger.warning(f"缓存检查异常：{cache_err}")

        return cached_docs, cached_answer

    async def _run_route_sync(self) -> "Command":
        """在线程池中运行同步路由"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, router_node, dict(self._state))

    def _emit_sources_event(self, docs):
        """推送文档来源元数据 SSE 事件"""
        sources = [
            {
                "source": doc.metadata.get("source", "未知来源"),
                "file_path": doc.metadata.get("file_path", ""),
            }
            for doc in docs
        ]
        if sources:
            return f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"
        return ""

    def _build_clarification_answer(self) -> str:
        """构建澄清追问回复（替代低分时的自由生成，消除幻觉出口）

        当检索文档评分极低或为空时，不进入 LLM 自由生成，
        而是返回结构化的澄清追问，引导用户补充信息。
        """
        question = self.question
        final_question = self._state.get("final_question") or question
        rewritten = self._state.get("rewritten_query") or question

        # 构建上下文提示：如果重写后的问题与原问题不同，说明依赖了对话历史
        context_hint = ""
        if final_question != question:
            context_hint = f"您提到的「{question}」，我理解为「{final_question}」。"

        return (
            f"{context_hint}"
            f"暂时没有在知识库中找到与您的问题直接相关的可靠资料。\n\n"
            f"为了更准确地帮助您，请补充以下信息：\n"
            f"1. 具体的症状表现（如头痛、发烧等）\n"
            f"2. 症状持续时间\n"
            f"3. 是否有已确诊的疾病\n"
            f"4. 想了解的具体方面（如治疗方法、用药建议、注意事项等）\n\n"
            f"如症状明显加重、持续不缓解或伴有高热/剧烈疼痛/呼吸困难，请及时就医。"
        )

    def _record_low_score_bad_case(self):
        """记录低分 bad case（检索低分但未触发澄清 → 幻觉出口）"""
        try:
            from app.memory import get_long_term_memory
            memory = get_long_term_memory()
            user_id = self._state.get("user_id") or "anonymous"
            memory.append_bad_case(
                case_type="low_score_no_clarify",
                original_query=self.question,
                rewritten_query=self._state.get("rewritten_query") or self.question,
                final_question=self._state.get("final_question") or self.question,
                route="rag_low_score",
                user_id=user_id,
                thread_id=self.thread_id,
                metadata={
                    "grade_next": "direct_answer",
                    "retrieved_docs_count": len(self._state.get("retrieved_docs") or []),
                },
            )
        except Exception as e:
            logger.warning(f"低分 bad case 记录失败：{e}")

    def _check_hallucination(self, answer: str):
        """检测答案中的医疗实体是否来自检索文档（忠实度检测）

        扫描答案中的药物/症状实体，与检索文档内容比对。
        如果答案提到文档中不存在的具体药物名，记录为疑似幻觉。
        """
        if not answer or not self._state.get("retrieved_docs"):
            return

        from app.graph.nodes.helpers import _DRUG_KEYWORDS

        docs_text = " ".join(
            doc.page_content for doc in self._state["retrieved_docs"]
            if hasattr(doc, "page_content")
        )

        drugs_in_answer = {kw for kw in _DRUG_KEYWORDS if kw in answer}
        drugs_in_docs = {kw for kw in _DRUG_KEYWORDS if kw in docs_text}
        unsupported_drugs = drugs_in_answer - drugs_in_docs

        if unsupported_drugs:
            logger.info(
                f"疑似幻觉检测：答案提到 {unsupported_drugs}，"
                f"但检索文档中未出现这些药物名"
            )
            try:
                from app.memory import get_long_term_memory
                memory = get_long_term_memory()
                user_id = self._state.get("user_id") or "anonymous"
                memory.append_bad_case(
                    case_type="hallucination_suspected",
                    original_query=self.question,
                    rewritten_query=self._state.get("rewritten_query") or self.question,
                    final_question=self._state.get("final_question") or self.question,
                    answer_preview=answer[:200],
                    user_id=user_id,
                    thread_id=self.thread_id,
                    metadata={
                        "unsupported_drugs": list(unsupported_drugs),
                        "retrieved_docs_count": len(self._state["retrieved_docs"]),
                    },
                )
            except Exception as e:
                logger.warning(f"幻觉 bad case 记录失败：{e}")

    def _record_retrieval_miss(self):
        """记录检索完全失败（零文档召回）的 bad case"""
        try:
            from app.memory import get_long_term_memory
            memory = get_long_term_memory()
            user_id = self._state.get("user_id") or "anonymous"
            memory.append_bad_case(
                case_type="retrieval_miss",
                original_query=self.question,
                rewritten_query=self._state.get("rewritten_query") or self.question,
                final_question=self._state.get("final_question") or self.question,
                user_id=user_id,
                thread_id=self.thread_id,
                metadata={
                    "has_drugs_in_query": any(
                        kw in self.question for kw in [
                            "布洛芬", "对乙酰氨基酚", "阿莫西林", "头孢", "阿司匹林"
                        ]
                    ),
                    "retrieval_attempts": self._state.get("retrieval_attempts", 0),
                },
            )
        except Exception as e:
            logger.warning(f"检索 miss bad case 记录失败：{e}")

    def _check_route_misclassification(self, route_type: str):
        """检测路由异常：问题含症状词但被路由到 direct_answer"""
        if route_type != "direct_answer":
            return

        _core_symptoms = [
            "头痛", "头疼", "肚子疼", "腹痛", "胃痛", "发烧", "发热",
            "咳嗽", "恶心", "呕吐", "腹泻", "拉肚子", "胸闷", "胸痛",
            "过敏", "头晕", "失眠", "乏力", "咽痛", "喉咙痛", "流鼻涕",
            "关节痛", "肌肉酸痛", "呼吸困难", "心悸", "便血", "皮疹",
        ]
        has_symptom = any(s in self.question for s in _core_symptoms)
        if not has_symptom:
            return

        logger.info(
            f"路由异常检测：问题含症状词但路由到 direct_answer，"
            f"question={self.question[:30]}..."
        )
        try:
            from app.memory import get_long_term_memory
            memory = get_long_term_memory()
            user_id = self._state.get("user_id") or "anonymous"
            memory.append_bad_case(
                case_type="route_misclassification",
                original_query=self.question,
                rewritten_query="",
                final_question="",
                route=route_type,
                user_id=user_id,
                thread_id=self.thread_id,
                metadata={
                    "matched_symptoms": [s for s in _core_symptoms if s in self.question],
                },
            )
        except Exception as e:
            logger.warning(f"路由异常 bad case 记录失败：{e}")

    async def _run_rag_pipeline(self, route_node: str):
        """执行 RAG 流水线：症状解析 → 重写 → 检索 → 评分（含自纠正）"""
        if route_node == "symptom_analysis":
            symptom_state = symptom_analysis_node(self._state)
            if symptom_state:
                self._state.update(symptom_state)

        rewrite_state = query_rewrite_node(self._state)
        if rewrite_state:
            self._state.update(rewrite_state)

        retrieval_state = knowledge_retrieval_node(self._state)
        if retrieval_state:
            self._state.update(retrieval_state)

        grade_command = grade_documents_node(self._state)
        grade_update = getattr(grade_command, "update", None) or {}
        if grade_update:
            self._state.update(grade_update)

        grade_next = getattr(grade_command, "goto", "answer_generation")
        logger.info(
            f"文档评分完成：request_id={self.request_id}, thread_id={self.thread_id}, "
            f"next_node={grade_next}, docs={len(self._state.get('retrieved_docs') or [])}"
        )

        # 自纠正：文档不相关时重写重试一次（含振荡检测）
        if grade_next == "query_rewrite":
            self._state["retrieval_attempts"] = self._state.get("retrieval_attempts", 0) + 1

            rewrite_state = query_rewrite_node(self._state)
            if rewrite_state:
                self._state.update(rewrite_state)

            retrieval_state = knowledge_retrieval_node(self._state)
            if retrieval_state:
                self._state.update(retrieval_state)

            grade_command = grade_documents_node(self._state)
            grade_update = getattr(grade_command, "update", None) or {}
            if grade_update:
                self._state.update(grade_update)
            grade_next = getattr(grade_command, "goto", "answer_generation")

        return grade_next

    async def _save_history(self):
        """保存对话历史到 checkpointer，触发后台快照更新"""
        try:
            graph = await get_graph()
            checkpoint_config = {"configurable": {"thread_id": self.thread_id}}

            # 追加用户消息
            await graph.aupdate_state(
                checkpoint_config,
                {"messages": [HumanMessage(content=self.question)]},
                as_node="memory_load",
            )

            # 追加 AI 回答
            if self._full_answer:
                await graph.aupdate_state(
                    checkpoint_config,
                    {"messages": [AIMessage(content=self._full_answer)]},
                    as_node="answer_generation",
                )

            # 后台快照更新（不阻塞响应）
            asyncio.create_task(self._background_snapshot_update(graph, checkpoint_config))
            logger.info("对话历史已保存")
        except Exception as e:
            logger.warning(f"保存对话历史失败（不影响当前回答）：{e}")

    async def _background_snapshot_update(self, graph, checkpoint_config):
        """后台异步更新临床快照"""
        try:
            await asyncio.wait_for(self._snapshot_lock.acquire(), timeout=0.01)
        except asyncio.TimeoutError:
            logger.info(f"快照更新已在进行中，跳过本次（thread_id={self.thread_id}）")
            return

        try:
            state_snapshot = await graph.aget_state(checkpoint_config)
            if not state_snapshot or not state_snapshot.values:
                return

            bg_state = {
                "messages": state_snapshot.values.get("messages", []),
                "clinical_checkpoint": state_snapshot.values.get("clinical_checkpoint"),
            }

            snapshot_decision = should_update_snapshot(bg_state)
            if snapshot_decision != "update_snapshot":
                logger.info("快照已由前次任务处理，无需重复更新")
                return

            logger.info("后台触发临床状态快照更新")
            snapshot_result = update_clinical_snapshot_node(bg_state)
            if snapshot_result:
                await graph.aupdate_state(
                    checkpoint_config,
                    snapshot_result,
                    as_node="update_snapshot",
                )
                removed = len(snapshot_result.get("messages", []))
                logger.info(f"临床状态快照已更新，删除 {removed} 条早期消息")
        except Exception as e:
            logger.warning(f"后台快照更新失败（不影响后续对话）：{e}")
        finally:
            self._snapshot_lock.release()

    async def run(self):
        """主入口：执行完整的流式编排，yield SSE 事件字符串"""
        try:
            # 阶段 1：加载初始状态
            await self._load_initial_state()

            has_profile = bool(self._state.get("user_profile"))

            # 阶段 2：路由优先 → 按路由类型决定缓存策略
            route_command = None
            cached_docs = None
            cached_answer = None

            if self.image_base64:
                route_command = router_node(self._state)
            else:
                cache_check_start = time.time()

                # 先跑路由（规则+上下文 0ms，LLM 1-2s）
                route_command = await self._run_route_sync()
                route_type = getattr(route_command, "goto", "direct_answer") if route_command else "direct_answer"

                # 按路由类型决定缓存深度，避免无意义的 embedding API 调用
                # symptom/direct_answer：追问高度个性化，语义缓存几乎不可能命中 → 仅 L0
                # knowledge：知识查询常重复 → L0 + L2 语义缓存
                if route_type == "knowledge":
                    cached_docs, cached_answer = await self._check_cache(has_profile)
                elif not has_profile:
                    # symptom/general: 只查 L0 答案缓存（无需 embedding API）
                    cached_answer = await self._check_l0_cache()

                cache_check_ms = (time.time() - cache_check_start) * 1000
                logger.info(f"路由+缓存耗时：{cache_check_ms:.2f}ms（route={route_type}）")

            # 阶段 3：按缓存/路由结果分发处理
            if cached_answer:
                self._full_answer = cached_answer
                yield await self._emit(cached_answer)

            elif cached_docs:
                self._state["retrieved_docs"] = cached_docs
                self._state["rewritten_query"] = self.question
                logger.info(
                    f"缓存命中，直接进入答案生成：request_id={self.request_id}, "
                    f"docs={len(cached_docs)}"
                )
                sources_event = self._emit_sources_event(cached_docs)
                if sources_event:
                    yield sources_event
                self._full_answer = ""
                async for token in stream_answer_generation(self._state):
                    self._full_answer += token
                    yield await self._emit(token)
                if self._full_answer:
                    can_cache, revision_sse = await self._run_safety_review()
                    if revision_sse:
                        yield revision_sse
                    if can_cache and not has_profile:
                        _save_answer_cache(self.question, self._full_answer)
                    self._check_hallucination(self._full_answer)

            else:
                next_node = (
                    getattr(route_command, "goto", "direct_answer")
                    if route_command
                    else "direct_answer"
                )
                logger.info(
                    f"流式请求路由完成：request_id={self.request_id}, "
                    f"thread_id={self.thread_id}, next_node={next_node}"
                )

                if next_node == "direct_answer":
                    self._check_route_misclassification(next_node)
                    self._full_answer = ""
                    async for token in stream_direct_answer(self._state):
                        self._full_answer += token
                        yield await self._emit(token)
                    if self._full_answer:
                        can_cache, revision_sse = await self._run_safety_review()
                        if revision_sse:
                            yield revision_sse
                        if can_cache and not has_profile:
                            _save_answer_cache(self.question, self._full_answer)

                elif next_node == "vision_analysis":
                    logger.info(
                        f"开始流式图片问诊：request_id={self.request_id}, "
                        f"thread_id={self.thread_id}"
                    )
                    self._full_answer = ""
                    async for token in stream_vision_answer(self._state):
                        self._full_answer += token
                        yield await self._emit(token)
                    if self._full_answer:
                        can_cache, revision_sse = await self._run_safety_review()
                        if revision_sse:
                            yield revision_sse

                else:
                    grade_next = await self._run_rag_pipeline(next_node)

                    if grade_next == "answer_generation" and self._state.get("final_answer"):
                        self._full_answer = self._state["final_answer"]
                        yield await self._emit(self._full_answer)
                        can_cache, revision_sse = await self._run_safety_review()
                        if revision_sse:
                            yield revision_sse
                        if can_cache and not has_profile:
                            _save_answer_cache(self.question, self._full_answer)

                    elif self._state.get("retrieved_docs"):
                        logger.info(
                            f"开始流式生成 RAG 答案：request_id={self.request_id}, "
                            f"thread_id={self.thread_id}"
                        )
                        sources_event = self._emit_sources_event(
                            self._state.get("retrieved_docs", [])
                        )
                        if sources_event:
                            yield sources_event
                        self._full_answer = ""
                        async for token in stream_answer_generation(self._state):
                            self._full_answer += token
                            yield await self._emit(token)
                        if self._full_answer:
                            can_cache, revision_sse = await self._run_safety_review()
                            if revision_sse:
                                yield revision_sse
                            if can_cache and not has_profile:
                                _save_answer_cache(self.question, self._full_answer)
                            self._check_hallucination(self._full_answer)

                    else:
                        # 无检索文档 → 澄清追问而非自由生成（消除幻觉出口）
                        logger.info(
                            f"无检索文档，返回澄清追问：request_id={self.request_id}, "
                            f"thread_id={self.thread_id}"
                        )
                        self._record_low_score_bad_case()
                        self._record_retrieval_miss()
                        clarification = self._build_clarification_answer()
                        self._full_answer = clarification
                        yield await self._emit(clarification)

            # 阶段 4：收尾
            total_elapsed_ms = (time.time() - self.request_start_time) * 1000
            logger.info(
                f"流式请求完成：request_id={self.request_id}, "
                f"thread_id={self.thread_id}, total_elapsed_ms={total_elapsed_ms:.2f}"
            )
            yield "data: [DONE]\n\n"

            await self._save_history()

            # 档案提取（回答后异步，不阻塞用户）
            try:
                profile_state = profile_extraction_node(self._state)
                if profile_state:
                    self._state.update(profile_state)
                    logger.info(f"档案提取完成（回答后）：request_id={self.request_id}")
            except Exception as profile_err:
                logger.warning(f"档案提取失败（不影响回答）：{profile_err}")

        except Exception as e:
            logger.error(
                f"流式处理失败：request_id={self.request_id}, error={e}", exc_info=True
            )
            yield f"data: [ERROR] {str(e)}\n\n"
