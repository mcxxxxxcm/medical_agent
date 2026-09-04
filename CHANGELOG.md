# 系统优化更新日志

## v9.40 - 修复本地模型查询改写全失败导致的多轮对话幻觉（多个文件）

背景：多轮对话第二轮追问（如"服用这个药需要注意啥？"）总是检索出冠心病/糖尿病等无关文档并生成幻觉答案。`logs/error.log` 每轮都刷 `查询重写失败：QueryRewriteOutput 结构化输出失败（2次尝试 × 3种策略），最后错误：None`。根因是**设计矛盾**：`QUERY_REWRITE_PROMPT` 要求模型输出 `FINAL:`/`SEARCH:` 两行纯文本，但 `query_rewrite_node` 却用 `get_local_llm_json()` —— 该函数强制 `response_format={"type":"json_object"}`。本地 qwen2.5:1.5b 被夹在矛盾指令中间，输出的 JSON 键名永远是 `FINAL`/`SEARCH`，与 `QueryRewriteOutput` 期望的 `final_question`/`search_keywords` 对不上，三层降级策略全部校验失败（最后错误为 None 说明是解析/校验失败而非请求失败）。except 分支随后把 `rewritten_query=final_question=原残缺问题` 拿去检索，缺实体（无"二甲双胍"）召回无关文档，幻觉由此产生。

- **对齐 Prompt 与 JSON Schema**：`QUERY_REWRITE_PROMPT`（app/graph/nodes/prompts.py）改为要求输出合法 JSON 对象，键名与 Pydantic 模型完全一致（`final_question`/`search_keywords`），并保留 Ollama json_object 识别所需的 "JSON" 字样；示例中的 JSON 字面量用 `{{}}` 转义避免 `str.format` 展开报错。模型从此不再收到矛盾的输出格式指令。
- **search_keywords 输入容错**：`QueryRewriteOutput`（app/graph/nodes/models.py）的 `strip_search_prefix` 校验器扩展为兼容 list/dict 输入——qwen 若把关键词输出成数组或字典，统一归一为空格分隔字符串，避免 `model_validate` 因类型不符而失败。
- **改写失败时的上下文规则兜底**：`query_rewrite_node` 的 except 分支（app/graph/nodes/nodes.py）不再裸用残缺问题，改为 `_context_fallback_rewrite`——复用 AC 自动机（`get_drug_matcher`/`get_symptom_matcher`）从最近 6 条历史提取药物/症状实体（如"二甲双胍"），拼进 `final_question`（"…（历史提到二甲双胍）"）与 `rewritten_query`（"… 二甲双胍"），保证至少能召回历史实体对应的文档，杜绝检索无关内容。仅当问题含指代词/省略结构且历史能补出实体时触发，否则保持原题（语义不变）。
- 语义不变：不改检索环节其他逻辑；改写走通时完全走原路径，兜底仅在最坏情况下介入。

<footer>查询改写修复 · 改动文件：`app/graph/nodes/prompts.py`、`app/graph/nodes/models.py`、`app/graph/nodes/nodes.py`</footer>

## v9.39 - 修复元数据交叉校验总体置信度统计异常（app/rag/metadata_extractor.py）

背景：入库时日志反复刷 `文档元数据自动提取失败（不影响入库）：'tuple' object has no attribute 'endswith'`（每篇文档一条）。虽然被外层 `loader.py` 的 try/except 降级接住不影响入库，但 `overall_confidence` 永远算不出来，属于跨战评审时无法给出整体可信度的隐性 bug。

- **根因**：`cross_validate_metadata` 统计置信度字段数时，第 `493` 行误写成 `for k in result.items()`，迭代出来的是 `(key, value)` 元组，再对其调 `.endswith("_confidence")` 抛 `'tuple' object has no attribute 'endswith'`。同段第 `492` 行是正确的双变量解包 `for k, v in result.items()`。
- **修复**：改为 `for k in result`（迭代 dict 的 key），置信度统计恢复正常。
- 全仓已扫描无其他 `for x in dict.items()` 单变量误用。
- 语义不变：仅修正统计逻辑，提取规则与字段写入完全不动。

<footer>元数据修复 · 改动文件：`app/rag/metadata_extractor.py`</footer>

## v9.37 - 前端 AI 回复结构化排版美化（app/static/index.html）

背景：此前 `formatAnswer` 仅把换行转 `<br>`、把 `###`/`**` 转 `<strong>`，AI 回复里常见的分级标题、有序/无序列表、代码块、引用块全部堆成纯文本；且直接拼接 `innerHTML`、未做 HTML 转义，若回复含标签存在 XSS 隐患。

- **引入标准 Markdown 渲染库 markdown-it + DOMPurify**：库文件**内置到本地** `app/static/lib/`（`markdown-it.min.js` + `purify.min.js`，合计约 145KB），head 引入，运行时**不依赖外网 CDN**。`formatAnswer` 用 `markdownit({html:false, breaks:true, linkify:true})` 渲染完整 CommonMark（标题、有序/无序列表、代码块、引用、表格、粗斜体、行内代码、链接），再经 `DOMPurify.sanitize` 二次加固；渲染库加载失败时降级为换行+粗体。`html:false` + 转义双保险杜绝注入。
- **免责声明（⚠️）提取为 Alert 框**：在**原文阶段**（markdown-it 渲染前）就按行切走独立的 `⚠️ ...` 段，单独渲染为 `<div class="md-alert">`，再用私有区占位符回填原位置。相比在渲染后 HTML 上做正则，此方案**彻底规避"⚠️ 被 `**` 加粗包裹"导致失配**的问题（多数模型会加粗免责声明）。内联靠句尾的免责声明（如"…退烧药。⚠️ 以上建议仅供参考"）经 `normalizeMarkdown` 补换行自动独立成块，始终切入警示框。
- **AI 涂抹格式的前端兜底**：新增 `normalizeMarkdown()`，在渲染前修复模型常见的冒格式问题——① 句末标点后紧跟 `- `/`1. ` 而未换行时（如"…促进新陈代谢。- 饮食：…"）补空行，使粘连的列表项正确渲染为 `ul/li`；② 列表行后紧跟非列表正文时补空行，避免 markdown 惰性续行把后续段落并进列表项；③ 内联 ⚠️ 补换行独立成块，并先剥离 `** ⚠️ **` 的粗体标记（防止正文残留孤立的 `**`）。全部用 `[^\S\n]*` 不跨行，避免误伤本就正确的换行结构。
- **参考来源改为胶囊标签**：`renderSources` 改为「📚 参考来源」标签 + 一条胶囊条目（`border-radius:999px`、紫色底、数字圆角徽章），hover 反色，替代原来的细长条形引用。
- **排版润色**：正文 `line-height:1.7`、段落间 `margin-bottom:10px`、正文色 `#333`、标题/列表/表格/引用/链接统一风格化。
- **顺带修复既有 XSS**：此前 LLM/用户内容直接拼 `innerHTML`，现一律经 markdown-it 转义 + DOMPurify 清洗。
- 语义不变：纯前端渲染层增强，服务端回复内容与接口完全不动。

<footer>前端渲染美化 · 改动文件：`app/static/index.html`（新增 `app/static/lib/`）</footer>

## v9.38 - 知识库重建增加实时进度反馈（app/static/admin.html）

背景：重建接口 `kb_rebuild` 是同步跑完才返回的（`async def` 所有步骤做完才 return），前端此前点「重建知识库」只发一次 POST 就静默挂起——期间 `_kb_update_status.progress` 明明在更新，却没有刷新入口。用户点击后完全不知道是否触发、跑到哪一步。

- **轮询进度**（`confirmRebuild` → 新增 `pollRebuildProgress`）：点重建时立即开 `setInterval(1s)` 并行轮询 `/api/admin/kb/status`，读取 `update_status.updating/progress/error`，实时刷新进度条与文案；POST 返回后 `stopRebuildPolling()` 停止轮询。
- **进度条 UI**：在文档列表操作区下新增 `#rebuildProgress` 进度条（复用 `.upload-progress/.progress-bar` 样式）。能解析出「（done/total）」的阶段（写入影子集合）精确填充百分比；其余阶段（加载/切分/构建索引/校验/切换/清理）用 `indeterminate` 不确定动画。按钮在运行期间置灰防重复触发（后端本有 409 兜底）。
- **完成/失败态**：后端 `finally` 保留 `progress="完成"/"失败：…"`,轮询据此展示「重建完成/失败:…」后隐藏进度条;POST 返回后叠加成功/失败 toast 与 `refreshStatus()`。
- 语义不变：纯前端反馈增强,重建流程与后端逻辑零改动。
- **修复确认弹窗回调被吞的既有 bug**：`executeConfirm()` 原为 `closeConfirm(); if (_confirmAction) _confirmAction();`,而 `closeConfirm()` 先把 `_confirmAction` 置 `null`,导致**所有确认弹窗操作（重建/回滚/删除）点「确认」后回调永不执行**、不发任何请求。改为先捕获 action 再关闭弹窗。此前"点击重建无响应、Network 无新增请求"正是此 bug 所致。

<footer>进度反馈优化 · 改动文件：`app/static/admin.html`</footer>

## v9.36 - RAG 多轮增量：改写后主动澄清 + 显式话题轨迹（多轮对话）

背景：对照"RAG 多轮对话四层策略"分析，补齐两个真缺口——改写层缺"检索前拦截式澄清"，状态层缺"跨轮显式话题轨迹"。按**最保守**门槛落地，绝不骚扰已确立话题的正常对话。

- **改写后主动澄清**（`nodes.py` `query_rewrite_node` + `graph.py`）：改写完成后，仅当「追问(含指代/省略) + 改写结果与原文一致(补不出实体) + 历史/临床快照/症状确无领域实体」三重条件**同时**满足时，拦截短路为澄清，返回 `refusal_type="clarify"` + 澄清文案 + `messages`，复用现有 refusal 短路语义。`graph.py` 将 `query_rewrite → question_decompose` 固定边改为 `route_after_rewrite` 条件路由，命中澄清直接 `END`，不误进 answer_generation 二次生成（symptom 与 knowledge 路径都经过 query_rewrite，统一覆盖）。澄清文案由新增 `_build_clarify_answer` 生成，不透传 LLM 原文、不编造事实。
  - 触发不依赖大改 schema：`models.py` 的 `QueryRewriteOutput` 不变，判断基于改写结果 + 现有 `_has_anaphora_pattern`/`is_same_query`/`_build_rewrite_context`/`_DOMAIN_ENTITY_KEYWORDS`/`clinical_checkpoint`。
  - Bad Case 闭环：澄清触发处新增 `case_type="clarify_triggered"` 记录，供回归审查触发是否恰当。
- **显式话题轨迹**（`state.py` + `nodes.py`）：新增跨轮 state 字段 `current_topic` + `topic_trajectory`（轨迹栈 `[{topic_id, ts, turns}...]`，最近 8 条）。生命周期对齐 `clinical_checkpoint`——每轮由 `query_rewrite_node` 读旧栈、覆盖写回，而非追加纯文本。话题 id 由新增 `_detect_topic` 判定（优先级 症状实体 > 药物(helpers `_DRUG_KEYWORDS`) > 疾病(`_DOMAIN_ENTITY_KEYWORDS`) > 路由类型 > general）。该字段本轮作为对话级结构化基础设施落地，并为将来"会话内检索复用"打底。
- **语义不变**：澄清为额外短路，非澄清查询流程与原先完全一致；话题轨迹为新增旁路字段，不修改检索/改写结果；general/direct 路径不经 query_rewrite，不触发澄清、不更新轨迹。
- **回归测试**：`tests/test_nodes.py` 新增 `TestTopicTrajectory`（话题 id 判定、栈 turns/push/clip）与 `TestProactiveClarify`（追问澄清触发、首轮含实体不澄清、纯指代无历史不澄清、`_context_has_entity` 拦截、`route_after_rewrite` 路由），共 14 例全绿。

<footer>RAG 多轮增量 · 改动文件：`app/graph/state.py`、`app/graph/nodes/nodes.py`、`app/graph/graph.py`、`tests/test_nodes.py`</footer>

## v9.35 - L2 语义缓存判空改为 O(1) + 答案生成异常补栈（性能/健壮性）

- **L2 语义缓存判空 O(1) 化**（`hybrid_retriever.py` + `semantic_cache.py`）：此前每次检索请求都为了「判断 L2 是否为空」对 Redis 前缀做**全量 SCAN**（`while cursor: scan(count=100)` 多批往返，缓存键多时 O(n)）。写入侧本就维护 LRU Sorted Set 与旧 Set，改为新增 `SemanticCache.has_any_key()`（`zcard + scard`，O(1)）在进入 Embedding API 前判空，大幅削减每次检索的 Redis 开销。
- **答案生成异常补 traceback**（`nodes.py` `answer_generation_node`）：顶层 `except` 打日志加 `exc_info=True`，记录完整堆栈。此前仅 `logger.error(str(e))`，偶发的 `KeyError('key')` 只在日志留下孤零零的 `'key'` 字串、完全无法定位根因；带栈后再次出现可直接 grep 定位。
- 语义不变：判空逻辑等价（空则跳过 Embedding 计算）；异常改日志仅加栈、不改行为；`has_any_key()` 在 Redis 不可用时安全降级返回 False（视为空，与旧 SCAN 结果等价）。

<footer>性能与健壮性小项 · 改动文件：`app/rag/hybrid_retriever.py`、`app/cache/semantic_cache.py`、`app/graph/nodes/nodes.py`</footer>

## v9.34 - 成熟度专项：检索并行化 + 高并发数据竞争修复 + 限流泄漏 + PG 降级（性能/健壮性）

背景：对照 RAG 系统成熟度分析（回答准确率 / 响应速率 / 高并发 / 健壮性）落地的第一批 no-regret 修复。聚焦四块：降低检索链路耗时、消除共享可变单例被多请求并发改写的数据竞争、修复限流中间件 IP 表无界膨胀的内存泄漏、补齐 PostgreSQL 不可用时的降级（此前 Redis 有降级而 PG 无，不对称）。

- **检索链路并行化**（`hybrid_retriever.py`）：BM25 稀疏检索提前提交到共享线程池 `_SPARSE_EXECUTOR`（`max_workers=4`），与随后的 Embedding 计算（网络 200~400ms）及 Dense 检索重叠执行，替代原先 dense→sparse 的严格串行，降低检索链路耗时、缩短 TTFB。检索结果由 `_sparse_future.result()` 汇总，语义等价只改时序。
- **共享态数据竞争修复**（`hybrid_retriever.py` + `semantic_cache.py`）：
  - 新增 `_isolate_docs()`：RRF 融合后对候选文档做 metadata 浅拷贝隔离。下游 Reranker（写 `rerank_score`）、来源多样性、版本去重改写的都是副本，不再污染 BM25 复用文档、消除多请求并发 data race。
  - `_EmbeddingLRUCache` 补上 `threading.Lock`（原 docstring 声明"线程安全"但实现无锁），get/put/clear 全部互斥，多请求共享该模块级单例时安全。
  - 移除对 `semantic_cache._embedding_cache`（无界无锁 dict）的写入，并删除该已弃用字段的定义，消除本地 embedding 缓存的无界增长点。
- **限流中间件内存泄漏修复**（`routes.py` `RateLimitMiddleware`）：原 `_cleanup` 仅清理当前请求 IP，其它来源 IP 的旧记录常驻内存、随来源 IP 增多无界膨胀。改为 `_cleanup_all` 定期全量清理（5s 闸门避免每请求 O(n)）+ `MAX_TRACKED_IPS=20000` 上限保护（超限淘汰最久未访问 IP），并用 `threading.Lock` 保护 `_requests` 的 check-then-append 并发。
- **PostgreSQL 检查点内存降级**（`checkpointer.py`）：`get_checkpointer` 的 PG `AsyncPostgresSaver` 初始化包进 try/except，连接失败时降级为 `InMemorySaver`（内存检查点）并打 error 日志，保证对话服务不因数据库故障整体不可用（与 Redis 降级策略对称）；代价是重启后历史会话状态丢失。降级初始化拆出独立 `_init_postgres_checkpointer()`。

语义不变：并行只改检索时序不改变检索结果；文档隔离只隔离对象不改变内容；限流阈值/窗口不变；PG 正常时行为与原先完全一致（降级仅在连接失败时触发）。

<footer>性能与健壮性专项 · 改动文件：`app/rag/hybrid_retriever.py`、`app/cache/semantic_cache.py`、`app/api/routes.py`、`app/memory/checkpointer.py`</footer>

## v9.33 - 首 token 提速：答案生成改为"按段先校验后流出"（性能优化）

背景：v9.31 为根治幻觉段，把答案生成改成"整段缓冲 → `_sanitize_answer` 清洗 → 一次性发出"，代价是生成期前端全程空白（感知首 token ≈ 20s）。经 metrics 定位：`answer_generation` 平均 14s，其内 LLM 生成期因整段缓冲而零流出，成了首 token 的隐形大头。v9.32 确认此慢**不是** L1 上下文压缩（`context_manager.py` 的 L1/L2/L3/L4 压缩在 RAG 流程里运行，但都是廉价字符串操作，L4 LLM 仅超阈值触发）。

- **新增 `_SegmentedEmitter` 分段发射器**（`nodes.py`）：在 LangGraph 流式下一边生成一边攒，攒成一个**逻辑块**就以纯规则走同一套 `_sanitize_answer` 校验，干净才放行：
  - 逻辑块 = 一个 bullet 及其续行（至下一 bullet / 空行 / 标题），散文按句末符（。！？）分段保证响应。
  - 越界症状小节（白名单外）整块剔除、文档外药物整句剔除——**"幻觉段不流出前端"**这一 v9.31 承诺以更细粒度保留。
  - 非流式（ainvoke）下 emit 为 no-op，`clean_parts` 仍累计出清洗后的完整答案。
- **`answer_generation_node` 与 L2 缓存路径 `stream_answer_generation` 统一改用 `_SegmentedEmitter`**：首 token（首个逻辑块）进入流式即流出，不必等全量；同时两路径行为一致。LLM 流式仍打 `TAG_NOSTREAM` 压制消息通道中继，最终文案统一经 `_emit_chunk` 分小段打字机发出。
- **回归测试**：`tests/test_nodes.py` 新增 `TestOffDocMedicationStrip`（文档外用药整句剔除、无残句、文档内药保留、混句摘药魔、用药整行删除）与 `TestSegmentedEmitter`（首 bullet 未等整段即流出、白名单外小节剔除、同义词保留、bullet+散文保留、空白名单不动）。
- 语义不变：幻觉段兜底剔除、多症状白名单约束、安全/分诊/用药三套规则 skill 全部保留，只是把"清洗时点"从"全量之后"前移到"每个逻辑块之后"。

## v9.32 - RAG 回答结构放开：保留逐点 bullet、小节变可选，不再固定三段式（体验优化）

用户反馈：回答每次都固定套用「处理建议 / 须立即就医的情况 / 当前资料未提及的关键方面」三段模板，机械。希望允许适当自由发挥、按点论述，但不要每次都出现这三节。

- **`prompts.py` `RAG_ANSWER_PROMPT` 组织指令重写**：
  - 正文一律用 bullet（"- "）逐条论述、观点分明；小节标题与先后顺序交给模型自由设计，不再强制固定模板；相关才写、无关略去，不为其凑结构硬写。
  - 「**需立即就医**」与「**当前资料未提及的关键方面**」降为**可选小节**：仅当文档确实提到红旗征/危险信号、或确实缺少关键信息时才用小节简述，否则省略。
  - 自由发挥仅限**组织与措辞**，事实性底线不变：每条护理/用药措施仍须逐字出自【文档】，禁止凭常识补数字/时长/步骤。
  - 多症状部分仍强制**bullet（"- 症状名：…"）**形式——这是 L1 白名单指令与 L2 白名单清洗器（`_extract_bullet_symptom_token`/`_strip_out_of_scope_symptom_sections`）能识别并兜底剔除越界症状小节的格式前提，未因自由化而丢失。
- **不变量**：仅改组织方式，事实约束、L1 白名单、安全/分诊/用药三套规则 skill 全部不变。

## v9.31 - 用工程化三层防线彻底根治"多症状幻觉"（Prompt 约束 → 确定性校验 → 全链路清洗）

背景：v9.27/v9.30 只是**提示词约束"多症状逐一作答只针对本次问题"**，LLM 是概率性的，总有一次交互会忽略约束、把【L2 临床快照】里的既往症状（头痛/腹痛/头晕…）当成"本次诉求"逐一作答——用户要求以工程思维**彻底避免**这类幻觉，而非再叠一层 prompt。方案：确定性校验（纯字符串，不依赖 LLM）取代概率性约束作为兜底层。

- **L1·输入侧白名单提纯**（`nodes.py`）：新增 `_SYMPTOM_WHITELIST_WORDS`（内置常见症状词）与 `_get_question_symptom_whitelist()`。白名单 = 症状解析规则结果 `symptoms["symptoms"]`，为空则回退问题文本症状词匹配。白名单由**代码确定**、注入 prompt（`build_rag_prompt` → `{symptom_whitelist_section}`，`prompts.py` `RAG_ANSWER_PROMPT` 新增占位符），限定"本次唯一允许逐一作答的症状集合"，从输入侧压掉根因。
- **L2·生成端确定性校验**（`nodes.py`）：
  - `_norm_symptom` + `_SYMPTOM_CANONICAL`（发烧↔发热、头疼↔头痛、肚子疼↔腹痛…）：统一"问题措辞"与"模型措辞"。
  - `_extract_bullet_symptom_token`：识别"多症状逐一作答"里每个 `- 症状：` 小节标题。
  - `_strip_out_of_scope_symptom_sections`：确定性剔除白名单外（临床快照/历史近年既往、或内置症状词里非本次）的症状小节及其续行；**白名单为空则整段保留**，不冒险误删。
  - `_strip_off_doc_medications`：确定性剔除**检索文档中未出现**的药物（仅当文档可比对时）。
  - `_sanitize_answer` 串起上述两步，返回 `(cleaned, removed_sections, removed_meds)` 供日志/bad-case。
  - **`answer_generation_node` 改造为先校验后发出**：LLM 流式打 `TAG_NOSTREAM`（LangGraph 官方抑制 `stream_mode="messages"` 中继的原语，见 `langgraph/_messages.py StreamMessagesHandler.on_chat_model_start`）→ 原始 token 不再实时流到前端；缓冲全量 → `_sanitize_answer` 清洗 → 清洗后的文本经 `get_stream_writer()`（custom 事件）分小段发出（打字机效果）。前端看到的永远是**校验通过**的内容。自定义事件用 `{"answer_chunk": text}` 包裹，与图片摘要预览（不进累计）区分。
  - `streaming.py` custom 分支识别 `answer_chunk`：累计进 `_full_answer`（供持久化/缓存/后置审查），并逐段 SSE 发出。
  - **L2 语义缓存命中路径同步加固**（`nodes.py` `stream_answer_generation`）：该生成器此前逐 token 裸流 → 改为与 `answer_generation_node` 同一套"缓冲→`_sanitize_answer`→分小段发出"，使 L2 缓存命中走此生成器时同样不会把幻觉段传出。
- **L3·缓存/非流式全链路统一清洗**（`streaming.py` + `nodes.py`）：新增 `sanitize_cached_answer`——L0 答案缓存命中与无 token 兜底 final 同过白名单症状校验；因其无检索文档，**跳过文档外用药剔除**（避免把用户合法用药当幻觉误删，`_strip_off_doc_medications` 需 docs 才能判定）。旧的（修复前）缓存答案同样被清洗。
- **回归测试**：`tests/test_nodes.py` 新增 `TestSymptomWhitelistSanitizer`，覆盖词匹配白名单、同义词归一化、越界小节剔除、同名高频发热保留、空白名单不动、无文档不误删用药。
- 语义不变：真实"多症状当前问题"白名单含多个症状，仍逐一作答；单症状问题只答该症状。

## v9.30 - 单症状问题不再被临床快照带偏成"多症状逐一作答"（Bug 修复）

基于 errorLog 分析（多轮线程 `thread_test_user` 中仅问"发烧了怎么办？"，答案却按"多症状逐一作答"列出头痛/发热/腹痛/头晕四条）：`build_rag_prompt` 会把整张【L2 临床快照】注入 prompt，罗列该线程累计的所有症状；v9.27 的"多症状逐一作答"指令让模型把这些既往背景症状也当成"本次要处理的多症状"逐一给建议。检索端无问题（该请求查询自包含、跳过重写，单凭症状"发热"命中缓存）。

- **生成端限定范围**（`prompts.py` `RAG_ANSWER_PROMPT`）：明确"多症状逐一作答"只针对【问题】里这次明确提到的症状；若本次只提一个症状，就只答该症状；【L2 临床快照】【L3 对话历史】里记录的既往症状只是背景（供判断用药/禁忌/相互关注），不得伪装成本次诉求逐条列处理建议。
- 真实的"多症状当前问题"不受影响，仍逐一作答。

## v9.29 - 隐藏答案文末的临床快照 JSON 后缀 + 答案禁止输出 JSON（Bug 修复）

基于 errorLog 分析：v9.28 只解决了模型在正文**前**输出 JSON 前缀，但模型偶尔会在正文写完、免责声明之后**尾随复述一段临床快照结构化 JSON**（`{"chief_complaint":...}`，甚至重复两次），v9.28 的前缀剥离对文末 JSON 无能为力，导致答案末尾裸露 JSON。

- **生成端根治**（`prompts.py` `RAG_ANSWER_PROMPT` / `DIRECT_ANSWER_PROMPT`）：严格禁止清单新增"只输出自然语言，禁止任何 JSON/结构化数据/代码块，禁止复述【L2 临床快照】等结构化数据"→ 新生成答案不再携带尾部 JSON；prompt 版本绑定失效 L0/L2 旧缓存。
  - 注意：模板是 `ChatPromptTemplate.format_messages()` 渲染，prompt 内一律不要出现字面 `{}` 花括号（如 `{"key":...}`），否则会被当占位符解析抛 `KeyError('"key"')`、触发"答案生成时错误"。禁令用纯文字措辞规避。
  - 复查同类隐患：`ROUTER_PROMPT` 的 `{"question_type":...}` 也是未转义字面花括号，导致 `_llm_route`（`ROUTER_PROMPT.format_messages`）每次抛 `KeyError`、被外层 except 吞掉恒降级 general（仅规则/上下文路由命中的查询不受影响）→ 已改为 `{{...}}` 转义，模型看到的 JSON 示例文本不变，LLM 路由恢复正常。凡需在模板里展示 JSON 示例，一律用 `{{`/`}}` 转义。
- **兜底·尾部剥离**（`streaming.py` + `nodes.py`）：新增 `_trailing_json_block_start`/`_strip_trailing_json_block`（从右向左用花括号深度逐个匹配完整对象，支持连续多个尾部 JSON 对象；要求末尾块含引号键值形态方认定是 JSON，避免误伤正文普通花括号文本），与 v9.28 的前缀剥离成对使用。
- **接线**：answer_generation_node / direct_answer_node 非流式 final、L0 答案缓存读取、流式无 token 兜底 final，均改为先剥前缀再剥后缀。
- 说明：流式 live token 受"已发送不可回收"限制，主要靠生成端 prompt 根治；缓存/非流式/历史答案已全覆盖。

## v9.28 - 隐藏答案开头的结构化 JSON 前缀（流式 + 兜底，Bug 修复）

用户反馈"每次系统回答时都会出现 json 字符串，隐藏它们"。模型在输出处理建议正文前，有时会先输出一段结构化 JSON（用户档案提取结果如 `{"name":...}`、问题拆解结果如 `{"问题拆解结果":...}`）。前端 index.html 只把流式 token 直接拼接进答案，这些 JSON 前缀会原样显示在正文上方造成污染。

- **streaming.py·逐 token 剥离 `_strip_leading_json_token`**：跨 token 累积缓冲，新增 `_json_prefix_done`/`_json_prefix_buf` 字段；用花括号深度扫描（非正则）识别连续前导 JSON 块，检测到正文起点才放行，通篇纯 JSON 则整体丢弃。接线到 native graph 流式 token 出口与 L2 语义缓存 token 出口（先剥 JSON 前缀、再剥 [来源] 标记）。
- **streaming.py·整段剥离 `_strip_leading_json_block`**：L0 `cached_answer` 命中与 final_answer 无 token 兜底路径，对整个答案做同样剥离。
- **nodes.py·非流式兜底**：新增同名纯函数 `_json_prefix_end`/`_strip_leading_json_block`（避免循环 import），接线到两处生成后的 `answer = _strip_leading_json_block(strip_source_markers(full_answer.strip()))`。
- 用花括号深度而非正则匹配 JSON，兼容含中文、含嵌套数组的 JSON；独立单元测试覆盖"双 JSON 前缀 + 正文""无前缀正文""通篇纯 JSON（罕见，保留）"等用例，全部通过。

## v9.27 - 复合问题多症状不漏答（检索 + 生成两端，Bug 修复）

基于 errorLog 分析（"我对花粉过敏，现在有点肚子疼怎么办？"）：v9.26 修复后复合问题改走单问题链路，但单问题检索用整句 query，背景症状或偏强的症状会把 Dense/Reranker 检索方向带偏，另一诉求症状文档进不了候选，答案只围绕其一处理（"只答花粉过敏"）。这是复合问题的通用缺陷，不局限于过敏。

- **检索端·剥离过敏背景 `_strip_allergic_background`**（`nodes.py`）：匹配并剥离查询中的"对X过敏"背景结构（X=过敏原，如花粉/芒果），预处理后、增强前接入检索 `search_query`；空查询时回退原句。仅改检索词，答案 prompt 仍用完整原问题（过敏背景不丢失）。
- **检索端·按诉求症状分别检索 `_build_symptom_sub_queries`**：多症状且未拆成多子问题时，把每个诉求症状构造成独立检索 query（并剔除过敏类背景症状），走并行检索，保证每个症状都有对应文档进候选，而非整句检索只偏其中一个。
- **生成端·强制逐一作答**（`prompts.py` `RAG_ANSWER_PROMPT`）：新增"多症状逐一作答"指令——问题涉及多个症状时对每个症状分别给处理建议（各成一条/一节）；某症状在文档找不到对应处理时明确写"文档未提供该症状的具体处理建议"，绝不只挑一个回答而遗漏其他。通用兜底，覆盖所有漏答场景；prompt 版本绑定失效缓存，L0/L2 旧缓存自动作废。

## v9.26 - 修复长问题拆解残句吞症状 + 多子问题置信度失真（Bug 修复）

基于 errorLog 分析（"我对芒果过敏，现在发烧了怎么办？"）：LLM 拆解失败降级规则拆解时，把"过敏"当切分边界切出"过敏，现在"残句，导致发烧被弱化/展示忽略；同时多子问题并检索后合并列表首位为跳过 Reranker 的无分文档时，置信度被压虚低（rerank=0.0000）。

- **修复规则拆解残句过滤**（`app/graph/nodes/nodes.py` `_rule_based_decompose`）：按症状边界切出的片段须含疑问/处理结构才计为子问题，无意义残句（如"过敏，现在"）丢弃；过滤后不足 2 个子问题则不强拆、退回单问题链路 → "我...过敏，现在发烧了怎么办？"由单问题按症状"发烧"增强检索词，发烧成为检索核心，不再被过敏背景吞掉。
- **修复多子问题置信度失真**（`nodes.py` v9.15 检索置信度段）：多子问题合并时按 `sub_question` 分组取每组最大 Reranker 分作为代表分，用各组代表分 Top2 算 rerank/gap，避免无分文档排首位导致 rerank=0、置信度虚低。
- 说明：低置信（<0.4）由"仅记拒答日志"升级为真正拦截/提示，涉及生成端取舍与误杀面，属需单独校准的一轮，暂缓。

## v9.25 - 修复版本去重误杀父子检索章节（Bug 修复）

基于 errorLog 分析：父子检索 + 邻域扩展会把同一文档按章节切成多个 Parent 块，它们共享同一 `source`、没有 `doc_version`，原 `_dedup_by_version` 却把"同名"一律当"多版本"，除排序第一的章节外全部标记废弃并逐条打日志。

- **修复 `_dedup_by_version`**（`app/graph/nodes/nodes.py`）：仅当文档携带版本标识（`doc_version` 或 `doc_effective_date` 非空）时才执行同源版本去重、旧版标记 `_superseded`；无版本标识的父章节块原样保留，不再互相标记废弃 → 兄弟章节扩展不再被误删，也不会再出现"旧版文档已标记废弃 (version=)"的重复刷屏。

## v9.24 - 答案来源展示优化（去重 + 正文去来源）

基于 errorLog 分析（首 token 8.7s 中 Reranker 占 2.8s、LLM TTFT 占 2s）的配套展示优化：

- **参考来源去重**：`knowledge_retrieval_node` / `format_retrieved_sources` / `_emit_sources_event` / 前端 `renderSources` 均按来源文档名去重 → 父子索引+兄弟章节扩展产生的 15 个同源条目收敛为 2 个唯一文档名，`📚 参考来源` 不再重复刷屏。
- **正文剥离 [来源:文档名]**：`RAG_ANSWER_PROMPT` 移除"标注来源文档名"指令，改为明确禁止正文出现来源引用；新增 `strip_source_markers`（非流式兜底）与 `streaming.py._strip_source_markers_stream`（流式跨 token 剥离，兼容标记被拆分），覆盖 graph 流式 / L0 缓存 / L2 缓存 / 无 token 兜底全部路径。
- 说明：L0 答案缓存 key 已绑定 `prompt_version`（prompts.py 文件 MD5），本次 prompt 改动会自动失效旧缓存，无需手动清理。

## v9.23 - README 同步实际进度（文档更新）

对照代码与 CHANGELOG 修订 README，消除与实际实现的偏差：

- **图片问诊**：「🖼️ 图片识别（规划中）」→「已实现」，补充 `image_base64` 完整流程（VLM 结构化提取 → OCR 数值校准 → 追问闭环 → RAG 续查），标注 `_vision_fallback_goto` 安全收尾。
- **API 文档**：移除不存在的 `/api/upload/analyze` 端点示例，改为聊天接口传 `image_base64` 的实际用法（该端点代码中不存在）。
- **新增「知识库管理与零停机重建」章节**：增量双缓冲、版本化（`version_id`）、影子集合原子切换、软删除/恢复/回滚、一致性校验、审计日志；补全 `/api/admin/kb/*` 九大管理接口清单与文件名路径穿越防护。
- **新增「安全检查引擎」章节**：`medication_guide_engine`（剂量/禁忌人群/重复用药/相互作用/5 字段）、`symptom_triage_engine`（危险症状组合 + 建议就诊）、`safety_review_engine`（紧急信号 + LLM 深度审查 + 拒答）。
- **新增「评估与迭代」章节**：RAGAS 四维指标、版本化 A/B 对比、Bad Case 回归、feedback/metrics 接口。
- **项目结构修正**：移除不存在的 `app/vision/` 目录，补 `skills/`、`evaluation/`、`models/`；vision 逻辑标注在 `graph/nodes/nodes.py`。
- **路线图**：勾选最近完成项（零停机重建、增量双缓冲、KB 管理 API、安全检查引擎、拒答机制、缓存版本化）。
- **模型清单修正**：本地模型 `qwen2.5:3b` → 实际 `qwen2.5:1.5b`；补充视觉模型 `glm-4v-plus`；`LOCAL_MODEL_ENABLED=False` 降级说明。
- **缓存体系**：L2 语义缓存阈值 `0.75` → 实际 `0.92`；补充缓存 key 绑定 `kb_version`/`prompt_version` 与有历史时跳过缓存复用。
- **安全防护**：补全 `X-Admin-API-Key` 覆盖范围（`/api/cache/*`、`/api/admin/kb/*`、`/api/admin/refusal/*`、`/api/metrics/*`）。

## v9.22 - 第二轮中风险项修复（18/18）

承接 26/8/16 审计中风险清单，全部 18 项已修复并逐项验证：

- **M1 自纠正重试沿用旧子问题**：`question_decompose_node` 改用 `final_question or question`，`query_rewrite_node` 返回 `sub_questions: None` 重置 → 重试轮用新关键词重新检索。
- **M2 ROUTER_PROMPT 与 JSON 提取冲突**：新增 `_ROUTE_ALIASES` + `_parse_route_text()`，`_llm_route` 在 `invoke_structured` 失败时回退 `llm.invoke` 纯文本解析 → 纯类型名输出不再降级 direct_answer。
- **M3 L0 答案缓存跨用户串用**：`streaming.py` 新增 `_has_history` 标志，有 thread 历史即跳过 L0/L2 缓存复用与写入。
- **M4 vision 追问不写 messages**：vision 追问/低置信/错误分支的 update 补 `messages: [HumanMessage, AIMessage]` → 图片问诊追问进入 checkpointer，下一轮上下文衔接。
- **M5 历史 RAG 文档正则失配**：重写 `strip_rag_documents_from_history` 匹配 `[{source}]\n{content}` 实际格式，doc_id 改用 `hashlib.md5` 确定性生成 → 上下文 token 不再膨胀、孤儿 doc 消除。
- **M6 grade 无覆盖保留 sources**：5 处 no-coverage Command update 补 `sources: []` → 不再"没查到却给来源"。
- **M7 `get_symptom_history` 读错命名空间**：改读 `("symptom_events", user_id)` 并过滤 `event_type=="symptom_report"` → 长期症状历史恢复可用。
- **M8 `document_cache` 无界增长**：补 `_NAMESPACE_RETENTION=7天`、`_MAX_ITEMS=500`，新增 `_prune_document_cache()`（超 2 倍阈值触发，摊销清理）。
- **M9 fallback flush 非幂等 + 无 busy_timeout**：新增 `_connect()`（`timeout=5` + `PRAGMA busy_timeout=5000`），flush 传 `event_id` 幂等 → 崩溃重跑不重复、并发不丢事件。
- **M10 药物相互作用去重方向反**：去重查询改为 `ia["drug_a"]==drug_a and ia["drug_b"]==drug_b` → 对称互列不再重复告警。
- **M11 "岁"误判儿童禁忌**：移除单字"岁"，新增 `age_match` 正则，`<16` 儿童 / `>=60` 老年 → 阿司匹林误报消除。
- **M12 紧急信号参与决策**：`answer_emergency_symptoms` 合并 `emergency_in_snapshot`，`needs_alert = has_emergency and not answer_addressed` → 回答含"胸痛"未给就医指引时主路径追加紧急提示。
- **M13 同步 /api/chat 丢 image_base64**：`input_state` 补传 `image_base64` → 非流式图片问诊恢复。
- **M14 配置定义不生效**：Settings 补 `MAX_MESSAGES`/`SUMMARY_TRIGGER` 字段；ChatRequest `_question_max_len` 取配置；`request_timing_middleware` 加 content-length 校验（413）。
- **M15 reload_config 假热更新**：变更时调用 `clear_llm_caches()` 清空各 `get_*_llm` 的 lru_cache → 热更新对 LLM/中间件生效。
- **M16 adaptive_threshold 首个校准点延迟**：校准条件改为"达到 MIN_SAMPLES 且（从未校准或距上次满间隔）" → 冷启动 100 样本即校准，符合注释宣称。
- **M17 双重初始化竞态**：`get_long_term_memory` 加 `threading.Lock` 双重检查、`get_checkpointer` 加 `asyncio.Lock` → 连接池不再泄漏。
- **M18 metrics 相对路径依赖 CWD**：`metrics.py` 基于 `PROJECT_ROOT` 拼 `data/metrics/metrics.db` → 任意启动目录写同一 DB。

## v9.21 - 第二轮高风险项修复（14/14）

承接 26/8/16 审计高风险清单，全部 14 项已修复并逐项验证：

- **H1 评估生成恒空**：`evaluation.py:run_generation` 同步调用 async 节点得 coroutine → 改用 `asyncio.run(answer_generation_node(state, config))`，生成链路恢复。
- **H2 临床快照字段读写不一致**：快照症状实际存于 `symptom_timeline`，读端统一改为从 `symptom_timeline` 提取（`_get_checkpoint_symptoms`/`check_emergency_signals`/`_detect_user_conditions`），并补读 `confirmed_facts` 用于禁忌检测。
- **H3 rerank_top_k 不生效**：删除硬编码 `RERANKER_INPUT_CAP=7`，改用 `candidates[:max(self.rerank_top_k, self.k)]`。
- **H4 剂量不累计每日总剂量**：新增频次×单次剂量检测（"每次600mg，每日3次"→1800mg>1200mg 告警）。
- **H5 增量入库后 BM25 不同步**：upload 端点补齐 `bm25_index.pkl` 删除 + `reset_hybrid_retriever()`。
- **H6 L2 无序 SET 取键**：`_find_similar_query` 改从 LRU Sorted Set `ZREVRANGE` 取最近键（旧 Set 降级兼容）。
- **H7 L0 答案缓存绕过版本化**：key 改为 `answer:{md5(question:kb_version:prompt_version)[:16]}`，KB/Prompt 变更自动失效。
- **H8 L2 写失败关闭全局缓存**：移除 `self._cache._available=False`，写失败仅跳过本次。
- **H9 metrics/拒答接口无鉴权**：`_verify_admin_key` 覆盖全部 `/api/metrics/*`、`/api/admin/refusal/*`、`/api/admin/kb/*`（复核确认 13 个 admin 端点全带鉴权）。
- **H10 上传/删除路径穿越**：新增 `_sanitize_kb_filename`（拒 `/`、`\`、盘符、`.`、`..`），upload + delete 双端点应用。
- **H11 非流式绕过安全检查**：`/api/chat` 在 `ENABLE_SAFETY_CHECK=False` 时补齐后置 `safety_check_node`，与流式一致。
- **H12 振荡检测字段不在 schema**：`_prev_max_score`/`_prev_relevant_count` 声明进 `MedicalAssistantState`+`InputSchema`，并每轮重置。
- **H13 诊断断言修订删疾病名**：含通配符模式仅替换关键字前缀（`就是.{0,6}病`→只替换"就是"为"可能是"），保留疾病名。
- **H14 分诊 duration 类型不匹配**：`assess_duration` 兼容 `{症状:{iso,ts,precision}}` 字典值，提取 `ts` 计算。

## 26/8/16待修复 - 第二轮全链路审计问题清单

承接 26/8/15 审计（P0/P1/P2 已全部修复），再次并行审查 5 条链路，新发现约 45 个问题：14 高风险 / 约 18 中风险 / 约 10 低风险。其中 4 项已亲自复核代码确认（标注 ⚠️已复核）。逐项修复后在本条目内更新状态。

### 高风险（14 项）

- **H1 ⚠️已复核【✅已修复】评估生成环节恒返回空**：`evaluation.py:263-275` `run_generation` 同步调用 `answer_generation_node`，而该节点是 `async def` 且需 `(state, config)` → 得到 coroutine，`.get("final_answer")` 抛错被 except 吞掉，恒返回 `""`。Faithfulness/Relevance 全部基于空答案，评估链路形同虚设。
- **H2 ⚠️已复核【✅已修复】临床快照字段读写不一致**：`models.py:151-162` 快照 schema 只有 `chief_complaint/symptom_timeline/medication_history/red_flags/confirmed_facts/ruled_out/symptom_onset_dates`；`nodes.py:849-861`/`2183-2186` 却读不存在的顶层 `symptoms`/`body_parts`/`severity` → 恒空。多轮追问症状继承（`_symptoms_with_checkpoint_fallback`）静默失效；`safety_check` 里 `_get_checkpoint_symptoms` 恒空 → `run_symptom_triage` 危险组合/就诊时限核查从不触发。症状实际存于 `symptom_timeline` 数组。
- **H3 ⚠️已复核【✅已修复】rerank_top_k 完全不生效**：`hybrid_retriever.py:545` 候选截断 `RERANKER_INPUT_CAP=7` 硬编码，`rerank` 用 `top_k=self.k`，`rerank_top_k`/`config.RERANKER_TOP_K` 从未使用。symptom 类型 k=8 被压到 ≤7，多跳症状查询召回不足。
- **H4 ⚠️已复核【✅已修复】剂量核查不累计每日总剂量**：`medication_guide_engine.py:348-364` 只比对单次剂量 > 日上限；"每次1g，每日3次"（3000mg > 对乙酰氨基酚 2000 上限）`exceeds_limit=False` 不告警。超量漏报。
- **H5【✅已修复】增量入库后 BM25 与知识库永久不同步**：`routes.py:887-918` `add_documents_to_store` 只失效语义缓存，不触发 `reset_hybrid_retriever` 也不删 `bm25_index.pkl`（带 `active_only` 标记的缓存仍判定有效）→ 新文档仅 dense 可召回、sparse 永不召回。
- **H6【✅已修复】L2 语义缓存从无序 SET 任意取前 100 键**：`semantic_cache.py:179-190` `smembers` 返回 Python set 无序，`list(all_keys)[:top_k*10]` 任意截取；真正维护的 `_keys_zset`（LRU）从未用于查找排序 → 缓存 >100 条后命中变"概率事件"，L2 命中率随缓存增长崩塌。
- **H7【✅已修复】L0 答案缓存绕过版本化 key**：`streaming.py:39-53,172-200` 用 `answer:{question}` 直接 Redis get/setex，绕过 redis_cache 的 `_generate_key`（kb_version/prompt_version 绑定）；KB 更新时无任何路由清空 `answer:*` → 知识库更新后旧答案最长残留 30 分钟。
- **H8【✅已修复】L2 写失败关闭全局 L0 `_available`**：`semantic_cache.py:380` `set()` 捕获写异常后把**共享 Redis 单例** `self._cache._available=False`，该标志同时决定 hybrid_retriever 是否走 L2 → 一次瞬断/序列化异常拖垮整个 L0 答案缓存 + L2，降级 30 秒。
- **H9【✅已修复】拒答日志/metrics 接口无鉴权**：`routes.py:663-678` `get_refusal_stats`/`export_refusal_logs` 声明了 `request: Request` 但从不调用 `_verify_admin_key`（同文件其余 `/api/admin/*` 均校验），`/api/metrics/*` 系列同样无鉴权 → 患者原始提问（question/request_id/thread_id）明文可导出，违反医疗数据合规。
- **H10【✅已修复】知识库上传/删除路径穿越**：`routes.py:793,972` `target_path = docs_dir / filename`、`file_path = docs_dir / filename`，filename 直接来自 multipart/URL 参数未 `Path.name` 净化、未拒 `..` → 可写入/删除 `docs_dir` 之外任意文件，潜在 RCE/服务瘫痪。
- **H11【✅已修复】ENABLE_SAFETY_CHECK=False 时非流式 /api/chat 完全绕过安全检查**：`routes.py:318-372` `chat` 端点 `graph.ainvoke` 直连，图内 safety_check 不可达；流式路径有后置补偿（streaming.py 调 safety_check_node），非流式没有 → 默认配置下同步接口返回的答案无任何规则引擎/用药核查/LLM 审查，安全策略按入口不一致。
- **H12【✅已修复】`_prev_max_score`/`_prev_relevant_count` 不在 state schema，自纠正振荡检测死代码**：`nodes.py:1392-1566` grade_documents 通过 `Command(update={"_prev_max_score":...})` 写入，LangGraph 对未知键静默丢弃 → 重试轮 `state.get("_prev_max_score")` 恒 0，两个"改善不足即早退"分支永不触发，每次重试都跑满 rewrite+检索 2 轮（徒增 2s+）。
- **H13【✅已修复】诊断断言修订删除疾病名**：`safety_review_engine.py:176-187` 模式 `就是.{0,6}病` 替换查表返回兜底"可能"、不保留疾病名；且多模式匹配基于原始串位置、先替换靠后位置后文本偏移 → "这肯定就是肺炎"→"这可能可能"，疾病名丢失、句子破碎。
- **H14【✅已修复】症状分诊持续时间维度类型不匹配**：`symptom_triage_engine.py:172-177` `assess_duration` 用 `isinstance(onset_ts,(int,float))` 判断，但快照 `symptom_onset_dates` 实为 `{症状:{iso,ts,precision}}`，值恒为 dict → 条件恒 False，72 小时就诊阈值分支从不触发，"症状持续 3 天"被漏判为 🟢。

### 中风险（18 项，已全部修复 ✅，见 v9.22）

- **M1【✅已修复】** 自纠正重试沿用旧 `sub_questions`：`nodes.py:878-883` question_decompose 见已有列表即 `return {}`，query_rewrite 不重置 → 拆解过的复合问题重试时用旧子问题再查，新关键词检索被丢弃。
- **M2【✅已修复】** ROUTER_PROMPT 要求"只返回类型名称"，与 JSON 结构化提取冲突：`prompts.py:79-93` vs `nodes.py:655-678`，本地模型照 prompt 输出纯 `symptom` 时三层解析全失败 → 降级 general→direct_answer，症状/知识类问题被弱化回答；`parse_router_output` 兼容函数未被 `_llm_route` 使用。
- **M3【✅已修复】** L0 答案缓存以问题文本为 key 跨用户串用：`streaming.py:166-181` 仅 `not self._has_profile` 时查询，但有 thread 历史无档案的用户也命中 → 前一会话基于不同主诉的医疗答案被 30 分钟内复用。
- **M4【✅已修复】** vision 追问/低置信度回答不写 messages：`nodes.py:2537-2553` 只返回 final_answer/warnings，无 messages → 图片问诊追问不进 checkpointer，下一轮上下文断裂。
- **M5【✅已修复】** `strip_rag_documents_from_history` 正则与真实格式不匹配：`nodes.py:1668-1712` 正则要求 `[文档N 来源...]`，实际注入格式是 `[{source}]` → 历史 RAG 文档块永不被占位符替换，上下文 token 膨胀，Redis 产生孤儿 doc，MicroCompact 失效。
- **M6【✅已修复】** grade 无覆盖兜底时保留 sources：`nodes.py:1395-1406` 返回"无命中结果"答案但不清除 sources → 用户可见"没查到却给了来源"。
- **M7【✅已修复】** `get_symptom_history` 读的 `("symptom_history", user_id)` 命名空间无任何写入者：`long_term_memory.py:55` 恒空（写入在 `symptom_events`），潜伏 bug。
- **M8【✅已修复】** `document_cache` 命名空间无 TTL/无 prune：`long_term_memory.py:105-125` 不在 `_NAMESPACE_RETENTION/_MAX_ITEMS`，Postgres 无界增长且全库共享。
- **M9【✅已修复】** fallback flush 无幂等 + 无 busy_timeout：`fallback_buffer.py:118-191` 先写 L1 再 DELETE 非原子，崩溃重跑产生重复事件；sqlite 未设 busy_timeout，flush 持锁期间 enqueue 撞 `database is locked` 静默丢事件。
- **M10【✅已修复】** 药物相互作用去重方向写反：`medication_guide_engine.py:270-282` `already` 查 `(drug_b,drug_a)` 而首段 append `(drug_a,drug_b)` → 对称互列两药重复警告两条。
- **M11【✅已修复】** "儿童"禁忌含单字"岁"误判成人：`medication_guide_engine.py:46,188-193` 年龄字段几乎必含"岁" → 任何成年档案命中"儿童"，阿司匹林误报禁忌。
- **M12【✅已修复】** 紧急信号 `answer_has_emergency` 计算后未参与决策：`safety_review_engine.py:128-138` `needs_alert` 只看 `emergency_in_snapshot` → 用户快照无紧急症状、仅回答含"胸痛"未给就医指引时主路径不追加紧急提示，拦截落空。
- **M13【✅已修复】** 同步 `/api/chat` 静默丢弃 `image_base64`：`routes.py:52-57` ChatRequest 定义了但 `chat()` input_state 不传 → 图片问诊走非流式接口图片被忽略。
- **M14【✅已修复】** 多处配置项定义但不生效：`.env` `MAX_MESSAGES=20`/`SUMMARY_TRIGGER=14` 在 Settings 无字段（extra=ignore 静默丢弃）；`MAX_CONTENT_LENGTH`/`MAX_QUESTION_LENGTH` 定义后无消费。
- **M15【✅已修复】** reload_config 假热更新：`llm.py` 各 get_llm 均 `lru_cache` 用旧配置冻结实例，`RateLimitMiddleware.max_requests` 构造时捕获 → 热更新对已缓存 LLM/中间件不生效。
- **M16【✅已修复】** adaptive_threshold 首个校准点在 1000 而非注释宣称的 100：`adaptive_threshold.py:118-120` 校准条件被 `RECALIBRATE_INTERVAL=1000` 门控，`MIN_SAMPLES=100` 常规路径形同虚设，冷启动期固定阈值长期不校准。
- **M17【✅已修复】** checkpointer/long_term_memory 双重初始化竞态：`checkpointer.py:48-64`、`long_term_memory.py:645-658` 首个并发请求各自建连接池，后写覆盖全局、先建的从不 `__exit__`，连接池泄漏。
- **M18【✅已修复】** metrics 相对路径依赖进程 CWD：`metrics.py:46` `data/metrics/metrics.db` 未基于 PROJECT_ROOT 拼接 → 不同启动目录写不同 DB，统计丢失/对不上。

### 低风险 / 清理（约 10 项）

- **L1** qa_chain.invoke 双重检索：`qa_chain.py:235-238` 先取 docs 作 sources，chain 内 `RunnablePassthrough.assign` 又调一次 retriever → 检索两次、展示来源与生成所用文档可能不一致。
- **L2** `run_rule_based_review` 文档声明支持 block，代码永不返回 block：`safety_review_engine.py:248-251` block 全依赖 LLM 深度审查，规则层无法兜底拒答。
- **L3** `_FORCE_STRATEGY` 恒 None 且不读 env：`structured_output.py:47` 文档宣称可被 `STRUCTURED_OUTPUT_STRATEGY` 覆盖，实际不存在该能力。
- **L4** `loader_txt_only.py` 为重复死文件：与 `loader.py` `load_medical_documents` 重复且无 import。
- **L5** `get_collection_info` 返回绑定方法而非数值：`vector_store.py:277` `collection_info.count` 未调用 → 接入管理 API 时 JSON 序列化抛异常（当前无调用方）。
- **L6** 熔断器注册表静默忽略同名新参数：`circuit_breaker.py:122-135` name 已存在时直接返回旧实例，新 threshold/timeout 被丢弃无日志。
- **L7** 限流中间件 IP 桶全局共享 + 热更新无效：`routes.py:206-247` 以 `request.client.host` 为 key 未读 X-Forwarded-For，反代后全站同源 IP，单用户打满即全站 429。
- **L8** 主模型失败无备用切换：`nodes.py:3842-3857` 仅返回道歉文案，config 注释中的"备选模型"从未被代码切换，熔断器未接入 LLM 层。
- **L9** evaluation 每条样本重建 hybrid retriever：`evaluation.py:245-256` 每次 `get_hybrid_retriever(k=3,...)` 重建 BM25 索引，批量评估耗时线性放大。
- **L10** `_embedding_cache` 无上限只写不读：`semantic_cache.py:93` 进程内存泄漏；`record_feedback` 插入失败仍返回 id；`prune_namespace` 对空时间戳事件按空串排序反删最新。


## 26/8/15待修复 - 全链路审计发现的问题清单

并行审计 5 条链路（RAG 检索 / 图流程 / 缓存记忆 / 核心工具与 API / 加载评估与技能），发现约 30 个隐藏逻辑错误或功能目标落空的问题。分级记录如下，P0/P1/P2 已全部修复 ✅。

### P0 - 数据丢失 / 功能彻底失效（已全部修复 ✅）

**1.【✅ 已修复】增量上传丢文档内容**
- 根因：`routes.py` 增量上传时 `filter_unchanged_chunks` 把未变 chunk 跳过（只写变化块），激活新版本后 `deprecate_old_versions` 把旧版本**全部** chunk 置 deprecated，而检索层只查 `status=active` → 大文档改几行重传，未变内容整体不可见；且 deprecated 块 `is_deleted=False` 仍在 `get_existing_content_hashes` 里，二次修改仍跳过，永久丢失，只能全量重建恢复。
- 修复：`kb_updater.py` `deprecate_old_versions` 增加 `keep_hashes` 参数，仅废弃"新版本中 content_hash 不存在的旧块"，且同 content_hash 跨版本累积时只保留 version_id 最高的一份（避免重复）；`activate_document_version` 透传该参数；`get_existing_content_hashes` 改为只统计 `status=active`（deprecated 块不再挡"改回旧内容"的重新激活）；`routes.py` 激活时传新版本全部 chunk（changed+unchanged）的 hash 集合。

**1.5.【✅ 已修复】chromadb>=1.0 多键 where 语法不兼容（增量上传/软删除/版本查询实际全失效）**
- 根因：`kb_updater.py` 的 `activate_document_version`/`deprecate_old_versions`/`soft_delete_document`/`restore_deleted_document`/`get_document_version` 均用 dict 多键 `where={"source":..., "status":...}`，chromadb 1.5.5 要求多键用 `$and` 数组（`{"$and":[{...},{...}]}`），否则抛 `Expected where to have exactly one operator` → 这些功能在生产环境实际从未成功（activate 恒返回 0、chunk 停在 pending）。
- 修复：全部多键 where 改为 `$and` 数组结构（单键 where 不受影响）。

**2.【✅ 已修复】`reset_hybrid_retriever()` 无效，零停机重建被架空**
- 根因：`get_hybrid_retriever` 是 `@lru_cache(maxsize=8)`，`reset_hybrid_retriever` 只清实例字典和 embedding 缓存，未清 lru_cache；`HybridRetriever.__init__` 构造时就绑定 `get_vector_store()` → 切换/回滚后 lru 命中旧实例，仍读旧集合旧 BM25，旧目录 300s 后物理删除即报错。
- 修复：`hybrid_retriever.py` reset 时补 `get_hybrid_retriever.cache_clear()`。

**3.【✅ 已修复】fallback_buffer 每 5 分钟误删全部缓冲**
- 根因：`fallback_buffer.py` `cleanup_expired` 的 `cutoff = datetime.now()`（而非 7 天前），`DELETE WHERE created_at < cutoff` 清掉所有未 flush 事件，Redis/Postgres 故障期间 L1 数据静默丢失，`retry_count` 重试补写机制失效。
- 修复：cutoff 改为 `datetime.now() - timedelta(days=_MAX_AGE_DAYS)`，仅清理超过 7 天保留期的过期事件。

**P0 验证结果**（`scripts/verify_p0_fix.py` 全部通过 ✅）
- 增量上传：v1(A,B,C) → v2(改C→C') → v3(改回C) 全流程，active 始终 3 块、无重复 hash、旧块正确 deprecated、改回可恢复。
- reset：lru_cache currsize 归 0，双集合切换后检索器重建。
- fallback：8 天前事件删除、1 小时前事件保留。

### P1 - 功能目标落空（已全部修复 ✅）

**1.【✅ 已修复】`_is_self_contained` NameError**
- 根因：`nodes.py:2972` `if enable_hyde and not _is_self_contained` 引用了未定义变量，`ENABLE_HYDE=True` 时一进 HyDE 分支即 NameError（潜伏，默认 False 未触发）。
- 修复：改用已存在的 `_has_anaphora_pattern(final_question)`（返回 True 表示"不自包含"，与 `not _is_self_contained` 语义一致）。
- 验证：`tests/test_self_containment.py` 38 条 100% 通过。

**2.【✅ 已修复】keyword_matcher 边界检测恒 True**
- 根因：`_check_boundary` 所有分支都返回 True（恒接受），`use_boundary` 开关形同虚设；`build_route_symptom_matcher` 含单字"疼/痛"关键词 → "心疼"被误路由为 symptom；无否定词处理（"不痛"匹配"痛"）。
- 修复：`keyword_matcher.py` 重写 `_check_boundary`（否定前缀拒绝 + 前邻/后邻更长词拒绝 + 非 CJK 词内成分拒绝 + CJK 邻接放行"偏头痛"→"头痛"）；`build_route_symptom_matcher` 移除单字"疼/痛"；`nodes.py` 全部 7 处调用点（route 93/98、extract 136、拆解 902/975、实体扫描 3378/3379）`use_boundary=False`→`True`。
- 验证：`心疼`→不匹配；`不发烧`→不匹配；`偏头痛`→头痛；`喉咙痛`→嗓子疼；路由"心疼的滋味不好受"→None（不再误路由 symptom）；`safety_review_engine` 紧急识别保持 `use_boundary=False`（紧急宁可多报不漏报）。

**3.【✅ 已修复】rebuild 审计从不落库**
- 根因：`routes.py:1220` `log_kb_audit(vs, "full_rebuild", "rebuild", "success", details=...)`——首参应为 `doc_id: str` 却传了 vector store 对象，参数表无 `details`，`chunk_count` 被传成字符串"rebuild" → TypeError 被 `except: pass` 吞掉，审计日志永远少一条全量重建记录。
- 修复：改为 `log_kb_audit(doc_id="full_rebuild", change_type="rebuild", chunk_count=len(child_chunks), result="success", elapsed_ms=int((time.time()-started_at)*1000))`（`details` 参数不存在，审计表也无该列，弃用）。

**4.【✅ 已修复】用药指南/症状分诊引擎（单位 bug + 接入 graph）**
- 根因：`skills/*` 的 `run_medication_guide_review`/`run_symptom_triage` 全项目无调用，剂量/禁忌核查从未运行；且 `check_dosage_safety` 用 `re.search(r"(\d+)")` 提取上限，`"4g"` 解析成 4（实为 4000mg）→ 接线即把正常剂量误判超量；答案侧只匹配 `\d+ mg`，g/μg 单位剂量漏检。
- 修复：
  1. 单位：`medication_guide_engine.py` 新增 `_parse_dosage_mg()`（统一 g/mg/μg/微克 → mg，支持小数），`check_dosage_safety` 上限与答案剂量全部走该函数。验证："4g"→4000；阿莫西林"每次500mg"不再误报、每次5g 正确报超量。
  2. 接线：`safety_check_node` 规则引擎步骤扩展两个子核查——① `run_medication_guide_review`（回答含药物名时做剂量上限/禁忌人群/相互作用/5字段完整性核查，修订注入剂量/禁忌/相互作用警告与字段补全模板）；② `run_symptom_triage`（从临床快照取症状，仅响应**危险症状组合**与 **🟡建议就诊** 两类信号，注入紧凑警告，不注入整块分诊文本；单症状紧急信号仍由既有 `check_emergency_signals` 覆盖，避免重复与"腹痛→急性腹痛"子串误报）。触发风险标签后进入既有 LLM 深度审查路径。
- 验证：布洛芬回答补全 5 字段；阿莫西林 5g 注入剂量警告；头痛+发热+颈僵 注入"脑膜炎高风险"紧急提醒；无风险回答 status=pass 仅 ~4ms 开销、final_answer 不动。
- 说明：接线后用药类回答更频繁触发 LLM 深度审查（`qwen2.5:1.5b` 弱模型对剂量普遍过度保守、易判 block）——这是既有安全网行为，非接线引入的回归；若生产用小模型需评估是否放宽 deep-review 判 block 阈值。

**5.【✅ 已修复】metadata 单源误判 high + mtime 污染**
- 根因：`metadata_extractor.py` `_resolve_field` 在 `len(unique_values)==1` 时无条件返回 high，含 `len(sources)==1`；filesystem mtime 常为 effective_date 唯一来源 → 每文件写入 `effective_date=mtime` 且 direct 落库，重拷文件即被当"更新版本"。
- 修复：`_resolve_field` 在 `len(sources)==1` 时提前返回 confidence="low"（无法交叉验证，写入 `doc_{field}_pending` 待复核）；多来源一致仍为 high。
- 验证：单源 filesystem→low，双源一致→high，三源多数→mid，无源→none。

### P2 - 正确性缺陷（已全部修复 ✅）

1. **【✅ 已修复】BM25 绕过软删除过滤**
   - 根因：`vector_store.py:283` `load_all_documents` 无 where、`hybrid_retriever.py:322` `_sparse_search` 无过滤 → 已删除/废弃文档仍被 BM25 召回。
   - 修复：`vector_store.py` `load_all_documents` 带 `where={"status":"active"}`（legacy 数据无 status 时回退全量）；`hybrid_retriever.py` `_load_bm25_documents` 增加 cache 版本标记 `active_only`（旧缓存触发重建），BM25 只索引 active chunk。
2. **【✅ 已修复】RRF 去重 key 不一致**
   - 根因：dense 文档无 `.id`、BM25 文档有 `.id`，同一文档双份进 rerank。
   - 修复：`_reciprocal_rank_fusion` doc_key 优先 `chunk_id` metadata（dense/BM25 一致），同一文档不再双份。
3. **【✅ 已修复】adaptive_threshold 观察值取错字段**
   - 根因：`hybrid_retriever.py:560` 读 `relevance_score`，`reranker.py:221` 写 `rerank_score`；`config.py:63` `RERANKER_THRESHOLD=0.005` 未生效（注册默认硬编码 0.02）。
   - 修复：`hybrid_retriever.py` 读 `rerank_score`；`evaluation.py` 同步修复；`adaptive_threshold.py` RERANKER_THRESHOLD 注册默认值改为 `get_config().RERANKER_THRESHOLD`。
4. **【✅ 已修复】sources/warnings 跨轮次累积**
   - 根因：`state.py` 用 `add` reducer 但 InputSchema 不重置 → 用户可见陈旧引用无限增长。
   - 修复：新增自定义 reducer `_resetable_list_add`（`right is None → []`），`warnings`/`sources` 改用；`graph.py`/`streaming.py`/`routes.py` 三处 `ainvoke` input_state 每轮传 `None` 重置。验证：带 checkpointer 两轮，第二轮仅含本轮 warnings。
5. **【✅ 已修复】流式 token 统计失效**
   - 根因：`token_tracker.py` 读 `response_metadata["token_usage"]`，langchain-openai 1.x 实际在 `usage_metadata` → 用量恒 0。
   - 修复：优先读 `AIMessage.usage_metadata`（`input_tokens`/`output_tokens`），兼容旧 `token_usage`/`usage` 格式。验证：三种格式提取正确、无 token 信息跳过。
6. **【✅ 已修复】prune 死代码**
   - 根因：`long_term_memory.py` `prune_namespace` 无任何调用 → 长期记忆无界膨胀。
   - 修复：新增 `_prune_if_oversize`，在 `save_query_record`/`append_symptom_event`/`append_medication_event`/`save_bad_case` 写入后触发，条目超 `_NAMESPACE_MAX_ITEMS` 才 prune。验证：超上限触发、未超跳过、未注册 namespace 跳过。
7. **【✅ 已修复】匿名用户共用 thread**
   - 根因：`graph.py:231` 所有匿名请求落 `thread_default` → 跨用户医疗对话互相加载。
   - 修复：`routes.py` 新增 `_resolve_thread_id`（有 thread 用 thread；无 thread 有 user 用 `thread_{user}`；都无则生成独立 `thread_anon_{uuid}`）；`graph.py` `run_graph` 同步。验证：匿名每次独立会话。
8. **【✅ 已修复】缓存命中不写对话历史**
   - 根因：`streaming.py:499` L0/L2 命中直接 yield 不进 checkpointer → 下一轮失去语境。
   - 修复：新增 `_persist_conversation`，L0/L2 命中后用 `graph.aupdate_state(as_node="answer_generation")` 把本轮问答写入 checkpointer。验证：两轮 update_state 后 messages 追加（4 条）而非覆盖。
9. **【✅ 已修复】父对象跨请求变异**
   - 根因：`parent_child_store.py:208` 直接改写 store 内 Document 的 metadata（rerank_score 等）→ 语义污染。
   - 修复：`get_parents` 始终返回 `model_copy()` + 独立 metadata dict，rerank_score 写副本。验证：store 共享对象无 rerank_score 残留、两次请求评分独立。
10. **【✅ 已修复】复合症状只增强首个**
    - 根因：`nodes.py:1033-1037` 命中第一个症状即 return，"发烧头痛怎么办"只追加发热词。
    - 修复：`_enrich_treatment_query` 合并所有命中症状的护理词（保序去重）。验证："发烧头痛怎么办" 同时含发热+头痛护理词。
11. **【✅ 已修复】vision 安全关闭时静默终止**
    - 根因：`nodes.py:2464` `goto="safety_check"` 无出边，图片追问不落库。
    - 修复：新增 `_vision_fallback_goto()`，`ENABLE_SAFETY_CHECK=False` 时追问/低置信度/异常 `goto=END` 收尾（不再卡在无出边的 safety_check）。验证：True→safety_check，False→END。
12. **【✅ 已修复】safety_review 降级分支误报紧急**
    - 根因：`safety_review_engine.py:132-137` 缩进错误，`append` 在命中条件外 → 异常时把全部紧急症状塞入快照。
    - 修复：`emergency_in_snapshot.append` 移入 `emerg in answer` 条件内。验证：回答"胸痛..." 仅 append 胸痛，不再塞全部紧急症状。
13. **【✅ 已修复】增量更新 build_index 重置父索引**
    - 根因：`routes.py:867` 增量用线上单例 `build_index(changed_chunks)` 会清空全库 parent store → 单文档更新后全库父还原能力退化。
    - 修复：`parent_child_store.py` 新增 `update_index`（仅删变更文档旧版本 parent + 写入新版本，其余文档保留）；`routes.py` 增量路径改用 `update_index`。验证：全量 A+B 后增量更新 B，A 的 parent 仍可检索。

## v9.22 - 知识库不停机重建：双集合机制真正生效

### 问题

v9.21 之前，全量重建知识库只有一条路：`scripts/rebuild_vector_store.py` 的 `force_rebuild` 路径——**删目录再建**。Windows 下运行中的服务锁住 `data/chroma_db` 的文件时直接失败（`PermissionError: WinError 32`），必须停服才能重建。

项目里本有一套为不停机设计好的机制（`DualCollectionManager` + `/api/admin/kb/rebuild` 接口：影子集合 → 校验 → 原子切换指针 → 延迟清理），但**从未真正生效**，存在多个断点：

1. **加载路径不读指针**：`switch_active_collection()` 写 `kb_active.json`（`active_persist_dir` 指向影子目录）并重置全局 manager，但 `get_vector_store()` 永远从 `config.PERSIST_DIRECTORY`（`data/chroma_db`）加载，**从不消费 `kb_active.json`** → 切了等于没切，应用继续读旧集合。
2. **延迟清理误删基础目录**：首次切换时 `previous_persist_dir` 默认指向 `data/chroma_db` 根目录，5 分钟后 `schedule_cleanup_old_collection` 会 `rmtree` 它——而影子集合是它的子目录，等于**删除还在用的活跃集合**。
3. **`build_shadow_collection` 调 `shadow_vs.persist()`**：langchain_chroma 新版本已移除 `persist()` 方法，接口一旦真正被调用必崩。
4. **`_write_config_atomic` 用 `Path.rename`**：Windows 上目标文件已存在时抛 `FileExistsError`，第二次写 `kb_active.json` 必崩。
5. **重建接口 `enrich_chunk_metadata(child_chunks, source="rebuild", ...)`**：`kb_updater.py` 用 `source` 拼 `doc_id/chunk_id`，一次性传 `"rebuild"` 会把全部 chunk 的 `doc_id` 覆盖成 `"rebuild"`，丢失真实文档来源。
6. **父索引污染线上**：接口直接用线上检索的单例 `get_parent_child_manager().build_index()`，`build_index` 会重置 store，重建期间并发检索可能读到空父索引。

### 方案

1. **加载路径消费指针**（`app/rag/vector_store.py`）：新增 `_resolve_active_collection()`，`VectorStoreManager.__init__` 在 `persist_directory` 为空（默认路径）时从 `kb_active.json` 解析 `active_persist_dir + active_collection`；回滚到默认集合时 active_collection 为哨兵名 `medical_kb_default`（与真实默认集合 `langchain` 不一致），此时回退到 `config.PERSIST_DIRECTORY` + 默认集合。`create_vector_store` 仅在解析到活跃集合时传 `collection_name`（None 时不传，避免 chromadb 报 `NoneType`）。
2. **清理跳过基础目录**：`schedule_cleanup_old_collection` 当 `old_path == chroma_base_dir` 时跳过，保留默认集合及其影子子目录。
3. **移除 `shadow_vs.persist()`**（新版本 from_documents 已自动持久化）。
4. **`_write_config_atomic` 改用 `os.replace`**（Windows/POSIX 均可原子覆盖已存在文件），并补 `import os`。
5. **重建接口按真实 source 分组增强元数据**（`app/api/routes.py`），保证每个 chunk 的 `doc_id` 反映文档来源。
6. **父索引安全换入**：重建用全新 `ParentChildManager()` 构建（不污染线上单例），切换后 `pcs._parent_child_manager = parent_manager` 原子替换全局单例再 `save_to_disk`。
7. **重建脚本确定化**（`scripts/rebuild_vector_store.py`）：开头删除 `kb_active.json`（回退默认集合），`get_vector_store` 显式传 `persist_directory=config.PERSIST_DIRECTORY`，手动全量重建始终针对默认集合，不被指针劫持。

### 验证

- 零停机流程单测（`scripts/verify_zerodowntime.py`）：基线默认集合 `langchain`(386) → 建 mini 影子集合(3 chunks) → 原子切换 → `get_vector_store()` 加载影子集合并检索到测试内容 → 回滚 → 回退默认集合 `langchain`(386)，全部通过 ✅
- 回归：检索单测（5 症状各自召回治疗内容、无跨症状污染）+ 端到端"发烧怎么办"（减少衣物/散热，无"增加衣物保暖"幻觉）均通过 ✅
- 生产路径：`/api/admin/kb/rebuild` 走影子集合构建 → 校验 → 原子切换，服务不中断；切换后 300s 延迟清理旧影子集合（跳过基础目录）

## v9.21 - 医疗幻觉修复：发热问诊不再出现"增加衣物保暖"（与文档相反的危险建议）

### 问题

errorLog 中 AI 对"发烧怎么办"回复"适当增加衣物，注意保暖"，但知识库《发热诊断与家庭护理指南》明确写"减少衣物：不要捂汗，适当减少衣物散热"——**AI 输出了与文档相反的危险建议**。根因链：

1. **检索召回不到治疗内容（主因）**：`dense(k*2=12)+BM25(k*2=12) → RRF → RERANKER_INPUT_CAP=7 → rerank → parent-child`。"发烧怎么办"的 fusion 候选里，治疗 chunk（物理降温/减少衣物/退热药）排 16-21 位，进不了 reranker 的 7 个候选位，LLM 的 context 只有文档标题+发热概述，回答护理措施时只能靠自己的（错误）常识填补。v9.20 的通用增强词（治疗 处理 药物 用药 缓解…）对发热 case 无效——与文档"物理降温/减少衣物/温水擦浴"真实词汇不匹配。
2. **L2 语义缓存污染**：`.env` 的 `SEMANTIC_CACHE_THRESHOLD=0.80` 覆盖代码默认 0.92，不同症状查询互相命中（头痛↔咳嗽↔腹泻 80-85% 相似），返回完全错误内容（咳嗽问诊返回糖尿病指南）。
3. **LLM 凭空编造护理建议**：`RAG_ANSWER_PROMPT` 格式指令强制要求"物理/家庭护理措施"，文档没有时 LLM 用错误常识填补，system 层"禁止编造"约束在格式压力下失效。
4. **版本污染 + 数据卫生**：全库 1014 个 chunk 全部无 `status/is_deleted` 元数据（旧管线重建不走版本化 route）→ 版本软删除失效，文档多版本堆叠，标题 chunk 反复挤占 fusion top3。

### 方案

1. **症状感知护理词映射**（`app/graph/nodes/nodes.py`）：`_enrich_treatment_query` 由通用增强词改为 `_SYMPTOM_CARE_KEYWORDS` 症状→护理词映射（发热→"物理降温 散热 减少衣物 温水擦浴 退热药…"，头痛→"止痛 药物选择 布洛芬…"，咳嗽/腹泻/流鼻血等 13 类），每类症状用向量库实证的该症状治疗 section 真实词汇，未命中已知症状时用通用词兜底。已实证：发热追加护理词后"减少衣物/退热药"chunk 从 fusion rank 16-21 升到 top5。只改 `search_query`，不改 `rewritten_query`/`final_question`。
2. **`RAG_ANSWER_PROMPT` 安全硬化**（`app/graph/nodes/prompts.py`）：system 层新增"禁止补充文档未记载的护理/用药措施（如衣物增减、物理降温方法、饮食建议）——护理措施必须逐字出自【文档】"；human"处理建议"节改为每条措施必须能在【文档】找到原文，某类措施未记载则该条不写并在"当前资料未提及的关键方面"说明"文档未提供此方面具体建议"。
3. **L2 语义缓存修复**（`.env` + `app/rag/hybrid_retriever.py`）：阈值 0.80→0.92 消除跨症状污染；`cache_query` 改用实际检索词（原用 `original_query`，检索逻辑改动后旧差结果仍持续返回）。
4. **全库重索引**（`scripts/rebuild_vector_store.py`）：新增按 source 分组写版本元数据（`enrich_chunk_metadata` status=active/version_id=1）+ 删除旧 `bm25_index.pkl`。修复 `create_vector_store` 的 `force_rebuild=True` 未先删旧集合的 bug（`Chroma.from_documents` 的 get_or_create_collection 会把新 chunk 追加到旧集合，导致 1400=1014 旧+386 新 堆叠），重建后每文档 1 份 chunk、全部 status=active，版本软删除机制恢复可用。
5. **清空 L2 + L0 缓存**：新检索词/新阈值立即生效。

### 验证

- 检索单测：发烧/头痛/咳嗽/腹泻/流鼻血各自召回自己的治疗内容（发烧含"减少衣物/散热"），无跨症状污染
- 端到端"发烧怎么办"：护理措施包含"减少衣物/散热"，**不再出现"增加衣物保暖"**
- 缓存阈值 0.92 后，头痛/咳嗽/腹泻查询不再互相命中（相似度 80-85% < 92%）
- 重索引后每文档 chunk 数 = 唯一 chunk 数，metadata 含 status=active
- greeting / knowledge 类型回归通过，不受影响

## v9.20 - 答案质量修复 + 直接建议输出格式

### 核心改进：问诊答案从"病情介绍"升级为"可执行的处理建议"

**问题**：部分问诊只介绍病情、不给实用处理方式。以"发烧怎么办"为例，答案只列了"需立即就医"的警示征，完全没有退热药、物理降温、护理步骤等实用建议。根因是**检索查询词术语鸿沟**：用户问"发烧怎么办"，但文档治疗内容关键词是"退热药/布洛芬/物理降温"，dense+BM25 把"定义/警示"类 chunk 排在治疗 chunk 前面，治疗 chunk 没进候选集，LLM 拿不到处理信息（知识库本身内容充足）。

**方案**：

1. **症状"怎么办"类问题检索词增强**（`app/graph/nodes/nodes.py`）：新增 `_enrich_treatment_query`，当 `question_type == "symptom"` 且查询含治疗意图（怎么办/怎么处理/怎么缓解/用药/治疗等）时，在**检索查询**上追加"治疗 处理 药物 用药 缓解 护理 注意事项 家庭护理 康复"，让治疗 chunk 进入候选集。仅改 `search_query`（传给 retriever），不改 `rewritten_query`/`final_question`（这两个字段仍被答案 prompt 用作问题，避免注入词污染最终回答）。规则式零额外延迟。
2. **答案输出格式改为直接建议**（`app/graph/nodes/prompts.py` 重写 `RAG_ANSWER_PROMPT`）：删除"第一步提取事实 / 第二步组织回答 / 第三步完整性评估"的三步式指令，改为直接输出：**处理建议**（药物+剂量按原文、物理/家庭护理、生活调整）→ **需立即就医的情况**（红旗征，文档有才写）→ **当前资料未提及的关键方面**（一句话说明不编造）。保留来源标注 `[来源:文档名]`、`⚠️ 以上建议仅供参考` 免责声明。system 层安全约束（仅基于文档、禁止编造、禁止假设人群）原样保留。
3. **清理 L2 语义缓存**：TTL 3600s 的旧差结果会持续返回，改动后清理 `semantic_cache:*` 与 `medical_assistant:answer:*` 键，让新检索词立即生效。

**验证**：

- 检索增强后 top5 立即包含物理降温、退热药物选择、热性惊厥处理等治疗类 chunk
- 端到端"发烧怎么办"：答案包含退热药（布洛芬/对乙酰氨基酚）、温水擦拭、饮食调整等实用处理，为直接建议格式（无步骤式分析），首 token 约 2.1s
- greeting / knowledge 类型回归通过，不受影响

## v9.18 - 流式接口迁移到原生 LangGraph 流式（P0-1）

### 核心改进：消除双维护，graph.py 成为唯一节点编排来源

**问题**：`StreamingOrchestrator`（`app/graph/streaming.py`）手动逐个调用节点（router → symptom\_analysis → query\_rewrite → question\_decompose → knowledge\_retrieval → grade\_documents → answer\_generation），与 `graph.py` 的边定义重复维护，容易漂移。

**方案**：改为驱动 `graph.astream(stream_mode=["messages", "updates", "custom"])` 原生流式：

- `messages` 模式：捕获 LLM token（仅 `answer_generation` / `direct_answer` 两个节点使用 `get_llm(streaming=True)` 会流出 token，中间节点用非流式本地模型，不污染 token 流）
- `updates` 模式：捕获 `knowledge_retrieval` 节点更新，用于推送 sources 事件
- `custom` 模式：捕获 `get_stream_writer` 事件（图片分析摘要）

### 节点改造

- `answer_generation_node` / `direct_answer_node` 改为 async + `get_llm(streaming=True).astream(prompt, config=config)`，显式传入 config 以兼容 Python < 3.11 的 token 捕获
- `vision_analysis_node` 通过 `get_stream_writer` 流式输出图片分析摘要（原 `stream_vision_answer` 的摘要逻辑提取为 `_build_vision_summary` 纯函数）
- `state.py` 新增 `thread_id` / `request_id` 字段，供 token 追踪和指标关联贯穿全链路

### 删除的冗余代码（双维护来源）

- `stream_direct_answer`、`stream_vision_answer`：被原生流式 + `get_stream_writer` 取代
- `_run_rag_pipeline`、`_save_history`、`_background_snapshot_update`、`_build_clarification_answer`、`_record_low_score_bad_case`、`_emit_progress`：手动编排/后处理逻辑
- `validate_streaming_sync`、`_STREAMING_REQUIRED_NODES`：失效的同步校验补丁
- `_get_snapshot_lock` / `_snapshot_locks`（routes.py）：后台快照锁，随 `_background_snapshot_update` 移除

### 行为差异（迁移带来的已知变化）

- **澄清追问文案**：无检索文档时由 `build_no_results_answer`（graph 内 grade\_documents 节点）替代原 `_build_clarification_answer`
- **档案提取时机**：由「回答后异步」改为「graph 内 pre-router」（`profile_extraction_node` 本就在 graph 中，且只依赖 question 不依赖 answer，无语义损失）
- **进度事件移除**：`_emit_progress`（正在分析/检索/生成）不再发出；如需恢复可用 `get_stream_writer` 在节点内补发
- **安全审查**：`ENABLE_SAFETY_CHECK=False`（默认）时由编排器后置审查；`=True` 时由 graph 内 `safety_check_node` 原生审查
- **快照更新**：由「后台异步」改为「graph 内同步」（`should_update_snapshot → update_snapshot` 边），仅滑动窗口触发时增加少量延迟
- **对话历史**：由 checkpointer 自动持久化（节点返回 messages），替代手动 `_save_history`
- **图片问诊历史**：vision 路径下 question 被改写为 RAG 查询后写入历史，与原「存原问题」略有差异

### Bug 修复

- **L2 语义缓存从未生效**：原 `_check_cache` 判断 `route_type == "knowledge"`，但 router 的 `goto` 实际是 `"query_rewrite"`，条件永不匹配 → 改为 `route_type == "query_rewrite"`，知识类查询现在能命中 L2 语义缓存
- **答案生成异常处理**：修复 `answer_generation_node` except 分支引用未定义 `question` 变量（改为 `original_question`）
- **启动阻塞：`build_conflict_answer` f-string 含反斜杠转义**：`nodes.py` 中 `f"- {c.get('source', '未...')}..."` 在 `{}` 表达式内出现 `\uXXXX` 转义，Python < 3.12 直接 SyntaxError 导致项目无法启动 → 将默认值提取到普通字符串变量后拼入 f-string
- **启动阻塞：视觉 Prompt dict 漏闭合花括号**：`prompts.py` 的 `VISION_STRUCTURED_EXTRACT_PROMPT`（254 行）与 `VISION_OCR_INJECTED_PROMPT`（288 行）把 `{"type": "text", "text": """...""")` 的 dict 闭合 `}` 误写成 `)`，导致括号不匹配 SyntaxError → 修正为 `"""},`

## v9.17 - 上下文管理优化：动态压缩 + 原文存档

### P1: 动态压缩阈值（70%窗口替代固定轮数）

- 上下文占用超过LLM窗口70%时才触发压缩，而非固定max\_rounds=2
- 短对话可保留更多轮（充分利用窗口），长对话自动缩减（防止溢出）
- `get_conversation_history_text` 默认 max\_rounds=0（动态计算）
- `update_clinical_snapshot_node` 从固定6条消息触发改为token占用率触发
- 优点：上下文利用率最优化，不会过早截断也不会过晚
- 缺点：token估算是粗略的（字符数×1.5），实际占比可能有±10%偏差；需要配置正确的LLM\_CONTEXT\_WINDOW\_TOKENS

### P2: 压缩时原文存档（不丢弃）

- 快照提取前，将早期消息原文完整存入磁盘（data/persisted\_outputs/archives/）
- clinical\_checkpoint 中记录 archive\_ids 列表，便于按需回溯
- 优点：历史对话可回溯，用户问"之前说的症状具体是什么"时可加载原文
- 缺点：磁盘占用随时间增长（三个月约44MB）；回溯时需加载原文到上下文，可能再次触发压缩

## v9.16 - 性能优化：检索提速 + 准确率提升

### 响应速率优化

**P1: Embedding缓存统一**（预计省 200\~600ms）

- 语义缓存和 HybridRetriever 的 embedding LRU 缓存双向同步
- 避免同一 query 在两套缓存中各存一份、重复计算
- 优点：零成本提速，无精度损失
- 缺点：两套缓存的淘汰策略不同（LRU vs 无淘汰），极端情况下缓存一致性可能漂移

**P4: Reranker预热**（首次请求省 \~4s）

- 应用启动时触发 Reranker 首次推理，避免冷启动
- 优点：消除首请求冷启动延迟
- 缺点：启动时间增加 \~4s；如果 Reranker 不常用则浪费资源

### 准确率优化

**P2: Reranker阈值校准 + 动态K值**

- Reranker 阈值从 0.02 调整为 0.005（原阈值过高导致合理文档被过滤后降级兜底）
- symptom 类型检索 K=8（多跳推理需更多候选），knowledge 类型 K=5
- 优点：减少误过滤，多跳问题召回率提升
- 缺点：K 值增大后 Reranker 输入增多，单次推理延迟略增（\~100ms）；阈值降低可能引入更多噪声文档

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
- symptom/knowledge类型 + 知识库无覆盖 → 拒答（不再走direct\_answer）

**拒答日志**：SQLite持久化（refusal\_logs表 + v\_refusal\_daily视图）

- 高频拒答聚类 → 定向补充知识库
- 误拒答回收 → 校准阈值参数
- 与node\_metrics通过request\_id关联，可追溯完整链路

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

| 字段                          | 用途          | 来源            |
| --------------------------- | ----------- | ------------- |
| doc\_version                | 版本冲突裁决      | 文件名/PDF属性/LLM |
| doc\_effective\_date        | 时效性校验       | 文件名/PDF属性/LLM |
| doc\_authority\_level       | 权威优先级       | 文件名/PDF属性/LLM |
| doc\_issuing\_body          | 发布机构        | PDF属性/LLM     |
| doc\_medical\_system        | 医学体系区分      | 文件名/LLM       |
| doc\_applicable\_population | 适用人群        | 文件名/LLM       |
| doc\_expire\_date           | 自动计算（生效+3年） | 派生            |
| doc\_meta\_confidence       | 整体置信度       | 交叉校验          |
| doc\_needs\_meta\_review    | 需人工审核       | 交叉校验          |

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

| 文件                           | 改动                                                                                       |
| ---------------------------- | ---------------------------------------------------------------------------------------- |
| `app/graph/nodes/models.py`  | 新增 `VisionAnalysisOutput` Pydantic模型（7字段+2校验器）                                           |
| `app/graph/nodes/prompts.py` | 新增 `VISION_STRUCTURED_EXTRACT_PROMPT` + `VISION_OCR_INJECTED_PROMPT`                     |
| `app/graph/nodes/nodes.py`   | 重构 `vision_analysis_node`（4步流程+Command路由）；新增5个辅助函数；重构 `stream_vision_answer`（图片摘要+流式RAG） |
| `app/graph/graph.py`         | `vision_analysis` 从固定边改为 Command 动态路由（→knowledge\_retrieval 或 →safety\_check）            |

***

## v9.12 - 路由评估体系：意图识别可量化、可归因、可迭代

### 核心改进：评估基础设施 + 分层归因 + Bad Case 反哺闭环

**问题**：意图识别（route\_node）的三层路由（规则→上下文→LLM）没有准确率度量，无法判断：

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

| 文件                                | 功能                                 |
| --------------------------------- | ---------------------------------- |
| `tests/data/route_test_set.jsonl` | 路由评估测试集（85条：54条golden + 31条边界case） |
| `scripts/evaluate_router.py`      | 路由评估脚本（规则/上下文/LLM三层评估+指标+对比基线）     |

### 改动文件

| 文件                                  | 改动                                                             |
| ----------------------------------- | -------------------------------------------------------------- |
| `app/graph/nodes/nodes.py`          | route\_node 增加 route\_layer 记录 + \_record\_route\_metrics 指标采集 |
| `app/evaluation/bad_case_runner.py` | 新增 run\_single\_route / run\_batch\_route 路由回归测试               |
| `app/core/metrics.py`               | 新增 get\_route\_stats() 路由分层统计查询                                |
| `tests/test_nodes.py`               | 新增 TestDetectRouteFromContext 7个测试 + 规则层6个扩展测试                 |

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
- 错误用例明细（含boundary\_reason归因）

***

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

| 阶段     | 操作                     | 用户感知        |
| ------ | ---------------------- | ----------- |
| **构建** | 影子集合写入，对线上不可见          | 始终走旧集合，完整结果 |
| **校验** | chunk数 + 抽样召回率 + 模型一致性 | 同上          |
| **切换** | 别名指针原子更新 + 刷新全局实例      | 毫秒级切换到新集合   |
| **清理** | 5 分钟后删除旧集合             | 无感知         |

### 新增组件

| 组件                                  | 说明                                                |
| ----------------------------------- | ------------------------------------------------- |
| `DualCollectionManager`             | 影子集合创建/校验/切换/回滚/清理                                |
| `kb_active.json`                    | 别名指针配置（active\_collection + previous\_collection） |
| `validate_shadow_collection()`      | 四维校验（chunk数/召回率/模型一致/pending状态）                   |
| `switch_active_collection()`        | 原子切换（temp文件→rename）                               |
| `schedule_cleanup_old_collection()` | 延迟5分钟清理旧集合                                        |
| `rollback_to_previous()`            | 紧急回滚到上一活跃集合                                       |

### 新增 API

| 接口                              | 方法   | 说明            |
| ------------------------------- | ---- | ------------- |
| `/api/admin/kb/rebuild`         | POST | 双集合重建（替代原地重建） |
| `/api/admin/kb/rollback`        | POST | 紧急回滚到上一集合     |
| `/api/admin/kb/collection-info` | GET  | 查询当前/上一集合信息   |

### 改动文件

- `vector_store.py`：新增 `DualCollectionManager`（\~340 行）
- `routes.py`：`kb_rebuild` 改用双集合流程 + 新增回滚/集合信息 API
- `hybrid_retriever.py`：新增 `reset_hybrid_retriever()` + `get_cached_hybrid_retriever()`
- `admin.html`：重建按钮改为"零停机" + 新增回滚按钮/集合信息按钮

***

## v9.10 - 双缓冲：消除知识库更新检索空窗期

### 核心改进：pending → active → deprecated 状态机

**问题**：v9.9 的增量更新流程是"先软删旧版本 → 再写新版本"，中间存在 \~1s 检索空窗期，
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

| 函数                            | 说明                              |
| ----------------------------- | ------------------------------- |
| `activate_document_version()` | 校验并激活新版本：pending→active + 废弃旧版本 |
| `deprecate_old_versions()`    | 旧版本 active→deprecated（非删除，可回滚）  |
| `cleanup_deprecated_chunks()` | 清理超过 5 分钟的 deprecated chunk     |

### 改动文件

- `kb_updater.py`：新增 3 个双缓冲函数 + `enrich_chunk_metadata` 增加 `status` 参数
- `hybrid_retriever.py`：ChromaDB 查询加 `status=active` 过滤
- `routes.py`：上传接口改用双缓冲流程
- `index.html`：发送消息前检测更新状态，非阻断提示

***

## v9.9 - 知识库更新架构优化（参考日志1最佳实践）

### 核心改进：6 大问题修复

| # | 日志1 指出的问题         | 修复                                               | 效果            |
| - | ----------------- | ------------------------------------------------ | ------------- |
| 1 | 只插入新向量，不删除旧向量     | 软删除旧版本 + version\_id +1                          | 修改5次不会留下5个版本  |
| 2 | Embedding 模型混用    | embedding\_model/dimension 元数据 + 一致性校验           | 索引=查询模型不一致时告警 |
| 3 | Chunk 策略变了历史数据不重建 | chunk\_strategy 版本化 + reconciliation 检测          | 策略变更触发全量重建    |
| 4 | 文档删除后仍被召回         | is\_deleted 软删除 + 检索过滤                           | 删除文档不再被检索命中   |
| 5 | 变更检测漏检            | 变更检测 API（磁盘 mtime vs 索引 updated\_at）             | 轮询兜底防漏检       |
| 6 | 更新无审计记录           | SQLite 审计日志（doc\_id/change\_type/result/elapsed） | 出问题可定位到具体环节   |

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

| 接口                                   | 方法     | 说明                              |
| ------------------------------------ | ------ | ------------------------------- |
| `/api/admin/kb/status`               | GET    | 知询知识库状态 + 一致性校验 + Embedding 信息  |
| `/api/admin/kb/upload`               | POST   | 上传文档（增量去重 + 版本管理 + 软删除旧版本 + 审计） |
| `/api/admin/kb/documents/{filename}` | DELETE | 软删除文档（is\_deleted=True）         |
| `/api/admin/kb/restore/{filename}`   | POST   | 恢复误删文档                          |
| `/api/admin/kb/rebuild`              | POST   | 重建知识库（审计日志）                     |
| `/api/admin/kb/audit-log`            | GET    | 查询审计日志                          |
| `/api/admin/kb/reconcile`            | GET    | 一致性校验（磁盘/向量库/Embedding/策略）      |
| `/api/admin/kb/stale-detect`         | GET    | 变更检测（陈旧文档）                      |

### 增量更新流程（content\_hash 去重）

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
- BM25 在重建时跳过 is\_deleted=True 的文档
- 软删除后 30 天由 `physical_cleanup_stale_deletes()` 物理清理

**新增文件**：

- `app/rag/kb_updater.py`：知识库更新管理核心模块

**改动文件**：

- `routes.py`：6 个新接口 + 上传/删除集成 kb\_updater
- `hybrid_retriever.py`：ChromaDB 查询加 is\_deleted 过滤

***

## v9.8 - 知识库管理 API + 并发安全

### 新增1：知识库管理接口

| 接口                                   | 方法     | 说明                                    |
| ------------------------------------ | ------ | ------------------------------------- |
| `/api/admin/kb/status`               | GET    | 知询知识库状态（文档列表、向量数、kb\_version、更新状态）    |
| `/api/admin/kb/upload`               | POST   | 上传文档（multipart/form-data，支持多文件），增量入库  |
| `/api/admin/kb/documents/{filename}` | DELETE | 删除指定文档（ChromaDB chunks + 磁盘文件 + 缓存清除） |
| `/api/admin/kb/rebuild`              | POST   | 重建知识库（清空→重新加载→切分→索引→写入）               |

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

| 机制                    | 实现                  | 保证                                |
| --------------------- | ------------------- | --------------------------------- |
| **asyncio.Lock 写锁**   | `_kb_update_lock`   | 同一时间只有一个更新操作（上传/删除/重建）            |
| **更新状态追踪**            | `_kb_update_status` | 前端可查询进度，更新中拒绝新的更新请求（409 Conflict） |
| **非阻塞检索**             | 检索不加锁               | 更新期间检索继续使用当前数据，新数据就绪后原子切换         |
| **run\_in\_executor** | CPU 密集操作在线程池执行      | 不阻塞 FastAPI 事件循环，其他请求正常处理         |
| **缓存清除**              | 更新完成后统一清除           | 防止旧缓存返回过期答案                       |

**并发场景处理**：

| 场景         | 行为                                     |
| ---------- | -------------------------------------- |
| 重建中 + 用户查询 | 查询走旧数据（ChromaDB persist），重建完成后下次查询走新数据 |
| 重建中 + 再次重建 | 返回 409 Conflict，提示"知识库正在更新中"           |
| 上传中 + 用户查询 | 查询走当前数据，上传完成后下次查询包含新文档                 |
| 上传中 + 删除   | 写锁排队，串行执行                              |

**改动文件**：

- `routes.py`：新增 4 个知识库管理接口 + `_kb_update_lock` + `_kb_update_status`

***

## v9.7 - 表格数据知识库处理（行级切片+双格式）

### 设计原则（参考日志1：复杂表格入库最佳实践）

> 核心原则：单个切片被检索出来时，仍然能够解释自己在原表中的位置和含义。

**当前实现与日志1方案的差异对比**：

| 维度     | 日志1推荐             | v9.7 旧方案（整表保留）  | v9.7 新方案（行级切片）                                    |
| ------ | ----------------- | --------------- | ------------------------------------------------- |
| 切片粒度   | 按行/按业务分组          | 整表（≤1500字符）或按行组 | **每行独立 chunk + 概览 chunk**                         |
| 上下文完整性 | 每个切片携带表头+标题+单位+页码 | HTML 注释上下文提示    | **自然语言摘要 + 字段路径 + 表格标题**                          |
| 合并单元格  | fill-down 继承      | 不处理             | **fill-down 自动继承**                                |
| 可追溯性   | 文档名+表格标题+页码+行号    | 仅 source        | **row\_primary\_key + row\_index + table\_title** |
| 检索精度   | 行级精确检索            | 整表返回（含无关行）      | **行级精确检索，对比查询走概览 chunk**                          |

### 新增1：Excel/CSV/Markdown 文档加载器

**支持格式**：

| 格式      | 加载器         | 依赖                | 说明                                  |
| ------- | ----------- | ----------------- | ----------------------------------- |
| `.xlsx` | `load_xlsx` | pandas + openpyxl | 每个 Sheet → 一个 Document（Markdown 表格） |
| `.xls`  | `load_xlsx` | pandas + xlrd（降级） | 旧版 Excel 格式                         |
| `.csv`  | `load_csv`  | pandas            | 自动检测编码（utf-8/gbk/gb2312/latin-1）    |
| `.md`   | `load_md`   | 无                 | 保留原始 Markdown 格式（含表格）               |

**转换流程**：Excel/CSV → DataFrame → Markdown 表格格式 → Document（含表格元数据）

**表格元数据**：

- `is_table: True` — 标记为表格文档
- `table_headers: List[str]` — 列名列表（检索增强用）
- `table_row_count / table_col_count` — 行列数
- `table_header_summary: str` — 表头摘要（如"表格列：药物, 剂量, 频次"）
- `sheet_name: str` — Excel Sheet 名

**LOADERS 字典扩展**：`.txt` `.pdf` `.docx` **`.md`** **`.xlsx`** **`.xls`** **`.csv`**

### 新增2：行级切片 + 双格式（核心改动）

**问题**：原有方案"整表保留"→ 查询"布洛芬的每日最大量"返回整个对比表（7 行），LLM 需自己从表格中提取，容易出错；且 Embedding/BM25 对表格格式文本检索效果差

**解决**：行级切片 + 双格式策略

| chunk 类型     | 数量    | 用途    | 示例查询             |
| ------------ | ----- | ----- | ---------------- |
| **概览 chunk** | 1 个/表 | 对比类查询 | "布洛芬和对乙酰氨基酚哪个好？" |
| **行级 chunk** | N 个/表 | 精确查询  | "布洛芬的每日最大量是多少？"  |

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

**自然语言摘要** **`_generate_row_summary()`**：

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

| 元数据字段             | 说明                                  | 示例                       |
| ----------------- | ----------------------------------- | ------------------------ |
| `chunk_type`      | `"table_row"` or `"table_overview"` | `table_row`              |
| `table_title`     | 最近的上层 Markdown 标题                   | `"对乙酰氨基酚 vs 布洛芬对比"`      |
| `table_headers`   | 完整表头列名                              | `["项目","对乙酰氨基酚","布洛芬"]`  |
| `row_index`       | 行号（0-based）                         | `5`                      |
| `row_primary_key` | 第一列值（行主键）                           | `"每日最大量"`                |
| `row_summary`     | 自然语言摘要                              | `"在对乙酰氨基酚...布洛芬为1200mg"` |

**改动文件**：

- `loader.py`：新增 `load_xlsx`、`load_csv`、`load_md`、`_dataframe_to_markdown_table`（含 fill-down）
- `loader.py`：新增 `_detect_markdown_tables`、`_enrich_table_metadata`、`_extract_table_title`
- `loader.py`：新增 `_split_table_aware`（行级切片）、`_generate_row_chunks`、`_generate_row_summary`
- `loader.py`：新增 `_segment_by_table`（表格/非表格分段）
- `loader.py`：修改 `split_documents`、`_split_by_markdown_headers`（含表格章节走行级切片）
- `loader.py`：增强 `add_metadata`（溯源元数据：file\_type/file\_size/doc\_hash/source\_trace）
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

| 元数据            | 说明          | 溯源用途   | 示例                               |
| -------------- | ----------- | ------ | -------------------------------- |
| `source`       | 文档文件名       | 定位文档   | `"发热诊断与家庭护理指南.txt"`              |
| `file_path`    | 完整路径        | 打开原文   | `"d:/Agent/.../发热诊断与家庭护理指南.txt"` |
| `file_type`    | 文件类型        | 选择打开方式 | `"txt"`                          |
| `file_size`    | 文件大小        | 完整性校验  | `15234`                          |
| `doc_hash`     | 内容 MD5 前8位  | 防篡改校验  | `"a3f7b2c1"`                     |
| `page_number`  | 页码（PDF/扫描件） | 定位页码   | `5`                              |
| `source_trace` | 溯源路径        | 一键追溯   | `"指南.txt \| 药物对比 \| 行5: 每日最大量"`  |

**`source_trace`** **格式**：`文档名 | 表格标题 | 行号: 行主键`

- 表格行级 chunk：`"发热诊断与家庭护理指南.txt | 对乙酰氨基酚 vs 布洛芬对比 | 行5: 每日最大量"`
- 表格概览 chunk：`"发热诊断与家庭护理指南.txt | 对乙酰氨基酚 vs 布洛芬对比 | 概览（前3行）"`
- 非表格 chunk：无 `source_trace`（通过 `source` + `page_number` 追溯）

***

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
   - 例："发烧怎么处理便秘怎么处理" → \["发烧怎么处理", "便秘怎么处理"]

### 修复2：子问题检索串行执行导致首字延迟翻倍

**问题**：`knowledge_retrieval_node` 中多子问题用 `for` 循环串行调用 `retriever.invoke()`，2 个子问题串行约 4s

**修复**：改用 `ThreadPoolExecutor` 真正并行检索，检索耗时从 \~4s 降到 \~2s

### 修复3：adaptive\_threshold 竞态导致 Reranker 被整体跳过

**问题**：`ThreadPoolExecutor` 并行检索时，`get_adaptive_threshold()` 阈值尚未注册，触发 `KeyError("未注册的阈值：RERANKER_THRESHOLD")`，整个 Reranker 被 `except` 跳过，返回 7 篇未精排的低质量文档 → 启发式过滤 10→2 → LLM 说"文档未提及"

**修复**（双重保险）：

1. `adaptive_threshold.py`：`get_adaptive_threshold()` 改为双重检查锁定（DCL），确保初始化+注册原子完成
2. `hybrid_retriever.py`：阈值获取失败时**用默认值**（RERANKER\_THRESHOLD=0.02, HIGH\_CONFIDENCE\_THRESHOLD=0.08），而非让整个 Reranker 被跳过。这是防御性修复，即使 DCL 失败也能保证 Reranker 正常执行

### 修复4：邻域扩展导致 sub\_question 与文档内容不匹配，启发式过滤误杀关键文档

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

***

## v9.6.1 - 问题拆解延迟修复 + 拆解条件修复

### 修复1：移除长度阈值（"发烧？便秘？"仅 8 字符但也应拆解）

**根因**：`len(question) <= 20` 跳过了 14 字符的复合问题"发烧怎么处理？便秘怎么处理？"
**修复**：移除长度阈值，仅用问号数 ≥ 2 或连接词作为复合检测条件

### 修复2：来源多样性过滤 max\_per\_source 2→3

**根因**：多子问题检索后，5 篇文档经 `max_per_source=2` 过滤只剩 2 篇（且同源），便秘文档被丢弃
**修复**：`max_per_source=3`，给多子问题检索更多空间

### 修复3：问题拆解节点延迟 4850ms → \~1200ms

**根因**（日志1.txt 分析）：

1. `invoke_structured` 尝试 2 次 × 3 种策略 = 6 次 LLM 调用，Ollama 本地模型不支持 Tool Calling/JSON Mode，前两层各报错，Layer 3 也解析失败 → 4 次 HTTP × \~1.2s = 4.85s
2. 子问题2 语义缓存误命中子问题1 的结果（83.86% 相似度命中），导致药物文档未检索
3. 文档去重过度：6 篇 → 3 篇 → 2 篇

**修复**：

- `max_attempts=2 → 1`，`force_strategy="text_only"`：跳过 Tool Calling 和 JSON Mode，直接用 Layer 3 纯文本解析
- 2 秒超时保护：超过 2s 直接降级规则拆解
- 子问题检索用 `original_query=sub_q`（子问题自身），避免缓存误命中
- 降级规则拆解按问号切分（无需 LLM，0ms）

***

## v9.6 - 长问题拆解 + 多子问题并行检索

### 新增：question\_decompose\_node 长问题拆解节点

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
- `models.py`：`QuestionDecomposeOutput`（need\_decompose + sub\_questions + field\_validator）
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

***

## v9.5 - 四层上下文压缩策略（L1→L3→L2→L4）

### 新增：context\_manager.py 四层上下文压缩模块

**背景**：对话历史随轮次增长，messages 列表无限膨胀导致 Prompt 超出 4K token 限制，LLM 遗忘早期对话信息

**架构**：四层压缩 Pipeline，执行顺序 L1 → L3 → L2 → L4

| 层级 | 策略       | 触发条件          | 是否用 LLM | 效果                            |
| -- | -------- | ------------- | ------- | ----------------------------- |
| L1 | 中间输出清除   | 消息数 > 3       | 否       | 中间 AI 回答只保留首句摘要               |
| L3 | 大输出持久化   | 单条 > 30KB     | 否       | 写入磁盘，占位符 `<persisted-output>` |
| L2 | 工具调用裁剪   | RAG 文档块/工具痕迹  | 否       | `[参考了 N 篇文档]` 占位符             |
| L4 | LLM 摘要压缩 | 总量 > 50000 字符 | 是       | 保留 5 类关键信息，原始存 transcript     |

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

- `ContextSummaryOutput` — L4 摘要结构化输出（5 个字段 + field\_validator）

***

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

***

## v9.3 - 前端来源展示与反馈按钮修复

### Bug1：SSE 完成事件类型不匹配导致来源和反馈按钮丢失

**问题**：后端发送的完成事件为 JSON `{"type": "done", "request_id": "..."}` ，但前端检查的是字符串 `data === '[DONE]'`，两者永远不匹配

- 导致来源展示代码和 `addFeedbackButtons()` 调用从未执行
- 用户看不到文档来源，也看不到 👍/👎 反馈按钮

**修复**：

1. **完成事件处理**：`if (data === '[DONE]')` → `else if (parsed.type === 'done')`，正确匹配 JSON 完成事件
2. **request\_id 传递**：从 done 事件中提取 `request_id`，传入 `addFeedbackButtons()`，反馈提交时关联请求
3. **来源渲染**：新增 `renderSources()` 函数，来源显示为带编号的蓝色标签 `[1] xxx.md [2] yyy.txt`
4. **来源样式**：新增 `.source-item` 样式（蓝底圆角标签）+ `.sources` 左边框加粗
5. **👍 按钮修复**：之前只切换 UI 状态不提交反馈 → 现在调用 `submitFeedbackAPI('up', ...)` 真正提交
6. **反馈数据修复**：`submitFeedback()` 之前缺失 `rating` 和 `request_id` 字段 → 现在完整发送 `rating`/`request_id`/`question`/`reason`/`note`/`answer_preview`
7. **同步请求**：同样适配 `renderSources()` 和 `addFeedbackButtons()` 新参数

**效果**：流式和同步模式下都能正确显示文档来源 + 反馈按钮，反馈数据完整关联 request\_id

***

## v9.2 - 时间退化风险修复（5 项安全加固）

### 🔴 漏洞1：语义缓存毒化（最高优先级）✅ 已修复

**问题**：语义缓存 key 仅含 `md5(query)`，无知识库版本绑定。知识库更新后旧缓存仍返回过期答案 → 医疗安全风险

- 衰减速度：1\~2 周
- 安全影响：⚠️ 用药剂量/禁忌症过期

**修复**：

1. **`vector_store.py`** **新增** **`get_kb_version()`**：基于 ChromaDB 所有 doc\_id 排序哈希 + 文档数量生成 8 位版本指纹
   - 首次调用计算并缓存，后续 O(1) 读取
   - `add_documents()` / `delete_collection()` 后自动调用 `invalidate_kb_version()` 使指纹失效
   - 知识库更新 → kb\_version 变化 → 旧缓存 key 不再匹配 → 自动失效
2. **`semantic_cache.py`** **修复**：
   - `set()`：缓存 key 从 `md5(query)` 改为 `md5(query:kb_version)`，写入时记录 `kb_version`
   - `get()`：命中相似查询后校验 `cached_kb_version == current_kb_version`，不匹配则删除过期条目并 miss
3. **`redis_cache.py`** **修复**：
   - `_generate_key()`：自动注入 `kb_version` 到 key 哈希输入，L0 检索缓存同样防毒化

**效果**：知识库每次更新后，L0（Redis 缓存）和 L2（语义缓存）自动失效，不再返回过期医疗答案

### 🔴 漏洞2：临床快照状态腐烂 ✅ 已修复

**问题**：`clinical_checkpoint` 由 LLM 增量更新，无字段级合并策略，`medication_history`/`red_flags` 被全量覆盖

- LLM 可能遗忘已有用药记录、篡改 chief\_complaint、凭空编造药物
- 无快照历史版本、无回滚机制、无上限约束
- 衰减速度：2\~4 周
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
   - medication\_history ≤ 15, red\_flags ≤ 10, symptom\_timeline ≤ 20, confirmed\_facts ≤ 20, ruled\_out ≤ 15
   - 防止 LLM 编造过多条目导致 Prompt token 膨胀
3. **合并变更日志**：每次合并后记录字段条目数变化，便于审计

**效果**：过敏史/既往史不会因 LLM "遗忘" 而丢失，用药记录不会因全量覆盖而消失，主诉不会在追问中被篡改

### 🟡 漏洞3：PostgresStore Append-Only 无界膨胀 ✅ 已修复

**问题**：`symptom_events`/`medication_events`/`bad_cases`/`query_history` 四个命名空间只有 append，无 prune/compact

- `get_symptom_events()` 全量加载后截断，数据量增大后性能退化
- 衰减速度：1\~2 个月
- 安全影响：性能退化 + 早期记录被遗忘

**修复**：

1. **Prune 机制**（`long_term_memory.py` 新增）：
   - `prune_namespace()`：按保留天数 + 条目上限双重清理，自动删除过期和超量记录
   - `prune_all_namespaces()`：批量清理用户全部命名空间
   - `prune_all_users()`：管理员接口，清理多用户数据
   - 各命名空间默认保留天数：symptom\_events=90, medication\_events=90, bad\_cases=180, query\_history=30
   - 各命名空间条目上限：symptom\_events=500, medication\_events=300, bad\_cases=500, query\_history=200
2. **查询优化**（提前截断）：
   - `get_symptom_events()`：最多读取 `limit * 3` 条后排序截断，避免全量加载
   - `get_query_history()` / `get_bad_cases()` / `get_medication_events()`：最多读取 `limit * 2` 条
   - 降低内存峰值，减少排序耗时

**效果**：数据量 3 个月后稳定在配额内，查询性能不再随时间退化

### 🟡 漏洞4：硬编码阈值随数据分布漂移失效 ✅ 已修复

**问题**：6 个关键阈值（0.08/0.02/0.01/0.05/0.92）在特定数据分布下调优，知识库扩张后静默失效

- `HIGH_CONFIDENCE_THRESHOLD=0.08`：稀疏空间"极度相似" → 密集空间"有点相关"
- 衰减速度：1\~3 个月
- 安全影响：⚠️ 跳过 Reranker 导致幻觉

**修复**：

1. **新增** **`app/core/adaptive_threshold.py`**：`AdaptiveThreshold` 自适应阈值管理器
   - 基于运行时百分位统计动态调整阈值
   - 冷启动：前 100 个样本使用默认值
   - 自动校准：每 1000 个样本重新计算百分位数
   - 持久化：校准值写入 SQLite（`data/adaptive_thresholds.db`），重启后恢复
   - 管理员接口：`force_recalibrate()` 手动触发校准，`get_stats()` 查看统计
2. **注册的三个自适应阈值**：
   | 阈值                          | 默认值  | 策略         | 百分位 | 范围             |
   | --------------------------- | ---- | ---------- | --- | -------------- |
   | HIGH\_CONFIDENCE\_THRESHOLD | 0.08 | percentile | P5  | \[0.01, 0.20]  |
   | RERANKER\_THRESHOLD         | 0.02 | percentile | P5  | \[0.005, 0.10] |
   | SEMANTIC\_CACHE\_THRESHOLD  | 0.92 | percentile | P95 | \[0.85, 0.99]  |
3. **`hybrid_retriever.py`** **修改**：
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

***

## v9.0 - RAG 流水线性能优化（TTFT 预计 -600\~900ms）

### 优化1：Reranker 三阶段化（970ms → \~300ms）

**问题**：Reranker 入参 10\~20 篇文档，max\_length=512，CPU 推理 400\~970ms

**修复**：

- RRF 融合后先轻量截断 top 8（`RERANKER_INPUT_CAP=8`），再进 Reranker 精排
- `max_length` 512 → 256（200 字中文 ≈ 256 tokens，覆盖 95%+ 关键信息）
- `MAX_RERANK_DOC_CHARS` 300 → 200（头 134 + 尾 66）
- `DEFAULT_K` 5 → 3（减少送入 LLM 的文档数，缩短 Prompt token）
- `rerank_top_k` 5 → 8（入参数，RRF 融合后截断数）

**预期**：Reranker 970ms → \~300ms，TTFT -600ms

### 优化2：Embedding LRU 缓存（重复查询 0ms vs API 200\~400ms）

**问题**：每次查询调智谱 embedding-3 API，网络延迟 200\~400ms

**修复**：

- 新增 `_EmbeddingLRUCache` 类（LRU，128 条上限，30 分钟 TTL）
- 相同查询复用 embedding 向量，命中时 0ms
- 跨请求复用（同一用户多次查询相同问题）
- 约占 1MB 内存（128 \* 2048 \* 4 bytes）

**预期**：重复查询 Embedding 400ms → 0ms

### 优化3：邻域扩展字符上限（2000 → 1500）

**问题**：邻域扩展后文档过长，Prompt token 数膨胀，TTFT 增加

**修复**：`MAX_SIBLING_CHARS` 2000 → 1500

**预期**：TTFT -200\~300ms

### 优化4：知识库无覆盖早退（避免无效重试 1\~2s）

**问题**：Reranker 最高分 < 0.01 时仍走自纠正循环，浪费 1\~2s

**修复**：新增 `RERANK_NO_COVERAGE_THRESHOLD=0.01`，低于此值直接走 `direct_answer`

**预期**：知识库无覆盖场景 TTFT -1000\~2000ms

### 优化5：查询预处理（纠错 + 同义词 + 语气词清理）

**问题**：口语化查询（"头疼咋办啊"）BM25 命中率低，Embedding 噪声大

**修复**：

- 同义词标准化（"头疼"→"头痛"、"拉肚子"→"腹泻"、"退烧"→"退热" 等 18 条）
- 语气词前缀清理（"我想问一下"→""）
- 语气词后缀清理（"啊呀呢吧"→""）
- 仅对未重写的原始查询生效，重写查询已是高质量查询

**预期**：口语化查询检索召回率 +10\~15%

### 优化6：pyahocorasick C 扩展安装

**问题**：纯 Python AC 自动机实现，关键词库扩展后性能差距大

**修复**：`pip install pyahocorasick==2.3.1`，`keyword_matcher.py` 自动使用 C 扩展版

**预期**：规则层关键词匹配提速 5\~10x（大规模关键词库时）

### 优化7：黄金测试集 + 评估模块适配

**新增**：

- `tests/data/golden_test_set.jsonl`：53 条人工精选黄金测试集（覆盖用药安全/症状分诊/急救/慢性病/剂量/知识问答）
- `scripts/generate_golden_test_set.py`：自动生成脚本（73 条模板）
- `evaluation.py`：新增 `query` 字段和 `key_facts` 字段支持

### 优化8：渐进式进度反馈（SSE progress events）

**问题**：用户发出问题后等 2\~3s 才看到第一个 token，期间无任何反馈

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

**问题**：只有 timing\_decorator 写日志，无结构化指标，无法做 P50/P95/P99 分析

**新增**：

- `app/core/metrics.py`：`MetricsCollector` 类，SQLite 存储（3 张表）
  - `node_metrics`：节点级耗时，支持 P50/P95/P99 分析
  - `token_usage`：LLM Token 用量，成本估算，按模型/节点/每日趋势
  - `feedback`：用户反馈闭环，满意度率，差评原因分布
- `timing_decorator` / `async_timing_decorator` 自动写入 node\_metrics
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

- `app/core/circuit_breaker.py`：`CircuitBreaker` 类，CLOSED → OPEN → HALF\_OPEN 三态
- 集成到 `hybrid_retriever.py` 的 Embedding API 调用处
- 连续 3 次失败 → OPEN 状态（快速失败，跳过 API 调用）
- 30 秒后 → HALF\_OPEN（放行一次探测请求）
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
- SQLite 存储：`token_usage` 表（request\_id, model, prompt\_tokens, completion\_tokens, estimated\_cost）
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
- `llm.invoke()` 同时支持 str 和 List\[BaseMessage]，无缝兼容

**Prompt 清单**：

| Prompt                           | 角色                  | 变量                                                                             |
| -------------------------------- | ------------------- | ------------------------------------------------------------------------------ |
| RAG\_ANSWER\_PROMPT              | system + human      | context, question, frozen\_profile, time\_facts, checkpoint, history, followup |
| RAG\_ANSWER\_NO\_CONTEXT\_PROMPT | system + human      | question, frozen\_profile, history                                             |
| DIRECT\_ANSWER\_PROMPT           | system + human      | question, frozen\_profile, checkpoint, history                                 |
| ROUTER\_PROMPT                   | system + human      | question                                                                       |
| QUERY\_REWRITE\_PROMPT           | system + human      | history\_summary, question                                                     |
| SAFETY\_CHECK\_PROMPT            | system + human      | answer, clinical\_snapshot                                                     |
| PROFILE\_EXTRACTION\_PROMPT      | system + human      | question                                                                       |
| CHECKPOINT\_UPDATE\_PROMPT       | system + human      | existing\_snapshot, new\_messages                                              |
| CHECKPOINT\_NEW\_PROMPT          | system + human      | new\_messages                                                                  |
| HYDE\_PROMPT                     | system + human      | question                                                                       |
| VISION\_ANALYSIS\_PROMPT         | system + human(多模态) | question, image\_url                                                           |

### 优化14：Pydantic 结构化输出校验补全

**问题**：7 个 Pydantic 模型中只有 3 个被实际使用，路由和查询重写无校验

**修复**：

1. **路由节点**：f-string + `parse_router_output()` 正则 → `ROUTER_PROMPT` + `invoke_json_once_with_fallback` + `RouterOutput`
   - `RouterOutput.question_type`：Literal\["symptom", "knowledge", "general"]
   - `field_validator`：兼容中文/复数/变体输入（"症状"→"symptom"，"symptoms"→"symptom"）
   - 校验失败 → 兜底 "general"
2. **查询重写节点**：正则提取 `FINAL:` / `SEARCH:` → `QUERY_REWRITE_PROMPT` + `invoke_json_once_with_fallback` + `QueryRewriteOutput`
   - `QueryRewriteOutput` 扩展：`rewritten_query` → `final_question` + `search_keywords`
   - `field_validator`：自动去除 `FINAL:` / `SEARCH:` 残留前缀
   - 支持 `max_attempts=2` 重试
3. **models.py 增强**：
   - `RouterOutput`：新增 `normalize_question_type` field\_validator
   - `QueryRewriteOutput`：扩展为双字段 + 两个 field\_validator
   - `ProfileExtractionOutput`：新增 `coerce_age`（"30岁"→30）、`coerce_allergies`（"青霉素,头孢"→\["青霉素","头孢"]）
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
- 当前方案完全依赖后处理（extract\_json\_block + json\_repair + Pydantic），是"打补丁"思维
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

| 模型                     | Layer 1        | Layer 2     | Layer 3 |
| ---------------------- | -------------- | ----------- | ------- |
| glm-4-flash            | ✅ Tool Calling | ✅ JSON Mode | 兜底      |
| glm-4-plus             | ✅ Tool Calling | ✅ JSON Mode | 兜底      |
| Ollama qwen2.5 (v0.3+) | ✅ Tool Calling | ✅ JSON Mode | 兜底      |
| Ollama qwen2.5 (旧版)    | ❌ 不支持          | ✅ JSON Mode | 兜底      |

### 累计性能演进

| 版本       | TTFT (自包含)   | TTFT (追问)    | 核心优化                         |
| -------- | ------------ | ------------ | ---------------------------- |
| v4.3     | \~4000ms     | \~6000ms     | 症状短路+缓存+Prompt精简             |
| v5.5     | \~2700ms     | \~4000ms     | 自包含跳过重写                      |
| v8.4     | \~2700ms     | \~3500ms     | 症状移除LLM+HyDE短路               |
| v8.5     | \~2700ms     | \~3500ms     | HyDE默认关闭                     |
| **v9.0** | **\~2000ms** | **\~2800ms** | Reranker三阶段+Embedding缓存+邻域缩减 |

***

## v8.5 - 指代词误判修复 + HyDE 移除 + Faithfulness 提升

### Bug1：\_ANAPHORA\_PATTERNS 多义字导致自包含查询强制重写

"流鼻血了怎么办？"被判定为"含指代词/省略结构"→ 强制调用 LLM 重写（3008ms）→ 记录 Bad case。

修复：移除 `"呢"`、`"再"`、`"也"` 多义字；补充实体词表

### Bug2：自包含查询跳过重写但没跳过 HyDE（5180ms 白跑）

修复：自包含查询统一跳过重写 + HyDE

### 决策：HyDE 基于实测数据默认关闭

A/B 测试结果：Recall -13.3%，耗时 +1574ms，4 条负向仅 2 条正向。

处置：`ENABLE_HYDE=False` 默认关闭

### Bug3：规则引擎中文分词失效（评估分数全为 0）

`_tokenize_chinese` 用空格分词，中文无空格 → 整段文本变成 1 个 token → 不重叠 → 分数为 0

修复：改用 jieba 精确模式分词 + 停用词过滤。修复后分数：faithfulness 0.36, relevance 0.81, context\_relevance 0.22, context\_precision 0.48

### Faithfulness 提升：强化 RAG Prompt 忠实度约束

| 维度   | 旧 Prompt   | 新 Prompt                             |
| ---- | ---------- | ------------------------------------ |
| 约束   | "基于文档回答"   | "严格基于【文档】内容回答，不得编造文档中未提及的药物/剂量/治疗方案" |
| 无信息时 | "无相关信息则说明" | "明确告知'根据现有资料无法回答'，不要用自身知识补充"         |
| 引用约束 | 无          | "引用药物/剂量时，必须与文档原文一致"                 |

### RAGAS 评估模块

| 问题    | 回答                                            |
| ----- | --------------------------------------------- |
| 需要魔法？ | 不需要——评估 LLM 用智谱 API（`get_llm()`），不用 OpenAI    |
| 免费？   | RAGAS 框架免费（MIT），LLM API 调用少量成本（\~¥0.04/10条）   |
| 评估模式  | RAGAS（`pip install ragas`）/ 规则引擎（默认，jieba 分词） |

### Bug2：自包含查询跳过重写但没跳过 HyDE（5180ms 白跑）

"便秘怎么办？"：重写正确跳过，但 HyDE 未跳过（HyDE 跳过条件与重写条件独立）

修复：自包含查询统一跳过重写 + HyDE

### 决策：HyDE 基于实测数据默认关闭

**A/B 测试结果（10 条典型医疗查询）**：

| 指标            | 无 HyDE | 有 HyDE  | 差异         |
| ------------- | ------ | ------- | ---------- |
| 平均 Recall     | 56.7%  | 43.3%   | **-13.3%** |
| 平均耗时          | 1473ms | 3048ms  | +1574ms    |
| Recall 正向     | —      | 2 条     | —          |
| Recall **负向** | —      | **4 条** | —          |
| 文档重叠度         | 10%    | —       | 几乎完全不同     |

**结论**：HyDE 在当前架构下为负收益组件。原因：

1. 规则引擎 + 症状解析已做查询标准化，填平了"查询-文档语义鸿沟"
2. 现代中文 Embedding（bge/m3e）语义理解已足够强
3. 混合检索（BM25 + Dense）精确匹配不依赖语义桥接
4. Ollama 1.5b 本地模型生成质量有限，假想答案含噪反而污染检索

**处置**：默认关闭（`ENABLE_HYDE=False`），保留开关供未来长尾模糊查询按需启用---|------|
\| `app/graph/nodes/nodes.py` | `_ANAPHORA_PATTERNS` 移除多义字；`_DOMAIN_ENTITY_KEYWORDS` 新增 7 个症状关键词；HyDE 跳过逻辑改为自包含判断 |

***

## v8.4 - 症状解析架构瘦身 + TTFT 优化（6.3s → 2.7s）

### 核心设计决策：移除症状解析 LLM 兜底

**问题**：症状解析和查询改写做的是同一件事（理解查询医学语义），但分别调用 LLM，导致：

- 症状解析 LLM 兜底 2.8s，结果常校验失败（如 severity="轻至中度"），白跑
- 查询改写对自包含查询跳过，但 HyDE 仍执行 734ms
- 两次 LLM 调用做重复劳动，合计 3.5s

**决策**：症状解析只保留规则引擎（<5ms），移除 LLM 兜底。规则未命中时降级为原始查询检索，下游完全能处理。

**理由**：

1. symptoms=None 时：`_build_followup_hints()` 返回空，RAG prompt 不追加追问，LLM 照常生成答案
2. 时间锚定已独立于症状关键词（新增 `_extract_time_grounding()`），规则未命中时 onset\_ts 仍能计算
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

| 节点         | 修复前           | 修复后          | 节省               |
| ---------- | ------------- | ------------ | ---------------- |
| 症状解析       | 2809ms（LLM白跑） | <5ms（规则/降级）  | 2804ms           |
| HyDE       | 734ms（未跳过）    | 0ms（跳过）      | 734ms            |
| 知识检索       | 1559ms        | 1559ms       | —                |
| LLM 首token | \~1042ms      | \~1042ms     | —                |
| **TTFT**   | **6269ms**    | **\~2700ms** | **3538ms（-56%）** |

### 修改文件

| 文件                            | 改动                                                                                               |
| ----------------------------- | ------------------------------------------------------------------------------------------------ |
| `app/graph/nodes/nodes.py`    | `symptom_analysis_node` 移除 LLM 兜底；新增 `_extract_time_grounding()`；`_hyde_symptom_words` 补充"流血""血" |
| `app/core/keyword_matcher.py` | 症状映射新增 7 个关键词                                                                                    |

### 后续优化方向

| 方向                       | 预期收益              | 复杂度         |
| ------------------------ | ----------------- | ----------- |
| 本地 Embedding 模型          | 730ms → <50ms     | 中（需 GPU 内存） |
| 症状解析与检索并行                | 串行 4.3s → 并行 2.8s | 高（需重构节点依赖）  |
| Reranker max\_length=256 | 777ms → \~400ms   | 低           |

***

## v8.3 - 邻域扩展（Sibling Expansion）：跨章节信息补全 + 幻觉消除

### 背景

用户问"头痛怎么办？"，答案天然分布在"危险信号（排除禁忌）"+"药物选择（治疗）"+"非药物治疗（辅助）"等多个 Parent 中。但 Reranker 只返回了 1 个 Parent（"头痛危险信号"），而"头痛的药物选择"（含布洛芬等药物信息）不在检索候选中（Dense 排名 #50+）。

LLM 拿到不含药物的文档，被迫靠自身知识编造布洛芬/对乙酰氨基酚 → 幻觉检测误报。

根因：Embedding 对"怎么办"和"药物选择"的语义映射不够，Dense 召回不了跨章节的相关内容。这不是调参能解决的——调大 top\_k 从 6 到 50 不现实。

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

| 维度    | 旧方案（v8.2）             | 新方案（v8.3）                             |
| ----- | --------------------- | ------------------------------------- |
| 检索粒度  | 仅 Reranker 返回的 Parent | Parent + 相邻兄弟章节                       |
| 跨章节信息 | 缺失 → LLM 编造 → 幻觉      | 邻域扩展自动补全                              |
| 新增元数据 | 无                     | `doc_id`（所属文档）+ `section_index`（章节序号） |
| 字符上限  | 无                     | `MAX_SIBLING_CHARS=2000`（防撑爆 LLM 上下文） |
| 额外延迟  | —                     | <1ms（内存查找）                            |
| 幻觉检测  | 误报（文档不含布洛芬）           | 正确（扩展后含布洛芬）                           |

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

| 配置                  | 默认值  | 说明                |
| ------------------- | ---- | ----------------- |
| `SIBLING_WINDOW`    | 1    | 邻域窗口大小，1=前后各取1个章节 |
| `MAX_SIBLING_CHARS` | 2000 | 扩展后最大总字符数         |

### 修改文件

| 文件                                | 改动                                                                                            |
| --------------------------------- | --------------------------------------------------------------------------------------------- |
| `app/rag/parent_child_store.py`   | 新增 `expand_with_siblings()` 方法；`build_index()` 写入 `doc_id`+`section_index` 元数据；持久化格式升级（含章节索引） |
| `app/rag/hybrid_retriever.py`     | parent 映射后调用 `expand_with_siblings()`                                                         |
| `app/core/config.py`              | 新增 `SIBLING_WINDOW`、`MAX_SIBLING_CHARS` 配置项                                                   |
| `scripts/rebuild_vector_store.py` | 重建前删除旧 parent\_store.pkl，避免旧数据叠加                                                              |

### 兼容性

- 旧版 `parent_store.pkl`（无 `doc_id`/`section_index`）自动降级：`_rebuild_index_from_store()` 从 Parent 元数据中重建索引
- 无章节索引时跳过邻域扩展，不影响现有功能

***

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

| 维度                     | 旧代码                                               | 实际                       |
| ---------------------- | ------------------------------------------------- | ------------------------ |
| 调用方法                   | `Chroma.similarity_search_by_vector_with_score()` | **此方法不存在！**              |
| `hasattr` 检查           | `if hasattr(...)` → `False` → 跳过                  | 永远跳过带分数的分支               |
| 回退路径                   | `similarity_search_by_vector()` → 无分数             | `top1_score` 保持默认值 `0.0` |
| High-Confidence Bypass | `0 <= 0.0 < 0.08` = `True` → 跳过 Reranker          | **每次请求都跳过 Reranker！**    |

**langchain\_chroma.Chroma 的正确方法名**：

- `similarity_search_with_score` → 文本查询 + ChromaDB distance ✅
- `similarity_search_by_vector_with_relevance_scores` → 向量查询 + 归一化分数 ✅
- ~~`similarity_search_by_vector_with_score`~~ → **不存在** ❌

### 根因 2：幻觉检测读取过滤后的文档

| 维度   | 旧实现                                   | 新实现                                |
| ---- | ------------------------------------- | ---------------------------------- |
| 检测依据 | `state["retrieved_docs"]`（过滤后）        | `state["all_retrieved_docs"]`（过滤前） |
| 问题   | grade\_documents\_node 过滤后只剩 1 篇不相关文档 | 全量文档包含所有检索结果                       |
| 后果   | 布洛芬在神经系统文档中，但该文档已被过滤 → 误报幻觉           | 正确识别布洛芬在检索文档中存在                    |

### 根因 3：`has_query_overlap` 口语虚词误判

| 维度          | 旧实现                                                  | 新实现                              |
| ----------- | ---------------------------------------------------- | -------------------------------- |
| 查询 "头痛怎么办？" | tokens = \["头痛", "怎么办"]                              | tokens = \["头痛"]（过滤"怎么办"虚词）      |
| 神经系统文档      | "头痛"匹配1个，"怎么办"不匹配 → match=1 → `1 >= 1.5` = **False** | "头痛"匹配1个 → `1 >= 1` = **True** ✅ |
| 后果          | 含"头痛"的正确文档被误过滤掉                                      | 正确保留                             |

### 修复 1：三层回退策略获取真实 distance

| 优先级 | 策略                 | 方法                                                       | 返回值                        |
| --- | ------------------ | -------------------------------------------------------- | -------------------------- |
| 1   | 底层 collection 直接查询 | `col.query(query_embeddings=..., include=["distances"])` | ChromaDB cosine distance ✅ |
| 2   | LangChain 向量查询     | `similarity_search_by_vector_with_relevance_scores`      | 归一化分数 → 转换为 distance       |
| 3   | 无分数兜底              | `similarity_search_by_vector`                            | 默认值 1.0（不触发 Bypass）        |

### 修复 2：`top1_score` 默认值 0.0 → 1.0

| 维度    | 旧实现                     | 新实现                      |
| ----- | ----------------------- | ------------------------ |
| 默认值   | `0.0`                   | `1.0`                    |
| 异常时返回 | `([], 0.0)` → 触发 Bypass | `([], 1.0)` → 不触发 Bypass |

### 修复 3：幻觉检测使用过滤前文档 + 口语虚词过滤

- 新增 `all_retrieved_docs` state 字段，保存知识检索节点的完整结果
- 幻觉检测优先读 `all_retrieved_docs`
- `has_query_overlap` 增加 `ORAL_FILLERS` 过滤"怎么办/怎么样/好不好"等口语虚词
- 过滤后只需 1 个实质关键词命中即判定相关

### 修复后 Dense 检索实际排名（查询"头痛怎么办？"）

| 排名 | distance | 文档         | Reranker 执行             |
| -- | -------- | ---------- | ----------------------- |
| 1  | 0.4212   | 皮肤疾病诊疗指南   | ✅ 正常执行                  |
| 5  | 0.4646   | 神经系统症状鉴别指南 | ✅ Reranker 可将其提升至 Top-3 |

### 修改文件

| 文件                            | 改动                                                       |
| ----------------------------- | -------------------------------------------------------- |
| `app/rag/hybrid_retriever.py` | `_dense_search()` 重写：三层回退策略获取真实 distance；默认值 0.0→1.0     |
| `app/graph/state.py`          | 新增 `all_retrieved_docs` 字段                               |
| `app/graph/nodes/nodes.py`    | 知识检索节点保存 `all_retrieved_docs`；`has_query_overlap` 口语虚词过滤 |
| `app/graph/streaming.py`      | 幻觉检测优先读 `all_retrieved_docs`                             |

### 验证方式

```bash
D:\Agent\software\envs\my_medical_env\python.exe scripts\diagnose_dense.py
```

***

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

| 维度     | 旧实现                                               | 新实现                                                            |
| ------ | ------------------------------------------------- | -------------------------------------------------------------- |
| 超时参数   | `request_timeout=10`                              | `httpx.Timeout(connect=5, read=10, write=10, pool=5)`          |
| 实际生效   | ❌ 新版 openai>=1.0 已弃用 `request_timeout`，回退到默认 600s | ✅ 显式 `httpx.Timeout` 强制各阶段超时                                   |
| 60s 卡顿 | API 响应慢时阻塞 60 秒无超时                                | 最长 10s 超时 + 1 次重试 = 最多 20s                                     |
| 兼容性    | 无                                                 | 旧版 langchain-openai 不支持 `timeout` 参数时自动回退 `request_timeout=15` |

**改动**：[embeddings.py](file:///d:/Agent/medical_assistant_agent/app/core/embeddings.py)

### 修复 2：High-Confidence Bypass 误排除完美匹配（distance=0.0）

| 维度           | 旧实现                                                | 新实现                            |
| ------------ | -------------------------------------------------- | ------------------------------ |
| 条件           | `top1_dense_score > 0 and top1_dense_score < 0.08` | `0 <= top1_dense_score < 0.08` |
| distance=0.0 | ❌ `0.0 > 0` = False，不触发跳过                          | ✅ `0 <= 0.0` = True，正确跳过       |
| 后果           | 完美匹配仍跑 Reranker（浪费 2s）                             | 完美匹配直接跳过 Reranker              |

**根因**：ChromaDB cosine distance 中 `0.0` 是完美匹配（两向量完全一致），但旧代码 `> 0` 把它排除了。正确逻辑：`distance ∈ [0, 0.08)` 都是高置信度，应跳过 Reranker。

**改动**：[hybrid\_retriever.py](file:///d:/Agent/medical_assistant_agent/app/rag/hybrid_retriever.py#L276)

### 修复后预期

| 指标               | 修复前        | 修复后                                          |
| ---------------- | ---------- | -------------------------------------------- |
| Embedding API 超时 | 60s+ (无超时) | ≤10s (强制超时)                                  |
| 完美匹配时 Reranker   | 仍执行 (2s)   | 跳过 (0ms)                                     |
| 理想 TTFT          | 62s        | ≤3s (API正常时 \~500ms embedding + 跳过 reranker) |

***

## v8.0 - Skills 体系扩展 + Bad Case 回归测试闭环 + RAGAS 评估重写

### 背景

项目已具备基础的医疗安全审查 Skill、Bad Case 自动采集和 RAGAS 评估模块，但三个方向均存在明显短板：

- **Skills**：仅有 `medical_safety_review` 1 个 Skill，用药指导和症状分诊等高频场景无覆盖
- **Bad Case**：自动采集（4 触发点）+ 导出脚本 + 人工审核 API 已有，但缺回归测试运行器，无法验证修复效果
- **RAGAS 评估**：基础代码存在但硬编码 `gpt-4o`、兼容层冗余、无增量评估、无版本对比、未接入 bad case 数据

### 改动 1：新增 2 个结构化 Skill（Anthropic 范式）

#### 1.1 用药指导 Skill（`medication_guide`）

| 维度       | 说明                                                 |
| -------- | -------------------------------------------------- |
| Trigger  | 用户问题涉及药物用法用量、相互作用、禁忌人群时触发                          |
| Workflow | 药物实体识别 → 禁忌人群交叉检查 → 相互作用初筛 → 用量安全验证 → 规范性校验        |
| 输出       | `{status: pass/revise, revised_answer, risk_tags}` |

规则引擎覆盖：

- **药物实体识别**：AC 自动机匹配 + 正则兜底（剂型后缀"XX片""XX胶囊"）
- **禁忌人群交叉检查**：6 类人群（孕妇/儿童/老年/肝肾功能不全/消化道溃疡）与 4 种常见药物安全规则交叉
- **药物相互作用**：布洛芬↔阿司匹林/华法林等已知高风险组合
- **用量安全**：检测回答中剂量是否超过每日上限（如布洛芬 1200mg）
- **5 字段完整性**：适应症/用法用量/注意事项/禁忌/如症状持续请就医

#### 1.2 症状分诊 Skill（`symptom_triage`）

| 维度       | 说明                                                             |
| -------- | -------------------------------------------------------------- |
| Trigger  | 路由为 `symptom_analysis` 时触发                                     |
| Workflow | 紧急度分级 → 危险症状组合检测 → 持续时间评估 → 分诊建议生成                             |
| 输出       | `{status: pass/revise, triage_result, advice_text, risk_tags}` |

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

| 维度         | 旧方案                   | 新方案                                    |
| ---------- | --------------------- | -------------------------------------- |
| 采集         | ✅ 4 触发点自动采集           | ✅ 不变                                   |
| 导出         | ✅ JSONL 导出脚本          | ✅ 不变                                   |
| 人工审核       | ✅ API + PostgresStore | ✅ 不变                                   |
| **回归测试**   | ❌ 无                   | ✅ `BadCaseRegressionRunner`            |
| **统计报告**   | ❌ 无                   | ✅ 按类型/通过率/失败分布                         |
| **CLI 运行** | ❌ 无                   | ✅ `scripts/run_bad_case_regression.py` |

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

| 维度          | 旧方案                | 新方案                     |
| ----------- | ------------------ | ----------------------- |
| 评估 LLM      | 硬编码 `gpt-4o`       | 项目配置 `get_llm()`        |
| RAGAS 兼容    | 逐指标 try/except 冗长链 | 顶层一次性导入，仅支持 `>=0.1`     |
| 降级策略        | 无（报错退出）            | 规则引擎简易评估（关键词覆盖率）        |
| 增量评估        | ❌ 每次全量重跑           | ✅ 按 question 去重跳过已评估    |
| 版本对比        | ❌ 无                | ✅ A/B 对比 + delta + 改进方向 |
| 测试集         | 内嵌 5 条             | JSONL 文件 10 条（4 类场景）    |
| Bad Case 接入 | ❌ 不支持              | ✅ 自动识别 bad case 格式      |

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

| 文件                                      | 说明                          |
| --------------------------------------- | --------------------------- |
| `app/skills/medication_guide.md`        | 用药指导 Skill 定义（Anthropic 范式） |
| `app/skills/medication_guide_engine.py` | 用药指导规则引擎                    |
| `app/skills/symptom_triage.md`          | 症状分诊 Skill 定义（Anthropic 范式） |
| `app/skills/symptom_triage_engine.py`   | 症状分诊规则引擎                    |
| `app/evaluation/__init__.py`            | 评估模块入口                      |
| `app/evaluation/bad_case_runner.py`     | Bad Case 回归测试运行器            |
| `scripts/run_bad_case_regression.py`    | Bad Case 回归测试 CLI           |
| `tests/data/rag_eval_test_set.jsonl`    | RAGAS 评估测试集（10 条/4 类场景）     |

### 修改文件

| 文件                        | 改动                                    |
| ------------------------- | ------------------------------------- |
| `app/skills/__init__.py`  | 新增用药指导 + 症状分诊导出                       |
| `app/rag/evaluation.py`   | 重写：项目 LLM + 增量评估 + 版本对比 + bad case 接入 |
| `scripts/evaluate_rag.py` | 重写：新参数 + 版本对比 CLI                     |

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

***

## v7.0 - 系统性代码质量审计：P0/P1 缺陷修复 + AC 自动机引擎

### 背景

对项目所有模块进行了系统性审查，识别出 30 个"朴素实现可用更优算法替代"的问题（类似"300 字符截断"模式）。按优先级从 P0 开始修复，本次修复 4 项。

### 修复 1：warnings 字段覆盖→累积（P0 Bug）

| 维度  | 旧实现                                                        | 新实现                                   |
| --- | ---------------------------------------------------------- | ------------------------------------- |
| 声明  | `warnings: List[str]  # 覆盖警告信息`                            | `warnings: Annotated[List[str], add]` |
| 行为  | 后序节点的 warnings 覆盖前序节点                                      | 多个节点的 warnings 自动合并累积                 |
| Bug | `knowledge_retrieval_node` 的检索警告被 `safety_check_node` 覆盖丢失 | 所有节点的 warnings 都保留                    |

**改动**：[state.py](file:///d:/Agent/medical_assistant_agent/app/graph/state.py#L65) 1 行

### 修复 2：onset\_dates 合并语义错误→取最早 ts（P0 Bug）

| 维度   | 旧实现                                         | 新实现                             |
| ---- | ------------------------------------------- | ------------------------------- |
| 合并方式 | `{**a, **b}` 简单覆盖                           | `_merge_onset_dates(a, b)` 深度合并 |
| 同一症状 | 后值覆盖前值（L2 覆盖 L1）                            | 取 `ts` 更小（更早首发）的记录              |
| Bug  | L1 记录"头痛首次出现在3天前"，L2 记录"头痛出现在1天前"→ 1天前覆盖3天前 | 保留3天前（更早的首发时间）                  |

**改动**：[nodes.py](file:///d:/Agent/medical_assistant_agent/app/graph/nodes/nodes.py#L2074-L2104) 新增 `_merge_onset_dates()` 函数，替换 2 处简单字典合并

### 修复 3：关键词匹配→AC 自动机（P1 性能+精度）

| 维度    | 旧实现                                                 | 新实现                            |
| ----- | --------------------------------------------------- | ------------------------------ |
| 算法    | `any(keyword in text for keyword in keywords)` 线性扫描 | Aho-Corasick 自动机 O(m) 一次扫描     |
| 复杂度   | O(n×m)（n=关键词数，m=文本长度）                               | O(m)（与关键词数无关）                  |
| 误匹配   | "心疼"命中"疼"→误判为症状                                     | AC 自动机 + 边界检测消除子串误匹配           |
| 关键词维护 | 5 个文件各自维护一份列表                                       | 集中式 `keyword_matcher.py`，单一真相源 |
| 降级策略  | 无                                                   | pyahocorasick 未安装时自动降级为线性扫描    |

**受影响的模块**：

| 模块                           | 旧方式                                 | 新方式                                                             |
| ---------------------------- | ----------------------------------- | --------------------------------------------------------------- |
| `detect_rule_based_route`    | 2 个关键词列表 × 线性扫描                     | `get_route_symptom_matcher()` / `get_route_knowledge_matcher()` |
| `_extract_symptoms_by_rules` | 70+ 条 `symptom_map` × 线性遍历          | `get_symptom_matcher().get_matched_keywords()`                  |
| `_build_rewrite_context`     | 2 个内联关键词集合 × 线性扫描                   | `get_drug_matcher()` / `get_symptom_matcher()`                  |
| `_check_emergency_risks`     | 双向子串 `emerg in sym or sym in emerg` | `get_emergency_matcher().contains_any()`                        |

**新增文件**：[keyword\_matcher.py](file:///d:/Agent/medical_assistant_agent/app/core/keyword_matcher.py) — AC 自动机引擎 + 5 个集中式匹配器构建器

**新增依赖**：`pyahocorasick`（纯 C 实现，<1MB，自动降级）

### 修复 4：语义缓存伪 LRU→真 LRU（P1 正确性）

| 维度     | 旧实现                              | 新实现                             |
| ------ | -------------------------------- | ------------------------------- |
| 数据结构   | Redis Set（无序）                    | Redis Sorted Set（score=访问时间戳）   |
| 淘汰策略   | `list(all_keys)[:20%]` → 本质是随机淘汰 | `ZRANGE` 按 score 升序 → 淘汰最久未访问的  |
| 访问刷新   | 无（命中不更新顺序）                       | 命中时 `ZADD` 更新 score → 最近访问的排在后面 |
| 与注释一致性 | 注释说"LRU"但实际是随机淘汰 ❌               | 真正的 LRU 行为 ✅                    |

**改动**：[semantic\_cache.py](file:///d:/Agent/medical_assistant_agent/app/cache/semantic_cache.py)

- 新增 `_keys_zset` Sorted Set 索引
- `set()` 写入时 `ZADD {key: timestamp}`
- `get()` 命中时 `ZADD` 刷新时间戳（读时刷新）
- 淘汰时 `ZRANGE` 按 score 升序取最早的 20%
- `clear()` 同时清理 ZSET

### 修改文件清单

| 文件                                   | 改动类型                                   |
| ------------------------------------ | -------------------------------------- |
| `app/graph/state.py`                 | Bug 修复：warnings 覆盖→累积                  |
| `app/graph/nodes/nodes.py`           | Bug 修复：onset\_dates 深度合并；AC 自动机集成（4 处） |
| `app/core/keyword_matcher.py`        | 新增：AC 自动机引擎 + 5 个集中式匹配器                |
| `app/skills/safety_review_engine.py` | 优化：紧急症状检测使用 AC 自动机                     |
| `app/cache/semantic_cache.py`        | 修复：伪 LRU→真 LRU（Sorted Set + 读时刷新）      |

***

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

| 文件                              | 说明                                                |
| ------------------------------- | ------------------------------------------------- |
| `app/rag/parent_child_store.py` | ParentChildManager：父子索引管理器（InMemoryStore + 磁盘持久化） |

### 修改文件

| 文件                                | 改动                                                                               |
| --------------------------------- | -------------------------------------------------------------------------------- |
| `app/rag/hybrid_retriever.py`     | Reranker 后增加 `parent_manager.get_parents()` 映射；BM25 缓存兼容检测（无 parent\_id 视为旧缓存重建） |
| `app/graph/nodes/nodes.py`        | `build_rag_prompt` 移除 300 字符截断，仅对 >2000 字符做安全兜底                                  |
| `app/rag/__init__.py`             | 导出 ParentChildManager                                                            |
| `scripts/rebuild_vector_store.py` | 重建脚本支持父子索引：Parent 切分 → Child 入库 → Parent 持久化                                     |

### 技术栈对比

| 维度          | 旧方案                    | 新方案                            |
| ----------- | ---------------------- | ------------------------------ |
| 索引粒度        | 单层（大 chunk \~400 字符）   | 双层（Parent \~400 + Child \~150） |
| 检索精度        | 大 chunk 语义模糊           | Child 小块精准匹配                   |
| 上下文完整性      | 截断到 300 字符             | 完整 Parent 注入                   |
| Reranker 性能 | 5 篇 × 300 字符 = \~969ms | 5 篇 × \~80 字符 = 预期 \~400ms     |
| 信息丢失风险      | 高（第 301+ 字符丢失）         | 无（Parent 完整保留）                 |

### 使用方式

```bash
# 重建向量库（自动构建父子索引）
python scripts/rebuild_vector_store.py
```

### 兼容性

- 未重建索引时，系统自动降级为旧模式（child 文档无 parent\_id → 跳过 parent 映射）
- BM25 缓存自动检测旧版数据并重建

***

## v6.1 - 性能修复：Dense 检索 759ms → \~100ms（三个根因修复）

### 背景

日志分析发现 Dense 检索耗时 759ms（244 篇文档的向量库，正常应 <100ms）。排查发现三个叠加问题导致 Chroma 被重复实例化、Embedding API 被重复调用。

### 三个根因与修复

| 根因               | 文件                    | 问题                                                                                                                                                        | 修复                                                                                    |
| ---------------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Chroma 重复实例化     | `vector_store.py`     | `create_vector_store()` 每次调用都新建 Chroma 实例，即使已存在                                                                                                           | 增加 `if self.vector_store is not None and not force_rebuild: return self.vector_store` |
| lru\_cache 参数不匹配 | `routes.py`           | 启动预热 `get_hybrid_retriever()`（k=5, rerank\_top\_k=10）与搜索节点 `get_hybrid_retriever(k=3, rerank_top_k=5)` 参数不同 → 缓存未命中 → 新建 HybridRetriever → 触发 Chroma 重新加载 | 启动预热改为 `get_hybrid_retriever(k=3, alpha=0.5, use_reranker=True, rerank_top_k=5)`      |
| Embedding 未预计算   | `hybrid_retriever.py` | `elif query_embedding is None` 在 L2 缓存开启时永远不会执行（`if` 条件已为 True）→ `query_embedding` 为 None → Chroma 内部调 Embedding API（\~200-300ms）                         | `elif` 改为 `if`，L2 缓存为空时仍计算 embedding 供 Dense 复用                                       |

### 修复前后对比

| 指标               | 修复前              | 修复后（预期）            |
| ---------------- | ---------------- | ------------------ |
| Dense 检索耗时       | 759ms            | \~100ms            |
| Chroma 实例化       | 每次请求重新加载         | 启动时加载一次，后续复用       |
| Embedding API 调用 | Chroma 内部调用（不可控） | 预计算后传入 Chroma（可复用） |
| BM25 缓存加载        | 每次请求从磁盘重新加载      | lru\_cache 命中后跳过   |

### 修改文件

| 文件                            | 修改内容                                                          |
| ----------------------------- | ------------------------------------------------------------- |
| `app/rag/vector_store.py`     | `create_vector_store()` 增加实例缓存判断                              |
| `app/api/routes.py`           | 启动预热参数与搜索节点一致（k=3, rerank\_top\_k=5）                          |
| `app/rag/hybrid_retriever.py` | `elif query_embedding is None` → `if query_embedding is None` |

***

## v6.0 - Skill 增强：医疗合规与安全审查（结构化 Prompt 范式）

### 背景

项目原有的 `safety_check_node` 仅调用 LLM 做简单风险评估，追加 warnings，不修改回答内容。在医疗场景中，大模型即使拿到正确文档，仍可能在生成阶段出现超适应症建议、诊断性断言或遗漏紧急就医指引。本次升级基于 Anthropic Skill 范式（结构化 Prompt），将安全审查从"附加警告"升级为"三态决策阀门"。

### Skill 定义

新增 [medical\_safety\_review.md](file:///d:/Agent/medical_assistant_agent/app/skills/medical_safety_review.md)，按 Anthropic Skill 范式定义五大模块：

- 🎯 Trigger：答案生成后、缓存写入前自动触发
- ⚙️ Workflow：5 步审查流程（诊断断言→用药安全→紧急风险→免责声明→决策输出）
- 📤 Output：{status: pass|revise|block, revised\_answer, risk\_tags}
- 🛡️ Guardrails：禁止生成新医学建议、800ms 超时兜底、规则引擎优先

### 技术栈对比

| 维度    | 旧方案（v5.x）    | 新方案（v6.0）                               |
| ----- | ------------ | --------------------------------------- |
| 审查架构  | 仅 LLM 单步审查   | 规则引擎（0ms）+ LLM 深度审查（高风险时触发）             |
| 审查决策  | 仅追加 warnings | 三态决策：pass（透传）/ revise（修订）/ block（拦截）    |
| 诊断性断言 | 未检测          | 10 类正则模式检测 + 自动替换为风险提示句式                |
| 紧急风险  | 未关联快照        | 交叉检查 clinical\_checkpoint 中的危急重症信号      |
| 免责声明  | 固定追加         | 检测缺失后自动注入                               |
| 流式集成  | 未集成          | 流式结束后执行审查，修订时发送 safety\_revision SSE 事件 |
| 缓存保护  | 无            | block 的回答不写入 Redis，防止污染缓存               |

### 审查流程

```
答案生成 → [规则引擎 0ms] → 无风险 → pass → 缓存写入
                         ↓ 有风险
                    [LLM 深度审查] → revise → 修订后缓存 + 发送修正 SSE
                                  → block → 不缓存 + 返回安全拒答模板
```

### 新增文件

| 文件                                    | 说明                            |
| ------------------------------------- | ----------------------------- |
| `app/skills/__init__.py`              | Skills 模块入口                   |
| `app/skills/medical_safety_review.md` | 结构化 Skill 定义（Anthropic 范式）    |
| `app/skills/safety_review_engine.py`  | 规则引擎：诊断断言检测 + 紧急风险拦截 + 免责声明注入 |

### 修改文件

| 文件                         | 修改内容                                                                                     |
| -------------------------- | ---------------------------------------------------------------------------------------- |
| `app/graph/nodes/nodes.py` | `safety_check_node` 重写：规则引擎 → LLM 深度审查 → 三态决策                                            |
| `app/graph/streaming.py`   | 新增 `_run_safety_review()` 方法；所有答案路径（direct/vision/RAG/cached\_docs）流式结束后执行安全审查，block 不缓存 |

### 设计要点

1. **规则引擎优先**：0ms 正则检测诊断性断言、紧急症状、免责声明，仅高风险时才触发 LLM
2. **三态决策**：pass（透传）/ revise（自动替换诊断断言 + 注入紧急提示 + 补全免责声明）/ block（返回安全引导模板）
3. **流式安全修正**：流式输出已发送给用户后，如审查发现风险，发送 `safety_revision` SSE 事件推送修正
4. **缓存保护**：block 的回答不写入 Redis，避免错误回答被缓存后持续返回
5. **临床快照关联**：审查时读取 `clinical_checkpoint`，检测用户症状快照中的危急重症信号是否在回答中被遗漏

***

## v5.7 - 修复：LangGraph BaseStore.search() namespace\_prefix 位置参数兼容

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

***

## v5.6 - 查询重写切换本地模型 + 前端新会话按钮

### 1. 查询重写 LLM 从远端 API 切换到本地模型

`query_rewrite_node` 的重写和 HyDE 生成原来使用 `get_rewrite_llm()`（调用智譜 API，\~4.5s），
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
- 填入 thread\_id 输入框
- 清空对话界面，显示"新会话已创建"提示

与"清空对话"的区别：

- 清空对话：仅清 UI，thread\_id 不变 → checkpointer 历史继续累积
- 新会话：生成新 thread\_id → checkpointer 从零开始

### 修改文件

- `app/graph/nodes/nodes.py` — `query_rewrite_node` 和 HyDE 的 LLM 调用从 `get_rewrite_llm` 改为 `get_local_llm`
- `app/static/index.html` — 新增"新会话"按钮 + `newSession()` 函数

***

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

| 指标          | 修复前     | 修复后                |
| ----------- | ------- | ------------------ |
| 查询重写        | 4536ms  | **0ms**            |
| 重写引入幻觉症状    | ✅ 可能发生  | ❌ 杜绝               |
| TTFT        | 7269ms  | **\~2700ms**       |
| 智譜 API 调用次数 | 每请求 1 次 | 仅追问时 1 次（减少 \~70%） |

### 修改文件

- `app/graph/nodes/nodes.py` — `query_rewrite_node` 添加 `elif not _anaphora_detected` 短路分支

***

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

两层截断叠加后，Reranker 实际只看到文档的 **前 \~80 个中文字符**（约 16%），
如果头痛的治疗建议在后半段，模型根本看不到。

### 为什么 max\_length=128 对中文太短

| 维度            | 英文       | 中文（BERT tokenizer） |
| ------------- | -------- | ------------------ |
| 1 个 token 覆盖  | 0.75 个单词 | 0.5-0.7 个汉字        |
| 128 tokens 覆盖 | \~96 个单词 | **\~65-90 个汉字**    |
| 400 字文档覆盖率    | \~100%   | **\~20%**          |

中文在 BERT tokenizer 下每个汉字常被拆为 1-3 个 subword token，
128 token 窗口对英文够用但对中文严重不足。

### 修复

**1. tokenizer max\_length：128 → 512**

BGE-reranker 模型上限为 512 tokens，之前仅用了 25%。
512 tokens 可覆盖约 300 个中文字符 + query + 特殊 token。

**2. 截断策略：纯取头 → 头尾各取**

| 维度          | 修复前           | 修复后                        |
| ----------- | ------------- | -------------------------- |
| 策略          | `text[:400]`  | `text[:200] + text[-100:]` |
| 保留开头（诊断/主题） | ✅             | ✅                          |
| 保留结尾（治疗/药物） | ❌ 被丢弃         | ✅ 保留最后 100 字               |
| 文档有效覆盖率     | \~16%（80/500） | \~60%（300/500）             |

**3. MAX\_RERANK\_DOC\_CHARS：400 → 300**

300 字中文 ≈ 450-500 tokens，加上 query + 特殊 token 可稳定装入 512 窗口，
避免 BERT tokenizer 的二次截断。

### 修改文件

- `app/rag/reranker.py` — `max_length` 128→512；`truncate_for_rerank` 改为头尾各取；`MAX_RERANK_DOC_CHARS` 400→300

***

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

| 模型                      | 训练目标           | 输出    |
| ----------------------- | -------------- | ----- |
| OpenAI text-embedding-3 | 余弦相似度对比学习      | 自动归一化 |
| 智谱 embedding-3          | 余弦相似度对比学习      | 自动归一化 |
| BGE / M3E 系列            | InfoNCE + 余弦损失 | 自动归一化 |

所有主流 Embedding 模型的训练目标都是**最小化正样本对的余弦距离、最大化负样本对的余弦距离**。归一化后的向量落在单位超球面上，此时 L2² = 2×(1−cos\_sim)，两种度量等价——但前提是**索引和查询使用同一种距离**。ChromaDB 默认 L2，而模型为余弦优化，这正是错配的根源。

**一句话总结：语义存在于方向中，不在长度中。余弦相似度度量的是"两个文本在说什么"，L2 度量的是"两个向量有多长"。RAG 需要前者。**

### 三个叠加问题

**1. ChromaDB 距离度量错误（主因）**

| 维度                          | 修复前                   | 修复后                    |
| --------------------------- | --------------------- | ---------------------- |
| ChromaDB 空间                 | `l2`（默认）              | `cosine`               |
| 距离 0.9333 的含义               | L2² 距离 ≈ 余弦相似度 0.53   | cosine 距离 ≈ 余弦相似度 0.88 |
| HIGH\_CONFIDENCE\_THRESHOLD | 0.08（注释写余弦但实际对 L2 无效） | 0.08（与 cosine 距离正确对应）  |

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

***

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
- 反馈以 `user_negative_feedback` 类型写入 bad\_cases
- 后端新增 `POST /api/feedback` 端点

**2. 正例测试集（18 条自包含查询）**

- 新增 `bc_pos_001` \~ `bc_pos_018`，均为自包含的医疗查询
  （如"布洛芬的副作用是什么？""头痛怎么缓解？"）
- 测试集从 20 条扩至 38 条：负例 20 + 正例 18

**3. 修复** **`_has_anaphora_pattern`** **误杀**

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

***

## v5.1 - Bad Case 采集面扩展（3 个新采集点）

### 问题

Bad case 只覆盖了查询重写环节（3 个采集点），以下关键失败模式完全未被采集：

- LLM 在答案中编造检索文档不存在的药物（幻觉）
- 检索返回零文档（索引/查询词匹配问题）
- 含症状词的问题被错误路由到 direct\_answer

### 新增采集点

| 采集点  | 触发条件                      | case\_type                | 位置                               |
| ---- | ------------------------- | ------------------------- | -------------------------------- |
| 幻觉检测 | 答案含药物名但检索文档中未出现           | `hallucination_suspected` | `streaming.py` RAG 答案生成后         |
| 检索失败 | RAG 管道返回零文档               | `retrieval_miss`          | `streaming.py` 零文档分支             |
| 路由异常 | 问题含症状词但路由到 direct\_answer | `route_misclassification` | `streaming.py` direct\_answer 分支 |

### 配套更新

- `long_term_memory.py`：`append_bad_case` docstring 补充 5 个新 case\_type
- `scripts/export_bad_cases.py`：`--case-type` 选项新增所有类型

### 修改文件

- `app/graph/streaming.py` — 新增 `_check_hallucination`、`_record_retrieval_miss`、`_check_route_misclassification` 三个方法；在 `run()` 的关键路径插入调用
- `app/memory/long_term_memory.py` — 更新 case\_type 文档
- `scripts/export_bad_cases.py` — 扩展 `--case-type` 选项

***

## v5.0 - 自包含性检测 + Bad Case 采集 + 低分澄清

### 问题

"语法完整性陷阱"：查询"还有其他什么可以吃的吗？"语法完美但语义残缺，
缺少核心实体（头痛/缓解药物），传统基于"查询质量/长度/语法"的静态规则完全失效。
低分检索结果直接进入 LLM 自由生成，产生幻觉回答。

### 修复

**1. 自包含性前置检测（方案A P0）**

| 维度      | 修复前 | 修复后                              |
| ------- | --- | -------------------------------- |
| 指代词检测   | ❌ 无 | ✅ 15个指代词黑名单（其他/还有/这个/那个/它/呢/...） |
| 极短查询    | ❌ 无 | ✅ <15字 + 有历史 → 强制重写              |
| 疑问词+缺实体 | ❌ 无 | ✅ 以"怎么/如何/什么/哪些"开头但缺少领域实体 → 强制重写 |

三层检测逻辑：`_has_anaphora_pattern(query)` → 误杀代价远小于漏改导致的幻觉

**2. Bad Case 自动采集**

| 采集点 | 触发条件               | case\_type                 |
| --- | ------------------ | -------------------------- |
| 重写后 | 指代词检测命中但重写结果与原问题一致 | `rewrite_same_as_original` |
| 重写后 | 指代词检测命中但重写后仍缺领域实体  | `rewrite_missed_anaphora`  |
| 低分时 | 检索低分但未触发澄清         | `low_score_no_clarify`     |

存储：PostgresStore `("bad_cases", user_id)` 命名空间，支持人工审核补填 `expected_rewrite` 和 `is_self_contained`

**3. 低分澄清机制（消除幻觉出口）**

| 场景    | 修复前                | 修复后                |
| ----- | ------------------ | ------------------ |
| 无检索文档 | 降级为 LLM 自由生成（幻觉风险） | 返回结构化澄清追问          |
| 低分检索  | 直接生成兜底答案           | 记录 bad case + 澄清追问 |

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

***

## v4.4 - 修复：`_build_rewrite_context` 截断导致药物名丢失

### 问题

`_build_rewrite_context` 对 AI 回复做头尾截断（保留前 2/3 + 后 1/3）时，
LLM 推荐的具体药品名称可能出现在回复的**中间部分**（如药理说明段落），截断后丢失。
后续用户追问"还有什么药可以吃？"时，重写提示词中看不到第一次推荐的药物名，
只能依赖用户问题中残留的关键词，造成上下文断层。

### 修复

**截断前全文扫描提取医疗实体**：

1. AI 回复**截断前**，先扫描全文匹配药物关键词（与 `_DRUG_KEYWORDS` 对齐，\~45个）和症状关键词（\~30个）
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

***

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

**3. 重写提示词精简**（3577ms → \~1500ms）

- `query_rewrite_node`：Prompt 从 \~1500 字缩减到 \~300 字
- 移除冗长规则说明和重复示例，保留核心输出格式
- 减少 token 数 → 降低 LLM 首 token 延迟

### 预期收益

| 阶段       | 优化前        | 优化后                 |
| -------- | ---------- | ------------------- |
| 症状解析     | 2871ms     | 0ms (短路)            |
| L2 语义缓存  | 397ms      | 0ms (跳过)            |
| 查询重写     | 3577ms     | \~1500ms (短 prompt) |
| 知识检索     | 1675ms     | 1675ms (不变)         |
| 答案 LLM   | 817ms      | 817ms (不变)          |
| **TTFT** | **9341ms** | **\~4000ms**        |

### 修改文件

- `app/graph/nodes/nodes.py` — `symptom_analysis_node` 追问短路；`query_rewrite_node` prompt 精简
- `app/graph/streaming.py` — `run()` 路由优先 → 按类型缓存；新增 `_check_l0_cache()`

***

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

***

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

***

## v3.9 - 紧急修复：路由结果被丢弃导致 RAG 全部跳过

### 问题

`streaming.py` 的 `run()` 方法中，路由和缓存并行执行后，
`asyncio.gather(_check(), self._run_route_sync())` 的返回值未被接收。
`_run_route_sync()` 返回的 `Command`（含 `goto=symptom_analysis` 等路由目标）
被丢弃，导致 `route_command` 始终为 `None`。

下游判断逻辑 `route_command or "direct_answer"` 永远命中默认值，
**所有无缓存请求都走** **`direct_answer`，RAG 管道被完全绕过。**

### 修复

- `app/graph/streaming.py` — `_, route_command = await asyncio.gather(...)` 接收路由结果

***

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

***

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

***

## v3.6 - 质量保障：核心节点单元测试

### 优化背景

项目缺少单元测试。LangGraph 节点的纯函数特性非常适合单元测试，
但没有覆盖时，重构和回归都缺乏安全网。

### 修改内容

**新增文件**：

- `tests/__init__.py`
- `tests/conftest.py` — pytest 配置和共享 fixtures（mock\_llm, base\_state 等）
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

***

## v3.5 - 可靠性优化：自纠正循环振荡检测

### 优化背景

`grade_documents_node` 在检索结果不相关时触发自纠正（重写→检索→评分），上限 2 次。
但如果 Reranker 分数刚好在阈值附近反复横跳，重试不会改善结果，反而浪费 3-5s。

### 修改内容

**修改文件**：

- `app/graph/nodes/nodes.py` — `grade_documents_node()` 增加振荡检测：
  - 重试前记录 `_prev_max_score` 和 `_prev_relevant_count` 到状态
  - 重试后检测：score\_delta < 0.05 且 doc\_delta < 1 → 无改善，跳过二次重试
  - 无检索文档且前次有重试历史时同样检测
- `app/graph/streaming.py` — `_run_rag_pipeline()` 重试时递增 `retrieval_attempts`

**收益**：

- 避免无效重试，节省 3-5s 的无关等待
- 日志明确记录每次重试前后的分数变化

***

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

***

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
- 1000 条缓存时查找耗时从 \~50ms 降到 \~5ms

***

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

- nodes.py 从 2619 行缩减到 2382 行，移除了 \~240 行已提取的代码
- helpers.py 和 models.py 可独立导入和测试，无需加载整个节点模块
- 外部代码通过 `from app.graph.nodes import router_node` 继续工作，零破坏性变更

***

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

- routes.py 代码量减少 \~40%（845 → 500 行）
- 消除 routes.py 和 graph.py 的双维护问题——编排逻辑现在是 graph 定义的唯一消费者
- `validate_streaming_sync()` 仍作为安全网在启动时自动检测一致性

***

## v3.0 - 参考蚂蚁阿福方案的功能增强

### 增强背景

参考蚂蚁阿福（支付宝医疗AI）的技术方案，从功能维度增强医疗助手：

| 能力   | 蚂蚁阿福      | 优化前本项目 | 优化后本项目          |
| ---- | --------- | ------ | --------------- |
| 主动追问 | 多轮追问补全信息  | 无      | 症状模糊时追加追问引导     |
| 图片问诊 | OCR+VLM混合 | 不支持    | VLM直接理解图片       |
| 循证标注 | 证据等级A/B/C | 无      | RAG回答标注来源和证据等级  |
| 安全拒答 | 超范围问题拒答   | 无      | LLM路由新增refuse类型 |

***

### 增强项 1：主动追问机制

**参考**：蚂蚁阿福的"模拟真人医生问诊逻辑"，当用户描述模糊时主动追问补全关键信息。

**方案**：新增 `_build_followup_hints()` 函数，根据症状提取结果检测缺失字段（部位、持续时间、严重程度），在 RAG prompt 末尾追加追问引导。

**修改文件**：

- `app/graph/nodes.py` - 新增 `_build_followup_hints()` 函数
- `app/graph/nodes.py` - `build_rag_prompt()` 新增 `symptoms` 参数和追问逻辑

**效果示例**：

- 用户："我肚子不舒服"
- 回答末尾追加："💡 为了更准确地帮助您，您可以补充以下信息：具体部位（如：头部、腹部、四肢等）、持续时间（如：3天、1周等）、严重程度（如：轻微、中等、剧烈）。"

***

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

***

### 增强项 3：循证医学标注

**参考**：蚂蚁阿福的回答标注证据等级（A级=随机对照试验，B级=学会共识，C级=临床经验）。

**方案**：在 `build_rag_prompt` 的回答要求中添加循证标注指令：

- `[来源：文档N]` — 标注建议来源
- `[证据等级：A/B/C]` — 标注证据可信度

**修改文件**：

- `app/graph/nodes.py` - `build_rag_prompt()` 添加循证标注要求

**效果示例**：

> 建议多饮水、注意休息 \[来源：文档1] \[证据等级：B]

***

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

***

### 增强项 5：症状解析快速模型

**方案**：规则未命中时调用 `glm-4-flash`（智谱最快模型）替代主模型 `glm-4.5-air`。

**修改文件**：

- `app/core/config.py` - 新增 `SYMPTOM_MODEL_NAME: str = "glm-4-flash"`
- `app/core/llm.py` - 新增 `get_symptom_llm()` 函数
- `app/graph/nodes.py` - `symptom_analysis_node()` 规则未命中时调用 `get_symptom_llm()`

**效果**：规则未命中时从 \~13秒 降至 **\~2-3秒**。

***

### 修改文件汇总

| 文件                   | 修改类型  | 说明                                                                                                                                                         |
| -------------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `app/core/config.py` | 新增    | `VISION_MODEL_NAME`、`SYMPTOM_MODEL_NAME`                                                                                                                   |
| `app/core/llm.py`    | 新增    | `get_vision_llm()`、`get_symptom_llm()`                                                                                                                     |
| `app/graph/state.py` | 新增    | `image_base64` 字段                                                                                                                                          |
| `app/graph/nodes.py` | 新增+修改 | `_build_followup_hints`、`vision_analysis_node`、`stream_vision_answer`、`build_rag_prompt`（追问+循证）、`router_node`（vision+refuse）、`symptom_analysis_node`（快速模型） |
| `app/graph/graph.py` | 新增    | `vision_analysis` 节点和边                                                                                                                                     |
| `app/api/routes.py`  | 新增+修改 | `ChatRequest.image_base64`、`event_generator` vision分支                                                                                                      |

***

### 后续优化方向

1. **OCR结构化方案**：体检报告场景，OCR提取指标 → 知识图谱校验 → LLM解读（参考阿福的报告解读架构）
2. **药品知识图谱**：药盒识别 → 国药准字匹配 → 禁忌/相互作用检查
3. **多轮追问**：当前为单次追问，后续可改为多轮对话式追问
4. **并行检索**：Dense + Sparse 检索改为并行执行
5. **全节点异步化**：所有节点改为 async，支持并发请求

***

## v4.0 - RAG Pipeline 深度优化 + 三层记忆架构

### 更新背景

针对首 token 响应时间（TTFT）过长（9s+）、3B 小模型结构化输出不稳定、跨会话记忆丢失三大核心问题，进行 RAG Pipeline 深度优化和三层记忆架构重构。

### 核心指标变化

| 指标            | 优化前   | 优化后           | 降幅       |
| ------------- | ----- | ------------- | -------- |
| TTFT（明确查询）    | \~9s  | \~3-4s        | **56%**  |
| TTFT（追问查询）    | \~9s  | \~4-5s        | **44%**  |
| 3B模型 JSON 合法率 | \~60% | \~95%         | **+35%** |
| 跨会话症状记忆       | ❌ 丢失  | ✅ L1持久化       | 新增       |
| Reranker 跳过率  | 0%    | \~40%（高置信度查询） | 新增       |

***

### 更新项 1：3B 小模型结构化输出（方案1+3+4组合）

**问题**：Qwen2.5:3b 输出 JSON 格式不稳定，`"symptoms": "膝盖摔伤"`（字符串而非列表）、缺少引号、多余逗号等。

**原方案**：`get_local_llm()` + `with_structured_output()`（3B 模型不支持 function calling）

**新方案**：三层防线

| 层级 | 方案        | 技术                                                                  | 兜底场景            |
| -- | --------- | ------------------------------------------------------------------- | --------------- |
| L1 | JSON Mode | `response_format={"type": "json_object"}`                           | 采样层提高 JSON 字符权重 |
| L2 | 鲁棒解析      | `extract_json_block`（json.loads → json\_repair → ast.literal\_eval） | 修复单引号、多余逗号、缺少引号 |
| L3 | 分隔符降级     | `parse_symptom_text`（`症状：xxx\n部位：xxx`）                              | JSON 完全无法解析时    |

**修改文件**：

- `app/core/llm.py` — 新增 `get_local_llm_json()`（JSON Mode）
- `app/graph/nodes.py` — `extract_json_block` 三层解析 + `_coerce_list_fields` 自动修复 + `parse_symptom_text` 分隔符降级
- `app/graph/nodes.py` — 症状解析/快照更新/档案提取/安全检查节点均改用 `get_local_llm_json()` + `invoke_json_once_with_fallback()`

**依赖新增**：`json_repair~=0.60.1`

***

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

- `app/graph/nodes.py` — 删除 `_should_rewrite_query`（\~80行）、`_detect_current_question_route`（\~25行）
- `app/graph/nodes.py` — 新增 `_build_rewrite_context`、`_rewrite_guard_check`
- `app/graph/nodes.py` — `query_rewrite_node` 重写为 LLM 判断模式

**删除的硬编码规则**：

| 删除的函数                            | 行数    | 问题            |
| -------------------------------- | ----- | ------------- |
| `_should_rewrite_query`          | \~80行 | 8层规则，字数/特征词陷阱 |
| `_detect_current_question_route` | \~25行 | 药物名+意图词硬编码    |

***

### 更新项 3：模型切换（本地3B → glm-4-flash API）

**问题**：RTX 3050 Laptop 4GB VRAM，3B模型 GPU/CPU 混合推理，查询重写耗时 2.7-3.4s。

**原方案**：

| 节点    | 模型             | 耗时         |
| ----- | -------------- | ---------- |
| 查询重写  | Qwen2.5:3b（本地） | \~2.7-3.4s |
| HyDE  | Qwen2.5:3b（本地） | \~2-3s     |
| 症状解析  | Qwen2.5:3b（本地） | \~3s       |
| 快照/档案 | Qwen2.5:3b（本地） | \~2-3s     |

**新方案**：

| 节点    | 模型                          | 耗时                   |
| ----- | --------------------------- | -------------------- |
| 查询重写  | **glm-4-flash（API）**        | \~0.5-0.8s           |
| HyDE  | **glm-4-flash（API）**        | \~0.5-0.8s           |
| 症状解析  | Qwen2.5:1.5b（本地）+ 规则优先      | 0ms（规则命中）/ \~1s（LLM） |
| 快照/档案 | Qwen2.5:1.5b（本地）+ JSON Mode | \~1-1.5s             |

**修改文件**：

- `app/core/config.py` — `LOCAL_MODEL_NAME: "qwen2.5:1.5b"`
- `app/graph/nodes.py` — `query_rewrite_node` / HyDE 改用 `get_rewrite_llm()`（glm-4-flash）

**本地模型对比**：

| <br />  | qwen2.5:3b           | qwen2.5:1.5b       |
| ------- | -------------------- | ------------------ |
| 模型大小    | 1.9 GB               | 986 MB             |
| VRAM 需求 | \~2.5-3 GB（4GB卡装不下）  | \~1.2 GB（4GB卡纯GPU） |
| 推理速度    | 15-20 tokens/s（混合模式） | 50+ tokens/s（纯GPU） |

***

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

**回退的错误逻辑**：~~`candidate_count <= k * 2`~~ ~~→ 跳过重排~~（数量少不等于质量高，忽略 Lost in the Middle 和噪声过滤）

***

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

| 层级 | 工具           | 场景                | 示例              |
| -- | ------------ | ----------------- | --------------- |
| L1 | `dateparser` | 标准相对时间表达          | "前天"→2026-06-22 |
| L2 | 中文数字正则       | dateparser 不覆盖的中文 | "持续三天了"→3天      |
| L3 | 默认锚定         | 未提及任何时间           | "我现在头痛"→当前时刻    |

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
- `app/graph/nodes.py` — `ClinicalCheckpointOutput` 新增 `symptom_onset_dates` 字段（结构升级为 Dict\[str, Dict]）
- `requirements.txt` — 新增 `dateparser~=1.4.1`

***

### 更新项 6：三层记忆协同架构

**问题**：症状首发时间只存在 L2 快照（绑定 thread\_id），用户开新会话后完全失忆。

**原方案**：

| 信息类型      | L1 Profile | L2 Snapshot |
| --------- | ---------- | ----------- |
| 姓名/年龄/过敏史 | ✅          | ✅           |
| 症状首发时间    | ❌          | ✅（跨会话丢失）    |
| 用药记录      | ❌          | ✅（跨会话丢失）    |

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

| 场景   | 数据流                                                     | 结果        |
| ---- | ------------------------------------------------------- | --------- |
| 新会话  | L1.get\_all\_symptom\_onsets → L2.symptom\_onset\_dates | ✅ 跨会话记忆   |
| 当前会话 | 规则提取 → L2 → 快照更新时同步L1                                   | ✅ 双写保障    |
| 快照更新 | L2合并 → L1.append（保留最早记录）                                | ✅ 不覆盖更早记录 |
| 用药记录 | L2.medication\_history → L1.append\_medication\_event   | ✅ 跨会话可查   |

**修改文件**：

- `app/memory/long_term_memory.py` — 新增6个方法：
  - `append_symptom_event` / `get_symptom_events` / `get_latest_symptom_onset` / `get_all_symptom_onsets`
  - `append_medication_event` / `get_medication_events`
- `app/graph/nodes.py` — `memory_load_node` 新增 L1→L2 症状首发时间合并
- `app/graph/nodes.py` — `update_clinical_snapshot_node` 新增 L2→L1 异步同步（症状+用药）

**L1 新增命名空间**：

| 命名空间                          | 用途      | 数据格式                                                                              |
| ----------------------------- | ------- | --------------------------------------------------------------------------------- |
| `symptom_events/{user_id}`    | 症状报告事件流 | `{event_type, symptom, onset_iso, onset_ts, precision, source_query, created_at}` |
| `medication_events/{user_id}` | 用药记录事件流 | `{event_type, drug, dosage, effect, source_query, created_at}`                    |

***

### 更新项 7：L0 缓存日志可见性

**问题**：`has_profile=True` 时 L0 缓存完全跳过，日志中无任何 L0 相关记录。

**原方案**：有用户档案时静默跳过 L0

**新方案**：添加3条日志

- `L0答案缓存命中` / `L0答案缓存未命中` / `L0答案缓存跳过（用户有档案）`

**修改文件**：

- `app/api/routes.py` — L0 缓存检查处添加日志

***

### 修改文件汇总

| 文件                               | 修改类型 | 说明                                          |
| -------------------------------- | ---- | ------------------------------------------- |
| `app/core/llm.py`                | 新增   | `get_local_llm_json()`（JSON Mode）           |
| `app/core/config.py`             | 修改   | `LOCAL_MODEL_NAME: "qwen2.5:1.5b"`          |
| `app/graph/nodes.py`             | 重构   | 查询重写（LLM判断）、时间锚定、三层记忆、Reranker Bypass、结构化输出 |
| `app/graph/state.py`             | 修改   | `hyde_answer` 字段                            |
| `app/memory/long_term_memory.py` | 新增   | 6个事件流读写方法                                   |
| `app/rag/hybrid_retriever.py`    | 修改   | `_dense_search` 返回 top1 score               |
| `app/api/routes.py`              | 修改   | L0 缓存日志                                     |
| `requirements.txt`               | 新增   | `dateparser~=1.4.1`、`json_repair~=0.60.1`   |

***

### 依赖变化

| 依赖            | 版本       | 用途                        |
| ------------- | -------- | ------------------------- |
| `dateparser`  | \~1.4.1  | 相对时间→绝对时间解析（200+语言）       |
| `json_repair` | \~0.60.1 | 3B模型输出 JSON 修复（单引号、多余逗号等） |

***

### 后续优化方向

1. **痊愈/恢复事件**：用户说"我头不痛了"时记录恢复时间，形成完整的症状生命周期
2. **时间范围过滤检索**：RAG 检索时支持按时间范围过滤文档
3. **L1 事件过期清理**：超过6个月的症状事件自动归档/清理
4. **异步 L1 写入**：快照更新时 L1 写入改为 `asyncio.create_task` 真正异步
5. **用户修改时间**："不是前天，是大前天" → 更新 L1 中已有的事件

