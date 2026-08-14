import asyncio, sys, os
sys.path.insert(0, ".")
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def main():
    from app.rag.hybrid_retriever import get_cached_hybrid_retriever
    from app.graph.nodes.nodes import build_rag_prompt
    retriever = get_cached_hybrid_retriever(k=5)
    docs = await asyncio.to_thread(retriever.invoke, "头痛怎么处理？")
    print(f"检索返回文档数: {len(docs)}")
    total = 0
    for d in docs:
        total += len(d.page_content)
        print(f"  [{d.metadata.get('source')}] {len(d.page_content)}字符")
    print(f"文档总字符: {total}")
    state = {"question": "头痛怎么处理？", "user_profile": None, "clinical_checkpoint": None, "symptoms": None, "messages": []}
    messages = build_rag_prompt(question="头痛怎么处理？", retrieved_docs=docs, user_profile=None, state=state)
    chars = sum(len(m.content) for m in messages if hasattr(m, "content"))
    print(f"prompt 总字符: {chars}, 估算 tokens: {chars//2}")
    for i, m in enumerate(messages):
        print(f"  [{i}]{getattr(m,'type','?')} {len(m.content)}字符")

asyncio.run(main())
