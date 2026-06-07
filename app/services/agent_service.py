# DP: 核心 RAG Agent — 检索 + LLM + 工具调用 + 流式输出 + 来源过滤
import json
import asyncio
import jieba
import dashscope
from app.core.config import settings
from app.services.tools import search_internet, get_real_weather, AGENT_TOOLS, get_last_web_sources
from app.database.chroma_store import knowledge_collection
from app.database.sqlite_store import save_message_with_source, get_recent_messages

dashscope.api_key = settings.DASHSCOPE_API_KEY

# DP: 全局 HybridRetriever + Reranker 实例（懒加载）
_retriever = None
_reranker = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        from app.core.retriever import HybridRetriever
        # 从 ChromaDB 拉取所有文档用于 BM25 索引
        try:
            all_docs = knowledge_collection.get()["documents"] or []
        except Exception:
            all_docs = []
        _retriever = HybridRetriever(all_docs)
    return _retriever


def _get_reranker():
    """DP: 懒加载 Reranker — 阿里云 API，无需 device 参数"""
    global _reranker
    if _reranker is None:
        from app.core.reranker import DocumentReranker
        _reranker = DocumentReranker()
    return _reranker


def _hybrid_retrieve(query: str, top_k: int = 10) -> list[str]:
    """DP: 混合检索 = BM25关键词 + ChromaDB语义 → 去重 → Rerank精排 → Top2"""
    retriever = _get_retriever()
    reranker = _get_reranker()
    # 1. 混合检索
    candidates = retriever.hybrid_search(query, top_k=top_k)
    if not candidates:
        return []
    # 2. Rerank 精排（DP: DashScope API，内置容错，失败自动回退 top_k）
    candidates = reranker.rerank(query, candidates, top_k=3)
    return candidates[:2]


# DP: 兼容 dashscope 返回的 dict 和 object 两种 tool_calls 格式
def _get_attr(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# DP: 用 jieba 分词在所有检索文档中找最佳匹配作为来源，不相关则不显示
def _format_source(docs, query=""):
    if not docs:
        return ""
    # DP: 构建查询词集合（去停用词）
    stop = {"的", "了", "是", "吗", "呢", "吧", "啊", "在", "有", "我", "你", "他", "这", "那", "什么", "怎么", "哪个", "多少", "为什么", "可以", "一个"}
    query_tokens = {t for t in jieba.cut(query) if len(t) > 1 and t not in stop}

    if not query_tokens:
        # DP: 无有效查询词，直接返回第一篇摘要
        doc = docs[0]
        short = doc[:150].replace("\n", " ").strip()
        if len(doc) > 150:
            short += "..."
        return short

    # DP: 遍历所有文档找最佳匹配（而非只看第一篇）
    best_doc = None
    best_score = -1
    for doc in docs:
        score = sum(1 for t in query_tokens if t in doc[:400])
        if score > best_score:
            best_score = score
            best_doc = doc

    if best_score == 0 or best_doc is None:
        return ""  # DP: 所有文档都不相关，不展示来源

    short = best_doc[:150].replace("\n", " ").strip()
    if len(best_doc) > 150:
        short += "..."
    return short


# DP: 异步生成器 — 流式 SSE 输出给前端
async def qwen_llm_generator(query: str, session_id: str):
    final_text = ""
    source_text = ""

    try:
        # DP: 1. 混合检索 = BM25 + ChromaDB → Rerank 精排
        retrieved_docs = _hybrid_retrieve(query)
        source_text = _format_source(retrieved_docs, query)

        # DP: 2. 加载最近 6 条聊天历史
        history = get_recent_messages(session_id, limit=6)

        # DP: 3. 构建 RAG 上下文消息
        rag_context = ""
        if retrieved_docs:
            rag_context = "\n\n【本地知识库】\n" + "\n".join(retrieved_docs)

        system_prompt = (
            "你是企业AI助理。优先参考【本地知识库】回答。"
            "本地资料足够时不要调工具。无法回答时才调 search_internet。问天气调 get_real_weather。"
            "回答简洁、用中文。"
        )

        messages = [{"role": "system", "content": system_prompt}]
        for h in history:
            messages.append({"role": h["role"], "content": h["content"]})
        user_content = f"{rag_context}\n\n【用户问题】{query}" if rag_context else query
        messages.append({"role": "user", "content": user_content})

        # DP: 4. 先发 THINKING 让前端立即可见，再调 LLM（线程池避免阻塞）
        yield "data: [THINKING]: 正在分析...\n\n"
        await asyncio.sleep(0)

        response = await asyncio.to_thread(
            dashscope.Generation.call,
            model=settings.DEFAULT_MODEL,
            messages=messages,
            tools=AGENT_TOOLS,
            result_format='message'
        )

        if response.status_code != 200:
            yield f"data: LLM调用失败: {response.message}\n\n"
            save_message_with_source(session_id, "assistant", f"LLM调用失败: {response.message}", "")
            return

        msg = response.output.choices[0].message
        tool_calls = _get_attr(msg, 'tool_calls', None)

        # DP: 5. 工具调用分支（天气 / 联网搜索）
        if tool_calls:
            tc = tool_calls[0]
            func = _get_attr(tc, 'function', {})
            func_name = _get_attr(func, 'name', '')
            func_args_str = _get_attr(func, 'arguments', '{}')
            func_args = json.loads(func_args_str) if isinstance(func_args_str, str) else func_args_str
            tc_id = _get_attr(tc, 'id', 'call_001')

            yield f"data: [TOOL]: {func_name}\n\n"

            # DP: 工具在线程池执行，带超时
            try:
                if func_name == "get_real_weather":
                    tool_result = await asyncio.to_thread(get_real_weather, func_args.get("location", ""))
                    # DP: 天气查询来源标记
                    source_text = f"🌤️ 天气查询: {func_args.get('location', '')}"
                else:
                    tool_result = await asyncio.to_thread(search_internet, func_args.get("query", ""))
                    # DP: 联网搜索来源标记，显示搜索URL
                    web_urls = get_last_web_sources()
                    if web_urls:
                        source_text = "🌐 网络搜索: " + " | ".join(web_urls[:2])
                    else:
                        source_text = f"🌐 网络搜索: {func_args.get('query', '')}"
            except Exception as te:
                tool_result = f"调用失败: {str(te)}"

            # DP: 追加工具消息（OpenAI 兼容格式，tool_call_id 必填）
            messages.append({
                "role": "assistant",
                "content": _get_attr(msg, 'content', '') or '',
                "tool_calls": [{
                    "id": tc_id, "type": "function",
                    "function": {"name": func_name, "arguments": func_args_str}
                }]
            })
            messages.append({"role": "tool", "tool_call_id": tc_id, "content": str(tool_result)})

            # DP: 第二次 LLM 调用，整合工具结果
            final_resp = await asyncio.to_thread(
                dashscope.Generation.call,
                model=settings.DEFAULT_MODEL,
                messages=messages,
                result_format='message'
            )
            if final_resp.status_code == 200:
                final_text = final_resp.output.choices[0].message.content or ''
                for i in range(0, len(final_text), 6):
                    yield f"data: {final_text[i:i + 6]}\n\n"
                    await asyncio.sleep(0.005)
            else:
                yield f"data: 生成失败: {final_resp.message}\n\n"

            # DP: 工具场景结束后也发送来源
            if source_text:
                yield f"data: [SOURCE]: {source_text}\n\n"

        else:
            # DP: 6. 无工具 — 直接分块流式输出
            final_text = _get_attr(msg, 'content', '') or ''
            if final_text:
                for i in range(0, len(final_text), 6):
                    yield f"data: {final_text[i:i + 6]}\n\n"
                    await asyncio.sleep(0.005)
            else:
                yield "data: 抱歉，无法回答。\n\n"

            # DP: 7. 内容输出完后再发来源（在前端底部显示）
            if source_text:
                yield f"data: [SOURCE]: {source_text}\n\n"

    except Exception as e:
        yield f"data: 服务出错: {str(e)}\n\n"
        final_text = f"服务出错: {str(e)}"

    # DP: 8. 保存回答 + 来源到 SQLite（切换对话不丢失）
    if final_text and "服务出错" not in final_text and "LLM调用失败" not in final_text:
        save_message_with_source(session_id, "assistant", final_text, source_text)
