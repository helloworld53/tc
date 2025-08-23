# graph.py
from typing import Annotated, TypedDict, List, Dict, Any
from langchain_core.documents import Document
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

from tools import calc_residency_days, schengen_90_180_check, fee_estimator

VECTOR_DIR = st.secrets.get("VECTOR_DIR", "vectorstore")

# ---------- State ----------
class State(TypedDict):
    query: str
    messages: Annotated[List[AnyMessage], add_messages]
    docs: List[Document]
    sources: List[Dict[str, Any]]
    tool_calls: List[Dict[str, Any]]

# ---------- Models ----------
def get_llm():
    return ChatOpenAI(
        model="openai/gpt-oss-20b",
        temperature=0.2,
        streaming=True,
        base_url="https://api.together.xyz/v1",
        api_key=st.secrets["OPENAI_API_KEY"],
    )

def get_retriever():
    embeddings = OpenAIEmbeddings(
        model="BAAI/bge-base-en-v1.5",
        base_url="https://api.together.xyz/v1",
        api_key=st.secrets["OPENAI_API_KEY"],
    )
    vs = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
    # MMR retrieval for diversity
    return vs.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 18, "lambda_mult": 0.4})

# ---------- Prompt pieces ----------
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "You rewrite user queries for legal-immigration search in Lithuania. Be specific. Output only the rewritten query."),
    ("human", "{q}")
])

answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a careful legal assistant for Lithuanian immigration. "
     "Answer using the provided context. If unsure, say what is missing. "
     "Cite sources with [S#] where # is the source index."),
    ("human", "Question: {q}\n\nContext:\n{context}\n\nMake it actionable with steps and bullet points when helpful.")
])

# ---------- Nodes ----------
def router(state: State):
    q = state["query"].lower()
    # Route tool-y queries to tool loop, otherwise to RAG
    if any(kw in q for kw in ["calculate", "days", "schengen", "fee", "cost", "estimate"]):
        return "tools_loop"
    return "rewrite_query"

def rewrite_query(state: State):
    llm = get_llm()
    new_q = llm.invoke(rewrite_prompt.format_messages(q=state["query"])).content.strip()
    state["messages"].append(HumanMessage(content=f"[Rewritten Query] {new_q}"))
    state["query"] = new_q
    return state

def retrieve(state: State):
    retriever = get_retriever()
    docs = retriever.get_relevant_documents(state["query"])
    # Optional: post-filter by year/type if your metadata provides it.
    state["docs"] = docs
    # Prepare sources
    state["sources"] = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        state["sources"].append({
            "id": i,
            "source": meta.get("source", "N/A"),
            "page": meta.get("page"),
            "title": meta.get("title", "Untitled")
        })
    return state

def rag_answer(state: State):
    llm = get_llm()
    context = ""
    for i, d in enumerate(state["docs"], 1):
        context += f"[S{i}] {d.page_content}\n"
    msg = answer_prompt.format_messages(q=state["query"], context=context)
    ai = llm.invoke(msg)
    state["messages"].append(ai)
    return state

# ----- Tool-calling loop using ToolNode -----
tools = [calc_residency_days, schengen_90_180_check, fee_estimator]
tool_node = ToolNode(tools)

def llm_with_tools(state: State):
    llm = get_llm().bind_tools(tools)
    # Ask the model; if it decides to call a tool, messages will contain tool calls
    ai = llm.invoke(state["messages"] + [HumanMessage(content=state["query"])])
    state["messages"].append(ai)
    return state

def should_continue_tool_loop(state: State):
    last = state["messages"][-1]
    # Continue if the model issued any tool calls
    if isinstance(last, AIMessage) and last.tool_calls:
        return "call_tools"
    return "finalize"

def add_tool_results_to_messages(state: State):
    """ToolNode returns ToolMessage(s); append them to messages."""
    return state

def finalize(state: State):
    # If we’re here after tools, do a final LLM answer grounded in results
    llm = get_llm()
    # Build a compact context from tool messages (if any)
    tool_context = ""
    for m in state["messages"]:
        if isinstance(m, ToolMessage):
            tool_context += f"{m.content}\n"
    # If no tool context (e.g., no tools called), just echo last AI
    if not tool_context.strip():
        return state
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the tool results below to answer clearly and concisely."),
        ("human", "Tool results:\n{tool}\n\nUser question:\n{q}")
    ])
    ai = llm.invoke(prompt.format_messages(tool=tool_context, q=state["query"]))
    state["messages"].append(ai)
    return state

# ---------- Build the graph ----------
def build_graph():
    g = StateGraph(State)

    g.add_node("rewrite_query", rewrite_query)
    g.add_node("retrieve", retrieve)
    g.add_node("rag_answer", rag_answer)

    g.add_node("llm_with_tools", llm_with_tools)
    g.add_node("call_tools", tool_node)           # executes tools & returns ToolMessages
    g.add_node("finalize", finalize)

    g.set_entry_point("router")

    # Router (function) to choose path
    g.add_node("router", router)
    g.add_conditional_edges("router", lambda s: router(s), {
        "tools_loop": "llm_with_tools",
        "rewrite_query": "rewrite_query",
    })

    # RAG path
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("retrieve", "rag_answer")
    g.add_edge("rag_answer", END)

    # Tools loop: LLM → [maybe tool calls] → LLM → ...
    g.add_conditional_edges("llm_with_tools", should_continue_tool_loop, {
        "call_tools": "call_tools",
        "finalize": "finalize",
    })
    g.add_edge("call_tools", "llm_with_tools")   # feed tool results back
    g.add_edge("finalize", END)

    return g.compile()
