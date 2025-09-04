# app.py
import time
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from graph import build_graph

st.set_page_config(page_title="🇱🇹 Immigration Assistant (LangGraph RAG+Tools)", layout="wide")
st.title("🇱🇹 Lithuanian Immigration Assistant")
st.caption("Advanced RAG + Function Calling with LangGraph + Together AI")

# Simple rate limit (per session)
MAX_MSG_PER_MIN = 12
if "timestamps" not in st.session_state:
    st.session_state.timestamps = []

def too_many():
    now = time.time()
    st.session_state.timestamps = [t for t in st.session_state.timestamps if now - t < 60]
    if len(st.session_state.timestamps) >= MAX_MSG_PER_MIN:
        return True
    st.session_state.timestamps.append(now)
    return False

# Build graph once
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "history" not in st.session_state:
    st.session_state.history = []

# Chat UI
for role, content in st.session_state.history:
    st.chat_message(role).markdown(content)

q = st.chat_input("Ask about Lithuanian visas, residency days, Schengen 90/180, fees, etc.")
if q:
    if too_many():
        st.warning("Rate limit: please wait a few seconds.")
    else:
        st.session_state.history.append(("user", q))
        st.chat_message("user").markdown(q)

        with st.spinner("Thinking..."):
            # Run the graph; stream events for progress (optional)
            result = st.session_state.graph.invoke({
                "query": q,
                "messages": [],
                "docs": [],
                "sources": [],
                "tool_calls": [],
            })

        # Render final assistant message
        # Find last AIMessage in result['messages']
        last_ai = None
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                last_ai = msg
                break

        answer = last_ai.content if last_ai else "I couldn't form an answer."
        st.chat_message("assistant").markdown(answer)
        st.session_state.history.append(("assistant", answer))

        # Show sources if present (RAG path)
        if result.get("sources"):
            with st.expander("Sources"):
                for s in result["sources"]:
                    srcline = f"[S{s['id']}] {s.get('title','Untitled')} — {s.get('source','N/A')}"
                    if s.get("page") is not None:
                        srcline += f" (p.{s['page']})"
                    st.markdown(f"- {srcline}")

        # Show tool calls & results (tool path)
        tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        if tool_msgs:
            with st.expander("Tool Calls"):
                for tm in tool_msgs:
                    st.code(f"{tm.name}: {tm.content}", language="json")

# Footer
with st.sidebar:
    st.subheader("ℹ️ Tips")
    st.markdown("- Try: *Calculate residency days from 2024-01-10 to 2024-05-20*")
    st.markdown("- Try: *Check Schengen 90/180 for two stays...*")
    st.markdown("- Try: *Estimate fees for a work visa, fast track*")

