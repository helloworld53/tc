## graph2.py — React-style agent with LLM + Tools + MCP
import asyncio
from typing import Annotated, TypedDict, List, Dict, Any

from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
import streamlit as st

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools


# ---------- State ----------
class State(TypedDict):
    query: str
    messages: Annotated[List[AnyMessage], add_messages]
    tool_calls: List[Dict[str, Any]]


# ---------- LLM ----------
def get_llm():
    return ChatOpenAI(
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
        temperature=0.2,
        streaming=True,
        base_url="https://api.together.xyz/v1",
        api_key=st.secrets["OPENAI_API_KEY"],  # ✅ uses Streamlit secrets
    )


# ---------- Prompts ----------
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a careful legal assistant for Lithuanian immigration. "
               "Answer using tool results if available. If unsure, say what is missing. "
               "Cite sources with [S#] when possible."),
    ("human", "Question: {q}\n\nTool/context data:\n{context}")
])


# ---------- MCP Tool Discovery ----------
async def load_mcp_tools_from_server():
    server_params = StdioServerParameters(
        command="python",
        args=["rag_mcp.py"],   # ✅ your MCP server file
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)   # ✅ returns StructuredTools already
            print(f"Discovered {len(tools)} tools from rag_mcp.py")
            return tools


# ---------- Nodes ----------
def router(state: State):
    print("🔎 Router directing to: llm_with_tools")
    return "llm_with_tools"


async def llm_with_tools(state: State, tools):
    llm = get_llm().bind_tools(tools)
    ai = await llm.ainvoke(
        state["messages"] + [HumanMessage(content=state["query"])]
    )
    new_state = dict(state)   # ✅ return fresh dict
    new_state["messages"] = state["messages"] + [ai]
    return new_state


def should_continue_tool_loop(state: State):
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        print("🔄 LLM requested tool calls")
        return "call_tools"
    print("✅ No tool calls, moving to finalize")
    return "finalize"


async def finalize(state: State):
    llm = get_llm()
    tool_context = "".join(
        f"{m.content}\n" for m in state["messages"] if isinstance(m, ToolMessage)
    )

    if not tool_context.strip():
        return state

    msg = answer_prompt.format_messages(q=state["query"], context=tool_context)
    ai = await llm.ainvoke(msg)

    new_state = dict(state)   # ✅ return fresh dict
    new_state["messages"] = state["messages"] + [ai]
    return new_state


# ---------- Build graph ----------
async def build_graph():
    tools = await load_mcp_tools_from_server()

    async def llm_with_tools_node(state: State):
        return await llm_with_tools(state, tools)

    g = StateGraph(State)
    g.add_node("llm_with_tools", RunnableLambda(llm_with_tools_node))
    g.add_node("call_tools", ToolNode(tools))  # ✅ StructuredTools already
    g.add_node("finalize", RunnableLambda(finalize))
    g.add_node("router", router)

    g.set_entry_point("router")
    g.add_conditional_edges("router", lambda s: "llm_with_tools",
                            {"llm_with_tools": "llm_with_tools"})

    g.add_conditional_edges("llm_with_tools", should_continue_tool_loop, {
        "call_tools": "call_tools",
        "finalize": "finalize",
    })
    g.add_edge("call_tools", "llm_with_tools")
    g.add_edge("finalize", END)

    return g.compile()


# ---------- Run agent ----------
async def run_agent(query: str) -> str:
    graph = await build_graph()
    state = {"query": query, "messages": [], "tool_calls": []}
    result = await graph.ainvoke(state)   # ✅ async version
    return result["messages"][-1].content


# ---------- CLI Test ----------
if __name__ == "__main__":
    async def main():
        answer = await run_agent("Please calculate residency days from 2024-01-01 to 2024-03-01.")
        print("Answer:", answer)

    asyncio.run(main())
