import asyncio
from typing import Annotated, TypedDict, List, Dict, Any

from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate

# from fastmcp import Client
# from langchain_mcp_adapters import mcp_tools_to_langchain  # <-- adapter
# from langchain_mcp_adapters.tools import mcp_tools_to_langchain
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
        model="meta-llama/Llama-3-8b-chat-hf",
        temperature=0.2,
        streaming=True,
        base_url="https://api.together.xyz/v1",
        api_key="your_together_api_key",  # replace with st.secrets in Streamlit
    )

# ---------- Prompts ----------
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a careful legal assistant for Lithuanian immigration. "
               "Answer using tool results if available. If unsure, say what is missing. "
               "Cite sources with [S#] when possible."),
    ("human", "Question: {q}\n\nTool/context data:\n{context}")
])

# ---------- MCP Tool Discovery ----------
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








# async def load_mcp_tools():
#     client = Client("rag_mcp.py")   # your MCP server
#     async with client:
#         tools = await client.list_tools()
#         print(f"Discovered {len(tools)} tools")
#         return mcp_tools_to_langchain(client, tools)  # ✅ convert into LangChain StructuredTools

# ---------- Nodes ----------
def router(state: State):
    return "llm_with_tools"

def llm_with_tools(state: State, tools):
    llm = get_llm().bind_tools(tools)
    ai = llm.invoke(state["messages"] + [HumanMessage(content=state["query"])])
    state["messages"].append(ai)
    return state

def should_continue_tool_loop(state: State):
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "call_tools"
    return "finalize"

def finalize(state: State):
    llm = get_llm()
    tool_context = ""
    for m in state["messages"]:
        if isinstance(m, ToolMessage):
            tool_context += f"{m.content}\n"
    if not tool_context.strip():
        return state
    msg = answer_prompt.format_messages(q=state["query"], context=tool_context)
    ai = llm.invoke(msg)
    state["messages"].append(ai)
    return state

# ---------- Build graph ----------
async def build_graph():
    tools =  await load_mcp_tools_from_server() 

    g = StateGraph(State)
    g.add_node("llm_with_tools", lambda s,tools=tools: llm_with_tools(s, tools))
    g.add_node("call_tools", ToolNode(tools))  # ✅ now works because tools are StructuredTools
    g.add_node("finalize", finalize)
    g.add_node("router", router)

    g.set_entry_point("router")
    g.add_conditional_edges("router", lambda s: "llm_with_tools", {"llm_with_tools": "llm_with_tools"})

    g.add_conditional_edges("llm_with_tools", should_continue_tool_loop, {
        "call_tools": "call_tools",
        "finalize": "finalize",
    })
    g.add_edge("call_tools", "llm_with_tools")
    g.add_edge("finalize", END)

    return g.compile()

# ---------- Run test ----------
async def main():
    graph = await build_graph()
    state = {"query": "Please calculate residency days from 2024-01-01 to 2024-03-01.", "messages": []}
    result = graph.invoke(state)
    print("Final messages:")
    for msg in result["messages"]:
        print(f"{msg.type.upper()}: {msg.content}")

if __name__ == "__main__":
    asyncio.run(main())
