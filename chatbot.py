# chat_bot.py

from typing import TypedDict, List
from langchain_core.documents import Document
from langchain.tools import tool
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda

# 🧩 Chat State
class ChatState(TypedDict):
    query: str
    docs: List[Document]
    answer: str
    tool_calls: List[str]

# 🔧 Tools
@tool
def get_market_trends():
    """Returns mock market trend data."""
    return {"trend": "Tech stocks up 5% this week."}

@tool
def recommend_jobs(skills: str):
    """Suggests jobs based on skills."""
    return [f"AI Engineer with {skills}", f"Data Scientist with {skills}"]

@tool
def analyze_docs(text: str):
    """Performs basic NLP analysis."""
    return {"word_count": len(text.split()), "keywords": text.split()[:5]}

# 🔁 Nodes
def router_node(state: ChatState) -> str:
    if "trend" in state["query"]:
        return "tool_node"
    return "retriever_node"

def retriever_node(state: ChatState) -> ChatState:
    # Mock retrieval
    state["docs"] = [Document(page_content="AI is transforming industries.")]
    return state

def generator_node(state: ChatState) -> ChatState:
    context = " ".join([doc.page_content for doc in state["docs"]])
    state["answer"] = f"Based on docs: {context}"
    return state

def tool_node(state: ChatState) -> ChatState:
    state["tool_calls"] = ["get_market_trends"]
    state["answer"] = get_market_trends