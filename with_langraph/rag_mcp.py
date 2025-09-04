import os, json, faiss, numpy as np
from openai import OpenAI
from datetime import datetime, timedelta
from mcp.server.fastmcp import FastMCP
import streamlit as st 
# ==== Config ====
VECTOR_DIR = "vectorstore"
INDEX_PATH = os.path.join(VECTOR_DIR, "index.faiss")
DOCSTORE_PATH = os.path.join(VECTOR_DIR, "docstore.json")
EMBED_MODEL = "togethercomputer/m2-bert-80M-32k-retrieval"

# ==== Load FAISS + docstore ====
index = faiss.read_index(INDEX_PATH)
docstore = json.load(open(DOCSTORE_PATH, encoding="utf-8"))

# ==== Together client ====
client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"],
    base_url="https://api.together.xyz/v1"
)

# ==== Define MCP server ====
mcp = FastMCP("rag-mcp")

# ------------------ RAG TOOLS ------------------
@mcp.tool()
def embed_and_search(query: str, k: int = 5):
    """Embed a query and return top-k FAISS matches (IDs + distances)."""
    q_vec = client.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding
    D, I = index.search(np.array([q_vec], dtype="float32"), k)
    return {"hits": [{"id": int(i), "dist": float(d)} for i, d in zip(I[0], D[0])]}

@mcp.tool()
def get_context(doc_ids: list, window: int = 1, max_chars: int = 5000):
    """Return merged context for given doc IDs, expanding neighbors in same article."""
    expanded = []
    for i in doc_ids:
        base = docstore[int(i)]
        art_id = base["meta"].get("article_id")
        # expand all chunks from the same article
        article_chunks = [c for c in docstore if c["meta"].get("article_id") == art_id]
        expanded.extend(article_chunks)

    # Merge into one big context (truncate if too long)
    text = "\n".join([c["text"] for c in expanded])[:max_chars]
    return {"contexts": [text]}

# ------------------ VISA TOOLS ------------------
@mcp.tool()
def calc_residency_days(entry_date: str, exit_date: str):
    """Calculate number of days stayed between two dates (YYYY-MM-DD)."""
    d1 = datetime.strptime(entry_date, "%Y-%m-%d")
    d2 = datetime.strptime(exit_date, "%Y-%m-%d")
    if d2 < d1:
        return {"error": "exit_date must be after entry_date"}
    days = (d2 - d1).days
    return {"days": days}

@mcp.tool()
def schengen_90_180_check(stays: list):
    """
    Check Schengen 90/180 compliance.
    stays: list of {"entry":"YYYY-MM-DD","exit":"YYYY-MM-DD"} windows.
    """
    parsed = []
    for s in stays:
        a = datetime.strptime(s["entry"], "%Y-%m-%d")
        b = datetime.strptime(s["exit"], "%Y-%m-%d")
        if b < a:
            return {"error": "Invalid stay window"}
        parsed.append((a, b))

    if not parsed:
        return {"total_days_180": 0, "compliant": True}

    end = max(b for _, b in parsed)
    window_start = end - timedelta(days=180)
    total = 0
    for a, b in parsed:
        start_overlap = max(a, window_start)
        end_overlap = b
        if end_overlap >= start_overlap:
            total += (end_overlap - start_overlap).days
    return {"total_days_180": total, "compliant": total <= 90}

@mcp.tool()
def fee_estimator(visa_type: str, fast_track: bool = False):
    """Rough fee estimate (demo values). visa_type in {"work","student","schengen"}."""
    base = {"work": 140, "student": 60, "schengen": 80}.get(visa_type.lower(), 100)
    total = base + (50 if fast_track else 0)
    return {"visa_type": visa_type, "fast_track": fast_track, "estimate_eur": total}

# ------------------ MAIN ------------------
if __name__ == "__main__":
    mcp.run(transport="stdio")
