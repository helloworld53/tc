# ingest.py
import os
import glob
import streamlit as st
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from openai import OpenAI
DATA_DIR = "data"
VECTOR_DIR = st.secrets.get("VECTOR_DIR", "vectorstore")

def load_docs() -> List:
    docs = []
    for path in glob.glob(os.path.join(DATA_DIR, "**/*"), recursive=True):
        if path.lower().endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
        elif path.lower().endswith(".txt") or path.lower().endswith(".md"):
            docs.extend(TextLoader(path, encoding="utf-8").load())
    if not docs:
        raise RuntimeError("No documents found in ./data. Add PDFs or txt/md files.")
    return docs

def main():
    os.makedirs(VECTOR_DIR, exist_ok=True)
    docs = load_docs()

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"], base_url="https://api.together.xyz/v1")
    

    # # Embeddings via Together AI (OpenAI-compatible)
    # embeddings = OpenAIEmbeddings(
    #     # model="BAAI/bge-base-en-v1.5",
    #     model="togethercomputer/m2-bert-80M-32k-retrieval",
    #     base_url="https://api.together.xyz/v1",
    #     api_key=st.secrets["OPENAI_API_KEY"],
    # )
    for i, d in enumerate(chunks):
        try:
            # embeddings.embed_query(d.page_content[:2000]) 
            print(client.embeddings.create(model="BAAI/bge-base-en-v1.5", input=d.page_content[:2000])) # test short embed
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ Chunk {i} failed, content snippet:\n{d.page_content[:200]}")

    # Build FAISS (cosine via internal normalize)
    # vs = FAISS.from_documents(chunks, embeddings)
    # vs.save_local(VECTOR_DIR)
    # print(f"✅ Vector store saved to {VECTOR_DIR} with {len(chunks)} chunks.")

if __name__ == "__main__":
    main()
