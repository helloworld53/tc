import streamlit as st
import asyncio
from graph2 import run_agent  # ✅ import backend agent

st.set_page_config(page_title="Immigration Legal Assistant", page_icon="🛂")

st.title("🛂 Lithuanian Immigration Assistant")
st.write("Ask questions about immigration rules, residency days, Schengen compliance, and more.")

# ---- Input box ----
user_query = st.text_area("Your question:", placeholder="e.g., Calculate residency days from 2024-01-01 to 2024-03-01")

if st.button("Ask"):
    if not user_query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            # Run asyncio inside Streamlit
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            answer = loop.run_until_complete(run_agent(user_query))

        st.success("Answer:")
        st.write(answer)
