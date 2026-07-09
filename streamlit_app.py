
import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path
import google.generativeai as genai

# Ensure local modules in 'app' folder are importable
sys.path.append(os.path.abspath('.'))
from app.orchestrator import Orchestrator

st.set_page_config(page_title="Clinical Agent Explorer", page_icon="🏥", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Settings")
    api_key = st.text_input("Google API Key", type="password")
    if api_key:
        genai.configure(api_key=api_key)
        st.success("API Ready")
    
    st.divider()
    st.header("📊 Data Preview")
    data_files = list(Path('data').glob('*.csv'))
    if data_files:
        selected_csv = st.selectbox("View Table", [f.name for f in data_files])
        df_view = pd.read_csv(Path('data') / selected_csv)
        st.dataframe(df_view.head(5))

# --- Main UI ---
st.title("🏥 Multi-Agent Clinical RAG")
st.info("This system uses a Planner, Retriever, and Evaluator to answer patient-specific queries.")

orch = Orchestrator()
user_query = st.text_input("Ask a clinical question (e.g., 'What meds is P0001 taking?')")

if st.button("Run Analysis"):
    if not api_key:
        st.error("Please provide an API Key in the sidebar.")
    elif user_query:
        with st.spinner("Agents coordinating..."):
            res = orch.process_query(user_query)
            st.subheader("Final Answer")
            st.write(res['answer'])
            
            with st.expander("🔍 Grounding Evidence"):
                for src in res['sources']:
                    st.markdown(f"**Source: {src['metadata']['source']}**")
                    st.caption(src['text'])
    else:
        st.warning("Please enter a query.")
