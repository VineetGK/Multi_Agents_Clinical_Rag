import streamlit as st
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Import using the app package
from app.orchestrator import Orchestrator

st.set_page_config(
    page_title="Clinical RAG - Patient Query System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_orchestrator():
    return Orchestrator()

@st.cache_data
def get_patient_list():
    orch = get_orchestrator()
    return sorted(orch.patients_df['patient_id'].tolist())

@st.cache_data
def get_patient_demo(patient_id):
    orch = get_orchestrator()
    row = orch.patients_df[orch.patients_df['patient_id'] == patient_id]
    if row.empty:
        return None
    return row.iloc[0].to_dict()

def main():
    st.title("🏥 Clinical RAG Patient Query System")
    st.caption("Query 500 patients across diagnoses, labs, medications, encounters, and clinical notes")

    orch = get_orchestrator()
    patient_list = get_patient_list()

    with st.sidebar:
        st.header("Patient Selection")
        selected_pid = st.selectbox(
            "Choose Patient ID",
            patient_list,
            index=0,
            format_func=lambda x: f"{x} (Age: {get_patient_demo(x)['age']}, Sex: {get_patient_demo(x)['sex']})"
        )

        demo = get_patient_demo(selected_pid)
        if demo:
            st.markdown("---")
            st.subheader("Demographics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Age", demo['age'])
                st.metric("Sex", demo['sex'])
            with col2:
                st.metric("Race", demo['race'])
                st.metric("Chronic Conditions", demo['chronic_conditions_count'])

        st.markdown("---")
        st.header("Quick Actions")
        quick_queries = {
            "📋 Demographics": f"What is patient {selected_pid}'s age, sex, race, and chronic conditions count?",
            "🩺 Diagnoses": f"What diagnoses has patient {selected_pid} had, and when were they recorded?",
            "📝 Discharge Notes": f"What were patient {selected_pid}'s most recent discharge summary notes?",
            "🔬 Latest Labs": f"What are patient {selected_pid}'s latest lab results with abnormal flags?",
            "📈 A1c Trend": f"Show patient {selected_pid}'s A1c trend over time and diabetes control",
            "💊 Medications": f"What medications has patient {selected_pid} been prescribed?",
            "🏥 Encounters": f"How many encounters has patient {selected_pid} had?",
            "⚠️ Readmission Risk": f"What are patient {selected_pid}'s readmission risk factors?",
            "📋 Handoff Summary": f"Generate a handoff summary for patient {selected_pid}",
        }
        
        for label, query in quick_queries.items():
            if st.button(label, use_container_width=True):
                st.session_state['query_input'] = query
                st.rerun()

    st.header(f"Patient: {selected_pid}")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Query", "📋 Diagnoses", "🔬 Labs", "💊 Medications", "🏥 Encounters"])

    with tab1:
        st.subheader("Ask a Clinical Question")
        
        default_query = st.session_state.get('query_input', f"Patient {selected_pid}: ")
        query = st.text_area(
            "Enter your question",
            value=default_query,
            height=100,
            placeholder=f"e.g., What diagnoses has patient {selected_pid} had? Show A1c trend for patient {selected_pid}."
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            ask_btn = st.button("🔍 Ask", type="primary", use_container_width=True)
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state['query_input'] = f"Patient {selected_pid}: "
                st.rerun()

        if ask_btn and query.strip():
            with st.spinner("Processing query..."):
                result = orch.process_query(query)
            
            st.markdown("### Answer")
            st.write(result['answer'])
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Grounded", "✅ Yes" if result['grounded'] else "❌ No")
            with col2:
                st.metric("Sources", len(result['sources']))
            
            if result['sources']:
                with st.expander("📚 View Sources"):
                    for i, src in enumerate(result['sources']):
                        st.markdown(f"**Source {i+1}:** `{src['metadata']['source']}` (Patient: {src['metadata']['patient_id']})")
                        st.text(src['text'][:500] + ("..." if len(src['text']) > 500 else ""))

    with tab2:
        st.subheader("All Diagnoses")
        result = orch.process_query(f"What diagnoses has patient {selected_pid} had, and when were they recorded?")
        st.write(result['answer'])

    with tab3:
        st.subheader("All Lab Results")
        result = orch.process_query(f"What are patient {selected_pid}'s latest lab results with abnormal flags?")
        st.write(result['answer'])

    with tab4:
        st.subheader("Medications")
        result = orch.process_query(f"What medications has patient {selected_pid} been prescribed?")
        st.write(result['answer'])

    with tab5:
        st.subheader("Encounters")
        result = orch.process_query(f"How many encounters has patient {selected_pid} had, and what was the most recent?")
        st.write(result['answer'])

    st.markdown("---")
    st.markdown("### 💡 Example Questions")
    examples = [
        f"Show patient {selected_pid}'s A1c trend and diabetes control",
        f"Does patient {selected_pid} have heart failure evidence (CHF diagnosis, BNP, notes)?",
        f"Has patient {selected_pid} had multiple elevated creatinine results?",
        f"Show patient {selected_pid}'s clinical timeline",
        f"What follow-up occurred within 30 days of discharge for patient {selected_pid}?",
        f"Do patient {selected_pid}'s medications align with their chronic conditions?",
    ]
    
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        with cols[i % 2]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state['query_input'] = ex
                st.rerun()

if __name__ == "__main__":
    main()