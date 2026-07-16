# Clinical RAG Patient Query System

A RAG-based clinical question-answering system for 500 synthetic EHR patients. Query demographics, diagnoses, labs, medications, encounters, and clinical notes with natural language.

## 🌐 Live Demo

https://multiagentsclinicalrag-axnmlwsn8zpaxswr7zhpb3.streamlit.app/



## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app.py

# Or CLI query
python query.py "Patient P0001 age sex race chronic conditions"
```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Main file: `streamlit_app.py`
5. Requirements: `requirements.txt` (auto-detected)
6. Deploy!

## Features

- **500 patients** with full EHR data
- **6 data tables**: patients, diagnoses, labs, medications, encounters, clinical notes
- **20 query categories** covering all your clinical questions
- **Grounded answers** with source citations
- **Streamlit UI** with patient sidebar, quick actions, and tabbed views

## Example Queries

| Category | Query |
|----------|-------|
| Demographics | `Patient P0001 age sex race chronic conditions` |
| Diagnoses | `What diagnoses has patient P0001 had and when` |
| Labs | `Patient P0001 latest lab results abnormal flags` |
| A1c Trend | `Patient P0001 A1c trend diabetes control` |
| Renal | `Patient P0001 creatinine renal dysfunction` |
| Infection | `Patient P0001 WBC infection inflammation` |
| Anemia | `Patient P0001 hemoglobin anemia` |
| Heart Failure | `Patient P0001 heart failure CHF BNP` |
| CKD | `Patient P0001 CKD creatinine` |
| Diabetes | `Patient P0001 diabetes control glucose A1c` |
| Multimorbidity | `Patient P0001 multimorbidity` |
| Discharge Notes | `Patient P0001 discharge summary` |
| Readmission Risk | `Patient P0001 readmission risk factors` |
| Handoff | `Patient P0001 handoff summary` |
| Timeline | `Patient P0001 timeline recent clinical` |
| Encounters | `Patient P0001 encounters how many most recent` |
| Meds Alignment | `Patient P0001 medications align chronic conditions` |
| 30-day Follow-up | `Patient P0001 follow-up 30 day discharge` |

## Data Files

- `data/patients.csv` - 500 patients
- `data/diagnoses.csv` - ~1,500 diagnoses
- `data/labs.csv` - ~1,500 lab results
- `data/medications.csv` - ~500 medications
- `data/encounters.csv` - ~1,500 encounters
- `data/clinical_notes.csv` - ~1,000 notes
- `data/faiss_index.bin` - Vector index for notes
- `data/chunks_meta.pkl` - Chunk metadata

## Architecture

```
Query → Orchestrator
  ├── Count queries → Direct DataFrame ops
  ├── Complex clinical → ClinicalAnalyzer (full patient profile)
  │     ├── Demographics, Diagnoses, Labs, Meds, Encounters, Notes
  │     └── Analysis: Readmission, Diabetes, Renal, HF, Anemia, Infection, Multimorbidity, Timeline, Follow-up
  └── General RAG → Planner → FAISS Retriever → Evaluator
```

## Fine-Tuning Data

`ift_dataset_alpaca.json` - 900 instruction-following examples for LLM fine-tuning (50 patients × 18 question types).
