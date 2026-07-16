# Clinical RAG - Streamlit Cloud Deployment

## Quick Deploy to Streamlit Cloud

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Clinical RAG Streamlit app"
   git remote add origin https://github.com/YOUR_USERNAME/clinical-rag-standalone.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repo: `YOUR_USERNAME/clinical-rag-standalone`
   - Main file path: `streamlit_app.py`
   - Click "Deploy!"

3. **That's it!** Your app will be live at `https://YOUR_APP.streamlit.app`

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run streamlit_app.py
```

## Project Structure

```
clinical_rag_standalone/
├── streamlit_app.py          # Streamlit frontend (entry point)
├── requirements.txt          # Dependencies
├── DATASET_SCHEMA.md         # Dataset documentation
├── ift_dataset.jsonl         # Fine-tuning dataset (900 examples)
├── ift_dataset_alpaca.json   # Alpaca format
├── app/
│   ├── __init__.py
│   ├── orchestrator.py       # Main query routing
│   ├── clinical_analyzer.py  # Data analysis engine
│   ├── planner_agent.py      # Query planning
│   ├── retrieval_agent.py    # FAISS retrieval
│   └── evaluator_agent.py    # Answer evaluation
├── data/
│   ├── patients.csv          # 500 patients
│   ├── diagnoses.csv         # ~1,500 diagnoses
│   ├── labs.csv              # ~1,500 lab results
│   ├── medications.csv       # ~500 medications
│   ├── encounters.csv        # ~1,500 encounters
│   └── clinical_notes.csv    # ~1,000 notes
└── query.py                  # CLI query tool
```

## Requirements

All in `requirements.txt`:
```
streamlit
pandas
faiss-cpu
sentence-transformers
google-generativeai
pydantic
```

## Features

- **500 patients** with full clinical records
- **20 query categories** (demographics, diagnoses, labs, medications, encounters, notes, risk assessment, handoff summaries)
- **Quick-action buttons** for common queries
- **Tabbed patient views** (Diagnoses, Labs, Meds, Encounters)
- **Grounded answers** with source citations
- **Cached orchestrator** for fast responses

## Example Queries

The system handles queries like:
- "What is patient P0001's age, sex, race, and chronic conditions count?"
- "Show patient P0001's A1c trend and diabetes control"
- "Does patient P0001 have heart failure evidence (CHF, BNP, notes)?"
- "What are patient P0001's readmission risk factors?"
- "Generate a handoff summary for patient P0001"

## Data Privacy

⚠️ **This uses synthetic data only.** No real patient information.