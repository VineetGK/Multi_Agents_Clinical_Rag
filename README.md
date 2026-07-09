# Multi-Agent Clinical Document Assistant

## Project Overview
A portfolio-ready multi-agent Retrieval-Augmented Generation (RAG) project designed to answer healthcare questions from synthetic EHR data and clinical notes. The system utilizes three core agents (Planner, Retrieval, Evaluator) to ensure robust and grounded responses.

## Privacy Note
**All EHR data used and generated in this project is 100% synthetic.** No real patient data or Protected Health Information (PHI) is included.

## Architecture Diagram
```mermaid
graph TD
    User-->API
    API-->Orchestrator
    Orchestrator-->PlannerAgent
    Orchestrator-->RetrievalAgent
    Orchestrator-->EvaluatorAgent
    RetrievalAgent-->VectorStore[(FAISS Vector Store)]
```

## Folder Structure
```
multi-agent-rag/
├── app/
│   ├── planner_agent.py
│   ├── retrieval_agent.py
│   ├── evaluator_agent.py
│   ├── orchestrator.py
│   ├── ingest.py
│   └── api.py
├── data/
│   └── (Synthetic CSVs and FAISS Index)
├── notebooks/
│   └── demo.ipynb
├── evals/
│   ├── sample_questions.json
│   └── simple_eval.py
├── tests/
│   └── test_smoke.py
└── README.md
```

## Setup Steps for Colab
1. Upload the `multi-agent-rag` folder to your Google Colab environment.
2. Install dependencies: `!pip install pandas numpy faker fastapi uvicorn nest_asyncio pyngrok sentence-transformers faiss-cpu pydantic`
3. Set `PYTHONPATH` to include the folder: `import sys; sys.path.append('multi-agent-rag')`
4. Run the data generation and ingestion scripts if the index isn't present.
5. Start the FastAPI server using `uvicorn` and `nest_asyncio`.

## Example API Request/Response
**POST /ask**
```json
{
  "query": "What are the active medications for patient P0001?"
}
```
**Response**
```json
{
  "query": "What are the active medications for patient P0001?",
  "answer": "Based on the retrieved clinical records:...",
  "grounded": true,
  "sources": [...],
  "evaluation": {
    "issues": [],
    "suggested_retry": false
  }
}
```

## Sample Questions
- What factors suggest the patient is high risk for 30-day readmission?
- Summarize the diabetic patient's recent history and possible follow-up concerns.
- Did patient P0010 have any abnormal lab results recently?

## Limitations
- The current planner and evaluator use heuristics instead of an actual LLM to allow for lightweight, offline testing in Colab.
- Simple embedding models (`all-MiniLM-L6-v2`) are used for speed.

## Future Improvements
- Integrate LangGraph/LangChain with an actual LLM (e.g., OpenAI or Llama 3) for the Planner and Evaluator agents.
- Add a dedicated re-ranking model for better retrieval accuracy.
