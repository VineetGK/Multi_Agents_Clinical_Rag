from fastapi import FastAPI
from pydantic import BaseModel
from .orchestrator import Orchestrator

app = FastAPI(title="Multi-Agent Clinical Document Assistant")
orchestrator = Orchestrator()

class QueryRequest(BaseModel):
    query: str

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is healthy"}

@app.post("/ask")
def ask_question(req: QueryRequest):
    result = orchestrator.process_query(req.query)
    return result
