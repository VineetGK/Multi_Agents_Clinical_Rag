import re
from pydantic import BaseModel
from typing import List, Optional

class Plan(BaseModel):
    intent: str
    rewritten_query: str
    subqueries: List[str]
    target_sources: List[str]
    patient_id_filter: Optional[str] = None

class PlannerAgent:
    def __init__(self):
        pass

    def plan_query(self, user_query: str) -> Plan:
        query_lower = user_query.lower()
        
        # Extract Patient ID using Regex (e.g., P0001)
        id_match = re.search(r'[pP]\d{4}', user_query)
        patient_id = id_match.group(0).upper() if id_match else None

        intent = "general_inquiry"
        target_sources = ["clinical_notes"]
        subqueries = [user_query]

        if "risk" in query_lower or "readmission" in query_lower:
            intent = "risk_assessment"
            target_sources.extend(["encounters", "labs", "diagnoses"])

        elif "summarize" in query_lower or "history" in query_lower:
            intent = "patient_summarization"
            target_sources.extend(["medications", "diagnoses", "labs"])

        target_sources = list(set(target_sources))

        return Plan(
            intent=intent,
            rewritten_query=user_query,
            subqueries=subqueries,
            target_sources=target_sources,
            patient_id_filter=patient_id
        )
